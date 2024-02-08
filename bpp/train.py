import math
import os
import random
import time

from tqdm import tqdm
import numpy as np
import scipy
import torch

from net import Net
from aco import ACO
from utils import gen_instance, gen_pyg_data, load_val_dataset

import wandb


EPS = 1e-10
T = 5


def calculate_log_pb_uniform(sols: torch.Tensor):
    # sols.shape: (batch, max_length)
    # sols are start with 0 and end with 0
    _sols = sols.detach().cpu().numpy()

    n_nodes = np.count_nonzero(_sols, axis=1)
    _pi2 = _sols[:, 1:] - _sols[:, :-1]
    n_routes = np.count_nonzero(_pi2, axis=1) - n_nodes
    log_p_b_outer = torch.tensor(-scipy.special.gammaln(n_routes + 1), device=sols.device)

    zero_indices = [np.where(sol == 0)[0] for sol in _sols]
    inner_list = [-scipy.special.gammaln(zero_idx[1:] - zero_idx[:-1]).sum() for zero_idx in zero_indices]
    log_p_b_inner = torch.tensor(inner_list, device=sols.device)

    return log_p_b_outer + log_p_b_inner


def train_instance(
    model, optimizer, batch_size, n_nodes, n_ants, invtemp=1.0, guided_exploration=False, beta=5000.0, topk=5, it=0
):
    model.train()
    ##################################################
    # wandb
    _train_mean_cost = 0.0
    _train_min_cost = 0.0
    _train_mean_cost_ls = 0.0
    _train_min_cost_ls = 0.0
    _train_entropy = 0.0
    _logZ_mean = torch.tensor(0.0, device=DEVICE)
    _logZ_nls_mean = torch.tensor(0.0, device=DEVICE)
    ##################################################
    sum_loss = torch.tensor(0.0, device=DEVICE)
    sum_loss_ls = torch.tensor(0.0, device=DEVICE)

    for _ in range(batch_size):
        demands = gen_instance(n_nodes, DEVICE)
        pyg_data = gen_pyg_data(demands, DEVICE)

        heu_vec, logZs = model(pyg_data, return_logZ=True)
        if guided_exploration:
            logZ, logZ_ls = logZs
        else:
            logZ = logZs[0]
        heu_mat = heu_vec.reshape(n_nodes + 1, n_nodes + 1) + EPS
        # heu_mat = (heu_vec / (heu_vec.min() + EPS) + EPS).reshape(n_nodes + 1, n_nodes + 1)

        aco = ACO(demands, n_ants, heuristic=heu_mat, device=DEVICE)

        objs, log_probs, sols = aco.sample(invtemp=invtemp, return_sol=True)  # type: ignore
        baseline = objs.mean()

        forward_flow = log_probs.sum(0) + logZ.expand(n_ants)  # type: ignore
        backward_flow = calculate_log_pb_uniform(sols.T) + (objs - baseline).detach() * beta
        tb_loss = torch.pow(forward_flow - backward_flow, 2).mean()
        sum_loss += tb_loss

        ##################################################
        # Wandb
        _train_mean_cost += (-baseline).item()
        _train_min_cost += (-objs).min().item()
        ##################################################

        if guided_exploration:
            # TopK guided exploration (Note that this does not directly refine the solution)
            aco.update_pheromone(sols, objs)

            with torch.no_grad():
                objs_K, _, sols_K = aco.sample(return_sol=True, K=topk)  # type: ignore

            _, idx = objs_K.topk(topk, largest=True)
            rand_idx = torch.randint(0, len(objs_K), (n_ants - topk,), device=DEVICE)
            sols_best = sols_K[:, idx]
            sols_rand = sols_K[:, rand_idx]
            sols_ls = torch.cat([sols_best, sols_rand], dim=1)

            aco.pheromone = torch.ones(n_nodes + 1, n_nodes + 1, device=DEVICE)  # reset pheromone before calculating log_probs
            _, log_probs_ls = aco.gen_sol(require_prob=True, sols=sols_ls)

            objs_best = objs_K[idx]
            objs_rand = objs_K[rand_idx]
            objs_ls = torch.cat([objs_best, objs_rand], dim=0)
            baseline_ls = objs_ls.mean()

            forward_flow_ls = log_probs_ls.sum(0) + logZ_ls.expand(n_ants)  # type: ignore
            backward_flow_ls = calculate_log_pb_uniform(sols.T) + (objs_ls - baseline_ls).detach() * beta
            tb_loss_ls = torch.pow(forward_flow_ls - backward_flow_ls, 2).mean()
            sum_loss_ls += tb_loss_ls

            ##################################################
            # Wandb
            _train_mean_cost_ls += (-baseline_ls).item()
            _train_min_cost_ls += (-objs_ls).min().item()
            ##################################################

        ##################################################
        # Wandb
        normed_heumat = heu_mat / heu_mat.sum(dim=1, keepdim=True)
        entropy = -(normed_heumat * torch.log(normed_heumat)).sum(dim=1).mean()
        _train_entropy += entropy.item()

        _logZ_mean += logZ
        if guided_exploration:
            _logZ_nls_mean += logZ_ls  # type: ignore
        ##################################################

    sum_loss /= batch_size
    sum_loss_ls /= batch_size
    loss = sum_loss + sum_loss_ls

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=3.0, norm_type=2)  # type: ignore
    optimizer.step()

    ##################################################
    if USE_WANDB:
        wandb.log(
            {
                "train_mean_cost": _train_mean_cost / batch_size,
                "train_min_cost": _train_min_cost / batch_size,
                "train_mean_cost_ls": _train_mean_cost_ls / batch_size,
                "train_min_cost_ls": _train_min_cost_ls / batch_size,
                "train_entropy": _train_entropy / batch_size,
                "train_loss": sum_loss.item(),
                "train_loss_ls": sum_loss_ls.item(),
                "invtemp": invtemp,
                "logZ": _logZ_mean.item() / batch_size,
                "logZ_nls": _logZ_nls_mean.item() / batch_size,
                "beta": beta,
            },
            step=it,
        )
    ##################################################


def infer_instance(model, instance, n_ants):
    model.eval()
    demands = instance
    pyg_data = gen_pyg_data(demands, DEVICE)
    n = demands.size(0)

    heu_vec = model(pyg_data)
    heu_mat = heu_vec.reshape(n, n) + EPS
    # heu_mat = (heu_vec / (heu_vec.min() + EPS) + EPS).reshape(n + 1, n + 1)

    aco = ACO(demands, n_ants, heuristic=heu_mat, device=DEVICE)

    objs, _ = aco.sample()  # type: ignore
    baseline = objs.mean().item()
    best_sample_obj = objs.max().item()

    best_aco_1, diversity_1 = aco.run(1)
    best_aco_T, diversity_T = aco.run(T - 1)
    best_aco_1, best_aco_T = best_aco_1.item(), best_aco_T.item()  # type: ignore

    return np.array([baseline, best_sample_obj, best_aco_1, best_aco_T, diversity_1, diversity_T])


def train_epoch(
    n_nodes,
    n_ants,
    epoch,
    steps_per_epoch,
    net,
    optimizer,
    batch_size,
    invtemp=1.0,
    guided_exploration=False,
    beta=5000.0,
    topk=5,
):
    for i in tqdm(range(steps_per_epoch), desc="Train"):
        it = (epoch - 1) * steps_per_epoch + i
        train_instance(net, optimizer, batch_size, n_nodes, n_ants, invtemp, guided_exploration, beta, topk, it)


@torch.no_grad()
def validation(val_list, n_ants, net, epoch, steps_per_epoch):
    stats = []
    for _, instance in tqdm(enumerate(val_list)):
        stats.append(infer_instance(net, instance, n_ants))
    avg_stats = [i.item() for i in np.stack(stats).mean(0)]

    ##################################################
    print(f"epoch {epoch}:", avg_stats)
    # Wandb
    if USE_WANDB:
        wandb.log(
            {
                "val_baseline": -avg_stats[0],
                "val_best_sample_cost": -avg_stats[1],
                "val_best_aco_1": -avg_stats[2],
                "val_best_aco_T": -avg_stats[3],
                "val_diversity_1": avg_stats[4],
                "val_diversity_T": avg_stats[5],
                "epoch": epoch,
            },
            step=epoch * steps_per_epoch,
        )
    ##################################################

    return avg_stats[3]


def train(
        n_nodes,
        n_ants,
        n_val_ants,
        steps_per_epoch,
        epochs,
        lr=1e-4,
        batch_size=3,
        val_size=None,
        val_interval=5,
        pretrained=None,
        savepath="../pretrained/bpp",
        run_name="",
        invtemp_schedule_params=(0.8, 1.0, 5),  # (invtemp_min, invtemp_max, invtemp_flat_epochs)
        guided_exploration=False,
        beta_schedule_params=(5, 50, 5),  # (beta_min, beta_max, beta_flat_epochs)
        topk=5,
    ):
    savepath = os.path.join(savepath, str(n_nodes), run_name)
    os.makedirs(savepath, exist_ok=True)

    net = Net(gfn=True, Z_out_dim=2 if guided_exploration else 1).to(DEVICE)
    if pretrained is not None:
        raise NotImplementedError("Need some more work to load pretrained model")
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr * 0.1)

    val_list = load_val_dataset(n_nodes, DEVICE)
    val_list = val_list[:(val_size or len(val_list))]

    best_result = validation(val_list, n_val_ants, net, 0, steps_per_epoch)

    sum_time = 0
    for epoch in range(1, epochs + 1):
        # Heatmap Inverse Temperature Schedule
        invtemp_min, invtemp_max, invtemp_flat_epochs = invtemp_schedule_params
        invtemp = invtemp_min + (invtemp_max - invtemp_min) * min((epoch - 1) / (epochs - invtemp_flat_epochs), 1.0)

        # Beta Schedule
        beta_min, beta_max, beta_flat_epochs = beta_schedule_params
        beta = beta_min + (beta_max - beta_min) * min(math.log(epoch) / math.log(epochs - beta_flat_epochs), 1.0)

        start = time.time()
        train_epoch(
            n_nodes,
            n_ants,
            epoch,
            steps_per_epoch,
            net,
            optimizer,
            batch_size,
            invtemp,
            guided_exploration,
            beta,
            topk,
        )
        sum_time += time.time() - start

        if epoch % val_interval == 0:
            curr_result = validation(val_list, n_val_ants, net, epoch, steps_per_epoch)
            if curr_result > best_result:
                torch.save(net.state_dict(), os.path.join(savepath, "best.pt"))
                best_result = curr_result

            torch.save(net.state_dict(), os.path.join(savepath, f"{epoch}.pt"))

        scheduler.step()

    print('total training duration:', sum_time)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", metavar='N', type=int, help="Problem scale")
    parser.add_argument("-l", "--lr", metavar='η', type=float, default=3e-4, help="Learning rate")
    parser.add_argument("-d", "--device", type=str,
                        default=("cuda:0" if torch.cuda.is_available() else "cpu"), 
                        help="The device to train NNs")
    parser.add_argument("-p", "--pretrained", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("-a", "--ants", type=int, default=30, help="Number of ants (in ACO algorithm)")
    parser.add_argument("-va", "--val_ants", type=int, default=100, help="Number of ants for validation")
    parser.add_argument("-b", "--batch_size", type=int, default=5, help="Batch size")
    parser.add_argument("-s", "--steps", type=int, default=20, help="Steps per epoch")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Epochs to run")
    parser.add_argument("-t", "--val_size", type=int, default=10, help="Number of instances for validation")
    parser.add_argument("-o", "--output", type=str, default="../pretrained/bpp",
                        help="The directory to store checkpoints")
    parser.add_argument("--val_interval", type=int, default=5, help="The interval to validate model")
    ### Logging
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--run_name", type=str, default="", help="Run name")
    ### invtemp
    parser.add_argument("--invtemp_min", type=float, default=0.8, help='Inverse temperature min for GFACS')
    parser.add_argument("--invtemp_max", type=float, default=1.0, help='Inverse temperature max for GFACS')
    parser.add_argument("--invtemp_flat_epochs", type=int, default=5, help='Inverse temperature glat rpochs for GFACS')
    ### Top-k guided exploration
    parser.add_argument("--disable_guided_exp", action='store_true', help='Disable guided exploration for GFACS')
    parser.add_argument("--topk", type=int, default=5, help="TopK for guided exploration")
    ### GFACS
    parser.add_argument("--beta_min", type=float, default=None, help='Beta Min for GFACS')
    parser.add_argument("--beta_max", type=float, default=None, help='Beta Max for GFACS')
    parser.add_argument("--beta_flat_epochs", type=int, default=5, help='Beta Flat Epochs for GFACS')
    ### Seed
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    if args.beta_min is None:
        beta_min_map = {120: 500, 250: 500, 500: 500}
        args.beta_min = beta_min_map[args.nodes]
    if args.beta_max is None:
        beta_max_map = {120: 5000, 250: 5000, 500: 5000}
        args.beta_max = beta_max_map[args.nodes]

    DEVICE = args.device if torch.cuda.is_available() else "cpu"
    USE_WANDB = not args.disable_wandb

    # seed everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ##################################################
    # wandb
    run_name = f"[{args.run_name}]" if args.run_name else ""
    run_name += f"bpp{args.nodes}_sd{args.seed}"
    pretrained_name = (
        args.pretrained.replace("../pretrained/bpp/", "").replace("/", "_").replace(".pt", "")
        if args.pretrained is not None else None
    )
    run_name += f"{'' if pretrained_name is None else '_fromckpt-'+pretrained_name}"
    if USE_WANDB:
        wandb.init(project="gfacs-bpp", name=run_name)
        wandb.config.update(args)
        wandb.config.update({"T": T, "model": "GFACS"})
    ##################################################

    train(
        args.nodes,
        args.ants,
        args.val_ants,
        args.steps,
        args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_size=args.val_size,
        val_interval=args.val_interval,
        pretrained=args.pretrained,
        savepath=args.output,
        run_name=run_name,
        invtemp_schedule_params=(args.invtemp_min, args.invtemp_max, args.invtemp_flat_epochs),
        guided_exploration=(not args.disable_guided_exp),
        beta_schedule_params=(args.beta_min, args.beta_max, args.beta_flat_epochs),
        topk=args.topk,
    )
