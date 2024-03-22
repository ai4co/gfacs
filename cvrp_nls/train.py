import os
import math
import random
import time

from tqdm import tqdm
import scipy
import numpy as np
import torch

from net import Net
from aco import ACO
from utils import gen_pyg_data, load_val_dataset, gen_instance

import wandb


EPS = 1e-10
T = 5


def calculate_log_pb_uniform(paths: torch.Tensor):
    # paths.shape: (batch, max_tour_length)
    # paths are start with 0 and end with 0

    _pi1 = paths.detach().cpu().numpy()
    # shape: (batch, max_tour_length)

    n_nodes = np.count_nonzero(_pi1, axis=1)
    _pi2 = _pi1[:, 1:] - _pi1[:, :-1]
    n_routes = np.count_nonzero(_pi2, axis=1) - n_nodes
    _pi3 = _pi1[:, 2:] - _pi1[:, :-2]
    n_multinode_routes = np.count_nonzero(_pi3, axis=1) - n_nodes
    log_b_p = - scipy.special.gammaln(n_routes + 1) - n_multinode_routes * math.log(2)

    return torch.from_numpy(log_b_p).to(paths.device)


def train_instance(
        model,
        optimizer,
        data,
        n_ants,
        cost_w=1.0,
        invtemp=1.0,
        guided_exploration=False,
        shared_energy_norm=False,
        beta=100.0,
        it=0
    ):
    model.train()

    ##################################################
    # wandb
    _train_mean_cost = 0.0
    _train_min_cost = 0.0
    _train_mean_cost_nls = 0.0
    _train_min_cost_nls = 0.0
    _train_entropy = 0.0
    _logZ_mean = torch.tensor(0.0, device=DEVICE)
    _logZ_nls_mean = torch.tensor(0.0, device=DEVICE)
    ##################################################
    sum_loss = torch.tensor(0.0, device=DEVICE)
    sum_loss_nls = torch.tensor(0.0, device=DEVICE)
    count = 0

    for pyg_data, demands, distances, positions in data:
        heu_vec, logZ = model(pyg_data, return_logZ=True)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS
        if guided_exploration:
            logZ, logZ_nls = logZ
        else:
            logZ = logZ[0]

        aco = ACO(
            distances=distances.to(DEVICE),
            demand=demands.to(DEVICE),
            n_ants=n_ants,
            heuristic=heu_mat.to(DEVICE),
            device=DEVICE,
            swapstar=True,
            positions=positions
        )
        
        if guided_exploration:
            costs_nls, log_probs, paths_nls, costs_raw, paths = aco.sample_nls(invtemp)
            advantage_raw = (costs_raw - (costs_raw.mean() if shared_energy_norm else 0.0))
            advantage_nls = (costs_nls - (costs_nls.mean() if shared_energy_norm else 0.0))
            weighted_advantage = cost_w * advantage_nls + (1 - cost_w) * advantage_raw
        else:
            costs_raw, log_probs, paths = aco.sample(invtemp)
            advantage_raw = (costs_raw - (costs_raw.mean() if shared_energy_norm else 0.0))
            weighted_advantage = advantage_raw

        ##################################################
        # Loss from paths before local search
        forward_flow = log_probs.sum(0) + logZ.expand(n_ants)
        backward_flow = calculate_log_pb_uniform(paths.T) - weighted_advantage.detach() * beta
        tb_loss = torch.pow(forward_flow - backward_flow, 2).mean()
        sum_loss += tb_loss

        ##################################################
        # Loss from paths after local search
        if guided_exploration:
            n_ants_nls = n_ants
            _, log_probs_nls, feasible_idx = aco.gen_path(  # type: ignore
                require_prob=True,
                invtemp=1.0,  # invtemp is 1.0 here, otherwise gradients from offpolicy data will be overestimated
                paths=paths_nls,  # type: ignore
            )
            if len(feasible_idx) < n_ants:  # type: ignore
                paths_nls = paths_nls[:, feasible_idx]  # type: ignore
                costs_nls = costs_nls[feasible_idx]  # type: ignore
                advantage_nls = advantage_nls[feasible_idx]  # type: ignore
                n_ants_nls = len(feasible_idx)  # type: ignore

            forward_flow_nls = log_probs_nls.sum(0) + logZ_nls.expand(n_ants_nls)  # type: ignore
            backward_flow_nls = calculate_log_pb_uniform(paths_nls.T) - advantage_nls.detach() * beta  # type: ignore
            tb_loss_nls = torch.pow(forward_flow_nls - backward_flow_nls, 2).mean()
            sum_loss_nls += tb_loss_nls

        count += 1

        ##################################################
        # wandb
        if USE_WANDB:
            _train_mean_cost += costs_raw.mean().item()
            _train_min_cost += costs_raw.min().item()

            normed_heumat = heu_mat / heu_mat.sum(dim=1, keepdim=True)
            entropy = -(normed_heumat * torch.log(normed_heumat)).sum(dim=1).mean()
            _train_entropy += entropy.item()

            _logZ_mean += logZ
            if guided_exploration:
                _train_mean_cost_nls += costs_nls.mean().item()
                _train_min_cost_nls += costs_nls.min().item()
                _logZ_nls_mean += logZ_nls
        ##################################################

    sum_loss = sum_loss / count
    sum_loss_nls = sum_loss_nls / count if guided_exploration else torch.tensor(0.0, device=DEVICE)
    loss = sum_loss + sum_loss_nls

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=3.0, norm_type=2)  # type: ignore
    optimizer.step()

    ##################################################
    # wandb
    if USE_WANDB:
        wandb.log(
            {
                "train_mean_cost": _train_mean_cost / count,
                "train_min_cost": _train_min_cost / count,
                "train_mean_cost_nls": _train_mean_cost_nls / count,
                "train_min_cost_nls": _train_min_cost_nls / count,
                "train_entropy": _train_entropy / count,
                "train_loss": sum_loss.item(),
                "train_loss_nls": sum_loss_nls.item(),
                "cost_w": cost_w,
                "invtemp": invtemp,
                "logZ": _logZ_mean.item() / count,
                "logZ_nls": _logZ_nls_mean.item() / count,
                "beta": beta,
            },
            step=it,
        )
    ##################################################


def infer_instance(model, pyg_data, demands, distances, positions, n_ants):
    model.eval()
    heu_vec = model(pyg_data)
    heu_mat = model.reshape(pyg_data, heu_vec) + EPS

    aco = ACO(
        distances=distances,
        demand=demands,
        n_ants=n_ants,
        heuristic=heu_mat,
        device=DEVICE,
        swapstar=True,
        positions=positions,
    )

    costs, _, _ = aco.sample()
    baseline = costs.mean().item()
    best_sample_cost = costs.min().item()
    best_aco_1, diversity_1 = aco.run(n_iterations=1)
    best_aco_T, diversity_T = aco.run(n_iterations=T - 1)
    return np.array([baseline, best_sample_cost, best_aco_1, best_aco_T, diversity_1, diversity_T])


def generate_traindata(count, n_node, k_sparse):
    for _ in range(count):
        instance = demands, distance, _ = gen_instance(n_node, DEVICE, TAM)
        yield gen_pyg_data(demands, distance, DEVICE, k_sparse), *instance


def train_epoch(
    n_node,
    k_sparse,
    n_ants,
    epoch,
    steps_per_epoch,
    net,
    optimizer,
    batch_size=1,
    cost_w=0.98,
    invtemp=1.0,
    guided_exploration=False,
    shared_energy_norm=False,
    beta=100.0,
):
    for i in tqdm(range(steps_per_epoch), desc="Train"):
        it = (epoch - 1) * steps_per_epoch + i
        data = generate_traindata(batch_size, n_node, k_sparse)
        train_instance(net, optimizer, data, n_ants, cost_w, invtemp, guided_exploration, shared_energy_norm, beta, it)


@torch.no_grad()
def validation(val_list, n_ants, net, epoch, steps_per_epoch):
    stats = []
    for data, demands, distances, positions in tqdm(val_list, desc="Val"):
        stats.append(infer_instance(net, data, demands, distances, positions, n_ants))
    avg_stats = [i.item() for i in np.stack(stats).mean(0)]

    ##################################################
    print(f"epoch {epoch}:", avg_stats)
    # Wandb
    if USE_WANDB:
        wandb.log(
            {
                "val_baseline": avg_stats[0],
                "val_best_sample_cost": avg_stats[1],
                "val_best_aco_1": avg_stats[2],
                "val_best_aco_T": avg_stats[3],
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
        k_sparse,
        n_ants,
        n_val_ants,
        steps_per_epoch,
        epochs,
        lr=1e-4,
        batch_size=3,
        val_size=None,
        val_interval=5,
        pretrained=None,
        savepath="../pretrained/cvrp_nls",
        run_name="",
        cost_w_schedule_params=(0.5, 1.0, 5),  # (cost_w_min, cost_w_max, cost_w_flat_epochs)
        invtemp_schedule_params=(0.8, 1.0, 5),  # (invtemp_min, invtemp_max, invtemp_flat_epochs)
        guided_exploration=False,
        shared_energy_norm=False,
        beta_schedule_params=(50, 500, 5),  # (beta_min, beta_max, beta_flat_epochs)
    ):
    savepath = os.path.join(savepath, str(n_nodes), run_name)
    os.makedirs(savepath, exist_ok=True)

    net = Net(gfn=True, Z_out_dim=2 if guided_exploration else 1).to(DEVICE)
    if pretrained:
        net.load_state_dict(torch.load(pretrained, map_location=DEVICE))
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr * 0.1)

    val_list = load_val_dataset(n_nodes, k_sparse, DEVICE, TAM)
    val_list = val_list[:(val_size or len(val_list))]

    best_result = validation(val_list, n_val_ants, net, 0, steps_per_epoch)

    sum_time = 0
    for epoch in range(1, epochs + 1):
        # Cost Weight Schedule
        cost_w_min, cost_w_max, cost_w_flat_epochs = cost_w_schedule_params
        cost_w = cost_w_min + (cost_w_max - cost_w_min) * min((epoch - 1) / (epochs - cost_w_flat_epochs), 1.0)

        # Heatmap Inverse Temperature Schedule
        invtemp_min, invtemp_max, invtemp_flat_epochs = invtemp_schedule_params
        invtemp = invtemp_min + (invtemp_max - invtemp_min) * min((epoch - 1) / (epochs - invtemp_flat_epochs), 1.0)

        # Beta Schedule
        beta_min, beta_max, beta_flat_epochs = beta_schedule_params
        beta = beta_min + (beta_max - beta_min) * min(math.log(epoch) / math.log(epochs - beta_flat_epochs), 1.0)

        start = time.time()
        train_epoch(
            n_nodes,
            k_sparse,
            n_ants,
            epoch,
            steps_per_epoch,
            net,
            optimizer,
            batch_size,
            cost_w,
            invtemp,
            guided_exploration,
            shared_energy_norm,
            beta,
        )
        sum_time += time.time() - start

        if epoch % val_interval == 0:
            curr_result = validation(val_list, n_val_ants, net, epoch, steps_per_epoch)
            if curr_result < best_result:
                torch.save(net.state_dict(), os.path.join(savepath, f'best.pt'))
                best_result = curr_result

            torch.save(net.state_dict(), os.path.join(savepath, f"{epoch}.pt"))

        scheduler.step()

    print('\ntotal training duration:', sum_time)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", metavar='N', type=int, help="Problem scale")
    parser.add_argument("-k", "--k_sparse", type=int, default=None, help="k_sparse")
    parser.add_argument("-l", "--lr", metavar='η', type=float, default=5e-4, help="Learning rate")
    parser.add_argument("-d", "--device", type=str, 
                        default=("cuda:0" if torch.cuda.is_available() else "cpu"), 
                        help="The device to train NNs")
    parser.add_argument("-p", "--pretrained", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("-a", "--ants", type=int, default=20, help="Number of ants (in ACO algorithm)")
    parser.add_argument("-va", "--val_ants", type=int, default=100, help="Number of ants for validation")
    parser.add_argument("-b", "--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("-s", "--steps", type=int, default=20, help="Steps per epoch")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="Epochs to run")
    parser.add_argument("-v", "--val_size", type=int, default=5, help="Number of instances for validation")
    parser.add_argument("-o", "--output", type=str, default="../pretrained/cvrp_nls",
                        help="The directory to store checkpoints")
    parser.add_argument("--val_interval", type=int, default=5, help="The interval to validate model")
    ### Logging
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--run_name", type=str, default="", help="Run name")
    ### cost_w
    parser.add_argument("--cost_w_min", type=float, default=0.5, help='Cost Weight Min for GFACS')
    parser.add_argument("--cost_w_max", type=float, default=1.0, help='Cost Weight Max for GFACS')
    parser.add_argument("--cost_w_flat_epochs", type=int, default=5, help='Cost Weight Flat Epochs for GFACS')
    ### invtemp
    parser.add_argument("--invtemp_min", type=float, default=0.8, help='Inverse Temperature Min for GFACS')
    parser.add_argument("--invtemp_max", type=float, default=1.0, help='Inverse Temperature Max for GFACS')
    parser.add_argument("--invtemp_flat_epochs", type=int, default=5, help='Inverse Temperature Flat Epochs for GFACS')
    ### GFACS
    parser.add_argument("--disable_guided_exp", action='store_true', help='Disable guided exploration for GFACS')
    parser.add_argument("--disable_shared_energy_norm", action='store_true', help='Disable shared energy normalization for GFACS')
    parser.add_argument("--beta_min", type=float, default=None, help='Beta Min for GFACS')
    parser.add_argument("--beta_max", type=float, default=None, help='Beta Max for GFACS')
    parser.add_argument("--beta_flat_epochs", type=int, default=5, help='Beta Flat Epochs for GFACS')
    ### Seed
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    ### Dataset
    parser.add_argument("--tam", action="store_true", help="Use TAM dataset")

    args = parser.parse_args()

    if args.k_sparse is None:
        args.k_sparse = args.nodes // 5

    if args.beta_min is None:
        beta_min_map = {100: 200, 200: 500, 400: 500, 500: 500, 1000: 2000 if args.pretrained is None else 500}
        args.beta_min = beta_min_map[args.nodes]
    if args.beta_max is None:
        beta_max_map = {100: 1000, 200: 2000, 400: 2000, 500: 2000, 1000: 2000}
        args.beta_max = beta_max_map[args.nodes]

    DEVICE = args.device if torch.cuda.is_available() else "cpu"
    USE_WANDB = not args.disable_wandb
    TAM = args.tam

    # seed everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ##################################################
    # wandb
    run_name = f"[{args.run_name}]" if args.run_name else ""
    run_name += f"cvrp{args.nodes}{'-tam' if args.tam else ''}_sd{args.seed}"
    pretrained_name = (
        args.pretrained.replace("../pretrained/cvrp_nls/", "").replace("/", "_").replace(".pt", "")
        if args.pretrained is not None else None
    )
    run_name += f"{'' if pretrained_name is None else '_fromckpt-'+pretrained_name}"
    if USE_WANDB:
        wandb.init(project=f"gfacs-cvrp_nls", name=run_name)
        wandb.config.update(args)
        wandb.config.update({"T": T, "model": "GFACS", "tam": TAM})
    ##################################################

    train(
        args.nodes,
        args.k_sparse,
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
        cost_w_schedule_params=(args.cost_w_min, args.cost_w_max, args.cost_w_flat_epochs),
        invtemp_schedule_params=(args.invtemp_min, args.invtemp_max, args.invtemp_flat_epochs),
        guided_exploration=(not args.disable_guided_exp),
        shared_energy_norm=(not args.disable_shared_energy_norm),
        beta_schedule_params=(args.beta_min, args.beta_max, args.beta_flat_epochs),
    )
