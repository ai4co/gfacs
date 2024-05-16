import os
import random
import time

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

from net import Net
from aco import ACO, get_subroutes
from utils import load_test_dataset

from typing import Tuple, List


EPS = 1e-10


def validate_route(
    distance: torch.Tensor, demands: torch.Tensor, windows: torch.Tensor, routes: List[torch.Tensor]
) -> Tuple[bool, float]:
    raise NotImplementedError
    length = 0.0
    valid = True
    visited = {0}
    for r in routes:
        d = demands[r].sum().item()
        if d>1.000001:
            valid = False
        length += distance[r[:-1], r[1:]].sum().item()
        for i in r:
            i = i.item()
            if i<0 or i>=distance.size(0):
                valid = False
            else:
                visited.add(i)  # type: ignore
    if len(visited) != distance.size(0):
        valid = False
    return valid, length


@torch.no_grad()
def infer_instance(
    model, pyg_data, demands, distances, positions, windows, n_ants, t_aco_diff, local_search_params=None
):
    if model is not None:
        model.eval()
        heu_vec = model(pyg_data)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS
    else:
        heu_mat = None

    aco = ACO(
        n_ants=n_ants,
        heuristic=heu_mat.cpu() if heu_mat is not None else heu_mat,
        demands=demands.cpu(),
        distances=distances.cpu(),
        windows = windows.cpu(),
        device='cpu',
        local_search=True,
        local_search_params=local_search_params,
        positions=positions.cpu(),
    )

    results = torch.zeros(size=(len(t_aco_diff),))
    diversities = torch.zeros(size=(len(t_aco_diff),))
    for i, t in enumerate(t_aco_diff):
        results[i], diversities[i] = aco.run(t)
        # path = get_subroutes(aco.shortest_path)
        # valid, length = validate_route(distances, demands, windows, path)
        # assert (length - results[i].item()) < 1e-5  # FIXME: remove this line
        # if valid is False:
        #    print("invalid solution.")
    return results, diversities


@torch.no_grad()
def test(dataset, model, n_ants, t_aco, local_search_params):
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i + 1] - _t_aco[i] for i in range(len(_t_aco) - 1)]

    sum_results = torch.zeros(size=(len(t_aco_diff),))
    sum_diversities = torch.zeros(size=(len(t_aco_diff),))
    start = time.time()
    for pyg_data, demands, distances, positions, windows in tqdm(dataset, dynamic_ncols=True):
        results, diversities = infer_instance(
            model, pyg_data, demands, distances, positions, windows, n_ants, t_aco_diff, local_search_params
        )
        sum_results += results
        sum_diversities += diversities
    end = time.time()
    return sum_results / len(dataset), sum_diversities / len(dataset), end - start


def main(
    ckpt_path,
    n_nodes,
    k_sparse,
    size=None,
    n_ants=100,
    n_iter=10,
    guided_exploration=False,
    seed=0,
    local_search_params=None,
):
    test_list = load_test_dataset(n_nodes, k_sparse, DEVICE, TAM)
    test_list = test_list[:(size or len(test_list))]

    t_aco = list(range(1, n_iter + 1))
    print("problem scale:", n_nodes)
    print("checkpoint:", ckpt_path)
    print("number of instances:", size)
    print("device:", 'cpu' if DEVICE == 'cpu' else DEVICE+"+cpu" )
    print("n_ants:", n_ants)
    print("seed:", seed)

    if ckpt_path is not None:
        net = Net(gfn=True, Z_out_dim=2 if guided_exploration else 1).to(DEVICE)
        net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    else:
        net = None
    avg_cost, avg_diversity, duration = test(test_list, net, n_ants, t_aco, local_search_params)
    print('total duration: ', duration)
    for i, t in enumerate(t_aco):
        print(f"T={t}, avg. cost {avg_cost[i]}, avg. diversity {avg_diversity[i]}")

    # Save result in directory that contains model_file
    filename = os.path.splitext(os.path.basename(ckpt_path))[0] if ckpt_path is not None else 'none'
    dirname = os.path.dirname(ckpt_path) if ckpt_path is not None else f'../pretrained/cvrptw_nls/{args.nodes}/no_model'
    os.makedirs(dirname, exist_ok=True)

    result_filename = f"test_result_ckpt{filename}-cvrptw{n_nodes}-ninst{size}-nants{n_ants}-niter{n_iter}-seed{seed}"
    result_file = os.path.join(dirname, result_filename + ".txt")
    with open(result_file, "w") as f:
        f.write(f"problem scale: {n_nodes}\n")
        f.write(f"checkpoint: {ckpt_path}\n")
        f.write(f"number of instances: {len(test_list)}\n")
        f.write(f"device: {'cpu' if DEVICE == 'cpu' else DEVICE+'+cpu'}\n")
        f.write(f"n_ants: {n_ants}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"total duration: {duration}\n")
        for i, t in enumerate(t_aco):
            f.write(f"T={t}, avg. cost {avg_cost[i]}, avg. diversity {avg_diversity[i]}\n")

    results = pd.DataFrame(columns=['T', 'avg_cost', 'avg_diversity'])
    results['T'] = t_aco
    results['avg_cost'] = avg_cost
    results['avg_diversity'] = avg_diversity
    results.to_csv(os.path.join(dirname, result_filename + ".csv"), index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", type=int, help="Problem scale")
    parser.add_argument("-k", "--k_sparse", type=int, default=None, help="k_sparse")
    parser.add_argument("-p", "--path", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("-i", "--n_iter", type=int, default=10, help="Number of iterations of ACO to run")
    parser.add_argument("-n", "--n_ants", type=int, default=100, help="Number of ants")
    parser.add_argument("-d", "--device", type=str,
                        default=("cuda:0" if torch.cuda.is_available() else "cpu"), 
                        help="The device to train NNs")
    parser.add_argument("-s", "--size", type=int, default=None, help="Number of instances to test")
    ### GFACS
    parser.add_argument("--disable_guided_exp", action='store_true', help='True for model w/o guided exploration.')
    ### Seed
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    ### Dataset
    parser.add_argument("--tam", action="store_true", help="Use TAM dataset")
    ### LocalSearchParams
    parser.add_argument("--n_cpus", type=int, default=1, help="Number of cpus to use")
    parser.add_argument("--max_trials", type=int, default=10, help="Number of iterations to perform")
    parser.add_argument("--load_penalty", type=int, default=20, help="Initial load_penalty in training phase")
    parser.add_argument("--tw_penalty", type=int, default=20, help="Initial tw_penalty in training phase")
    parser.add_argument("--nb_granular", type=int, default=None, help="Granularity of neighbourhood search")

    args = parser.parse_args()

    if args.k_sparse is None:
        args.k_sparse = args.nodes // 5
    if args.nb_granular is None:
        args.nb_granular = args.nodes // 5

    DEVICE = args.device if torch.cuda.is_available() else 'cpu'
    TAM = args.tam

    # seed everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.path is not None and not os.path.isfile(args.path):
        print(f"Checkpoint file '{args.path}' not found!")
        exit(1)

    local_search_params = {
        "n_cpus": args.n_cpus,
        "max_trials": args.max_trials,
        "neighbourhood_params": {"nb_granular": args.nb_granular},
        "cost_evaluator_params": {"load_penalty": args.load_penalty, "tw_penalty": args.tw_penalty},
    }

    main(
        args.path,
        args.nodes,
        args.k_sparse,
        args.size,
        args.n_ants,
        args.n_iter,
        not args.disable_guided_exp,
        args.seed,
        local_search_params,
    )
