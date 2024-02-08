import os
import time
import random
import numpy as np
import torch
import pandas as pd

from net import Net
from aco import ACO
from utils import load_test_dataset

from tqdm import tqdm


EPS = 1e-10


def infer_instance(model, instance, n_ants, t_aco_diff):
    pyg_data, due_time, weights, processing_time = instance
    if model:
        model.eval()
        heu_vec = model(pyg_data)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS
        # heu_mat = heu_mat / (heu_vec.min() + EPS) + EPS
    else:
        heu_mat = None

    aco = ACO(due_time, weights, processing_time, n_ants, heuristic=heu_mat, device=DEVICE)

    results = torch.zeros(size=(len(t_aco_diff),), device=DEVICE)
    diversities = torch.zeros(size=(len(t_aco_diff),))
    for i, t in enumerate(t_aco_diff):
        results[i], diversities[i] = aco.run(t)
    return results, diversities


@torch.no_grad()
def test(dataset, model, n_ants, t_aco):
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i + 1] - _t_aco[i] for i in range(len(_t_aco) - 1)]

    sum_results = torch.zeros(size=(len(t_aco_diff),))
    sum_diversities = torch.zeros(size=(len(t_aco_diff),))
    start = time.time()
    for instance in tqdm(dataset):
        results, diversities = infer_instance(model, instance, n_ants, t_aco_diff)
        sum_results += results.cpu()
        sum_diversities += diversities
    end = time.time()

    return sum_results / len(dataset), sum_diversities / len(dataset), end - start


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", type=int, help="Problem scale")
    parser.add_argument("-p", "--path", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("-i", "--n_iter", type=int, default=10, help="Number of iterations of ACO to run")
    parser.add_argument("-n", "--n_ants", type=int, default=100, help="Number of ants")
    parser.add_argument("-d", "--device", type=str,
                        default=("cuda:0" if torch.cuda.is_available() else "cpu"), 
                        help="The device to train NNs")
    parser.add_argument("-s", "--size", type=int, default=100, help="Number of instances to test")
    ### GFACS
    parser.add_argument("--disable_guided_exp", action='store_true', help='True for model w/o guided exploration.')
    ### Seed
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    DEVICE = args.device if torch.cuda.is_available() else 'cpu'

    # seed everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    test_list = load_test_dataset(args.nodes, DEVICE)
    args.size = args.size or len(test_list)
    test_list = test_list[:args.size]

    if args.path is not None:
        net = Net(gfn=True, Z_out_dim=2 if (not args.disable_guided_exp) else 1).to(DEVICE)
        net.load_state_dict(torch.load(args.path, map_location=DEVICE))
    else:
        net = None

    t_aco = list(range(1, args.n_iter + 1))
    avg_cost, avg_diversity, duration = test(test_list, net, args.n_ants, t_aco)
    print('total duration: ', duration)
    for i, t in enumerate(t_aco):
        print(f"T={t}, avg. cost {avg_cost[i]}, avg. diversity {avg_diversity[i]}")

    # Save result in directory that contains model_file
    filename = os.path.splitext(os.path.basename(args.path))[0] if args.path is not None else 'none'
    dirname = os.path.dirname(args.path) if args.path is not None else f'../pretrained/smtwtp/{args.nodes}/no_model'
    os.makedirs(dirname, exist_ok=True)

    result_filename = f"test_result_ckpt{filename}-smtwtp{args.nodes}-ninst{args.size}-nants{args.n_ants}-niter{args.n_iter}-seed{args.seed}"
    result_file = os.path.join(dirname, result_filename + ".txt")
    with open(result_file, "w") as f:
        f.write(f"problem scale: {args.nodes}\n")
        f.write(f"checkpoint: {args.path}\n")
        f.write(f"number of instances: {len(test_list)}\n")
        f.write(f"device: {'cpu' if DEVICE == 'cpu' else DEVICE+'+cpu'}\n")
        f.write(f"n_ants: {args.n_ants}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"total duration: {duration}\n")
        for i, t in enumerate(t_aco):
            f.write(f"T={t}, avg. cost {avg_cost[i]}, avg. diversity {avg_diversity[i]}\n")

    results = pd.DataFrame(columns=['T', 'avg_cost', 'avg_diversity'])
    results['T'] = t_aco
    results['avg_cost'] = avg_cost
    results['avg_diversity'] = avg_diversity
    results.to_csv(os.path.join(dirname, result_filename + ".csv"), index=False)
