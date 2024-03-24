from datetime import timedelta
import os
import pickle

import numpy as np
from pyvrp import Model, ProblemData, Client, VehicleType
from pyvrp.stop import MaxIterations

from utils import load_val_dataset, load_test_dataset, get_capacity


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename

def save_dataset(dataset, filename):
    filedir = os.path.split(filename)[0]
    if not os.path.isdir(filedir):
        os.makedirs(filedir)
    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", type=int, help="Problem scale")
    parser.add_argument("--type", type=str, default="test", help="Dataset type")
    parser.add_argument("--result_dir", type=str, default="pyvrp/results", help="Result directory")
    parser.add_argument("--n_cpus", type=int, default=1, help="Number of cpus to use")
    parser.add_argument("--size", type=int, default=None, help="Number of instances to solve")
    parser.add_argument("--maxiter", type=int, default=1000, help="Number of iterations to perform")
    parser.add_argument("--tam", action="store_true", help="Use TAM dataset")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    opt = parser.parse_args()

    if opt.type == "val":
        dataset = load_val_dataset(opt.nodes, opt.nodes // 5, "cpu", tam=opt.tam)
    elif opt.type == "test":
        dataset = load_test_dataset(opt.nodes, opt.nodes // 5, "cpu", tam=opt.tam)
    else:
        raise ValueError("Invalid dataset type")
    capacity = get_capacity(opt.nodes, opt.tam)

    size = opt.size or len(dataset)
    dataset = dataset[:size]

    target_dir = os.path.join(opt.result_dir, f"{opt.type}Dataset{'-tam' if opt.tam else ''}-{opt.nodes}-{size}")
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    sol_file = os.path.join(target_dir, f"{opt.type}Dataset{'-tam' if opt.tam else ''}-{opt.nodes}-pyvrp_maxiter{opt.maxiter}.pkl")
    out_file = os.path.join(target_dir, f"{opt.type}Dataset{'-tam' if opt.tam else ''}-{opt.nodes}-pyvrp_maxiter{opt.maxiter}.txt")

    if os.path.isfile(sol_file):
        print(f"File {sol_file} already exists")
        with open(out_file, "r") as f:
            print(f.read())
        exit(0)

    capacity = 600
    demands = [(data[1][1:].numpy() * 600).astype(int) for data in dataset]
    distances = [(data[2].numpy() * 10**4).astype(int) for data in dataset]
    locs = [(data[3].numpy() * 10**4).astype(int) for data in dataset]

    pyvrp_dataset = [
        ProblemData(
            clients=[Client(x=_l[0], y=_l[1], demand=_d) for _l, _d in zip(_loc[1:], _demand)],
            depots=[Client(x=_loc[0][0], y=_loc[0][1])],
            vehicle_types=[
                VehicleType(len(_loc) - 1, capacity, 0, name=",".join(map(str, range(1, len(_loc)))))
            ],
            distance_matrix=_distance,
            duration_matrix=np.zeros_like(_distance),
        )
        for _loc, _demand, _distance in zip(locs, demands, distances)
    ]

    results = []
    costs = []
    runtimes = []
    for data in pyvrp_dataset:
        model = Model.from_data(data)
        result = model.solve(stop=MaxIterations(opt.maxiter), seed=opt.seed)
        results.append(result)
        costs.append(result.cost() / 10**4)
        runtimes.append(result.runtime)

    # Save the costs and runtimes to a txt file and results to a pkl file
    avg_cost = np.mean(costs)
    std_cost = 2 * np.std(costs) / np.sqrt(len(costs))
    avg_runtime = np.mean(runtimes)
    std_runtime = 2 * np.std(runtimes) / np.sqrt(len(runtimes))
    print(f"Average cost: {avg_cost} +- {std_cost}")
    print(f"Average runtime: {avg_runtime} +- {std_runtime}")
    print(f"Total runtime: {timedelta(seconds=int(np.sum(runtimes)))}")
    with open(out_file, "w") as f:
        f.write(f"Average cost: {avg_cost} +- {2 * std_cost}\n")
        f.write(f"Average runtime: {avg_runtime} +- {2 * std_runtime}\n")
        f.write(f"Total runtime: {timedelta(seconds=int(np.sum(runtimes)))}\n")
    save_dataset((results, opt.maxiter), sol_file)
