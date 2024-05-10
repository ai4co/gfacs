from datetime import timedelta
import argparse
import os
import pickle
import time

from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from subprocess import check_call, CalledProcessError
from tqdm import tqdm
import numpy as np

from utils import load_test_dataset, load_val_dataset


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


def load_dataset(filename):
    with open(check_extension(filename), 'rb') as f:
        return pickle.load(f)


def calc_tsp_length(loc, tour):
    assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
    assert len(tour) == len(loc)
    sorted_locs = np.array(loc)[np.concatenate((tour, [tour[0]]))]
    return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()


def write_tsplib(filename, loc, grid_size, name="problem"):
    with open(filename, 'w') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name),
                ("TYPE", "TSP"),
                ("DIMENSION", len(loc)),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
            )
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, int(x / grid_size * 10000000 + 0.5), int(y / grid_size * 10000000 + 0.5))  # tsplib does not take floats
            for i, (x, y) in enumerate(loc)
        ]))
        f.write("\n")
        f.write("EOF\n")


def read_concorde_tour(filename):
    with open(filename, 'r') as f:
        n = None
        tour = []
        for line in f:
            if n is None:
                n = int(line)
            else:
                tour.extend([int(node) for node in line.rstrip().split(" ")])
    assert len(tour) == n, "Unexpected tour length"
    return tour


def solve_concorde_log(executable, directory, name, loc, grid_size=1, disable_cache=False):
    problem_filename = os.path.join(directory, f"{name}.tsp")
    tour_filename = os.path.join(directory, f"{name}.tour")
    output_filename = os.path.join(directory, f"{name}.concorde.pkl")
    log_filename = os.path.join(directory, f"{name}.log")

    try:
        # May have already been run
        if os.path.isfile(output_filename) and not disable_cache:
            tour, duration = load_dataset(output_filename)
        else:
            write_tsplib(problem_filename, loc, grid_size, name=name)

            with open(log_filename, 'w') as f:
                start = time.time()
                try:
                    # Concorde is weird, will leave traces of solution in current directory so call from target dir
                    check_call([executable, '-s', '1234', '-x', '-o',
                                os.path.abspath(tour_filename), os.path.abspath(problem_filename)],
                               stdout=f, stderr=f, cwd=directory)
                except CalledProcessError as e:
                    # Somehow Concorde returns 255
                    assert e.returncode == 255
                duration = time.time() - start

            tour = read_concorde_tour(tour_filename)
            save_dataset((tour, duration), output_filename)

        return calc_tsp_length(loc, tour), tour, duration

    except Exception as e:
        raise e
        print("Exception occured")
        print(e)
        return None


def run_all_in_pool(func, directory, dataset, n_cpus=None, use_multiprocessing=True):
    num_cpus = os.cpu_count() if n_cpus is None else n_cpus

    w = len(str(len(dataset) - 1))
    pool_cls = (Pool if use_multiprocessing and num_cpus > 1 else ThreadPool)  # type: ignore
    with pool_cls(num_cpus) as pool:
        results = list(tqdm(pool.imap(
            func,
            [(directory, str(i).zfill(w), *problem) for i, problem in enumerate(dataset)],
        ), total=len(dataset)))

    failed = [str(i) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", type=int, help="Problem scale")
    parser.add_argument("--type", type=str, default="test", help="Dataset type")
    parser.add_argument("--result_dir", type=str, default="concorde/results", help="Result directory")
    parser.add_argument("--size", type=int, default=None, help="Number of instances to solve")

    opt = parser.parse_args()

    if opt.type == "val":
        dataset = load_val_dataset(opt.nodes, opt.nodes // 10, "cpu")
    else:
        dataset = load_test_dataset(opt.nodes, opt.nodes // 10, "cpu")

    size = opt.size or len(dataset)
    dataset = dataset[:size]
    
    target_dir = os.path.join(opt.result_dir, f"{opt.type}Dataset-{opt.nodes}-{size}")
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    out_file = os.path.join(target_dir, f"{opt.type}Dataset-{opt.nodes}-concorde.pkl")

    solver_input = [(data[0].x.tolist(),) for data in dataset]

    executable = os.path.join(os.path.dirname(__file__), "concorde/concorde/TSP/concorde")
    use_multiprocessing = False

    def run_func(args):
        directory, name, loc = args
        return solve_concorde_log(executable, directory, name, loc)

    results, parallelism = run_all_in_pool(
        run_func, target_dir, solver_input, n_cpus=None, use_multiprocessing=use_multiprocessing
    )

    costs, tours, durations = zip(*results)  # Not really costs since they should be negative
    print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))  # type: ignore
    print("Average serial duration: {} +- {}".format(
        np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))  # type: ignore
    print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
    print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))

    save_dataset((results, parallelism), out_file)
