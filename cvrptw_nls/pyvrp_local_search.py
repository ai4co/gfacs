from functools import partial
from typing import List, Tuple
import multiprocessing

import numpy as np

from pyvrp import Client, CostEvaluator, Depot, ProblemData, RandomNumberGenerator, Route, Solution, VehicleType
from pyvrp.search import LocalSearch, NODE_OPERATORS, ROUTE_OPERATORS, compute_neighbours, NeighbourhoodParams


def make_data(positions: np.ndarray, demands: np.ndarray, windows: np.ndarray, distances: np.ndarray) -> ProblemData:
    positions = (positions * 10**4).astype(int)
    windows = (windows * 10**4).astype(int)
    distances = (distances * 10**4).astype(int)

    capacity = 600
    demands = (demands * capacity).astype(int)

    return ProblemData(
        clients=[
            Client(x=pos[0], y=pos[1], delivery=d, tw_early=w[0], tw_late=w[1])
            for pos, d, w in zip(positions[1:], demands[1:], windows[1:])
        ],
        depots=[Depot(x=positions[0][0], y=positions[0][1], tw_late=windows[0][1])],
        vehicle_types=[
            VehicleType(len(positions) - 1, capacity, 0, name=",".join(map(str, range(1, len(positions)))))
        ],
        distance_matrix=distances,
        duration_matrix=np.zeros_like(distances),
    )


def make_solution(data: ProblemData, path: np.ndarray) -> Solution:
    # Split the paths into sub-routes by the zeros
    routes = [arr[1:].tolist() for arr in np.split(path, np.where(path == 0)[0]) if len(arr) > 1]
    return Solution(data, routes)


def make_search_operator(data: ProblemData, seed=0, neighbourhood_params: dict | None = None) -> LocalSearch:
    rng = RandomNumberGenerator(seed)
    neighbours = compute_neighbours(data, NeighbourhoodParams(**(neighbourhood_params or {})))
    ls = LocalSearch(data, rng, neighbours)
    for node_op in NODE_OPERATORS:
        ls.add_node_operator(node_op(data))
    for route_op in ROUTE_OPERATORS:
        ls.add_route_operator(route_op(data))
    return ls


def perform_local_search(
        ls_operator: LocalSearch, solution: Solution, cost_evaluator_params: dict, remaining_trials: int = 5
    ) -> Tuple[Solution, bool]:
    cost_evaluator = CostEvaluator(**cost_evaluator_params)
    improved_solution = ls_operator.search(solution, cost_evaluator)
    remaining_trials -= 1
    if is_feasible := improved_solution.is_feasible() or remaining_trials == 0:
        return improved_solution, is_feasible

    print("Warning: Infeasible solution found from local search.",
          "This will slow down the search due to the repeated local search runs.")
    # If infeasible run the local search again with a higher penalty
    cost_evaluator_params["load_penalty"] *= 10
    cost_evaluator_params["tw_penalty"] *= 10
    return perform_local_search(ls_operator, solution, cost_evaluator_params, remaining_trials=remaining_trials)


def pyvrp_local_search(
    path: np.ndarray,
    positions: np.ndarray,
    demands: np.ndarray,
    windows: np.ndarray,
    distances: np.ndarray,
    neighbourhood_params: dict | None = None,
    cost_evaluator_params: dict | None = None,
    allow_infeasible: bool = False,
    max_trials: int = 10,
    inference: bool = False,
    seed: int = 0,
) -> np.ndarray:
    data = make_data(positions, demands, windows, distances)
    solution = make_solution(data, path)
    ls_operator = make_search_operator(data, seed, neighbourhood_params)

    cost_evaluator_params = {
        "load_penalty": (200 if inference else 20) * 10**4,
        "tw_penalty": (200 if inference else 20) * 10**4,
        "dist_penalty": 0,
    }
    cost_evaluator_params.update(cost_evaluator_params or {})

    improved_solution, is_feasible = perform_local_search(
        ls_operator, solution, cost_evaluator_params, remaining_trials=max_trials
    )

    # Return the original path if no feasible solution is found
    if not is_feasible and not allow_infeasible:
        return path

    # Recover the path from the sub-routes in the solution
    route_list = [idx for route in improved_solution.routes() for idx in [0] + route.visits()] + [0]
    return np.array(route_list)


def pyvrp_batched_local_search(
    paths: np.ndarray,
    positions: np.ndarray,
    demands: np.ndarray,
    windows: np.ndarray,
    distances: np.ndarray,
    neighbourhood_params: dict | None = None,
    cost_evaluator_params: dict | None = None,
    allow_infeasible: bool = False,
    max_trials: int = 10,
    inference: bool = False,
    seed: int = 0,
    n_cpus: int = 1,
):
    # distance matrix diagonal should be zero
    np.fill_diagonal(distances, 0)

    max_trials = 1 if allow_infeasible else max_trials

    partial_func = partial(
        pyvrp_local_search,
        positions=positions,
        demands=demands,
        windows=windows,
        distances=distances,
        neighbourhood_params=neighbourhood_params,
        cost_evaluator_params=cost_evaluator_params,
        allow_infeasible=allow_infeasible,
        max_trials=max_trials,
        inference=inference,
        seed=seed,
    )
    if n_cpus > 1:
        pool = multiprocessing.Pool(n_cpus)
        new_paths = pool.map(partial_func, paths)
        pool.close()
        pool.join()
    else:
        new_paths = [partial_func(path) for path in paths]

    # padding with zero
    legnths = [len(path) for path in new_paths]
    max_length = max(legnths)
    new_paths = np.array(
        [np.pad(path, (0, max_length - length), mode="constant") for path, length in zip(new_paths, legnths)]
    )
    return new_paths