import torch
import numpy as np
import numba as nb
from torch.distributions import Categorical
from two_opt import batched_two_opt_python
import random
import concurrent.futures
from functools import cached_property
from itertools import combinations


class ACO():
    def __init__(
        self, 
        distances,
        n_ants=20, 
        decay=0.9,
        alpha=1,
        beta=1,
        elitist=False,
        min_max=False,
        pheromone=None,
        heuristic=None,
        min=None,
        two_opt=False, # for compatibility
        device='cpu',
        local_search: str | None = 'nls',
    ):

        self.problem_size = len(distances)
        self.distances = distances.to(device)
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.elitist = elitist
        self.min_max = min_max

        if min_max:
            if min is not None:
                assert min > 1e-9
            else:
                min = 0.1
            self.min = min
            self.max = None

        if pheromone is None:
            self.pheromone = torch.ones_like(self.distances)
            if min_max:
                self.pheromone = self.pheromone * self.min
        else:
            self.pheromone = pheromone.to(device)

        assert local_search in [None, "2opt", "nls"]
        self.local_search_type = '2opt' if two_opt else local_search

        self.heuristic = 1 / (distances + 1e-10) if heuristic is None else heuristic

        self.shortest_path = None
        self.lowest_cost = float('inf')

        self.device = device

    @torch.no_grad()
    def sparsify(self, k_sparse):
        '''
        Sparsify the TSP graph to obtain the heuristic information 
        Used for vanilla ACO baselines
        '''
        _, topk_indices = torch.topk(self.distances, 
                                        k=k_sparse, 
                                        dim=1, largest=False)
        edge_index_u = torch.repeat_interleave(
            torch.arange(len(self.distances), device=self.device),
            repeats=k_sparse
        )
        edge_index_v = torch.flatten(topk_indices)
        sparse_distances = torch.ones_like(self.distances) * 1e10
        sparse_distances[edge_index_u, edge_index_v] = self.distances[edge_index_u, edge_index_v]
        self.heuristic = 1 / sparse_distances

    def sample(self, invtemp=1.0, inference=False, start_node=None):
        if inference:
            probmat = (self.pheromone ** self.alpha) * (self.heuristic ** self.beta)
            paths = numba_sample(probmat.cpu().numpy(), self.n_ants, start_node=start_node)
            paths = torch.from_numpy(paths.T.astype(np.int64)).to(self.device)
            # paths = self.gen_path(require_prob=False, start_node=start_node)
            log_probs = None
        else:
            paths, log_probs = self.gen_path(invtemp=invtemp, require_prob=True, start_node=start_node)
        costs = self.gen_path_costs(paths)
        return costs, log_probs, paths

    def sample_nls(self, paths):
        paths = self.local_search(paths)
        costs = self.gen_path_costs(paths)
        return costs, paths

    def local_search(self, paths, inference=False):
        if self.local_search_type == "2opt":
            paths = self.two_opt(paths, inference)
        elif self.local_search_type == "nls":
            paths = self.nls(paths, inference)
        return paths

    @torch.no_grad()
    def run(self, n_iterations, inference=True, start_node=None):
        assert n_iterations > 0

        for _ in range(n_iterations):
            if inference:
                probmat = (self.pheromone ** self.alpha) * (self.heuristic ** self.beta)
                paths = numba_sample(probmat.cpu().numpy(), self.n_ants, start_node=start_node)
                paths = torch.from_numpy(paths.T.astype(np.int64)).to(self.device)
            else:
                paths = self.gen_path(invtemp=1.0, require_prob=False, start_node=start_node)
            _paths = paths.clone()  # type: ignore

            paths = self.local_search(paths, inference)
            costs = self.gen_path_costs(paths)

            best_cost, best_idx = costs.min(dim=0)
            if best_cost < self.lowest_cost:
                self.shortest_path = paths[:, best_idx]
                self.lowest_cost = best_cost.item()
                if self.min_max:
                    max = self.problem_size / self.lowest_cost
                    if self.max is None:
                        self.pheromone *= max/self.pheromone.max()
                    self.max = max

            self.update_pheromone(paths, costs)

        # Pairwise Jaccard similarity between paths
        edge_sets = []
        _paths = _paths.T.cpu().numpy()  # type: ignore
        for _p in _paths:
            edge_sets.append(set(map(frozenset, zip(_p[:-1], _p[1:]))))

        # Diversity
        jaccard_sum = 0
        for i, j in combinations(range(len(edge_sets)), 2):
            jaccard_sum += len(edge_sets[i] & edge_sets[j]) / len(edge_sets[i] | edge_sets[j])
        diversity = 1 - jaccard_sum / (len(edge_sets) * (len(edge_sets) - 1) / 2)

        return self.lowest_cost, diversity

    @torch.no_grad()
    def update_pheromone(self, paths, costs):
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
            costs: torch tensor with shape (n_ants,)
        '''
        self.pheromone = self.pheromone * self.decay 

        if self.elitist:
            best_cost, best_idx = costs.min(dim=0)
            best_tour= paths[:, best_idx]
            self.pheromone[best_tour, torch.roll(best_tour, shifts=1)] += 1.0/best_cost
            self.pheromone[torch.roll(best_tour, shifts=1), best_tour] += 1.0/best_cost

        else:
            for i in range(self.n_ants):
                path = paths[:, i]
                cost = costs[i]
                self.pheromone[path, torch.roll(path, shifts=1)] += 1.0/cost
                self.pheromone[torch.roll(path, shifts=1), path] += 1.0/cost

        if self.min_max:
            self.pheromone[(self.pheromone > 1e-9) * (self.pheromone) < self.min] = self.min
            self.pheromone[self.pheromone > self.max] = self.max  # type: ignore

    @torch.no_grad()
    def gen_path_costs(self, paths):
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
        Returns:
                Lengths of paths: torch tensor with shape (n_ants,)
        '''
        assert paths.shape == (self.problem_size, self.n_ants)
        u = paths.T # shape: (n_ants, problem_size)
        v = torch.roll(u, shifts=1, dims=1)  # shape: (n_ants, problem_size)
        # assert (self.distances[u, v] > 0).all()
        return torch.sum(self.distances[u, v], dim=1)

    def gen_numpy_path_costs(self, paths, numpy_distances):
        '''
        Args:
            paths: numpy ndarray with shape (n_ants, problem_size), note the shape
        Returns:
            Lengths of paths: numpy ndarray with shape (n_ants,)
        '''
        assert paths.shape == (self.n_ants, self.problem_size)
        u = paths
        v = np.roll(u, shift=1, axis=1)  # shape: (n_ants, problem_size)
        # assert (self.distances[u, v] > 0).all()
        return np.sum(numpy_distances[u, v], axis=1)

    def gen_path(self, invtemp=1.0, require_prob=False, paths=None, start_node=None):
        '''
        Tour contruction for all ants
        Returns:
            paths: torch tensor with shape (problem_size, n_ants), paths[:, i] is the constructed tour of the ith ant
            log_probs: torch tensor with shape (problem_size, n_ants), log_probs[i, j] is the log_prob of the ith action of the jth ant
        '''
        if paths is None:
            if start_node is None:
                start = torch.randint(low=0, high=self.problem_size, size=(self.n_ants,), device=self.device)
            else:
                start = torch.ones((self.n_ants,), dtype = torch.long, device=self.device) * start_node
        else:
            start = paths[0]

        mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        index = torch.arange(self.n_ants, device=self.device)
        prob_mat = (self.pheromone ** self.alpha) * (self.heuristic ** self.beta)

        mask[index, start] = 0

        paths_list = [] # paths_list[i] is the ith move (tensor) for all ants
        paths_list.append(start)

        log_probs_list = []  # log_probs_list[i] is the ith log_prob (tensor) for all ants' actions
        prev = start
        for i in range(self.problem_size - 1):
            dist = (prob_mat[prev] ** invtemp) * mask
            dist = dist / dist.sum(dim=1, keepdim=True)  # This should be done for numerical stability
            dist = Categorical(probs=dist)
            actions = paths[i + 1] if paths is not None else dist.sample() # shape: (n_ants,)
            paths_list.append(actions)
            if require_prob:
                log_probs = dist.log_prob(actions) # shape: (n_ants,)
                log_probs_list.append(log_probs)
                mask = mask.clone()
            prev = actions
            mask[index, actions] = 0

        if require_prob:
            return torch.stack(paths_list), torch.stack(log_probs_list)
        else:
            return torch.stack(paths_list)

    @cached_property
    def distances_numpy(self):
        return self.distances.detach().cpu().numpy().astype(np.float32)

    @cached_property
    def heuristic_numpy(self):
        return self.heuristic.detach().cpu().numpy().astype(np.float32)  # type: ignore

    @cached_property
    def heuristic_dist(self):
        return 1 / (self.heuristic_numpy / self.heuristic_numpy.max(-1, keepdims=True) + 1e-5)

    def two_opt(self, paths, inference = False):
        maxt = 10000 if inference else self.problem_size // 4
        best_paths = batched_two_opt_python(self.distances_numpy, paths.T.cpu().numpy(), max_iterations=maxt)
        best_paths = torch.from_numpy(best_paths.T.astype(np.int64)).to(self.device)

        return best_paths

    def nls(self, paths, inference=False, T_nls=5, T_p=20):
        maxt = 10000 if inference else self.problem_size // 4
        best_paths = batched_two_opt_python(self.distances_numpy, paths.T.cpu().numpy(), max_iterations=maxt)
        best_costs = self.gen_numpy_path_costs(best_paths, self.distances_numpy)
        new_paths = best_paths

        for _ in range(T_nls):
            perturbed_paths = batched_two_opt_python(self.heuristic_dist, new_paths, max_iterations=T_p)
            new_paths = batched_two_opt_python(self.distances_numpy, perturbed_paths, max_iterations=maxt)
            new_costs = self.gen_numpy_path_costs(new_paths, self.distances_numpy)

            improved_indices = new_costs < best_costs
            best_paths[improved_indices] = new_paths[improved_indices]
            best_costs[improved_indices] = new_costs[improved_indices]

        best_paths = torch.from_numpy(best_paths.T.astype(np.int64)).to(self.device)

        return best_paths


@nb.jit(nb.uint16[:](nb.float32[:,:],nb.int64), nopython=True, nogil=True)
def _numba_sample(probmat: np.ndarray, start_node=None):
    n = probmat.shape[0]
    route = np.zeros(n, dtype=np.uint16)
    mask = np.ones(n, dtype=np.uint8)
    route[0] = lastnode = start_node   # fixed starting node
    for j in range(1, n):
        mask[lastnode] = 0
        prob = probmat[lastnode] * mask
        rand = random.random() * prob.sum()
        for k in range(n):
            rand -= prob[k]
            if rand <= 0:
                break
        lastnode = route[j] = k  # type: ignore
    return route


def numba_sample(probmat: np.ndarray, count=1, start_node=None):
    n = probmat.shape[0]
    routes = np.zeros((count, n), dtype=np.uint16)
    probmat = probmat.astype(np.float32)
    if start_node is None:
        start_node = np.random.randint(0, n, size=count)
    else:
        start_node = np.ones(count, dtype=np.int16) * start_node
    if count <= 4 and n < 500:
        for i in range(count):
            routes[i] = _numba_sample(probmat, start_node[i])
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i in range(count):
                future = executor.submit(_numba_sample, probmat, start_node[i])
                futures.append(future)
            for i, future in enumerate(futures):
                routes[i] = future.result()
    return routes
