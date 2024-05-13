import concurrent.futures
import random
import time
from functools import cached_property
from itertools import combinations

import numpy as np
import numba as nb
import torch
from torch.distributions import Categorical

from two_opt import batched_two_opt_python


class ACO():
    def __init__(
        self, 
        distances: torch.Tensor,
        n_ants=20,
        heuristic: torch.Tensor | None = None,
        k_sparse=None,
        pheromone: torch.Tensor | None = None,
        decay=0.9,
        alpha=1,
        beta=1,
        # AS variants
        elitist=False,
        maxmin=False,
        rank_based=False,
        n_elites=None,
        smoothing=False,
        smoothing_thres=5,
        smoothing_delta=0.5,
        shift_cost=True,
        local_search_type: str | None = 'nls',
        device='cpu',
    ):
        self.problem_size = len(distances)
        self.distances = distances.to(device)
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.elitist = elitist or maxmin  # maxmin uses elitist
        self.maxmin = maxmin
        self.rank_based = rank_based
        self.n_elites = n_elites or n_ants // 10  # only for rank-based

        # Smoothing
        self.smoothing = smoothing
        self.smoothing_cnt = 0
        self.smoothing_thres = smoothing_thres
        self.smoothing_delta = smoothing_delta
        self.shift_cost = shift_cost
        self.device = device

        if pheromone is None:
            self.pheromone = torch.ones_like(self.distances)
            # if maxmin:
            #     self.pheromone = self.pheromone / ((1 - self.decay) * (self.problem_size ** 0.5))  # arbitrarily (high) value
        else:
            self.pheromone = pheromone.to(device)

        if heuristic is None:
            assert k_sparse is not None
            self.heuristic = self.simple_heuristic(distances, k_sparse)
        else:
            self.heuristic = heuristic.to(device)

        assert local_search_type in [None, "2opt", "nls"]
        self.local_search_type = local_search_type

        self.shortest_path = None
        self.lowest_cost = float('inf')

    @torch.no_grad()
    def simple_heuristic(self, distances, k_sparse):
        '''
        Sparsify the TSP graph to obtain the heuristic information 
        Used for vanilla ACO baselines
        '''
        _, topk_indices = torch.topk(distances, k=k_sparse, dim=1, largest=False)
        edge_index_u = torch.repeat_interleave(
            torch.arange(len(distances), device=self.device), repeats=k_sparse
        )
        edge_index_v = torch.flatten(topk_indices)
        sparse_distances = torch.ones_like(distances) * 1e10
        sparse_distances[edge_index_u, edge_index_v] = distances[edge_index_u, edge_index_v]
        heuristic = 1 / sparse_distances
        return heuristic

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
    def run(self, n_iterations, start_node=None):
        assert n_iterations > 0

        start = time.time()
        for _ in range(n_iterations):
            probmat = (self.pheromone ** self.alpha) * (self.heuristic ** self.beta)
            paths = numba_sample(probmat.cpu().numpy(), self.n_ants, start_node=start_node)
            paths = torch.from_numpy(paths.T.astype(np.int64)).to(self.device)
            _paths = paths.clone()  # type: ignore

            paths = self.local_search(paths, inference=True)
            costs = self.gen_path_costs(paths)

            best_cost, best_idx = costs.min(dim=0)
            if best_cost < self.lowest_cost:
                self.shortest_path = paths[:, best_idx]
                self.lowest_cost = best_cost.item()

            self.update_pheromone(paths, costs)
        end = time.time()

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

        return self.lowest_cost, diversity, end - start

    @torch.no_grad()
    def update_pheromone(self, paths, costs):
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
            costs: torch tensor with shape (n_ants,)
        '''
        deltas = 1.0 / costs
        delta_gb = 1.0 / self.lowest_cost
        if self.shift_cost:
            total_delta_phe = self.pheromone.sum() * (1 - self.decay)
            n_ants_for_update = self.n_ants if not (self.elitist or self.rank_based) else 1
            shifter = - deltas.mean() + total_delta_phe / (2 * n_ants_for_update * self.problem_size)

            deltas = (deltas + shifter).clamp(min=1e-10)
            delta_gb += shifter

        self.pheromone = self.pheromone * self.decay

        if self.elitist:
            best_delta, best_idx = deltas.max(dim=0)
            best_tour= paths[:, best_idx]
            self.pheromone[best_tour, torch.roll(best_tour, shifts=1)] += best_delta
            self.pheromone[torch.roll(best_tour, shifts=1), best_tour] += best_delta

        elif self.rank_based:
            # Rank-based pheromone update
            elite_indices = torch.argsort(deltas, descending=True)[:self.n_elites]
            elite_paths = paths[:, elite_indices]
            elite_deltas = deltas[elite_indices]
            if self.lowest_cost < costs.min():
                elite_paths = torch.cat([self.shortest_path.unsqueeze(1), elite_paths[:, :-1]], dim=1)  # type: ignore
                elite_deltas = torch.cat([torch.tensor([delta_gb], device=self.device), elite_deltas[:-1]])

            rank_denom = (self.n_elites * (self.n_elites + 1)) / 2
            for i in range(self.n_elites):
                path = elite_paths[:, i]
                delta = elite_deltas[i] * (self.n_elites - i) / rank_denom
                self.pheromone[path, torch.roll(path, shifts=1)] += delta
                self.pheromone[torch.roll(path, shifts=1), path] += delta

        else:
            for i in range(self.n_ants):
                path = paths[:, i]
                delta = deltas[i]
                self.pheromone[path, torch.roll(path, shifts=1)] += delta
                self.pheromone[torch.roll(path, shifts=1), path] += delta

        if self.maxmin:
            _max = 1 / ((1 - self.decay) * self.lowest_cost)
            p_dec = 0.05 ** (1 / self.problem_size)
            _min = _max * (1 - p_dec) / (0.5 * self.problem_size - 1) / p_dec
            self.pheromone = torch.clamp(self.pheromone, min=_min, max=_max)
            # check convergence
            if (self.pheromone[self.shortest_path, torch.roll(self.shortest_path, shifts=1)] >= _max * 0.99).all():  # type: ignore
                self.pheromone = 0.5 * self.pheromone + 0.5 * _max

        else:  # maxmin has its own smoothing
            # smoothing the pheromone if the lowest cost has not been updated for a while
            if self.smoothing:
                self.smoothing_cnt = max(0, self.smoothing_cnt + (1 if self.lowest_cost < costs.min() else -1))
                if self.smoothing_cnt >= self.smoothing_thres:
                    self.pheromone = self.smoothing_delta * self.pheromone + (self.smoothing_delta) * torch.ones_like(self.pheromone)
                    self.smoothing_cnt = 0

        self.pheromone[self.pheromone < 1e-10] = 1e-10

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

    def gen_numpy_path_costs(self, paths):
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
        return np.sum(self.distances_numpy[u, v], axis=1)

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
        best_costs = self.gen_numpy_path_costs(best_paths)
        new_paths = best_paths

        for _ in range(T_nls):
            perturbed_paths = batched_two_opt_python(self.heuristic_dist, new_paths, max_iterations=T_p)
            new_paths = batched_two_opt_python(self.distances_numpy, perturbed_paths, max_iterations=maxt)
            new_costs = self.gen_numpy_path_costs(new_paths)

            improved_indices = new_costs < best_costs
            best_paths[improved_indices] = new_paths[improved_indices]
            best_costs[improved_indices] = new_costs[improved_indices]

        best_paths = torch.from_numpy(best_paths.T.astype(np.int64)).to(self.device)
        return best_paths


class ACO_NP():
    """
    ACO class for numpy implementation
    """
    def __init__(
        self, 
        distances: np.ndarray,
        n_ants=20,
        heuristic: np.ndarray | None = None,
        k_sparse=None,
        pheromone: np.ndarray | None = None,
        decay=0.9,
        alpha=1,
        beta=1,
        # AS variants
        elitist=False,
        maxmin=False,
        rank_based=False,
        n_elites=None,
        smoothing=False,
        smoothing_thres=5,
        smoothing_delta=0.5,
        shift_cost=True,
        local_search_type: str | None = 'nls',
    ):
        self.problem_size = len(distances)
        self.distances = distances
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.elitist = elitist or maxmin  # maxmin uses elitist
        self.maxmin = maxmin
        self.rank_based = rank_based
        self.n_elites = n_elites or n_ants // 10  # only for rank-based
        self.smoothing = smoothing
        self.smoothing_cnt = 0
        self.smoothing_thres = smoothing_thres
        self.smoothing_delta = smoothing_delta
        self.shift_cost = shift_cost

        self.pheromone = pheromone or np.ones_like(self.distances)
        # if maxmin:
        #     self.pheromone = self.pheromone / ((1 - self.decay) * (self.problem_size ** 0.5))  # arbitrarily (high) value
        self.heuristic = heuristic if heuristic is not None else self.simple_heuristic(distances, k_sparse)

        assert local_search_type in [None, "2opt", "nls"]
        self.local_search_type = local_search_type

        self.shortest_path = None
        self.lowest_cost = float('inf')

    def simple_heuristic(self, distances, k_sparse):
        '''
        Sparsify the TSP graph to obtain the heuristic information 
        Used for vanilla ACO baselines
        '''
        assert k_sparse is not None

        topk_indices = np.argpartition(distances, k_sparse, axis=1)[:, :k_sparse]
        edge_index_u = np.repeat(np.arange(len(distances)), k_sparse)
        edge_index_v = np.ravel(topk_indices)
        sparse_distances = np.ones_like(distances) * 1e10
        sparse_distances[edge_index_u, edge_index_v] = distances[edge_index_u, edge_index_v]
        heuristic = 1 / sparse_distances
        return heuristic

    def sample(self, invtemp=1.0, inference=False, start_node=None, epsilon=None, K=1):
        assert inference
        assert epsilon is None

        if K > 1:
            n_ants = self.n_ants
            self.n_ants = K * n_ants

        probmat = (self.pheromone ** self.alpha) * (self.heuristic ** self.beta)
        paths = numba_sample(probmat, self.n_ants, invtemp=invtemp, start_node=start_node)
        paths = paths.T.astype(np.int64)

        costs = self.gen_path_costs(paths)

        if K > 1:
            self.n_ants = n_ants

        return costs, None, paths

    def sample_nls(self, paths: np.ndarray):
        # paths: (problem_size, n_ants)
        paths = self.local_search(paths.T)
        costs = self.gen_path_costs(paths)
        return costs, paths

    def local_search(self, paths: np.ndarray, inference=False):
        # paths: (n_ants, problem_size)
        if self.local_search_type == "2opt":
            paths = self.two_opt(paths, inference)
        elif self.local_search_type == "nls":
            paths = self.nls(paths, inference)
        return paths

    def run(self, n_iterations, start_node=None):
        assert n_iterations > 0

        start = time.time()
        for _ in range(n_iterations):
            probmat = (self.pheromone ** self.alpha) * (self.heuristic ** self.beta)
            paths = numba_sample(probmat, self.n_ants, start_node=start_node).astype(np.int64)
            # (n_ants, problem_size)
            _paths = paths.copy()

            paths = self.local_search(paths, inference=True)
            costs = self.gen_path_costs(paths)

            best_idx = costs.argmin()
            best_cost = costs[best_idx]
            if best_cost < self.lowest_cost:
                self.shortest_path = paths[best_idx, :]
                self.lowest_cost = best_cost

            self.update_pheromone(paths, costs)
        end = time.time()

        # Pairwise Jaccard similarity between paths
        edge_sets = []
        for _p in _paths:
            edge_sets.append(set(map(frozenset, zip(_p[:-1], _p[1:]))))  # frozenset to consider symmetry

        # Diversity
        jaccard_sum = 0
        for i, j in combinations(range(len(edge_sets)), 2):
            jaccard_sum += len(edge_sets[i] & edge_sets[j]) / len(edge_sets[i] | edge_sets[j])
        diversity = 1 - jaccard_sum / (len(edge_sets) * (len(edge_sets) - 1) / 2)

        return self.lowest_cost, diversity, end - start

    def update_pheromone(self, paths: np.ndarray, costs: np.ndarray):
        '''
        Args:
            paths: np.ndarray with shape (n_ants, problem_size)
            costs: np.ndarray with shape (n_ants,)
        '''
        deltas = 1.0 / costs
        delta_gb = 1.0 / self.lowest_cost
        if self.shift_cost:
            total_delta_phe = self.pheromone.sum() * (1 - self.decay)
            n_ants_for_update = self.n_ants if not (self.elitist or self.rank_based) else 1
            shifter = - deltas.mean() + total_delta_phe / (2 * n_ants_for_update * self.problem_size)

            deltas = (deltas + shifter).clip(min=1e-10)
            delta_gb += shifter

        self.pheromone = self.pheromone * self.decay

        if self.elitist:
            best_idx = deltas.argmax(axis=0)
            best_delta = deltas[best_idx]
            best_tour= paths[best_idx, :]
            self.pheromone[best_tour, np.roll(best_tour, shift=1)] += best_delta
            self.pheromone[np.roll(best_tour, shift=1), best_tour] += best_delta

        elif self.rank_based:
            # Rank-based pheromone update
            elite_indices = np.argsort(deltas)[::-1][:self.n_elites]
            elite_paths = paths[elite_indices, :]
            elite_deltas = deltas[elite_indices]
            if self.lowest_cost < costs.min():
                elite_paths = np.vstack([self.shortest_path[None, :], elite_paths[:-1]])  # type: ignore
                elite_deltas = np.concatenate([[delta_gb], elite_deltas[:-1]])

            rank_denom = (self.n_elites * (self.n_elites + 1)) / 2
            elite_deltas = elite_deltas * (self.n_elites - np.arange(self.n_elites)) / rank_denom

            _u = elite_paths
            _v = np.roll(_u, shift=1, axis=1)
            edges = np.stack([_u.flatten(), _v.flatten()], axis=1)
            edge1 = np.concatenate([edges[:, 0], edges[:, 1]], axis=0)
            edge2 = np.concatenate([edges[:, 1], edges[:, 0]], axis=0)
            np.add.at(self.pheromone, (edge1, edge2), elite_deltas.repeat(2 * self.problem_size))

        else:
            _u = paths
            _v = np.roll(_u, shift=1, axis=1)
            edges = np.stack([_u.flatten(), _v.flatten()], axis=1)
            edge1 = np.concatenate([edges[:, 0], edges[:, 1]], axis=0)
            edge2 = np.concatenate([edges[:, 1], edges[:, 0]], axis=0)
            np.add.at(self.pheromone, (edge1, edge2), deltas.repeat(2 * self.problem_size))

        if self.maxmin:
            _max = 1 / ((1 - self.decay) * self.lowest_cost)
            p_dec = 0.05 ** (1 / self.problem_size)
            _min = _max * (1 - p_dec) / (0.5 * self.problem_size - 1) / p_dec
            self.pheromone = self.pheromone.clip(min=_min, max=_max)
            # check convergence
            if (self.pheromone[self.shortest_path, np.roll(self.shortest_path, shift=1)] >= _max * 0.99).all():  # type: ignore
                self.pheromone = 0.5 * self.pheromone + 0.5 * _max

        else:  # maxmin has its own smoothing
            # smoothing the pheromone if the lowest cost has not been updated for a while
            if self.smoothing:
                self.smoothing_cnt = max(0, self.smoothing_cnt + (1 if self.lowest_cost < costs.min() else -1))
                if self.smoothing_cnt >= self.smoothing_thres:
                    self.pheromone = self.smoothing_delta * self.pheromone + (self.smoothing_delta) * np.ones_like(self.pheromone)
                    self.smoothing_cnt = 0

        self.pheromone[self.pheromone < 1e-10] = 1e-10

    def gen_path_costs(self, paths):
        '''
        Args:
            paths: numpy ndarray with shape (problem_size, n_ants) or (n_ants, problem_size)
        Returns:
            Lengths of paths: numpy ndarray with shape (n_ants,)
        '''
        u = paths.T if paths.shape == (self.problem_size, self.n_ants) else paths  # shape: (n_ants, problem_size)
        v = np.roll(u, shift=1, axis=1)  # shape: (n_ants, problem_size)
        # assert (self.distances[u, v] > 0).all()
        return np.sum(self.distances[u, v], axis=1)

    def gen_path(self, invtemp=1.0, require_prob=False, paths=None, start_node=None, epsilon=None):
        '''
        Tour contruction for all ants
        Returns:
            paths: np.ndarray with shape (problem_size, n_ants), paths[:, i] is the constructed tour of the ith ant
            log_probs: np.ndarray with shape (problem_size, n_ants), log_probs[i, j] is the log_prob of the ith action of the jth ant
        '''
        raise ValueError("Not implemented for numpy")

    @cached_property
    def heuristic_dist(self):
        return 1 / (self.heuristic / self.heuristic.max(-1, keepdims=True) + 1e-5)

    def two_opt(self, paths: np.ndarray, inference=False):
        maxt = 10000 if inference else self.problem_size // 4
        best_paths = batched_two_opt_python(self.distances, paths, max_iterations=maxt)
        best_paths = best_paths.astype(np.int64)
        return best_paths

    def nls(self, paths: np.ndarray, inference=False, T_nls=5, T_p=20):
        maxt = 10000 if inference else self.problem_size // 4
        best_paths = batched_two_opt_python(self.distances, paths, max_iterations=maxt)
        best_costs = self.gen_path_costs(best_paths)
        new_paths = best_paths

        for _ in range(T_nls):
            perturbed_paths = batched_two_opt_python(self.heuristic_dist, new_paths, max_iterations=T_p)
            new_paths = batched_two_opt_python(self.distances, perturbed_paths, max_iterations=maxt)
            new_costs = self.gen_path_costs(new_paths)

            improved_indices = new_costs < best_costs
            best_paths[improved_indices] = new_paths[improved_indices]
            best_costs[improved_indices] = new_costs[improved_indices]

        best_paths = best_paths.astype(np.int64)
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


def numba_sample(probmat: np.ndarray, count=1, invtemp=1.0, start_node=None):
    n = probmat.shape[0]
    routes = np.zeros((count, n), dtype=np.uint16)
    probmat = probmat.astype(np.float32) ** invtemp
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
