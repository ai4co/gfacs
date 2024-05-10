import concurrent.futures
import time
from functools import cached_property
from itertools import combinations

import torch
from torch.distributions import Categorical

from swapstar import swapstar


CAPACITY = 1.0 # The input demands shall be normalized


def get_subroutes(route, end_with_zero = True):
    x = torch.nonzero(route == 0).flatten()
    subroutes = []
    for i, j in zip(x, x[1:]):
        if j - i > 1:
            if end_with_zero:
                j = j + 1
            subroutes.append(route[i: j])
    return subroutes


def merge_subroutes(subroutes, length, device):
    route = torch.zeros(length, dtype = torch.long, device=device)
    i = 0
    for r in subroutes:
        if len(r) > 2:
            if isinstance(r, list):
                r = torch.tensor(r[:-1])
            else:
                r = r[:-1].clone().detach()
            route[i: i + len(r)] = r
            i += len(r)
    return route


class ACO():
    def __init__(
        self,  # 0: depot
        distances: torch.Tensor,  # (n, n)
        demand: torch.Tensor,  # (n, )
        capacity=CAPACITY,
        positions = None,
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
        self.distances = distances
        self.demand = demand
        self.capacity = capacity
        self.positions = positions
        
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.elitist = elitist or maxmin  # maxmin uses elitist
        self.maxmin = maxmin
        self.rank_based = rank_based
        self.n_elites = n_elites or n_ants // 10

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

        self.heuristic = 1 / (distances + 1e-10) if heuristic is None else heuristic

        self.shortest_path = None
        self.lowest_cost = float('inf')

        assert local_search_type in [None, "nls", "swapstar"]
        assert positions is not None if local_search_type is not None else True
        self.local_search_type = local_search_type

    @torch.no_grad()
    def simple_heuristic(self, distances, k_sparse):
        '''
        Sparsify the TSP graph to obtain the heuristic information 
        Used for vanilla ACO baselines
        '''
        n = self.demand.size(0)
        temp_dists = distances.clone()
        temp_dists[1:, 1:][torch.eye(n - 1, dtype=torch.bool, device=self.device)] = 1e9
        # sparsify
        # part 1:
        _, topk_indices = torch.topk(temp_dists[1:, 1:], k = k_sparse, dim=1, largest=False)
        edge_index_1 = torch.stack([
            torch.repeat_interleave(torch.arange(n - 1).to(topk_indices.device), repeats=k_sparse),
            torch.flatten(topk_indices)
        ]) + 1
        # part 2: keep all edges connected to depot
        edge_index_2 = torch.stack([ 
            torch.zeros(n - 1, device=self.device, dtype=torch.long), 
            torch.arange(1, n, device=self.device, dtype=torch.long),
        ])
        edge_index_3 = torch.stack([ 
            torch.arange(1, n, device=self.device, dtype=torch.long),
            torch.zeros(n - 1, device=self.device, dtype=torch.long), 
        ])
        edge_index = torch.concat([edge_index_1, edge_index_2, edge_index_3], dim=1)

        sparse_distances = torch.ones_like(distances) * 1e10
        sparse_distances[edge_index[0], edge_index[1]] = distances[edge_index[0], edge_index[1]]

        heuristic = 1 / sparse_distances
        return heuristic

    def sample(self, invtemp=1.0):
        paths, log_probs = self.gen_path(require_prob=True, invtemp=invtemp)  # type: ignore
        costs = self.gen_path_costs(paths)
        return costs, log_probs, paths

    def local_search(self, paths, inference=False):
        new_paths = paths.clone()
        self.multiple_swap_star(new_paths, inference=inference)
        return new_paths

    @torch.no_grad()
    def run(self, n_iterations):
        assert n_iterations > 0

        start = time.time()
        for _ in range(n_iterations):
            paths = self.gen_path(require_prob=False)
            _paths = paths.clone()   # type: ignore

            if self.local_search_type is not None:
                self.multiple_swap_star(paths, inference=True)
            costs = self.gen_path_costs(paths)

            best_cost, best_idx = costs.min(dim=0)
            if best_cost < self.lowest_cost:
                self.shortest_path = paths[:, best_idx].clone()  # type: ignore
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
        if self.shift_cost:
            total_delta_phe = self.pheromone.sum() * (1 - self.decay)
            n_ants_for_update = self.n_ants if not (self.elitist or self.rank_based) else 1
            shifter = - deltas.mean() + total_delta_phe / (2 * n_ants_for_update * (paths.shape[0] - 1))

            deltas = deltas + shifter
            deltas = deltas.clamp(min=1e-10)
            delta_gb = (1.0 / self.lowest_cost) + shifter

        self.pheromone = self.pheromone * self.decay

        if self.elitist:
            best_delta, best_idx = deltas.max(dim=0)
            best_tour= paths[:, best_idx]
            self.pheromone[best_tour[:-1], torch.roll(best_tour, shifts=1)[:-1]] += best_delta
            self.pheromone[torch.roll(best_tour, shifts=1)[:-1], best_tour[:-1]] += best_delta

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
                self.pheromone[path[:-1], torch.roll(path, shifts=1)[:-1]] += delta
                self.pheromone[torch.roll(path, shifts=1)[:-1], path[:-1]] += delta

        else:
            for i in range(self.n_ants):
                path = paths[:, i]
                delta = deltas[i]
                self.pheromone[path[:-1], torch.roll(path, shifts=1)[:-1]] += delta
                self.pheromone[torch.roll(path, shifts=1)[:-1], path[:-1]] += delta

        if self.maxmin:
            _max = _max = 1 / ((1 - self.decay) * self.lowest_cost)
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
        u = paths.permute(1, 0) # shape: (n_ants, max_seq_len)
        v = torch.roll(u, shifts=-1, dims=1)  
        return torch.sum(self.distances[u[:, :-1], v[:, :-1]], dim=1)

    def gen_path(self, require_prob=False, invtemp=1.0, paths=None):
        actions = torch.zeros((self.n_ants,), dtype=torch.long, device=self.device)
        visit_mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        visit_mask = self.update_visit_mask(visit_mask, actions)
        used_capacity = torch.zeros(size=(self.n_ants,), device=self.device)

        used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)

        prob_mat = (self.pheromone ** self.alpha) * (self.heuristic ** self.beta)
        prev = actions

        paths_list = [actions]  # paths_list[i] is the ith move (tensor) for all ants
        log_probs_list = []  # log_probs_list[i] is the ith log_prob (tensor) for all ants' actions
        done = self.check_done(visit_mask, actions)

        ##################################################
        # given paths
        i = 0
        feasible_idx = torch.arange(self.n_ants, device=self.device) if paths is not None else None
        ##################################################
        while not done:
            selected = paths[i + 1] if paths is not None else None
            actions, log_probs = self.pick_move(prob_mat[prev], visit_mask, capacity_mask, require_prob, invtemp, selected)
            paths_list.append(actions)
            if require_prob:
                log_probs_list.append(log_probs)
                visit_mask = visit_mask.clone()
            visit_mask = self.update_visit_mask(visit_mask, actions)
            used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)

            ##################################################
            # NLS may generate infeasible solutions
            if paths is not None:
                infeasible_idx = torch.where(capacity_mask.sum(-1) == 0)[0]
                # remove infeasible ants
                if len(infeasible_idx) > 0:
                    is_feasible = capacity_mask.sum(-1) > 0
                    feasible_idx = feasible_idx[is_feasible]  # type: ignore

                    actions = actions[is_feasible]
                    visit_mask = visit_mask[is_feasible]
                    used_capacity = used_capacity[is_feasible]
                    capacity_mask = capacity_mask[is_feasible]

                    paths_list = [p[is_feasible] for p in paths_list]
                    if require_prob:
                        log_probs_list = [l_p[is_feasible] for l_p in log_probs_list]
                    if paths is not None:
                        paths = paths[:, is_feasible]

                    self.n_ants -= len(infeasible_idx)
            ##################################################

            done = self.check_done(visit_mask, actions)
            prev = actions
            i += 1

        if require_prob:
            if paths is not None:
                return torch.stack(paths_list), torch.stack(log_probs_list), feasible_idx  # type: ignore
            return torch.stack(paths_list), torch.stack(log_probs_list)
        else:
            return torch.stack(paths_list)

    def pick_move(self, dist, visit_mask, capacity_mask, require_prob, invtemp=1.0, selected=None):
        dist = (dist ** invtemp) * visit_mask * capacity_mask  # shape: (n_ants, p_size)
        dist = dist / dist.sum(dim=1, keepdim=True)  # This should be done for numerical stability
        dist = Categorical(probs=dist)
        actions = selected if selected is not None else dist.sample()  # shape: (n_ants,)
        log_probs = dist.log_prob(actions) if require_prob else None  # shape: (n_ants,)
        return actions, log_probs

    def update_visit_mask(self, visit_mask, actions):
        visit_mask[torch.arange(self.n_ants, device=self.device), actions] = 0
        visit_mask[:, 0] = 1 # depot can be revisited with one exception
        visit_mask[(actions==0) * (visit_mask[:, 1:]!=0).any(dim=1), 0] = 0 # one exception is here
        return visit_mask
    
    def update_capacity_mask(self, cur_nodes, used_capacity):
        '''
        Args:
            cur_nodes: shape (n_ants, )
            used_capacity: shape (n_ants, )
            capacity_mask: shape (n_ants, p_size)
        Returns:
            ant_capacity: updated capacity
            capacity_mask: updated mask
        '''
        capacity_mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        # update capacity
        used_capacity[cur_nodes==0] = 0
        used_capacity = used_capacity + self.demand[cur_nodes]
        # update capacity_mask
        remaining_capacity = self.capacity - used_capacity # (n_ants,)
        remaining_capacity_repeat = remaining_capacity.unsqueeze(-1).repeat(1, self.problem_size) # (n_ants, p_size)
        demand_repeat = self.demand.unsqueeze(0).repeat(self.n_ants, 1) # (n_ants, p_size)
        capacity_mask[demand_repeat > remaining_capacity_repeat + 1e-10] = 0
        
        return used_capacity, capacity_mask
    
    def check_done(self, visit_mask, actions):
        return (visit_mask[:, 1:] == 0).all() and (actions == 0).all()
    
    @cached_property
    @torch.no_grad()
    def distances_cpu(self):
        return self.distances.cpu().numpy()
    
    @cached_property
    @torch.no_grad()
    def demand_cpu(self):
        return self.demand.cpu().numpy()
    
    @cached_property
    @torch.no_grad()
    def positions_cpu(self):
        return self.positions.cpu().numpy() if self.positions is not None else None

    @cached_property
    @torch.no_grad()
    def heuristic_dist(self):
        heu = self.heuristic.detach().cpu().numpy()  # type: ignore
        return (1 / (heu/heu.max(-1, keepdims=True) + 1e-5))

    @torch.no_grad()
    def multiple_swap_star(self, paths, indexes=None, inference=False):
        subroutes_all = []
        for i in range(paths.size(1)) if indexes is None else indexes:
            subroutes = get_subroutes(paths[:, i])
            subroutes_all.append((i, subroutes))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i, p in subroutes_all:
                count = 10000 if inference else self.problem_size // 10
                if self.local_search_type == "swapstar":
                    future = executor.submit(
                        swapstar, self.demand_cpu, self.distances_cpu, self.positions_cpu, p, count=count
                    )
                elif self.local_search_type == "nls":
                    future = executor.submit(
                        neural_swapstar,
                        self.demand_cpu,
                        self.distances_cpu,
                        self.heuristic_dist,
                        self.positions_cpu,
                        p,
                        limit=count
                    )
                futures.append((i, future))

            for i, future in futures:
                paths[:, i] = merge_subroutes(future.result(), paths.size(0), self.device)

def neural_swapstar(demand, distances, heu_dist, positions, p, disturb=5, limit=10000):
    p0 = p
    p1 = swapstar(demand, distances, positions, p0, count = limit)
    p2 = swapstar(demand, heu_dist, positions, p1, count = disturb)
    p3 = swapstar(demand, distances, positions, p2, count = limit)
    return p3
