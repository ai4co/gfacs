from itertools import combinations
import time
from typing import cast

import torch
from torch.distributions import Categorical


class ACO():
    def __init__(
        self,
        distances,
        prizes,
        penalties,
        n_ants=20,
        heuristic=None,
        pheromone=None,
        decay=0.9,
        alpha=1,
        beta=1,
        # AS variants
        elitist=False,
        maxmin=False,
        rank_based=False,
        n_elites=None,
        shift_cost=True,
        use_local_search=False,
        device='cpu',
    ):
        self.n = prizes.size(0)
        self.distances = distances
        self.prizes = prizes
        self.penalties = penalties
        self.min_prizes = self.n / 4
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.elitist = elitist or maxmin  # maxmin uses elitist
        self.maxmin = maxmin
        self.rank_based = rank_based
        self.n_elites = n_elites or n_ants // 10
        self.shift_cost = shift_cost
        self.use_local_search = use_local_search
        self.device = device

        self.pheromone = pheromone if pheromone is not None else torch.ones_like(self.distances)
        self.pheromone = self.pheromone.to(self.device)

        if heuristic is None:
            _distances = self.distances.clone()
            _distances[torch.arange(self.n), torch.arange(self.n)] = 1e9
            self.heuristic = (1e-10 + prizes.repeat(self.n, 1)) / _distances
        else:
            self.heuristic = heuristic.to(self.device)
        
        self.best_obj = 1e10
        self.best_sol = None

    def sample(self, invtemp=1.0):
        sols, log_probs = self.gen_sol(invtemp=invtemp, require_prob=True)
        objs = self.gen_sol_obj(sols)

        return objs, log_probs, sols

    @torch.no_grad()
    def run(self, n_iterations):
        start = time.time()
        for _ in range(n_iterations):
            sols = cast(torch.Tensor, self.gen_sol(require_prob=False))
            objs = self.gen_sol_obj(sols)
            _sols = sols.clone()

            if self.use_local_search:
                sols, objs = self.local_search(sols, inference=False)

            best_obj, best_idx = objs.min(dim=0)
            if best_obj < self.best_obj:
                self.best_obj = best_obj
                self.best_sol = sols[:, best_idx]

            self.update_pheromone(sols, objs)
        end = time.time()

        # Pairwise Jaccard similarity between paths
        edge_sets = []
        _sols = _sols.T.cpu().numpy()
        for _p in _sols:
            edge_sets.append(set(map(frozenset, zip(_p[:-1], _p[1:]))))

        # Diversity
        jaccard_sum = 0
        for i, j in combinations(range(len(edge_sets)), 2):
            jaccard_sum += len(edge_sets[i] & edge_sets[j]) / len(edge_sets[i] | edge_sets[j])
        diversity = 1 - jaccard_sum / (len(edge_sets) * (len(edge_sets) - 1) / 2)

        return self.best_obj, diversity, end - start

    @torch.no_grad()
    def update_pheromone(self, sols, objs):
        # sols.shape: (max_len, n_ants)
        sols = sols.clone()[:-1, :]  # 0-start, 0-end
        sol_gb = self.best_sol[:-1]  # type: ignore
        deltas = 1.0 / objs
        delta_gb = 1.0 / self.best_obj
        if self.shift_cost:
            total_delta_phe = self.pheromone.sum() * (1 - self.decay)
            n_ants_for_update = self.n_ants if not (self.elitist or self.rank_based) else 1
            shifter = - deltas.mean() + total_delta_phe / (2 * n_ants_for_update * sols.size(0))

            deltas = (deltas + shifter).clamp(min=1e-10)
            delta_gb += shifter

        self.pheromone = self.pheromone * self.decay

        if self.elitist:
            best_delta, best_idx = deltas.max(dim=0)
            best_sol = sols[:, best_idx]
            self.pheromone[best_sol, torch.roll(best_sol, shifts=1)] += best_delta
            self.pheromone[torch.roll(best_sol, shifts=1), best_sol] += best_delta

        elif self.rank_based:
            # Rank-based pheromone update
            elite_indices = torch.argsort(deltas, descending=True)[:self.n_elites]
            elite_sols = sols[:, elite_indices]
            elite_deltas = deltas[elite_indices]
            if self.best_obj < objs.min():
                diff_length = elite_sols.size(0) - sol_gb.size(0)
                if diff_length > 0:
                    sol_gb = torch.cat([sol_gb, torch.zeros(diff_length, device=self.device)])
                elif diff_length < 0:
                    elite_sols = torch.cat([elite_sols, torch.zeros((-diff_length, self.n_elites), device=self.device)], dim=0)

                elite_sols = torch.cat([sol_gb.unsqueeze(1), elite_sols[:, :-1]], dim=1)  # type: ignore
                elite_deltas = torch.cat([torch.tensor([delta_gb], device=self.device), elite_deltas[:-1]])

            rank_denom = (self.n_elites * (self.n_elites + 1)) / 2
            for i in range(self.n_elites):
                sol = elite_sols[:, i]
                delta = elite_deltas[i] * (self.n_elites - i) / rank_denom
                self.pheromone[sol, torch.roll(sol, shifts=1)] += delta
                self.pheromone[torch.roll(sol, shifts=1), sol] += delta

        else:
            for i in range(self.n_ants):
                sol = sols[:, i]
                delta = deltas[i]
                self.pheromone[sol, torch.roll(sol, shifts=1)] += delta
                self.pheromone[torch.roll(sol, shifts=1), sol] += delta

        if self.maxmin:
            _max = 1 / ((1 - self.decay) * self.best_obj)
            p_dec = 0.05 ** (1 / self.n)
            _min = _max * (1 - p_dec) / (0.5 * self.n - 1) / p_dec
            self.pheromone = torch.clamp(self.pheromone, min=_min, max=_max)
            # check convergence
            if (self.pheromone[sol_gb, torch.roll(sol_gb, shifts=1)] >= _max * 0.99).all():
                self.pheromone = 0.5 * self.pheromone + 0.5 * _max

        self.pheromone[self.pheromone < 1e-10] = 1e-10

    @torch.no_grad()
    def gen_sol_obj(self, solutions, n_ants=None):
        '''
        Args:
            solutions: (max_len, n_ants)
        '''
        n_ants = n_ants or self.n_ants
        u = solutions.T
        v = torch.roll(u, shifts=-1, dims=1)
        length = torch.sum(self.distances[u[:, :-1], v[:, :-1]], dim=1)
        saved_penalties = self.penalties.repeat(n_ants, 1).gather(1, u).sum(dim=1)
        return length + self.penalties.sum() - saved_penalties

    def gen_sol(self, invtemp=1.0, require_prob=False, sols=None, n_ants=None):
        n_ants = n_ants or self.n_ants
        solutions = []
        log_probs_list = []

        cur_node = torch.zeros(size=(n_ants,), dtype=torch.int64, device=self.device)
        solutions = [cur_node]
        
        visit_mask = torch.ones(size=(n_ants, self.n), device=self.device) # 1) mask the visted regular node; 2) once return to depot, mask all
        depot_mask = torch.ones(size=(n_ants, self.n), device=self.device) 
        depot_mask[: , 0] = 0 # unmask the depot when 1) enough prize collected; 2) all nodes visited
        
        collected_prize = torch.zeros(size=(n_ants,), device=self.device)
        done = False
        # construction
        i = 0
        while not done:
            guiding_node = sols[i + 1] if (sols is not None) and (i + 1 < sols.size(0)) else None
            cur_node, log_prob = self.pick_node(
                visit_mask, depot_mask, cur_node, require_prob, invtemp, guiding_node
            )
            i += 1

            # update solution and log_probs
            solutions.append(cur_node) 
            log_probs_list.append(log_prob)
            # update collected_prize and mask
            collected_prize += self.prizes[cur_node]
            if require_prob:
                visit_mask = visit_mask.clone()
                depot_mask = depot_mask.clone()
            visit_mask, depot_mask = self.update_mask(visit_mask, depot_mask, cur_node, collected_prize, n_ants)
            # check done
            done = self.check_done(cur_node)
        if require_prob:
            return torch.stack(solutions), torch.stack(log_probs_list)  # (max_len, n_ants)
        else:
            return torch.stack(solutions)
    
    def pick_node(self, visit_mask, depot_mask, cur_node, require_prob, invtemp=1.0, guiding_node=None):
        pheromone = self.pheromone[cur_node] 
        heuristic = self.heuristic[cur_node] 
        dist = (((pheromone ** self.alpha) * (heuristic ** self.beta)) ** invtemp) * visit_mask * depot_mask
        # set the prob of depot to 1 if dist is all 0
        dist[(dist==0).all(dim=1), 0] = 1
        dist = dist / dist.sum(dim=1, keepdim=True)  # This should be done for numerical stability
        dist = Categorical(dist)
        item = dist.sample() if guiding_node is None else guiding_node
        log_prob = dist.log_prob(item) if require_prob else None
        return item, log_prob  # (n_ants,)

    def update_mask(self, visit_mask, depot_mask, cur_node, collected_prize, n_ants):
        # mask regular visted node
        visit_mask[torch.arange(n_ants), cur_node] = 0
        # if at depot, mask all regular nodes, and unmask depot
        at_depot = cur_node == 0
        visit_mask[at_depot, 0] = 1
        visit_mask[at_depot, 1:] = 0
        # unmask the depot for in either case
        # 1) not at depot and enough prize collected
        depot_mask[(~at_depot) * (collected_prize > self.min_prizes), 0] = 1
        # 2) not at depot and all nodes visited
        depot_mask[(~at_depot) * ((visit_mask[:, 1:]==0).all(dim=1)), 0] = 1
        return visit_mask, depot_mask

    def check_done(self, cur_node):
        # is all at depot ?
        return (cur_node == 0).all()        

    ### Destroy & Repair local search
    def get_symmetric_sols(self, sols):
        """
        Args:
            sols: (max_len, n_ants)
        Return:
            symmetric_sols: (max_len, n_ants, n_symmetry)
        """
        n_tail_zeros = sols.size(0) - 1 - sols.count_nonzero(0)
        symmetric_solutions = sols.flip((0,))  # naive flipping cause the leading 0, so we need to roll it.
        for i in range(symmetric_solutions.size(1)):
            symmetric_solutions[:, i] = torch.roll(symmetric_solutions[:, i], shifts=1 - n_tail_zeros[i].item(), dims=0)
        return torch.stack([sols, symmetric_solutions], dim=2)

    @torch.no_grad()
    def local_search(self, sols: torch.Tensor, inference=False):
        """
        Destroy & Repair local search, considering symmetry.
        Args:
            sols: (max_len, n_ants)
        """
        N_ANTS = sols.size(1)
        N_ROUNDS = 10 if inference else 2
        DESTROY_RATE = 0.5
        PROTECTED_LEN = int(sols.shape[0] * DESTROY_RATE)  # protect the first half of the solution
        ROUND_BUDGET = self.n if inference else max(1, self.n // 5)
        TOPK = max(1, ROUND_BUDGET // 10)
        N_SYMMETRY = 2  # maximum number of symmetric solutions to consider, in PCTSP, obviously 2
        N_REPAIR = max(1, ROUND_BUDGET // N_SYMMETRY)  # number of repairs each round

        best_sols = sols.clone()
        best_objs = torch.ones(N_ANTS) * 1e10

        new_sols = sols.clone()
        for _ in range(N_ROUNDS):
            _n_ants = new_sols.size(1)
            _n_repair = max(1, N_REPAIR // (_n_ants // N_ANTS))  # _n_repair * _n_ants = N_REPAIR * N_ANTS
            symmetric_solutions = self.get_symmetric_sols(new_sols)
            # (max_len, N_ANTS (* TOPK), N_SYMMETRY)
            assert symmetric_solutions.size(2) == N_SYMMETRY

            # Slicing is the most simple way to destroy the solutions, but there could be more sophisticated ways.
            destroyed_sols = symmetric_solutions[:PROTECTED_LEN].unsqueeze(3).expand(-1, -1, -1, _n_repair)
            # (PROTECTED_LEN, N_ANTS (* TOPK), N_SYMMETRY, _n_repair)

            destroyed_sols = destroyed_sols.reshape(PROTECTED_LEN, _n_ants * N_SYMMETRY * _n_repair)
            new_sols = cast(torch.Tensor, self.gen_sol(require_prob=False, sols=destroyed_sols, n_ants=_n_ants * N_SYMMETRY * _n_repair))
            new_max_seq_len = new_sols.size(0)
            # (max_seq_len, N_ANTS (* TOPK) * N_SYMMETRY * _n_repair)
            new_objs = self.gen_sol_obj(new_sols, n_ants=_n_ants * N_SYMMETRY * _n_repair)
            # (N_ANTS (* TOPK) * N_SYMMETRY * _n_repair)

            new_sols = new_sols.view(new_max_seq_len, N_ANTS, -1)
            new_objs = new_objs.view(N_ANTS, -1)
            # (max_seq_len, N_ANTS, (TOPK *) N_SYMMETRY * _n_repair)

            best_idx = new_objs.argmin(dim=1)
            best_sols = new_sols[:, torch.arange(N_ANTS), best_idx]
            best_objs = new_objs[torch.arange(N_ANTS), best_idx]

            # Top-10% selection each ants
            topk_indices = torch.argsort(new_objs, dim=1)[:, :TOPK]
            new_sols = new_sols.gather(2, topk_indices.unsqueeze(0).expand(new_max_seq_len, -1, -1)).view(new_max_seq_len, -1)
            # (max_seq_len, N_ANTS * TOPK)

        return best_sols, best_objs
