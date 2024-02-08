from copy import deepcopy
from itertools import combinations

import torch
from torch.distributions import Categorical


class ACO():
    def __init__(self,
                 distances,
                 prizes,
                 penalties,
                 n_ants=20, 
                 decay=0.9,
                 alpha=1,
                 beta=1,
                 elitist=False,
                 min_max=False,
                 pheromone=None,
                 heuristic=None,
                 min=None,
                 device='cpu'
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
        self.elitist = elitist
        self.min_max = min_max
        
        self.ants_idx = torch.arange(n_ants)
        
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
            self.pheromone = pheromone
        
        _distances = deepcopy(self.distances)
        _distances[torch.arange(self.n), torch.arange(self.n)] = 1e9
        self.heuristic = (1e-10 + prizes.repeat(self.n, 1)) / _distances if heuristic is None else heuristic
        
        self.alltime_best_obj = 1e10
        self.alltime_best_sol = None

        self.device = device
    
    def sample(self, invtemp=1.0, return_sol=False, K=1):
        n_ants = self.n_ants
        self.n_ants = K * n_ants
        self.ants_idx = torch.arange(self.n_ants)

        sols, log_probs = self.gen_sol(invtemp=invtemp, require_prob=True)
        objs = self.gen_sol_obj(sols)

        self.n_ants = n_ants
        self.ants_idx = torch.arange(self.n_ants)

        if return_sol:
            return objs, log_probs, sols
        else:
            return objs, log_probs

    @torch.no_grad()
    def run(self, n_iterations):
        for _ in range(n_iterations):
            sols = self.gen_sol(require_prob=False)
            objs = self.gen_sol_obj(sols)
            _sols = sols.clone()  # type: ignore

            sols = sols.T  # type: ignore
            best_obj, best_idx = objs.min(dim=0)
            if best_obj < self.alltime_best_obj:
                self.alltime_best_obj = best_obj
                self.alltime_best_sol = sols[best_idx]
                if self.min_max:
                    max = (self.n - 1) / self.alltime_best_obj
                    if self.max is None:
                        self.pheromone *= max/self.pheromone.max()
                    self.max = max
            self.update_pheromone(sols, objs, best_obj, best_idx)

        # Pairwise Jaccard similarity between paths
        edge_sets = []
        _sols = _sols.T.cpu().numpy()  # type: ignore
        for _p in _sols:
            edge_sets.append(set(map(frozenset, zip(_p[:-1], _p[1:]))))

        # Diversity
        jaccard_sum = 0
        for i, j in combinations(range(len(edge_sets)), 2):
            jaccard_sum += len(edge_sets[i] & edge_sets[j]) / len(edge_sets[i] | edge_sets[j])
        diversity = 1 - jaccard_sum / (len(edge_sets) * (len(edge_sets) - 1) / 2)

        return self.alltime_best_obj, diversity

    @torch.no_grad()
    def update_pheromone(self, sols, objs, best_obj, best_idx):
        self.pheromone = self.pheromone * self.decay

        if self.elitist:
            best_sol= sols[best_idx]
            self.pheromone[best_sol[:-1], torch.roll(best_sol, shifts=-1)[:-1]] += 1.0/best_obj
        else:
            for i in range(self.n_ants):
                sol = sols[i]
                obj = objs[i]
                self.pheromone[sol[:-1], torch.roll(sol, shifts=-1)[:-1]] +=  1.0/obj

        if self.min_max:
            self.pheromone[(self.pheromone>1e-9) * (self.pheromone)<self.min] = self.min
            self.pheromone[self.pheromone>self.max] = self.max  # type: ignore

    @torch.no_grad()
    def gen_sol_obj(self, solutions):
        '''
        Args:
            solutions: (max_len, n_ants)
        '''
        u = solutions.permute(1, 0)
        v = torch.roll(u, shifts=-1, dims=1)
        length = torch.sum(self.distances[u[:, :-1], v[:, :-1]], dim=1)
        penalty_bool = self.gen_penalty_bool(u, self.n)
        penalty = []
        for ant_id in range(self.n_ants):
            ant_penalty = self.penalties[penalty_bool[ant_id]].sum()
            penalty.append(ant_penalty)
        return length + torch.stack(penalty)

    def gen_penalty_bool(self, sol, n):
        '''
        Args:
            sol: (n_ants, max_seq_len)
        '''
        n_ants = sol.size(0)
        seq_len = sol.size(1)
        expanded_nodes = torch.arange(n, device=self.device).repeat(n_ants, seq_len, 1) # (n_ants, seq_len, n)
        expanded_sol = torch.repeat_interleave(sol, n, dim=-1).reshape(n_ants, seq_len, n)
        return (torch.eq(expanded_nodes, expanded_sol)==0).all(dim=1)

    def gen_sol(self, invtemp=1.0, require_prob=False, sols=None):
        solutions = []
        log_probs_list = []

        cur_node = torch.zeros(size=(self.n_ants,), dtype=torch.int64, device=self.device)
        solutions = [cur_node]
        
        visit_mask = torch.ones(size=(self.n_ants, self.n), device=self.device) # 1) mask the visted regular node; 2) once return to depot, mask all
        depot_mask = torch.ones(size=(self.n_ants, self.n), device=self.device) 
        depot_mask[: , 0] = 0 # unmask the depot when 1) enough prize collected; 2) all nodes visited
        
        collected_prize = torch.zeros(size=(self.n_ants,), device=self.device)
        done = False
        # construction
        i = 0
        while not done:
            cur_node, log_prob = self.pick_node(
                visit_mask, depot_mask, cur_node, require_prob, invtemp, sols[i + 1] if sols is not None else None
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
            visit_mask, depot_mask = self.update_mask(visit_mask, depot_mask, cur_node, collected_prize)
            # check done
            done = self.check_done(cur_node)
        if require_prob:
            return torch.stack(solutions), torch.stack(log_probs_list)  # shape: [n_ant, max_seq_len]
        else:
            return torch.stack(solutions)
    
    def pick_node(self, visit_mask, depot_mask, cur_node, require_prob, invtemp=1.0, guiding_node=None):
        pheromone = self.pheromone[cur_node] 
        heuristic = self.heuristic[cur_node] 
        dist = (((pheromone ** self.alpha) * (heuristic ** self.beta)) ** invtemp) * visit_mask * depot_mask
        # set the prob of depot to 1 if dist is all 0
        dist[(dist==0).all(dim=1), 0] = 1
        dist = Categorical(dist)
        item = dist.sample() if guiding_node is None else guiding_node
        log_prob = dist.log_prob(item) if require_prob else None
        return item, log_prob  # (n_ants,)

    def update_mask(self, visit_mask, depot_mask, cur_node, collected_prize):
        # mask regular visted node
        visit_mask[self.ants_idx, cur_node] = 0
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
