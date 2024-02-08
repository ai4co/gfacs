from itertools import combinations

import torch
from torch.distributions import Categorical


class ACO():
    def __init__(self,
                 due_time, # [n,]
                 weights,  # [n,]
                 processing_time, # [n]
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
        
        self.n = len(due_time)
        self.due_time = due_time
        self.weights = weights
        self.processing_time = processing_time
        
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
            self.max = 1
        
        if pheromone is None:
            self.pheromone = torch.ones(size=(self.n + 1, self.n + 1), device=device)  # [n + 1, n + 1], includes dummy node 0
            if min_max:
                self.pheromone = self.pheromone * self.min
        else:
            self.pheromone = pheromone

        # A Weighted Population Update Rule for PACO Applied to the Single Machine Total Weighted Tardiness Problem
        # perfer jobs with smaller due time, [n + 1, n + 1], includes dummy node 0
        self.heuristic = (
            1 / torch.cat([torch.tensor([1], device=device), self.due_time])
        ).repeat(self.n + 1, 1) if heuristic is None else heuristic

        self.best_sol = None
        self.lowest_cost = float('inf')

        self.device = device
    
    def sample(self, invtemp=1.0, return_sol=False, K=1):
        n_ants = self.n_ants
        self.n_ants = K * n_ants

        sols, log_probs = self.gen_sol(invtemp=invtemp, require_prob=True)
        costs = self.gen_sol_costs(sols)

        self.n_ants = n_ants
        if return_sol:
            return costs, log_probs, sols
        else:
            return costs, log_probs

    @torch.no_grad()
    def run(self, n_iterations):
        for _ in range(n_iterations):
            sols = self.gen_sol(require_prob=False)
            costs = self.gen_sol_costs(sols)
            _sols = sols.clone()  # type: ignore

            best_cost, best_idx = costs.min(dim=0)
            if best_cost < self.lowest_cost:
                self.best_sol = sols[:, best_idx]  # type: ignore
                self.lowest_cost = best_cost
            self.update_pheromone(sols, costs)

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

        return self.lowest_cost, diversity
       
    @torch.no_grad()
    def update_pheromone(self, sols, costs):
        '''
        Args:
            sols: torch tensor with shape (problem_size, n_ants)
            costs: torch tensor with shape (n_ants,)
        '''
        self.pheromone = self.pheromone * self.decay 
        
        if self.elitist:
            best_cost, best_idx = costs.min(dim=0)
            best_tour= sols[:, best_idx]
            self.pheromone[best_tour[:-1], torch.roll(best_tour, shifts=-1)[:-1]] += 1.0/(best_cost + 1)
        
        else:
            for i in range(self.n_ants):
                sol = sols[:, i]
                cost = costs[i]
                self.pheromone[sol[:-1], torch.roll(sol, shifts=-1)[:-1]] += 1.0/(cost + 1)
        if self.min_max:
            self.pheromone[(self.pheromone > 1e-9) * (self.pheromone) < self.min] = self.min
            self.pheromone[self.pheromone > self.max] = self.max
    
    @torch.no_grad()
    def gen_sol_costs(self, sols):
        sols = (sols - 1).T  # due to the dummy node 0, (n_ants, problem_size)
        ants_time = self.processing_time[sols]  # corresponding processing time (n_ants, problem_size)
        ants_presum_time = torch.stack([ants_time[:, :i].sum(dim=1) for i in range(1, self.n + 1)]).T # presum (total) time (n_ants, problem_size)
        ants_due_time = self.due_time[sols]  # (n_ants, problem_size)
        ants_weights = self.weights[sols]  # (n_ants, problem_size)
        diff = ants_presum_time - ants_due_time
        diff[diff < 0] = 0
        ants_weighted_tardiness = (ants_weights * diff).sum(dim=1)
        return ants_weighted_tardiness  # (n_ants,)
        
    def gen_sol(self, invtemp=1.0, require_prob=False, sols=None):
        '''
        Tour contruction for all ants
        Returns:
            sols: torch tensor with shape (problem_size, n_ants), sols[:, i] is the constructed tour of the ith ant
            log_probs: torch tensor with shape (problem_size, n_ants), log_probs[i, j] is the log_prob of the ith action of the jth ant
        '''
        start = torch.zeros(size=(self.n_ants,), dtype=torch.long, device=self.device)
        
        visit_mask = torch.ones(size=(self.n_ants, self.n + 1), device=self.device)
        visit_mask[:, 0] = 0  # exlude the dummy node (starting node) 0
                
        sols_list = []  # sols_list[i] is the ith action (tensor) for all ants
        log_probs_list = []  # log_probs_list[i] is the ith log_prob (tensor) for all ants' actions
        
        prev = start
        for i in range(self.n):
            actions, log_probs = self.pick_move(
                prev, visit_mask, require_prob, invtemp, sols[i] if sols is not None else None
            )
            sols_list.append(actions)
            if require_prob:
                log_probs_list.append(log_probs)
                visit_mask = visit_mask.clone()
            prev = actions
            visit_mask[torch.arange(self.n_ants), actions] = 0
            
        if require_prob:
            return torch.stack(sols_list), torch.stack(log_probs_list)
        else:
            return torch.stack(sols_list)
        
    def pick_move(self, prev, mask, require_prob, invtemp=1.0, guiding_node=None):
        '''
        Args:
            prev: tensor with shape (n_ants,), previous nodes for all ants
            mask: bool tensor with shape (n_ants, p_size), masks (0) for the visited cities
        '''
        pheromone = self.pheromone[prev]  # shape: (n_ants, p_size)
        heuristic = self.heuristic[prev]  # shape: (n_ants, p_size)
        dist = (((pheromone ** self.alpha) * (heuristic ** self.beta)) ** invtemp) * mask  # shape: (n_ants, p_size)
        dist = Categorical(dist)
        actions = dist.sample() if guiding_node is None else guiding_node  # shape: (n_ants,)
        log_probs = dist.log_prob(actions) if require_prob else None  # shape: (n_ants,)
        return actions, log_probs
