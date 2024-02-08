from itertools import combinations

import torch
from torch.distributions import Categorical

class ACO():

    def __init__(self, 
                 distances,
                 prec_cons,
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
        
        self.problem_size = len(distances)
        self.distances  = distances
        self.prec_cons = prec_cons
        
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
            self.pheromone = pheromone

        self.heuristic = 1 / distances if heuristic is None else heuristic

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
                if self.min_max:
                    max = self.problem_size / self.lowest_cost
                    if self.max is None:
                        self.pheromone *= max/self.pheromone.max()
                    self.max = max
            
            self.update_pheromone(sols, costs)

        # Pairwise Jaccard similarity between paths
        edge_sets = []
        _sols = _sols.T.cpu().numpy()  # type: ignore
        for _p in _sols:
            edge_sets.append(set(zip(_p[:-1], _p[1:])))

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
            self.pheromone[best_tour[:-1], torch.roll(best_tour, shifts=-1)[:-1]] += 1.0/best_cost
        else:
            for i in range(self.n_ants):
                sol = sols[:, i]
                cost = costs[i]
                self.pheromone[sol[:-1], torch.roll(sol, shifts=-1)[:-1]] += 1.0/cost

        if self.min_max:
            self.pheromone[(self.pheromone > 1e-9) * (self.pheromone) < self.min] = self.min
            self.pheromone[self.pheromone > self.max] = self.max  # type: ignore

    @torch.no_grad()
    def gen_sol_costs(self, sols):
        '''
        Args:
            sols: torch tensor with shape (problem_size, n_ants)
        Returns:
                Lengths of sols: torch tensor with shape (n_ants,)
        '''
        assert sols.shape == (self.problem_size, self.n_ants)
        u = sols.T # shape: (n_ants, problem_size)
        v = torch.roll(u, shifts=-1, dims=1)  # shape: (n_ants, problem_size)
        return torch.sum(self.distances[u[:, :-1], v[:, :-1]], dim=1)

    def gen_sol(self, invtemp=1.0, require_prob=False, sols=None):
        '''
        Tour contruction for all ants
        Returns:
            sols: torch tensor with shape (problem_size, n_ants), sols[:, i] is the constructed tour of the ith ant
            log_probs: torch tensor with shape (problem_size, n_ants), log_probs[i, j] is the log_prob of the ith action of the jth ant
        '''
        start = torch.zeros(size=(self.n_ants,), dtype=torch.long, device=self.device)
        prec_cons_ants = self.prec_cons.repeat(self.n_ants, 1, 1)

        prec_cons_ants = self.update_prec_cons(prec_cons_ants, start)
        visit_mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        visit_mask[:, 0] = 0

        prec_mask = (prec_cons_ants == 0).all(dim=-1)  # [n_ant, problem_size]

        sols_list = []  # sols_list[i] is the ith move (tensor) for all ants
        sols_list.append(start)

        log_probs_list = []  # log_probs_list[i] is the ith log_prob (tensor) for all ants' actions

        prev = start
        for i in range(self.problem_size - 1):
            actions, log_probs = self.pick_move(
                prev, visit_mask, prec_mask, require_prob, invtemp, sols[i + 1] if sols is not None else None
            )
            sols_list.append(actions)
            if require_prob:
                log_probs_list.append(log_probs)

            prec_cons_ants = self.update_prec_cons(prec_cons_ants, actions)

            prev = actions

            if require_prob:
                visit_mask = visit_mask.clone()
                prec_mask = prec_mask.clone()

            visit_mask[torch.arange(self.n_ants, device=self.device), actions] = 0
            prec_mask = (prec_cons_ants == 0).all(dim=-1)

        if require_prob:
            return torch.stack(sols_list), torch.stack(log_probs_list)
        else:
            return torch.stack(sols_list)
        
    def pick_move(self, prev, mask1, mask2, require_prob, invtemp=1.0, guiding_node=None):
        '''
        Args:
            prev: tensor with shape (n_ants,), previous nodes for all ants
            mask: bool tensor with shape (n_ants, p_size), masks (0) for the visited cities
        '''
        pheromone = self.pheromone[prev]  # shape: (n_ants, p_size)
        heuristic = self.heuristic[prev]  # shape: (n_ants, p_size)  # type: ignore
        mask = mask1 * mask2
        dist = (((pheromone ** self.alpha) * (heuristic ** self.beta)) ** invtemp) * mask  # shape: (n_ants, p_size)
        dist = Categorical(dist)
        actions = dist.sample() if guiding_node is None else guiding_node  # shape: (n_ants,)
        log_probs = dist.log_prob(actions) if require_prob else None  # shape: (n_ants,)
        return actions, log_probs
    
    def update_prec_cons(self, prec_cons_ants, actions):
        '''
        Args:
            prec_cons_ants: [n_ants, p_size, p_size], [i,j,k] = 1 means:
                for ant i, node k precedes node j and k has yet been visited
            actions: [n_ants, ], the visited nodes for all ants
        '''
        prec_cons_ants[torch.arange(self.n_ants), : , actions] = 0
        return prec_cons_ants


if __name__ == '__main__':
    torch.set_printoptions(precision=3,sci_mode=False)
    from utils import gen_instance
    distances, adj, mask = gen_instance(5, 'cpu')
    aco = ACO(distances=distances, prec_cons=mask, n_ants=20)
    for i in range(20):
        cost = aco.run(1)
        print(cost)