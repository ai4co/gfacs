from itertools import combinations

import torch
from torch.distributions import Categorical


class ACO():
    def __init__(self,
                 distances,
                 prizes,
                 max_len,
                 n_ants=20, 
                 decay=0.9,
                 alpha=1,
                 beta=1,
                 elitist=False,
                 min_max=False,
                 pheromone=None,
                 heuristic=None,
                 min=None,
                 device='cpu',
                 k_sparse = None
                 ):

        self.n = len(prizes)
        self.distances = distances
        self.prizes = prizes
        self.max_len = max_len

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

        self.heuristic = self.prizes.unsqueeze(0) / self.distances if heuristic is None else heuristic

        self.Q = 1 / prizes.sum()

        self.alltime_best_sol = None
        self.alltime_best_obj = 0

        self.device = device

        if heuristic is None:
            assert k_sparse
            self.sparsify(k_sparse)
        self.add_dummy_node()

    def add_dummy_node(self):
        '''
        One has to sparsify the graph first before adding dummy node
        distance: 
                [[1e9 , x   , x   , 0  ],
                [x   , 1e9 , x   , 0  ],
                [x   , x   , 1e9 , 0  ],
                [1e10, 1e10, 1e10, 0  ]]
        pheromone: [1]
        heuristic: [>0]
        prizes: [x,x,...,0]
        '''
        self.prizes = torch.cat((self.prizes, torch.tensor([1e-10], device=self.device)))
        distances = torch.cat((self.distances, 1e10 * torch.ones(size=(1, self.n), device=self.device)), dim=0)
        self.distances = torch.cat((distances, 1e-10 + torch.zeros(size=(self.n + 1, 1), device=self.device)), dim=1)

        self.heuristic = torch.cat((self.heuristic, torch.zeros(size=(1, self.n), device=self.device)), dim=0) # cannot reach other nodes from dummy node
        self.heuristic = torch.cat((self.heuristic, torch.ones(size=(self.n + 1, 1), device=self.device)), dim=1)

        self.pheromone = torch.ones_like(self.distances)
        self.distances[self.distances == 1e-10] = 0
        self.prizes[-1] = 0

    @torch.no_grad()
    def sparsify(self, k_sparse):
        '''
        Sparsify the OP graph to obtain the heuristic information 
        used for vanilla ACO baselines
        '''
        _, topk_indices = torch.topk(self.distances, 
                                        k=k_sparse, # to include the dummy node 
                                        dim=1, largest=False)
        edge_index_u = torch.repeat_interleave(
            torch.arange(len(self.distances), device=self.device),
            repeats=k_sparse
            )
        edge_index_v = torch.flatten(topk_indices)
        sparse_distances = torch.ones_like(self.distances) * 1e10
        sparse_distances[edge_index_u, edge_index_v] = self.distances[edge_index_u, edge_index_v]
        self.heuristic = self.prizes.unsqueeze(0) / sparse_distances

    def sample(self, invtemp=1.0, return_sol=False, K=1):
        n_ants = self.n_ants
        self.n_ants = K * self.n_ants

        sols, log_probs = self.gen_sol(invtemp=invtemp, require_prob=True)
        objs = self.gen_sol_obj(sols)

        self.n_ants = n_ants

        if return_sol:
            return objs, log_probs, sols
        else:
            return objs, log_probs

    @torch.no_grad()
    def run(self, n_iterations):
        for _ in range(n_iterations):
            sols = self.gen_sol(require_prob=False)
            objs = self.gen_sol_obj(sols)
            _paths = sols.clone()  # type: ignore

            sols = sols.T  # type: ignore
            best_obj, best_idx = objs.max(dim=0)
            if best_obj > self.alltime_best_obj:
                self.alltime_best_obj = best_obj
                self.alltime_best_sol = sols[best_idx]
                if self.min_max:
                    max = self.alltime_best_obj * self.n * self.Q
                    if self.max is None:
                        self.pheromone *= max/self.pheromone.max()
                    self.max = max
            self.update_pheromone(sols, objs, best_obj, best_idx)

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

        return self.alltime_best_obj, diversity

    @torch.no_grad()
    def update_pheromone(self, sols, objs, best_obj, best_idx):
        self.pheromone = self.pheromone * self.decay 
        if self.elitist:
            best_sol= sols[best_idx]
            self.pheromone[best_sol[:-1], torch.roll(best_sol, shifts=-1)[:-1]] += self.Q * best_obj

        else:
            for i in range(self.n_ants):
                sol = sols[i]
                obj = objs[i]
                self.pheromone[sol[:-1], torch.roll(sol, shifts=-1)[:-1]] += self.Q * obj

        if self.min_max:
            self.pheromone[(self.pheromone>1e-9) * (self.pheromone)<self.min] = self.min
            self.pheromone[self.pheromone>self.max] = self.max  # type: ignore

    @torch.no_grad()
    def gen_sol_obj(self, solutions):
        '''
        Args:
            solutions: (max_len, n_ants)
        '''
        objs = self.prizes[solutions.T].sum(dim=1)
        return objs

    def gen_sol(self, invtemp=1.0, require_prob=False, sols=None):
        '''
        Solution contruction for all ants
        '''
        solutions = []
        log_probs_list = []

        solutions = [torch.zeros(size=(self.n_ants,), device=self.device, dtype=torch.int64)]
        mask = torch.ones(size=(self.n_ants, self.n + 1), device=self.device)
        done = torch.zeros(size=(self.n_ants,), device=self.device)
        travel_dis = torch.zeros(size=(self.n_ants,), device=self.device)
        cur_node = torch.zeros(size=(self.n_ants,), dtype=torch.int64, device=self.device)
        
        mask = self.update_mask(travel_dis, cur_node, mask)
        done = self.check_done(mask)
        # construction
        i = 0
        while not done:
            if sols is not None and i < len(sols) - 1:
                nxt_node, log_prob = self.pick_node(mask, cur_node, require_prob, guiding_node=sols[i + 1])
            else:
                nxt_node, log_prob = self.pick_node(mask, cur_node, require_prob, invtemp=invtemp)

            # update solution and log_probs
            solutions.append(nxt_node) 
            log_probs_list.append(log_prob)
            # update travel_dis, cur_node and mask
            travel_dis += self.distances[cur_node, nxt_node]
            cur_node = nxt_node
            if require_prob:
                mask = mask.clone()
            mask = self.update_mask(travel_dis, cur_node, mask)
            # check done
            done = self.check_done(mask)

            i += 1

        if require_prob:
            return torch.stack(solutions), torch.stack(log_probs_list)  # shape: [n_ant, max_seq_len]
        else:
            return torch.stack(solutions)

    def pick_node(self, mask, cur_node, require_prob, invtemp=1.0, guiding_node=None):
        pheromone = self.pheromone[cur_node]  # shape: (n_ants, p_size + 1)
        heuristic = self.heuristic[cur_node]  # shape: (n_ants, p_size + 1)
        dist = (((pheromone ** self.alpha) * (heuristic ** self.beta)) ** invtemp) * mask
        # set the prob of dummy node to 1 if dist is all 0
        dist[(dist==0).all(dim=1), -1] = 1
        dist = dist / dist.sum(dim=1, keepdim=True)  # This should be done for numerical stability
        dist = Categorical(dist)
        item = dist.sample() if guiding_node is None else guiding_node
        log_prob = dist.log_prob(item) if require_prob else None
        return item, log_prob  # (n_ants,)

    def update_mask(self, travel_dis, cur_node, mask):
        '''
        Args:
            travel_dis: (n_ants,)
            cur_node: (n_ants,)
            mask: (n_ants, n + 1)
        '''
        mask[torch.arange(self.n_ants), cur_node] = 0

        dist_mask = (travel_dis.unsqueeze(1) + self.distances[cur_node] + self.distances[:, 0].unsqueeze(0)) <= self.max_len
        mask = mask * dist_mask

        mask[:, -1] = 0 # mask the dummy node for all ants
        go2dummy = (mask[:, :-1] == 0).all(dim=1) # unmask the dummy node for these ants
        mask[go2dummy, -1] = 1
        return mask

    def check_done(self, mask):
        # is all masked ?
        return (mask[:, :-1] == 0).all()
