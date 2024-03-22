import torch
from torch.distributions import Categorical
import random
import itertools
import numpy as np
from swapstar import swapstar
from functools import cached_property
import concurrent.futures
from itertools import combinations


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
        distances, # (n, n)
        demand,   # (n, )
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
        adaptive=False,
        capacity=CAPACITY,
        swapstar = False,
        positions = None,
    ):
        
        self.problem_size = len(distances)
        self.distances = distances
        self.capacity = capacity
        self.demand = demand
        
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.elitist = elitist or adaptive
        self.min_max = min_max
        self.adaptive = adaptive
        self.swapstar = swapstar
        self.positions = positions

        assert positions is not None if swapstar else True
        
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
        
        if self.adaptive:
            self.elite_pool = []

        self.heuristic = 1 / distances if heuristic is None else heuristic

        self.shortest_path = None
        self.lowest_cost = float('inf')

        self.device = device

    def sample(self, invtemp=1.0):
        paths, log_probs = self.gen_path(require_prob=True, invtemp=invtemp)  # type: ignore
        costs = self.gen_path_costs(paths)
        return costs, log_probs, paths

    def sample_nls(self, invtemp=1.0):
        paths, log_probs = self.gen_path(require_prob=True, invtemp=invtemp)  # type: ignore
        paths_raw = paths.clone()
        costs_raw = self.gen_path_costs(paths_raw).detach()
        self.multiple_swap_star(paths)
        costs = self.gen_path_costs(paths).detach()
        return costs, log_probs, paths, costs_raw, paths_raw

    @torch.no_grad()
    def multiple_swap_star(self, paths, indexes=None, inference=False):
        subroutes_all = []
        for i in range(paths.size(1)) if indexes is None else indexes:
            subroutes = get_subroutes(paths[:, i])
            subroutes_all.append((i, subroutes))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i, p in subroutes_all:
                future = executor.submit(
                    neural_swapstar,
                    self.demand_cpu,
                    self.distances_cpu,
                    self.heuristic_dist,
                    self.positions_cpu,
                    p,
                    limit=10000 if inference else self.problem_size // 10
                )
                futures.append((i, future))
            for i, future in futures:
                paths[:, i] = merge_subroutes(future.result(), paths.size(0), self.device)

    @cached_property
    @torch.no_grad()
    def heuristic_dist(self):
        heu = self.heuristic.detach().cpu().numpy()  # type: ignore
        return (1 / (heu/heu.max(-1, keepdims=True) + 1e-5))

    @torch.no_grad()
    def run(self, n_iterations):
        for _ in range(n_iterations):
            paths = self.gen_path(require_prob=False)
            _paths = paths.clone()   # type: ignore

            if self.adaptive:
                self.improvement_phase(paths, costs)
            
            if self.swapstar:
                self.multiple_swap_star(paths, inference=True)

            costs = self.gen_path_costs(paths)

            improved = False
            best_cost, best_idx = costs.min(dim=0)
            if best_cost < self.lowest_cost:
                self.shortest_path = paths[:, best_idx].clone()  # type: ignore
                self.lowest_cost = best_cost.item()
                if self.adaptive:
                    self.intensification_phase(paths, costs, best_idx)
                if self.min_max:
                    max = self.problem_size / self.lowest_cost
                    if self.max is None:
                        self.pheromone *= max / self.pheromone.max()
                    self.max = max
                improved = True

            if not self.adaptive or improved:           
                self.update_pheromone(paths, costs)
                if self.adaptive:
                    self.elite_pool.insert(0, (self.shortest_path, self.lowest_cost))
                    if len(self.elite_pool) > 5:  # pool_size = 5
                        del self.elite_pool[5:]
            else:
                self.diversification_phase()

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
            best_tour = paths[:, best_idx]
            self.pheromone[best_tour[:-1], torch.roll(best_tour, shifts=-1)[:-1]] += 1.0/best_cost
        
        else:
            for i in range(self.n_ants):
                path = paths[:, i]
                cost = costs[i]
                self.pheromone[path[:-1], torch.roll(path, shifts=-1)[:-1]] += 1.0/cost
                
        if self.min_max:
            self.pheromone[(self.pheromone > 1e-9) * (self.pheromone) < self.min] = self.min
            self.pheromone[self.pheromone > self.max] = self.max  # type: ignore
        
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
    
    # ======== code for adaptive elitist AS ========
    # These code are unrelated to DeepACO, and are kept for comparisons.
    def insertion_single(self, route, index):
        # route starts from 0, terminates with 0
        insertion_cost = (((self.distances[p1, index]+self.distances[index,p2]-self.distances[p1, p2]).item(), i) 
                          for i,(p1,p2) in enumerate(zip(route,route[1:])))
        min_deltacost, min_index = min(insertion_cost)
        return min_index, min_deltacost
    
    def insertion(self, node_indexes, shuffle = False):
        route = [node_indexes[0].item()]*2
        cost = 0
        if shuffle:
            perm = torch.randperm(len(node_indexes)-1) + 1
            nodes = node_indexes[perm]
        else:
            nodes = node_indexes[1:]
        for i in nodes:
            bestpos, deltacost = self.insertion_single(route, i)
            route.insert(bestpos + 1, i.item())
            cost += deltacost
        return route, cost


    @torch.no_grad()
    def N1_neighbourhood(self, subroutes, demands, count = 5):
        # N1 neighbourhood: Pick a random node and insert it in other subroutes.
        best_insertion = (None, 0.0)
        for _ in range(count):
            subroute_index = random.randint(0, len(subroutes) - 1)
            route = subroutes[subroute_index]
            node_index = random.randint(1, len(route) - 2) # exclude depots
            pred, node, next = route[node_index - 1: node_index + 2]
            demand = self.demand[node]
            avaliable = demands + demand <= self.capacity
            avaliable[subroute_index] = False
            if not avaliable.any(): # no avaliable subroute
                continue
            cost = self.distances[pred, next] - self.distances[pred, node] - self.distances[node, next]
            for i, r in itertools.compress(enumerate(subroutes), avaliable):
                loc, insertion_cost = self.insertion_single(r, node)
                insertion_cost += cost
                if insertion_cost < best_insertion[1]:
                    best_insertion = ((subroute_index, node_index, i, loc + 1), insertion_cost)

        if best_insertion[0] is not None: # perform insertion
            sri, sni, tri, tni = best_insertion[0]
            subroutes = subroutes[:]
            source_route, target_route = subroutes[sri], subroutes[tri]
            node = subroutes[sri][sni]
            subroutes[tri] = torch.cat([target_route[:tni], node.unsqueeze(0), target_route[tni:]])
            if len(subroutes[sri])==3:
                del subroutes[sri]
            else:
                subroutes[sri] = torch.cat([source_route[:sni], source_route[sni + 1:]])
            return subroutes, best_insertion[1]
        else:
            return best_insertion
    
    @torch.no_grad()
    def N2_neighbourhood(self, subroutes, demands, count = 5):
        # N2 neighbourhood: Randomly swap 2 nodes and insert them in the best position.
        best_insertion = (None, 0.0)
        for _ in range(count):
            sr1_index, sr2_index = np.random.choice(len(subroutes), size=2, replace=False)
            sr1, sr2 = subroutes[sr1_index], subroutes[sr2_index]
            node1_index = random.randint(1, len(sr1) - 2)
            pred1, node1, next1 = sr1[node1_index - 1:node1_index + 2]
            demand1 = self.demand[node1]
            # avaliable nodes to swap
            avaliable = torch.bitwise_and(
                demands[sr2_index] + demand1 - self.demand[sr2] <= self.capacity,
                demands[sr1_index] - demand1 + self.demand[sr2] <= self.capacity,
            )
            avaliable[0] = avaliable[-1] = False
            if not avaliable.any():
                continue
            # remove node1 from sr1
            cost = self.distances[pred1, next1] - self.distances[pred1, node1] - self.distances[node1, next1]
            sr1_mod = torch.concat([sr1[:node1_index], sr1[node1_index + 1:]])
            # choose a node from sr2
            avaliable_index = torch.arange(len(sr2))[avaliable]
            node2_index = np.random.choice(avaliable_index)
            pred2, node2, next2 = sr2[node2_index - 1: node2_index + 2]
            # remove node2 from sr2
            cost += self.distances[pred2, next2] - self.distances[pred2, node2] - self.distances[node2, next2]
            sr2_mod = torch.concat([sr2[:node2_index], sr2[node2_index + 1:]])
            # insert node1 into sr2_mod
            loc1, inscost1 = self.insertion_single(sr2_mod, node1)
            cost += inscost1
            sr2_mod = torch.concat([sr2_mod[:loc1 + 1], node1.unsqueeze(0), sr2_mod[loc1 + 1:]])
            # insert node2 into sr1_mod
            loc2, inscost2 = self.insertion_single(sr1_mod, node2)
            cost += inscost2
            sr1_mod = torch.concat([sr1_mod[:loc2 + 1], node2.unsqueeze(0), sr1_mod[loc2 + 1:]])
            if cost < best_insertion[1]:
                best_insertion = ((sr1_index, sr1_mod, sr2_index, sr2_mod), cost)

        if best_insertion[0] is not None: # perform insertion
            sr1_index, sr1, sr2_index, sr2 = best_insertion[0]
            subroutes = subroutes[:]
            subroutes[sr1_index] = sr1
            subroutes[sr2_index] = sr2
            return subroutes, best_insertion[1]
        else:
            return best_insertion

    @torch.no_grad()
    def improvement_phase(self, paths, costs, topk = 5):
        # local search
        if topk <= 0 or topk >= self.n_ants:
            target_indexes = range(paths.size(1))
        else:
            target_indexes = costs.topk(5, largest=False).indices

        for i in target_indexes:
            subroutes = get_subroutes(paths[:, i], end_with_zero=False)
            # ILS (not implemented)
            pass
            # insertion
            new_subroutes = []
            new_cost=0
            for r in subroutes:
                new_subroute, c = self.insertion(r)
                new_cost += c
                new_subroutes.append(new_subroute)
            if new_cost < costs[i]:
                paths[:, i] = merge_subroutes(new_subroutes, paths.size(0), self.device)
                costs[i] = new_cost
    
    @torch.no_grad()
    def intensification_phase(self, paths, costs, best_idx):
        ogroute, ogcost = self.shortest_path, self.lowest_cost
        subroutes = get_subroutes(ogroute, end_with_zero=True)
        demands = torch.tensor([self.demand[r].sum() for r in subroutes], device=self.device)
        # print(*subroutes, sep='\n')
        best_neighbour = (None, 0.0)
        for func in [self.N1_neighbourhood, self.N2_neighbourhood]:
            route, cost = func(subroutes, demands)
            if cost < best_neighbour[1]:
                best_neighbour = (route, cost)
        if best_neighbour[0] is not None:
            self.shortest_path = merge_subroutes(
                best_neighbour[0], self.shortest_path.size(0), self.device  # type: ignore
            )
            self.lowest_cost = ogcost + best_neighbour[1]
            paths[:, best_idx] = self.shortest_path
            costs[best_idx] = self.lowest_cost

    @torch.no_grad()
    def diversification_phase(self):
        # reinitialize pheromone trails
        self.pheromone = self.pheromone * (self.decay * 0.5) + 0.01
        for path, cost in self.elite_pool:
            self.pheromone[path[:-1], torch.roll(path, shifts=-1)[:-1]] += 1.0 / cost


def neural_swapstar(demand, distances, heu_dist, positions, p, disturb=5, limit=10000):
    p0 = p
    p1 = swapstar(demand, distances, positions, p0, count = limit)
    p2 = swapstar(demand, heu_dist, positions, p1, count = disturb)
    p3 = swapstar(demand, distances, positions, p2, count = limit)
    return p3
