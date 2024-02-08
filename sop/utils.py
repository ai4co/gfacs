import pickle

import numpy as np
import torch
from torch_geometric.data import Data

def ordering_constraint_gen(n, rand=0.2, device='cpu'):
    adjmat = np.ones(shape=(n, n))
    adjmat[np.arange(n), np.arange(n)] = 0
    precmat = np.zeros(shape=(n, n))

    def update_adj_and_prec(adj_mat, mask, i, j):
        adj_mat[j, i] = 0
        mask[j, i] = 1
        return adj_mat, mask

    for i in range(1, n):
        adjmat, precmat = update_adj_and_prec(adjmat, precmat, 0, i)

    a = [i for i in range(1, n)]
    precede = [set() for _ in range(1, n)]
    for i in range(n - 3, -1, -1):
        for j in range(i + 1, n - 1):
            if np.random.rand() > rand:
                continue

            precede[i].add(j)
            for k in precede[j]:
                precede[i].add(k)

        for j in precede[i]:
            adjmat, precmat = update_adj_and_prec(adjmat, precmat, a[i], a[j])

    # A node with no out-going edges is determined as the last node
    # We add a dummy edge to the last node to make it have an out-going edge,
    # otherwise the model will have a problem with shape.
    # Note that this doesn't affect the solution thanks to the mask
    no_outgoing_edges = (adjmat.sum(axis=1) == 0)
    if no_outgoing_edges.any():
        idx = np.nonzero(no_outgoing_edges)[0]
        assert len(idx) == 1, "There should be only one node with no out-going edges"
        adjmat[idx, 0] = 1

    adjmat = torch.from_numpy(adjmat).to(device)
    precmat = torch.from_numpy(precmat).to(device)

    return adjmat, precmat


def cost_mat_gen(n, device):
    distances = torch.rand(size=(n, n), device=device)
    job_processing_cost = distances[0, :]
    distances[1:, :] += job_processing_cost
    return distances


def gen_instance(n, device):
    distance = cost_mat_gen(n, device=device)
    adj_mat, prec_mat = ordering_constraint_gen(n, device=device)
    return distance, adj_mat, prec_mat


def gen_pyg_data(distances, adj):
    edge_index = torch.nonzero(adj).T
    edge_attr = distances[adj.bool()].unsqueeze(-1)
    x = distances[0, :].unsqueeze(-1)
    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data


def load_val_dataset(n_node, device):
    with open(f"../data/sop/valDataset-{n_node}.pkl", "rb") as f:
        loaded_list = pickle.load(f)
    for i in range(len(loaded_list)):
        for j in range(len(loaded_list[0])):
            loaded_list[i][j] = loaded_list[i][j].to(device)
    return loaded_list


def load_test_dataset(n_node, device):
    with open(f"../data/sop/testDataset-{n_node}.pkl", "rb") as f:
        loaded_list = pickle.load(f)
    for i in range(len(loaded_list)):
        for j in range(len(loaded_list[0])):
            loaded_list[i][j] = loaded_list[i][j].to(device)
    return loaded_list


if __name__ == "__main__":
    import pathlib
    from tqdm import tqdm
    pathlib.Path('../data/sop').mkdir(parents=False, exist_ok=True) 

    problem_sizes = [100, 200, 500]
    for p_size in problem_sizes:
        torch.manual_seed(123456)
        np.random.seed(123456)
        testDataset = []
        for _ in tqdm(range(100)):
            distances, adj_mat, mask = gen_instance(p_size, 'cpu')
            testDataset.append([distances, adj_mat, mask])
        with open(f"../data/sop/testDataset-{p_size}.pkl", "wb") as f:
            pickle.dump(testDataset, f)

    for p_size in problem_sizes:
        torch.manual_seed(12345)
        np.random.seed(12345)
        valDataset = []
        for _ in tqdm(range(30)):
            distances, adj_mat, mask = gen_instance(p_size, 'cpu')
            valDataset.append([distances, adj_mat, mask])
        with open(f"../data/sop/valDataset-{p_size}.pkl", "wb") as f:
            pickle.dump(valDataset, f)
