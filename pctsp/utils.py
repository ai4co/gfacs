import torch
from torch_geometric.data import Data

K_n = {
    100: 4,
    200: 6,
    500: 9
}

def gen_prizes(n, device):
    prizes = torch.rand(size=(n,), device=device)
    return torch.cat((torch.tensor([0.], device=device), prizes))

def gen_penalties(n, device):
    K = K_n[n]
    beta = torch.rand(size=(n,), device=device) * 3 * K / n
    return torch.cat((torch.tensor([0.], device=device), beta))  # (n + 1,)

def gen_distance_matrix(coordinates):
    n_nodes = len(coordinates)
    distances = torch.norm(coordinates[:, None] - coordinates, dim=2, p=2)
    return distances

def gen_instance(n, device):
    coor = torch.rand((n + 1, 2), device=device)
    dist_mat = gen_distance_matrix(coor)
    prizes = gen_prizes(n, device)
    penalties = gen_penalties(n, device)
    return dist_mat, prizes, penalties

def gen_pyg_data(prizes, penalties, dist_mat):
    n_nodes = prizes.size(0)
    x = torch.stack((prizes, penalties)).permute(1, 0)  # (n + 1, 2)
    nodes = torch.arange(n_nodes, device=prizes.device)  # (n + 1,)
    v = nodes.repeat(n_nodes)
    u = torch.repeat_interleave(nodes, n_nodes)
    edge_index = torch.stack([u, v])  # (2, n + 1)
    edge_attr = dist_mat.reshape(-1,)
    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.unsqueeze(-1))
    return pyg_data

def load_val_dataset(n_nodes, device):
    '''Return [(pyg_data, distance matrix, prizes)]
    '''
    val_list = []
    dataset = torch.load(f'../data/pctsp/valDataset-{n_nodes}.pt', map_location=device)
    for inst in dataset:
        dist_mat, prizes, penalties = inst[:-2], inst[-2], inst[-1]
        val_list.append((dist_mat, prizes, penalties))
    return val_list

def load_test_dataset(n_nodes, device):
    test_list = []
    dataset = torch.load(f'../data/pctsp/testDataset-{n_nodes}.pt', map_location=device)
    for inst in dataset:
        dist_mat, prizes, penalties = inst[:-2], inst[-2], inst[-1]
        test_list.append((dist_mat, prizes, penalties))
    return test_list

if __name__ == "__main__":
    import pathlib
    pathlib.Path('../data/pctsp').mkdir(parents=False, exist_ok=True) 

    problem_sizes = [100, 200, 500]
    for n in problem_sizes:
        torch.manual_seed(123456)
        testDataset = []
        for _ in range(100):
            dist_mat, prizes, penalties = gen_instance(n, 'cpu')
            testDataset.append(torch.cat([dist_mat, prizes.unsqueeze(0), penalties.unsqueeze(0)], dim=0))
        torch.save(torch.stack(testDataset), f"../data/pctsp/testDataset-{n}.pt")

    for n in problem_sizes:
        torch.manual_seed(12345)
        valDataset = []
        for _ in range(30):
            dist_mat, prizes, penalties = gen_instance(n, 'cpu')
            valDataset.append(torch.cat([dist_mat, prizes.unsqueeze(0), penalties.unsqueeze(0)], dim=0))
        torch.save(torch.stack(valDataset), f"../data/pctsp/valDataset-{n}.pt")
