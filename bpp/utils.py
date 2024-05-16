import torch
from torch_geometric.data import Data


# Emanuel Falkenauer. A hybrid grouping genetic algorithm for bin packing. Journal of Heuristics,2:5–30, 1996.
CAPACITY = 150
DEMAND_LOW = 20
DEMAND_HIGH = 100


def gen_instance(n, device):
    demands = torch.randint(low=DEMAND_LOW, high=DEMAND_HIGH + 1, size=(n,), device=device)
    all_demands = torch.cat((torch.zeros((1,), device=device), demands)) / CAPACITY  # normalize using capacity
    return all_demands  # (n + 1)


def gen_pyg_data(demands, device='cpu'):
    n = demands.size(0)
    nodes = torch.arange(n, device=device)
    u = nodes.repeat(n)
    v = torch.repeat_interleave(nodes, n)
    edge_index = torch.stack((u, v))
    # if edge_attr > 1, remove the edge
    mask = (demands[edge_index[0]] + demands[edge_index[1]] <= 1).squeeze()
    edge_index = edge_index[:, mask]
    edge_attr = torch.ones((edge_index.size(1), 1), device=device)
    x = demands
    pyg_data = Data(x=x.unsqueeze(1), edge_attr=edge_attr, edge_index=edge_index)
    return pyg_data


def load_val_dataset(problem_size, device):
    dataset = torch.load(f'../data/bpp/valDataset-{problem_size}.pt', map_location=device)
    return dataset


def load_test_dataset(problem_size, device):
    dataset = torch.load(f'../data/bpp/testDataset-{problem_size}.pt', map_location=device)
    return dataset


if __name__ == '__main__':
    import pathlib
    pathlib.Path('../data/bpp').mkdir(parents=False, exist_ok=True) 

    for n in [120, 250, 500]:
        torch.manual_seed(123456)
        inst_list = []
        for _ in range(100):
            demands = gen_instance(n, 'cpu')
            inst_list.append(demands)
        testDataset = torch.stack(inst_list)
        torch.save(testDataset, f'../data/bpp/testDataset-{n}.pt')

    for n in [120, 250, 500]:
        torch.manual_seed(12345)
        inst_list = []
        for _ in range(100):
            demands = gen_instance(n, 'cpu')
            inst_list.append(demands)
        valDataset = torch.stack(inst_list)
        torch.save(valDataset, f'../data/bpp/valDataset-{n}.pt')
