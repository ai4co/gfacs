import os
import torch
from torch_geometric.data import Data


DEMAND_LOW = 1
DEMAND_HIGH = 9


def get_capacity(n: int, tam=False):
    if tam:
        capacity_list_tam = [
            (1, 10), (20, 30), (50, 40), (100, 50), (400, 150), (1000, 200), (2000, 300)  # (number of nodes, capacity)
        ]
        return list(filter(lambda x: x[0]<=n, capacity_list_tam))[-1][-1]

    capacity_dict = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.,
        200: 50.,
        500: 50.,
        1000: 50.,
        2000: 50.,
    }
    assert n in capacity_dict
    return capacity_dict[n]


def gen_instance(n, device, tam=False):
    """
    Implements data-generation method as described by Kool et al. (2019), Hou et al. (2023), and Son et al. (2023)

    * Kool, W., van Hoof, H., & Welling, M. (2019). Attention, Learn to Solve Routing Problems! (arXiv:1803.08475)
    * Hou, Q., Yang, J., Su, Y., Wang, X., & Deng, Y. (2023, February 1). Generalize Learned Heuristics to Solve Large-scale Vehicle Routing Problems in Real-time. The Eleventh International Conference on Learning Representations. https://openreview.net/forum?id=6ZajpxqTlQ
    * Son, J., et al. (2023). Meta-SAGE: Scale Meta-Learning Scheduled Adaptation with Guided Exploration for Mitigating Scale Shift on Combinatorial Optimization (arXiv:2306.02688)
    """
    locations = torch.rand(size=(n + 1, 2), device=device, dtype=torch.double)
    demands = torch.randint(low=DEMAND_LOW, high=DEMAND_HIGH + 1, size=(n, ), device=device, dtype=torch.double)
    demands_normalized = demands / get_capacity(n, tam)
    all_demands = torch.cat((torch.zeros((1, ), device=device, dtype=torch.double), demands_normalized))
    distances = gen_distance_matrix(locations)
    return all_demands, distances, locations


def gen_distance_matrix(tsp_coordinates):
    n_nodes = len(tsp_coordinates)
    distances = torch.norm(tsp_coordinates[:, None] - tsp_coordinates, dim=2, p=2, dtype=torch.double)
    distances[torch.arange(n_nodes), torch.arange(n_nodes)] = 1e-10  # note here
    return distances


def gen_pyg_data(demands, distances, device, k_sparse):
    n = demands.size(0)
    # First mask out self-loops by setting them to large values
    temp_dists = distances.clone()
    temp_dists[1:, 1:][torch.eye(n - 1, dtype=torch.bool, device=device)] = 1e9
    # sparsify
    # part 1:
    topk_values, topk_indices = torch.topk(temp_dists[1:, 1:], k = k_sparse, dim=1, largest=False)
    edge_index_1 = torch.stack([
        torch.repeat_interleave(torch.arange(n-1).to(topk_indices.device), repeats=k_sparse),
        torch.flatten(topk_indices)
    ]) + 1
    edge_attr_1 = topk_values.reshape(-1, 1)
    # part 2: keep all edges connected to depot
    edge_index_2 = torch.stack([ 
        torch.zeros(n - 1, device=device, dtype=torch.long), 
        torch.arange(1, n, device=device, dtype=torch.long),
    ])
    edge_attr_2 = temp_dists[1:, 0].reshape(-1, 1)
    edge_index_3 = torch.stack([ 
        torch.arange(1, n, device=device, dtype=torch.long),
        torch.zeros(n - 1, device=device, dtype=torch.long), 
    ])
    edge_index = torch.concat([edge_index_1, edge_index_2, edge_index_3], dim=1)
    edge_attr = torch.concat([edge_attr_1, edge_attr_2, edge_attr_2])

    x = demands
    # FIXME: append node type and coordinates into x
    pyg_data = Data(x=x.unsqueeze(1).float(), edge_attr=edge_attr.float(), edge_index=edge_index)
    return pyg_data


def load_test_dataset(n_node, k_sparse, device, tam=False):
    filename = f"../data/cvrp/testDataset-{'tam-' if tam else ''}{n_node}.pt"
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            f"File {filename} not found, please download the test dataset from the original repository."
        )
    dataset = torch.load(filename, map_location=device)

    test_list = []
    for i in range(len(dataset)):
        demands, position, distances = dataset[i, 0, :], dataset[i, 1:3, :], dataset[i, 3:, :]
        pyg_data = gen_pyg_data(demands, distances, device, k_sparse=k_sparse)
        test_list.append((pyg_data, demands, distances, position.T))
    return test_list


def load_val_dataset(n_node, k_sparse, device, tam=False):
    filename = f"../data/cvrp/valDataset-{'tam-' if tam else ''}{n_node}.pt"
    if not os.path.isfile(filename):
        dataset = []
        for i in range(50):
            demand, dist, position = gen_instance(n_node, device, tam)  # type: ignore
            instance = torch.vstack([demand, position.T, dist])
            dataset.append(instance)
        dataset = torch.stack(dataset)
        torch.save(dataset, filename)
    else:
        dataset = torch.load(filename, map_location=device)

    val_list = []
    for i in range(len(dataset)):
        demands, position, distances = dataset[i, 0, :], dataset[i, 1:3, :], dataset[i, 3:, :]
        pyg_data = gen_pyg_data(demands, distances, device, k_sparse=k_sparse)
        val_list.append((pyg_data, demands, distances, position.T))
    return val_list


if __name__ == '__main__':
    import pathlib
    import pickle
    pathlib.Path('../data/cvrp').mkdir(exist_ok=True)

    # TAM dataset
    for n in [100, 400, 1000]:  # problem scale
        torch.manual_seed(123456)
        inst_list = []
        for _ in range(100):
            demand, dist, position = gen_instance(n, 'cpu', tam=True)  # type: ignore
            instance = torch.vstack([demand, position.T, dist])
            inst_list.append(instance)
        testDataset = torch.stack(inst_list)
        torch.save(testDataset, f'../data/cvrp/testDataset-tam-{n}.pt')

    # main Dataset
    for scale in [200, 500, 1000]:
        with open(f"../data/cvrp/vrp{scale}_128.pkl", "rb") as f:
            dataset = pickle.load(f)

        inst_list = []
        for instance in dataset:
            depot_position, positions, demands, capacity = instance

            demands_torch = torch.tensor([0] + [d / capacity for d in demands], dtype=torch.float64)
            positions_torch = torch.tensor([depot_position] + positions, dtype=torch.float64)
            distmat_torch = gen_distance_matrix(positions_torch)
            inst_list.append(torch.vstack([demands_torch, positions_torch.T, distmat_torch]))

            test_dataset = torch.stack(inst_list)
            torch.save(test_dataset, f"../data/cvrp/testDataset-{scale}.pt")
