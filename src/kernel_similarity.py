import torch
import torch.nn.functional as F


def cosine_kernel(x, y):
    return F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0), dim=1)


def rbf_kernel(x, y, gamma=1.0):
    diff = torch.norm(x - y)
    return torch.exp(-gamma * diff ** 2)

def euclidean_kernel(x, y, sigma=1.0):
    distance_squared = torch.norm(x - y, p=2) ** 2
    
    # Compute the Euclidean kernel
    kernel_value = torch.exp(-distance_squared / (2 * sigma ** 2))
    
    return kernel_value


def update_node_features(data, iterations, alpha):
    x, edge_index = data.x, data.edge_index
    x = (x) / (x.norm(dim=1, keepdim=True) + 1e-6)
    num_nodes = x.size(0)
    new_features = torch.zeros_like(x)
    # new_features = data.x.clone().detach()
    
    for _ in range(iterations):

        # Sum weighted features of neighbors
        for i in range(num_nodes):
            neighbors = edge_index[1][edge_index[0] == i]  # Nodes where source nodes are i
            for neighbor in neighbors:
                # weight = rbf_kernel(x[i], x[neighbor])
                weight = cosine_kernel(x[i], x[neighbor])
                # weight = euclidean_kernel(x[i], x[neighbor])
                new_features[i] += weight * x[neighbor]
            new_features[i] = new_features[i] / len(neighbors)
            new_features[i] = alpha*x[i] + (1-alpha)*new_features[i]

        # Normalize features to prevent numerical instability
        new_features = (new_features) / (new_features.norm(dim=1, keepdim=True) + 1e-6)
        x = new_features
    
    return new_features


def KS(data, iterations, gamma):
    for _ in range(iterations):
        new_features = update_node_features(data, iterations, gamma)
    return new_features


def GCN_KS(data, iterations, gamma, gcn_model):

    new_features = gcn_model(data.x, data.edge_index)
    return new_features