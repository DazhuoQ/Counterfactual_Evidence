import random
import os
import numpy as np
import yaml
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, AmazonProducts, FacebookPagePage
from torch_geometric.transforms import NormalizeFeatures


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def find_pairs(nodes):
    pairs = []
    n = len(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((nodes[i], nodes[j]))
    return pairs


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def split_facebook_dataset(data, seed, test_size=1000):
    """
    Randomly split the FacebookPagePage dataset into train and test nodes.
    
    Parameters:
        data (torch_geometric.data.Data): PyG data object containing the graph.
        test_size (int): Number of test nodes to sample.
        seed (int): Random seed for reproducibility.

    Returns:
        train_mask (Tensor): Boolean mask for train nodes.
        test_mask (Tensor): Boolean mask for test nodes.
    """

    # Set random seed for reproducibility
    torch.manual_seed(seed)

    num_nodes = data.x.shape[0]  # Number of nodes in the graph
    indices = torch.randperm(num_nodes)  # Randomly shuffle indices
    
    test_indices = indices[:test_size]  # First 1000 nodes as test set
    train_indices = indices[test_size:]  # Remaining nodes as train set
    
    # Create boolean masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_indices] = True
    test_mask[test_indices] = True

    return train_mask, test_mask




def dataset_func(config, random_seed):
    # if config['data_name'] == 'Facebook' and config['exp_name']=='fair':
    #     return load_fair_data()
    if config['data_name'] == 'facebook100' and config['exp_name']=='fair':
        return torch.load('./datasets/facebook100/data.pt')
    elif config['data_name'] == 'aml' and config['exp_name']=='fair':
        return torch.load('./datasets/aml/data.pt')
    elif config['data_name'] == 'german' and config['exp_name']=='fair':
        return torch.load('./datasets/german/data.pt')
    elif config['data_name'] == 'bail' and config['exp_name']=='fair':
        return torch.load('./datasets/bail/data.pt')
    elif config['data_name'] == 'credit' and config['exp_name']=='fair':
        return torch.load('./datasets/credit/data.pt')
    elif config['data_name'] == 'facebookpagepage' and config['exp_name']=='fair':
        data = FacebookPagePage(root='./datasets/facebookpagepage')
        
        train_mask, test_mask = split_facebook_dataset(data, random_seed, config['num_test'])
        
        data.train_mask = train_mask
        # data.val_mask = val_mask
        data.test_mask = test_mask
        return data
    
    data_dir = "./datasets"
    data_name = config['data_name']
    data_size = config['data_size']
    num_class = config['output_dim']
    num_test = config['num_test']
    os.makedirs(data_dir, exist_ok=True)
    set_seed(random_seed)
    num_train_per_class = (data_size - num_test)//num_class
    data = Planetoid(root=data_dir, name=data_name, split='random', num_train_per_class=num_train_per_class, num_val=0, num_test=num_test)[0]
    return data

