import sys
from tqdm.std import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from src.utils import *
from src.kernel_similarity import KS
from src.model import get_model
from src.index_algorithm import *
from src.index_algorithm_wo_SP import weighted_kmeans_wo_SP
from src.index_algorithm_wo_WC import weighted_kmeans_wo_WC

def localIndex(config, probs, data, rangek, method):
    theta = config['theta']
    random_seed = config['random_seed']
    iterations = config['iterations']
    alpha = config['alpha']

    set_seed(random_seed)

    # query node v
    # test_nodes = torch.where(data.test_mask)[0]
    # v = test_nodes[random_seed]
    v = random_seed
    pred_label_v = probs[v].argmax().item()
    
    # compute KS
    print('start KS computation')
    new_features = KS(data, iterations, alpha)
    # feats = new_features[test_nodes.numpy()].numpy()
    feats = new_features

    # Perform weighted K-means clustering with cosine distance
    print('start index computation')
    if method == 'index_wo_WC':
        centroids, labels = weighted_kmeans_wo_WC(feats)
    elif method == 'index_wo_SP':
        centroids, labels = weighted_kmeans_wo_SP(feats)
    elif method == 'index':
        centroids, labels = weighted_kmeans(feats)

    optimal_cluster = None
    optimal_cluster_value = np.inf
    for partition in centroids:
        partition_value = np.min([cosine_distance(feats[v], c.reshape(1, -1)) for c in partition])
        cluster_index = np.argmin([cosine_distance(feats[v], c.reshape(1, -1)) for c in partition])
        cluster_indices = np.where(labels == cluster_index)[0]
        if partition_value <  optimal_cluster_value:
            optimal_cluster = cluster_indices
            optimal_cluster_value = partition_value
    cluster_indices = optimal_cluster

    # Start time
    elapsed_times = []
    avg_sims = []        
    for k in rangek:  
        start_time = time.time()

        rlt_dict = {}
        for u in tqdm(cluster_indices, desc='num_nodes'):
            if v == u:
                continue
            pred_label_u = probs[u].argmax().item()
            sim_score = F.cosine_similarity(new_features[v].unsqueeze(0), new_features[u].unsqueeze(0))
            if pred_label_v != pred_label_u:
                rlt_dict[(v, u)] = sim_score.item()

        # find top-k
        top_k = sorted(rlt_dict.items(), key=lambda item: item[1], reverse=True)[:k]

        # End time
        end_time = time.time()
        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        avg_sim = np.mean([value for _, value in top_k])
        elapsed_times.append(elapsed_time)
        avg_sims.append(avg_sim)


    return elapsed_times, avg_sims


def average_pairwise_cosine_similarity(feats, labels, k=10):
    """
    Calculate the average pairwise cosine similarity for each cluster and return
    the node pairs with the highest average similarity.
    
    Parameters:
        feats (Tensor): Feature matrix of shape (num_nodes, num_features)
        labels (Tensor): Tensor of cluster labels for each node
        k (int): Number of clusters
    
    Returns:
        avg_similarities (Tensor): Average pairwise cosine similarity for each cluster
        max_cluster (int): The cluster index with the highest average similarity
        max_similarity (float): The highest average cosine similarity score
        max_cluster_pairs (List[Tuple[int, int]]): List of node pairs in max_cluster
    """
    avg_similarities = []
    max_cluster_pairs = None  # To store the node pairs for the max cluster
    max_similarity = -1  # Initialize with a very low value

    # Iterate through each cluster
    for cluster_idx in range(k):
        # Get the nodes belonging to the current cluster
        cluster_indices = torch.nonzero(labels == cluster_idx).squeeze()  # Get original node indices for the cluster
        cluster_nodes = feats[cluster_indices]  # Get the feature vectors for the current cluster
        print(cluster_indices)
        
        if cluster_nodes.size(0) <= 1:
            avg_similarities.append(0.0)  # No pairwise comparison if only 1 or 0 nodes
            continue
        
        # Calculate all pairwise cosine similarities in the cluster
        similarities = []
        node_pairs = []  # To store the pairs of nodes
        num_nodes_in_cluster = cluster_nodes.size(0)
        print(num_nodes_in_cluster)

        for i in range(num_nodes_in_cluster):
            for j in range(i + 1, num_nodes_in_cluster):
                sim = F.cosine_similarity(cluster_nodes[i].unsqueeze(0), cluster_nodes[j].unsqueeze(0)).item()
                similarities.append(sim)
                node_pairs.append((cluster_indices[i].item(), cluster_indices[j].item()))  # Store the indices of the pair
        
        # Calculate the average similarity for the current cluster
        avg_similarity = torch.tensor(similarities).mean().item()
        avg_similarities.append(avg_similarity)
        
        # If this is the max cluster, store its node pairs
        if avg_similarity > max_similarity:
            max_similarity = avg_similarity
            max_cluster_pairs = node_pairs  # Store the pairs contributing to max similarity

    # Convert to a tensor for easy manipulation
    avg_similarities = torch.tensor(avg_similarities)

    # Find the cluster with the highest average similarity
    max_similarity, max_cluster = avg_similarities.max(0)
    
    return avg_similarities, max_cluster.item(), max_similarity.item(), max_cluster_pairs


def globalIndex(config, probs, data, rangek, method):

    random_seed = config['random_seed']
    iterations = config['iterations']
    alpha = config['alpha']

    set_seed(random_seed)

    test_nodes = torch.where(data.test_mask)[0]

    # compute KS
    print('start KS computation')
    new_features = KS(data, iterations, alpha)
    feats = new_features[test_nodes]

    # Perform weighted K-means clustering with cosine distance
    print('start index computation')
    if method == 'index_wo_WC':
        centroids, labels = weighted_kmeans_wo_WC(feats)
    elif method == 'index_wo_SP':
        centroids, labels = weighted_kmeans_wo_SP(feats)
    elif method == 'index':
        centroids, labels = weighted_kmeans(feats)

    print('start find opt cluster')
    avg_similarities, max_cluster, max_similarity, max_cluster_pairs = average_pairwise_cosine_similarity(feats, labels)
    print(avg_similarities)
    print(max_cluster)
    print(max_similarity)


    elapsed_times = []
    avg_sims = []
    for k in rangek:
        rlt_dict = {}
        start_time = time.time()
        for v, u in tqdm(max_cluster_pairs, desc='num_pairs'):
            if v == u:
                continue
            pred_label_v = probs[test_nodes[v]].argmax().item()
            pred_label_u = probs[test_nodes[u]].argmax().item()
            sim_score = F.cosine_similarity(feats[v].unsqueeze(0), feats[u].unsqueeze(0))
            if pred_label_v != pred_label_u:
                rlt_dict[(v, u)] = sim_score.item()
        # find top-k
        top_k = sorted(rlt_dict.items(), key=lambda item: item[1], reverse=True)[:k]
        elapsed_time = time.time() - start_time

        avg_sim = np.mean([value for _, value in top_k])
        elapsed_times.append(elapsed_time)
        avg_sims.append(avg_sim)
        

    return elapsed_times, avg_sims


def localCE(data, probs, iterations, alpha, theta, rangek, random_seed):
    # query node v
    # test_nodes = torch.where(data.test_mask)[0]
    # v = test_nodes[random_seed]
    v = random_seed
        
    if probs.shape[-1] ==1:
        pred_label_v = 1 if probs[v] > 0.5 else 0
    else:
        pred_label_v = probs[v].argmax().item() 

    new_features = KS(data, iterations, alpha)


    # Start time
    elapsed_times = []
    avg_sims = []
    for k in rangek:  
        start_time = time.time()

        rlt_dict = {}
        for u in tqdm(range(data.x.size(0)), desc='num_nodes'):
            if v == u:
                continue
            if probs.shape[-1] ==1:
                pred_label_u = 1 if probs[u] > 0.5 else 0
            else:
                pred_label_u = probs[u].argmax().item()
            sim_score = F.cosine_similarity(new_features[v].unsqueeze(0), new_features[u].unsqueeze(0))
            if pred_label_v != pred_label_u:
                rlt_dict[(v, u)] = sim_score.item()

        # find top-k
        top_k = sorted(rlt_dict.items(), key=lambda item: item[1], reverse=True)[:k]

        # End time
        end_time = time.time()
        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        avg_sim = np.mean([value for _, value in top_k])
        elapsed_times.append(elapsed_time)
        avg_sims.append(avg_sim)

    return elapsed_times, avg_sims


def globalCE(data, probs, iterations, alpha, theta, rangek):

    test_nodes = torch.where(data.test_mask)[0]
    candidate_pairs = find_pairs(test_nodes)
    # compute KS
    new_features = KS(data, iterations, alpha)
    elapsed_times = []
    avg_sims = []
    for k in rangek:
        rlt_dict = {}
        start_time = time.time()
        for v, u in tqdm(candidate_pairs, desc='num_pairs'):
            if v == u:
                continue
            pred_label_v = probs[v].argmax().item()
            pred_label_u = probs[u].argmax().item()
            sim_score = F.cosine_similarity(new_features[v].unsqueeze(0), new_features[u].unsqueeze(0))
            if pred_label_v != pred_label_u and sim_score.item() >= theta:
                rlt_dict[(v.item(), u.item())] = sim_score.item()
        # find top-k
        top_k = sorted(rlt_dict.items(), key=lambda item: item[1], reverse=True)[:k]
        elapsed_time = time.time() - start_time

        avg_sim = np.mean([value for _, value in top_k])
        elapsed_times.append(elapsed_time)
        avg_sims.append(avg_sim)

    return elapsed_times, avg_sims




def main(config, rangek, output_dir):
    # Load configuration
    data_name = config['data_name']
    model_name = config['model_name']
    iterations = config['iterations']
    alpha = config['alpha']
    random_seed = config['random_seed']
    k = config['k']
    theta = config['theta']
    exp_name = config['exp_name']
    fine_tune_size = config['fine_tune_size']
    set_seed(random_seed)


    os.makedirs(f'./results/{data_name}', exist_ok=True)
    
    # Get input graph
    data = dataset_func(config, random_seed)

    # Ready the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(config)
    model.load_state_dict(torch.load('models/{}_{}_model.pth'.format(data_name, model_name)))
    model.eval()
    model.to(device)

    out = model(data.x, data.edge_index)
    probs = out
    
    print('start algorithm')

    if config['task']=='local':
        if config["method"] == 'baseline':
            elapsed_times, avg_sims = localCE(data, probs, iterations, alpha, theta, rangek, random_seed)
        else:
            elapsed_times, avg_sims = localIndex(config, probs, data, rangek, config["method"])
    elif config['task']=='global':
        if config["method"] == 'baseline':
            elapsed_times, avg_sims = globalCE(data, probs, iterations, alpha, theta, rangek)
        else:
            elapsed_times, avg_sims = globalIndex(config, probs, data, rangek, config["method"])

    with open(config["output_path"], "a") as f:
        for i, k in enumerate(rangek):
            f.write(f'k:{k}, task:{config["task"]}, method:{config["method"]}, model_name:{model_name}, avg_sim:{avg_sims[i]}, elapsed_time:{elapsed_times[i]}\n')


    # Save experiment settings
    print('Dataset: '+str(config['data_name']))
    print('Model: '+str(config['model_name']))

if __name__ == "__main__":

    rangek = [100, 200, 300, 400, 500, 600]

    
    config = load_config("config.yaml")
    config["task"] = "global"
    config["output_path"] = "./results/{}/{}.txt".format(config["data_name"], config["task"])
    for method in ['baseline', 'index', 'index_wo_WC', 'index_wo_SP']: # 'baseline', 'index', 'index_wo_WC', 'index_wo_SP'
    # for method in ['index_wo_WC']:
        config["method"] = method
        main(config, rangek, "output_dir")
    