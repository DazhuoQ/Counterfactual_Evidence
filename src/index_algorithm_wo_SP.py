import numpy as np
import torch
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import set_seed, dataset_func
from scipy.special import betainc, gamma
import time
from src.kernel_similarity import KS

def normalize_features(x):
    """
    Normalize the features by z-score normalization (mean = 0, std = 1).
    
    Parameters:
        x (Tensor): Feature matrix of shape (num_nodes, num_features)
        
    Returns:
        x_normalized (Tensor): Normalized feature matrix
    """
    mean = x.mean(0)  # Compute the mean of each feature
    std = x.std(0)  # Compute the standard deviation of each feature
    x_normalized = (x - mean) / (std + 1e-8)  # Standardize the features (avoid division by zero)
    return x_normalized


def normalized_cal_intersection_ratio(n, theta, theta_v):
    """
    Calculate 1 - (intersection_area / cap_area) in a numerically stable way
    n: dimension
    theta: given theta (pi/4 in this case)
    theta_v: angle between the axes of the two caps
    """
    # Calculate theta_m
    theta_m = np.arctan((1 - np.cos(theta_v)) / np.sin(theta_v))

    # Calculate the regularized incomplete beta functions
    I_cap = betainc((n-1)/2, 1/2, np.sin(theta)**2)
    
    # Define the integral for the hyperspherical cap cut by a hyperplane
    def integral_J(alpha, beta):
        t = np.linspace(alpha, beta, 100)
        dt = (beta - alpha) / 100
        I = betainc((n-2)/2, 1/2, 1 - (np.tan(alpha) / np.tan(t))**2)
        J_integral = np.sum(np.sin(t)**(n-2) * I) * dt
        return J_integral

    # Calculate the integrals for J1 and J2
    J1 = integral_J(theta_m, theta)
    J2 = integral_J(theta_v - theta_m, theta)

    # Sum the integrals for the intersection area
    I_intersection = J1 + J2

    # Compute the ratio of the intersection area to the cap area
    ratio = I_intersection / I_cap

    # Return the desired value
    result = 1 - ratio
    return result

def cosine_distance(x, y):
    x = x.flatten()
    y = y.flatten()
    if np.linalg.norm(x) == 0 or np.linalg.norm(y) == 0:
        return 1.0
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def weighted_kmeans_wo_SP(data, K=10, max_iter=2, tol=1e-4):

    # data = normalize_features(data)
    num_nodes, num_features = data.shape
    centroids = [data[torch.randint(0, num_nodes, (K,))]]  # Randomly select k nodes as initial centroids

    # for iteration in range(max_iter):

    # Step 3: Assign each node to the nearest centroid
    distances = torch.cdist(data, centroids[0])  # Calculate pairwise distances between data points and centroids
    labels = torch.argmin(distances, dim=1)  # Assign the closest centroid to each node
    weights = torch.tensor(calculate_weights_as_angles(data, centroids[0], labels.numpy()), dtype=torch.float32)

    # Weighting the distances: use the weight vector to scale the features
    weighted_x = data * weights.unsqueeze(1)  # Shape: (num_nodes, num_features)

    # Step 4: Update centroids based on weighted mean
    new_centroids = torch.stack([
        weighted_x[labels == i].sum(dim=0) / weights[labels == i].sum() if (labels == i).sum() > 0 else centroids[0][i]
        for i in range(K)
    ])
    centroids = [new_centroids]
    
    return centroids, labels

def initialize_centroids(data, K):
    n_samples, _ = data.shape
    centroids = np.zeros((K, data.shape[1]))
    centroids[0] = data[np.random.choice(n_samples)]
    
    # for k in range(1, K):
    #     distances = np.min([[cosine_distance(x, c) for c in centroids[:k]] for x in data], axis=1)
    #     distances[distances < 0] = 0
    #     probs = distances / np.sum(distances)
    #     centroids[k] = data[np.random.choice(n_samples, p=probs)]
    
    return centroids

def calculate_weights_as_angles(data, centroids, labels, epsilon=1e-6):

    n = len(data[0])  # Dimension
    # r = 1  # Radius
    theta = np.pi / 3  # Given theta
    # cap_area = hyperspherical_cap_area(n, r, theta)

    # data_normalized = normalize(data)
    
    weights = np.zeros(data.shape[0])
    for i, centroid in enumerate(centroids):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) > 0:
            cluster_vectors = data[cluster_indices]
            cosine_similarities = cosine_similarity(cluster_vectors, centroid.reshape(1, -1)).flatten()
            cosine_similarities = np.clip(cosine_similarities, -1.0 + epsilon, 1.0 - epsilon)  # Avoid angles equal to 0
            angles = np.arccos(np.clip(cosine_similarities, -1.0, 1.0))  # Convert to angles

            vector_weight_lst = []
            for fi in angles:
                vector_weight_lst.append(normalized_cal_intersection_ratio(n, theta, fi))
            vector_weight_lst = np.array(vector_weight_lst)

            weights[cluster_indices] = vector_weight_lst
    return weights

def find_top_k_similar(data, labels, target_index, centroids, k=3):
    data_normalized = normalize(data)
    target_vector = data_normalized[target_index].reshape(1, -1)
    
    cluster_index = np.argmin([cosine_distance(target_vector, c.reshape(1, -1)) for c in centroids])
    cluster_indices = np.where(labels == cluster_index)[0]
    cluster_indices = cluster_indices[cluster_indices != target_index]
    
    if len(cluster_indices) == 0:
        return [], [], cluster_index
    
    cluster_vectors = data_normalized[cluster_indices]
    similarities = cosine_similarity(target_vector, cluster_vectors).flatten()
    
    top_k_indices_local = np.argsort(similarities)[-k:][::-1]
    top_k_indices_global = cluster_indices[top_k_indices_local]
    
    top_k_similar_vectors = data[top_k_indices_global]
    
    return top_k_similar_vectors, top_k_indices_global, cluster_index

