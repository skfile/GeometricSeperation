# graph_methods.py

import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from scipy import sparse

from src.utils import connected_comp_helper

def build_knn_graph(X, n_neighbors):
    """
    Build a k-nearest-neighbors graph from data points, returning a sparse adjacency.
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    distances, indices = nbrs.kneighbors(X)
    G = nx.Graph()
    for i in range(len(X)):
        for neighbor_idx, dist in zip(indices[i, 1:], distances[i, 1:]):
            G.add_edge(i, neighbor_idx, weight=dist)
    A_sparse = nx.adjacency_matrix(G)  # NxN sparse CSR
    return G, A_sparse

def prune_random(G, data, p):
    """
    Randomly prune edges from a graph.
    Returns a pruned graph and its adjacency in sparse form.
    """
    G_pruned = G.copy()
    preserved_nodes = set()
    preserved_edges = []
    edges = list(G_pruned.edges())
    for idx, edge in enumerate(edges):
        if np.random.rand() < p:
            G_pruned.remove_edge(*edge)
        else:
            preserved_nodes.add(edge[0])
            preserved_nodes.add(edge[1])
            preserved_edges.append(idx)

    if len(preserved_nodes) < G_pruned.number_of_nodes():
        missing_nodes = set(G.nodes()).difference(preserved_nodes)
        for node_idx in missing_nodes:
            isolated_node = data[node_idx]
            dists = np.linalg.norm(data - isolated_node, axis=1)
            dists[node_idx] = np.inf
            nearest_neighbor = np.argmin(dists)
            G_pruned.add_edge(node_idx, nearest_neighbor, weight=dists[nearest_neighbor])

    A_pruned_sparse = nx.adjacency_matrix(G_pruned)
    return G_pruned, A_pruned_sparse, preserved_edges

def prune_distance(G, data, thresh):
    """
    Prune edges whose distance exceeds a threshold.
    """
    G_pruned = G.copy()
    preserved_nodes = set()
    preserved_edges = []
    edges = list(G_pruned.edges())
    for idx, edge in enumerate(edges):
        v1, v2 = edge
        dist = np.linalg.norm(data[v1] - data[v2])
        if dist > thresh:
            G_pruned.remove_edge(v1, v2)
        else:
            preserved_nodes.add(v1)
            preserved_nodes.add(v2)
            preserved_edges.append(idx)

    if len(preserved_nodes) != len(G.nodes()):
        missing_nodes = set(G.nodes()).difference(preserved_nodes)
        for node_idx in missing_nodes:
            isolated_node = data[node_idx]
            dists = np.linalg.norm(data - isolated_node, axis=1)
            dists[node_idx] = np.inf
            nearest_neighbor = np.argmin(dists)
            G_pruned.add_edge(node_idx, nearest_neighbor, weight=dists[nearest_neighbor])

    A_pruned_sparse = nx.adjacency_matrix(G_pruned)
    return G_pruned, A_pruned_sparse, preserved_edges

def prune_bisection(G, data, n):
    """
    Prune edges using the bisection method.
    """
    G_pruned = G.copy()
    preserved_nodes = set()
    preserved_edges = []
    edges = list(G.edges())
    for idx, edge in enumerate(edges):
        x_i = data[edge[0]]
        x_j = data[edge[1]]
        x_ij = (x_i + x_j) / 2

        dists = np.linalg.norm(data - x_i, axis=1)
        nearest_neighbors = np.argsort(dists)[:n]
        e_i = np.mean(np.linalg.norm(data[nearest_neighbors] - x_i, axis=1))

        dists = np.linalg.norm(data - x_j, axis=1)
        nearest_neighbors = np.argsort(dists)[:n]
        e_j = np.mean(np.linalg.norm(data[nearest_neighbors] - x_j, axis=1))

        e_ij = min(e_i, e_j)

        bounding_box = np.zeros((data.shape[1], 2))
        for i_dim in range(data.shape[1]):
            bounding_box[i_dim, 0] = x_ij[i_dim] - e_ij
            bounding_box[i_dim, 1] = x_ij[i_dim] + e_ij
        c_ij = np.sum(np.all(data >= bounding_box[:, 0], axis=1) &
                      np.all(data <= bounding_box[:, 1], axis=1))
        if c_ij == 0:
            G_pruned.remove_edge(*edge)
        else:
            preserved_nodes.add(edge[0])
            preserved_nodes.add(edge[1])
            preserved_edges.append(idx)

    if len(preserved_nodes) != len(G.nodes()):
        missing_nodes = set(G.nodes()).difference(preserved_nodes)
        for node_idx in missing_nodes:
            isolated_node = data[node_idx]
            dists = np.linalg.norm(data - isolated_node, axis=1)
            dists[node_idx] = np.inf
            nearest_neighbor = np.argmin(dists)
            G_pruned.add_edge(node_idx, nearest_neighbor, weight=dists[nearest_neighbor])

    A_pruned_sparse = nx.adjacency_matrix(G_pruned)
    return G_pruned, A_pruned_sparse, preserved_edges

def prune_mst(G, data, thresh):
    """
    Prune the graph with MST-based method.
    """
    import networkx as nx
    G_pruned = G.copy()
    mst_1 = nx.minimum_spanning_tree(G)
    G_minus_mst_1 = nx.Graph(G)
    G_minus_mst_1.remove_edges_from(mst_1.edges())
    mst_2 = nx.minimum_spanning_tree(G_minus_mst_1)
    mst_combined = nx.compose(mst_1, mst_2)
    preserved_nodes = set()
    preserved_edges = []
    edges = list(G_pruned.edges())
    for idx, edge in enumerate(edges):
        v1, v2 = edge
        sp_len = nx.shortest_path_length(mst_combined, source=v1, target=v2, weight='weight')
        if sp_len > thresh:
            G_pruned.remove_edge(v1, v2)
        else:
            preserved_nodes.add(v1)
            preserved_nodes.add(v2)
            preserved_edges.append(idx)

    if len(preserved_nodes) != len(G.nodes()):
        missing_nodes = set(G.nodes()).difference(preserved_nodes)
        for node_idx in missing_nodes:
            isolated_node = data[node_idx]
            dists = np.linalg.norm(data - isolated_node, axis=1)
            dists[node_idx] = np.inf
            nearest_neighbor = np.argmin(dists)
            G_pruned.add_edge(node_idx, nearest_neighbor, weight=dists[nearest_neighbor])

    A_pruned_sparse = nx.adjacency_matrix(G_pruned)
    return G_pruned, A_pruned_sparse, preserved_edges

def prune_density(G, data, thresh):
    """
    Prune the graph based on density estimation (Gaussian KDE).
    """
    from scipy.stats import gaussian_kde
    G_pruned = G.copy()
    kde = gaussian_kde(data.T)
    preserved_nodes = set()
    preserved_edges = []
    edges = list(G_pruned.edges())
    for idx, edge in enumerate(edges):
        v1, v2 = edge
        x_i = data[v1]
        x_j = data[v2]
        x_ij = (x_i + x_j) / 2
        density_ij = kde.pdf(x_ij)
        if density_ij < thresh:
            G_pruned.remove_edge(v1, v2)
        else:
            preserved_nodes.add(v1)
            preserved_nodes.add(v2)
            preserved_edges.append(idx)

    if len(preserved_nodes) != len(G.nodes()):
        missing_nodes = set(G.nodes()).difference(preserved_nodes)
        for node_idx in missing_nodes:
            isolated_node = data[node_idx]
            dists = np.linalg.norm(data - isolated_node, axis=1)
            dists[node_idx] = np.inf
            nearest_neighbor = np.argmin(dists)
            G_pruned.add_edge(node_idx, nearest_neighbor, weight=dists[nearest_neighbor])

    A_pruned_sparse = nx.adjacency_matrix(G_pruned)
    return G_pruned, A_pruned_sparse, preserved_edges