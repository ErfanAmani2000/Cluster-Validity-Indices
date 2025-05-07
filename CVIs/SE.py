from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import mannwhitneyu, rankdata
from scipy.sparse import csr_matrix
from scipy.spatial import distance
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import faiss

np.random.seed(0)

class SEIndex:
    def __init__(self, df):
        """
        Initialize the SE Index class with the dataset.
        :param df: Pandas DataFrame containing the data points and their assigned cluster labels.
        """
        df.columns = list(df.columns[:-1]) + ['labels']
        self.df = df

    def calculate_Ei(self, cluster_i, n_neighbors=10):
        """
        Compute the average edge weight in the Minimum Spanning Tree (MST) of a given cluster.
        :param cluster_i: NumPy array of data points belonging to the cluster.
        :return: Tuple containing the average MST edge weight and a list of MST edge weights.
        """
        if self.df.shape[1] <= 25:
            distance_matrix = distance.cdist(cluster_i, cluster_i, 'euclidean')
            
            # Convert to sparse format to reduce memory usage
            sparse_distance_matrix = csr_matrix(distance_matrix)

            mst = minimum_spanning_tree(sparse_distance_matrix).toarray()
            
            non_zero_entries = mst.nonzero()
            Ei = mst[non_zero_entries].tolist()
            Ei_avg = mst.sum() / len(Ei) if Ei else 0  # Avoid division by zero
            return Ei_avg, Ei
        else:
            return self.approximate_mst(cluster_i)

    def approximate_mst(self, cluster_i, n_neighbors=10):
        """
        Construct an approximate MST using FAISS for nearest neighbors.
        :param cluster_i: NumPy array of points in the cluster.
        :param n_neighbors: Number of nearest neighbors to consider.
        :return: Tuple (average MST edge weight, list of MST edge weights)
        """
        n_samples, n_features = cluster_i.shape
        if n_samples < 2:
            return 0, []

        # Convert data to float32 for FAISS
        cluster_i = cluster_i.astype(np.float32)

        # Create a FAISS index for fast nearest neighbor search
        index = faiss.IndexFlatL2(n_features)  # L2 (Euclidean) distance
        index.add(cluster_i)

        # Search for k nearest neighbors
        distances, indices = index.search(cluster_i, n_neighbors + 1)  # +1 to ignore self

        # Build sparse graph for MST
        G = nx.Graph()
        for i in range(n_samples):
            for j, d in zip(indices[i][1:], distances[i][1:]):  # Skip self (first entry)
                G.add_edge(i, j, weight=np.sqrt(d))  # FAISS returns squared L2 distances

        # Compute MST
        mst = nx.minimum_spanning_tree(G)
        edge_weights = [d['weight'] for _, _, d in mst.edges(data=True)]
        Ei_avg = np.mean(edge_weights) if edge_weights else 0

        return Ei_avg, edge_weights

    def find_border_points(self, cluster_i, cluster_j):
        """
        Identify border points between two clusters based on mutual nearest neighbors.
        :param cluster_i: NumPy array of points in cluster i.
        :param cluster_j: NumPy array of points in cluster j.
        :return: Lists of border points in each cluster.
        """
        border_points_i, border_points_j = [], []
        
        for p in cluster_i:
            distances = distance.cdist([p], cluster_j)
            min_index = np.argmin(distances)
            q = cluster_j[min_index]
            
            distances = distance.cdist([q], cluster_i)
            min_index = np.argmin(distances)
            if np.array_equal(cluster_i[min_index], p):
                border_points_i.append(p)
                border_points_j.append(q)
        
        return border_points_i, border_points_j

    def calculate_Sij(self, border_points_i, border_points_j):
        """
        Compute the minimum distance between each border point in cluster i and cluster j.
        :return: List of minimum distances.
        """
        return [np.min(distance.cdist([x_i], border_points_j)) for x_i in border_points_i]

    def calculate_ranks(self, E_i, S_ij):
        """
        Compute ranks of the combined distances using the Mann-Whitney U test.
        :return: Sum of ranks for E_i and S_ij.
        """
        combined = E_i + S_ij
        ranks = rankdata(combined)
        
        R_i_2 = np.sum(ranks[:len(E_i)])
        R_ij_1 = np.sum(ranks[len(E_i):])
        
        return R_ij_1, R_i_2

    def calculate_Sepij(self, cluster_i, cluster_j):
        """
        Compute the separation index between two clusters.
        :return: Separation index value.
        """
        E_i = self.calculate_Ei(cluster_i)[1]
        border_ij, border_ji = self.find_border_points(cluster_i, cluster_j)
        S_ij = self.calculate_Sij(border_ij, border_ji)
        
        nij_1, ni_2 = len(S_ij), len(E_i)
        U_ij = min(
            nij_1 * ni_2 + nij_1 * (nij_1 + 1) / 2 - self.calculate_ranks(E_i, S_ij)[0],
            nij_1 * ni_2 + ni_2 * (ni_2 + 1) / 2 - self.calculate_ranks(E_i, S_ij)[1]
        )
        
        return 1 - U_ij / (nij_1 * ni_2) if nij_1 * ni_2 != 0 else 0

    def calculate_Scores(self, clusters):
        """
        Compute separation scores for all clusters.
        :return: List of separation scores.
        """
        Scores = []
        for i, cluster_i in enumerate(clusters):
            Sep_ij_list = [self.calculate_Sepij(cluster_i, clusters[j]) for j in range(len(clusters)) if i != j]
            Sep_i = min(Sep_ij_list) if Sep_ij_list else 0
            Ei_avg = self.calculate_Ei(cluster_i)[0]
            Scores.append(Sep_i / Ei_avg if Ei_avg != 0 else 0)
        return Scores

    def calculate_SE(self, clusters):
        """
        Compute the final cluster validity index (SE) for the given clusters.
        :return: SE index value.
        """
        total_points = sum(len(cluster) for cluster in clusters)
        return sum((len(cluster) / total_points) * score for cluster, score in zip(clusters, self.calculate_Scores(clusters)))

    def plot_clusters(self):
        """
        Visualize the clusters in a scatter plot.
        """
        plt.figure(figsize=(10, 8))
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        
        for i in self.df['cluster'].unique():
            plt.scatter(self.df[self.df['cluster'] == i]['X'], self.df[self.df['cluster'] == i]['Y'],
                        c=colors[i % len(colors)], alpha=0.7, label=f'Cluster {i}')
        plt.legend()
        plt.grid(False)
        plt.gca().set_facecolor('white')
        plt.show()

    def plot_mst(self, cluster):
        """
        Visualize the Minimum Spanning Tree (MST) for a given cluster.
        """
        dist_matrix = distance.cdist(cluster, cluster, 'euclidean')
        mst = minimum_spanning_tree(dist_matrix).toarray()
        
        plt.scatter(cluster[:, 0], cluster[:, 1], alpha=0.7)
        for i, j in zip(*np.where(mst > 0)):
            plt.plot([cluster[i, 0], cluster[j, 0]], [cluster[i, 1], cluster[j, 1]], 'k-', alpha=0.5)
        
        plt.grid(False)
        plt.gca().set_facecolor('white')
        plt.show()

    def df_to_clusters(self):
        """
        Convert the dataframe to a list of NumPy arrays, each representing a cluster.
        """
        return [self.df[self.df['labels'] == i].iloc[:, :-1].values for i in self.df['labels'].unique()]

    def run(self):
        """
        Execute the clustering validity index calculation.
        :return: SE index value.
        """
        return self.calculate_SE(self.df_to_clusters())
