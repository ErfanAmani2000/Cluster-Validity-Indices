from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import mannwhitneyu, rankdata
from scipy.spatial import distance
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


np.random.seed(0)


class SE_Index:
    def __init__(self, df):
        """
        Initialize the class with the data.
        """
        self.df = df


    def calculate_Ei(self, cluster_i):
        """
        Calculate the average distance in the Minimum Spanning Tree (MST) of a cluster.
        """
        distance_matrix = distance.cdist(cluster_i, cluster_i, 'euclidean')
        min_spanning_tree = minimum_spanning_tree(distance_matrix)
        Ei_avg = min_spanning_tree.sum() / (min_spanning_tree != 0).sum()
        non_zero_entries = min_spanning_tree.nonzero()
        Ei = min_spanning_tree[non_zero_entries].tolist()[0]
        return Ei_avg, Ei


    def find_border_points(self, cluster_i, cluster_j):
        """
        Identify border points between two clusters.
        """
        border_points_i = []
        border_points_j = []
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
        Calculate the minimum distances between border points of two clusters.
        """
        S_ij = []
        for x_i in border_points_i:
            distances = distance.cdist([x_i], border_points_j)
            min_distance = np.min(distances)
            S_ij.append(min_distance)
        return S_ij


    def calculate_ranks(self, E_i, S_ij):
        """
        Calculate the ranks of the combined distances for Mann-Whitney U test.
        """
        U, p_value = mannwhitneyu(E_i, S_ij, alternative='two-sided')
        combined = E_i + S_ij
        ranks = rankdata(combined)
        R_i_2 = ranks[:len(E_i)]
        R_ij_1 = ranks[len(E_i):]
        R_i_2 = np.sum(R_i_2)
        R_ij_1 = np.sum(R_ij_1)
        return R_ij_1, R_i_2


    def calculate_Sepij(self, cluster_i, cluster_j):
        """
        Calculate the separation index between two clusters.
        """
        E_i = self.calculate_Ei(cluster_i)[1]
        border_ij, border_ji = self.find_border_points(cluster_i, cluster_j)
        S_ij = self.calculate_Sij(border_ij, border_ji)
        nij_1 = len(S_ij)
        ni_2 = len(E_i)
        U_ij = min(nij_1 * ni_2 + nij_1 * (nij_1 + 1) / 2 - self.calculate_ranks(E_i, S_ij)[0],
                   nij_1 * ni_2 + ni_2 * (ni_2 + 1) / 2 - self.calculate_ranks(E_i, S_ij)[1])
        Sep_ij = 1 - U_ij / (nij_1 * ni_2)
        return Sep_ij


    def calculate_Scores(self, clusters):
        """
        Calculate the separation scores for all clusters.
        """
        Scores = []
        K = len(clusters)
        for i in range(K):
            Sep_ij_list = []
            for j in range(K):
                if i != j:
                    Sep_ij = self.calculate_Sepij(clusters[i], clusters[j])
                    Sep_ij_list.append(Sep_ij)
            Sep_i = min(Sep_ij_list)
            Ei_avg = self.calculate_Ei(clusters[i])[0]
            Scores.append(Sep_i / Ei_avg)
        return Scores


    def calculate_SE(self, clusters):
        """
        Calculate the final cluster validity index (SE) for the given clusters.
        """
        SE, X = 0, 0
        for i in range(len(clusters)):
            X += len(clusters[i])
        for i in range(len(clusters)):
            SE += (len(clusters[i]) / X) * self.calculate_Scores(clusters)[i]
        return SE


    def plot_clusters(self):
        """
        Plot the clusters in a scatter plot.
        """
        plt.figure(figsize=(10, 8))
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for i in self.df['cluster'].unique():
            plt.scatter(self.df[self.df['cluster'] == i]['X'], self.df[self.df['cluster'] == i]['Y'],
                        c=colors[i % len(colors)], alpha=0.7)
        plt.grid(False)
        plt.gca().set_facecolor('white')
        plt.show()


    def plot_mst(self, cluster):
        """
        Plot the Minimum Spanning Tree (MST) for a given cluster.
        """
        dist_matrix = distance.cdist(cluster, cluster, 'euclidean')
        mst = minimum_spanning_tree(dist_matrix).toarray()
        plt.scatter(cluster[:, 0], cluster[:, 1], alpha=0.7)
        for i in range(mst.shape[0]):
            for j in range(mst.shape[1]):
                if mst[i, j] != 0:
                    plt.plot([cluster[i, 0], cluster[j, 0]], [cluster[i, 1], cluster[j, 1]], 'k-', alpha=0.5)
        plt.grid(False)
        plt.gca().set_facecolor('white')
        plt.show()


    def df_to_clusters(self):
        """
        Convert the dataframe to a list of clusters.
        """
        clusters = [self.df[self.df['labels'] == i].values for i in self.df['labels'].unique()]
        return clusters 


    def run(self):
        """
        Execute the clustering and validity index calculation.
        """
        clusters = self.df_to_clusters()
        return self.calculate_SE(clusters)
