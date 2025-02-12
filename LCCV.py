from sklearn.metrics import pairwise_distances
import numpy as np


class LCCV_Index:
    def __init__(self, df):
        self.df = df


    def run(self):
        labels = self.df["labels"]
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)   
        compactness = 0
        volume = 0
        for label in unique_labels:
            cluster_points = self.df[labels == label]
            centroid = cluster_points.mean(axis=0).values
            distances = pairwise_distances(cluster_points, centroid.reshape(1, -1))
            compactness += distances.mean()
            min_values = cluster_points.min(axis=0)
            max_values = cluster_points.max(axis=0)
            volume += np.prod(max_values - min_values)
        lccv = (compactness + volume) / n_clusters
        return lccv