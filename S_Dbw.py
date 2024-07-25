from scipy.spatial.distance import euclidean
import numpy as np


class S_Dbw_Index:
    def __init__(self, df):
        self.df = df
        self.X = df.iloc[:, :-1].to_numpy()
        self.labels = df.iloc[:, -1].to_numpy()


    def run(self):
        unique_labels = np.unique(self.labels)
        n_clusters = len(unique_labels)
        Scatt = np.mean([np.mean([euclidean(self.X[i], self.X[j]) for j in range(len(self.X)) if self.labels[j] == label]) for i, label in enumerate(self.labels)])
        centroids = [np.mean(self.X[self.labels == label], axis=0) for label in unique_labels]
        Dens_bw = np.mean([euclidean(centroids[i], centroids[j]) for i in range(n_clusters) for j in range(i+1, n_clusters)])
        S_Dbw = Scatt + Dens_bw    
        return S_Dbw