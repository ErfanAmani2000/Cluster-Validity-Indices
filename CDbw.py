import importlib
import math
from collections import defaultdict, Counter
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

class CDbwIndex:
    def __init__(self, df, metric="euclidean", alg_noise='comb', intra_dens_inf=False, s=3, multipliers=False):
        """
        Initialize the CDbwIndex class with the given parameters.
        :param df: DataFrame containing features and labels
        :param metric: Distance metric (default: 'euclidean')
        :param alg_noise: Noise handling strategy ('comb', 'bind', 'filter')
        :param intra_dens_inf: Whether to consider infinite intra-cluster density
        :param s: Parameter controlling density calculations (must be >2)
        :param multipliers: Additional parameters for future enhancements (default: False)
        """
        self.df = df
        self.X = df.iloc[:, :-1].to_numpy()
        self.labels = df.iloc[:, -1].to_numpy()
        self.metric = metric
        self.alg_noise = alg_noise
        self.intra_dens_inf = intra_dens_inf
        self.s = s
        self.multipliers = multipliers
        self.distvec = self.gen_dist_func(metric)
        
        # Ensure valid cluster count
        unique_labels = set(self.labels)
        if len(unique_labels) < 2 or len(unique_labels) > len(self.X) - 1:
            raise ValueError("Number of unique labels must be >1 and < n_samples")
        if s < 2:
            raise ValueError("Parameter s must be >2")
        
        # Handle noise based on the selected strategy
        if alg_noise == 'bind':
            self.labels = self.bind_noise_lab()
        elif alg_noise == 'comb':
            self.labels = self.comb_noise_lab()
        elif alg_noise == 'filter':
            self.labels, self.X = self.filter_noise_lab()

    def gen_dist_func(self, metric):
        """
        Generate the distance function based on the given metric.
        :param metric: Distance metric to use
        :return: Corresponding scipy distance function
        """
        mod = importlib.import_module("scipy.spatial.distance")
        return getattr(mod, metric)
    
    def filter_noise_lab(self):
        """
        Filter out noise points (-1 labeled) from the dataset.
        :return: Filtered labels and feature matrix
        """
        filter_label = self.labels[self.labels != -1]
        filter_X = self.X[self.labels != -1]
        return filter_label, filter_X
    
    def bind_noise_lab(self):
        """
        Bind noise points to the nearest cluster based on distance.
        :return: Updated labels with noise reassigned
        """
        labels = self.labels.copy()
        if -1 not in labels:
            return labels
        if len(set(labels)) == 1 and -1 in labels:
            raise ValueError("Labels contain only noise points")
        
        label_id, label_new = [], []
        for i in range(len(labels)):
            if labels[i] == -1:
                point = np.array([self.X[i]])
                dist = cdist(self.X[labels != -1], point, metric=self.metric)
                closest_idx = np.argmin(dist)
                label_id.append(i)
                label_new.append(labels[labels != -1][closest_idx])
        
        labels[np.array(label_id)] = np.array(label_new)
        return labels
    
    def comb_noise_lab(self):
        """
        Combine noise points into a new cluster.
        :return: Labels with noise points assigned a new cluster ID
        """
        labels = self.labels.copy()
        max_label = np.max(labels)
        labels[labels == -1] = max_label + 1
        return labels
    
    def prep(self):
        """
        Prepare data structures for cluster validity index calculation.
        :return: Various computed values used in subsequent methods
        """
        dimension = self.X.shape[1]
        n_clusters = self.labels.max() + 1
        n_points_in_cl = np.array([Counter(self.labels).get(i, 0) for i in range(n_clusters)])
        
        std1_cl = np.array([
            math.sqrt(np.sum(np.std(self.X[self.labels == i], axis=0) ** 2) / dimension)
            for i in range(n_clusters)
        ])
        
        n_max = max(n_points_in_cl)
        coord_in_cl = np.full((n_clusters, n_max, dimension), np.nan)
        labels_in_cl = np.full((n_clusters, n_max), -1)
        
        for i in range(n_clusters):
            coord_in_cl[i, :n_points_in_cl[i]] = self.X[self.labels == i]
            labels_in_cl[i, :n_points_in_cl[i]] = np.where(self.labels == i)[0]
        
        return n_clusters, std1_cl, dimension, n_points_in_cl, n_max, coord_in_cl, labels_in_cl
    
    def density(self):
        """
        Calculate the CDbw density index.
        :return: Density index value
        """
        n_clusters, stdev, dimension, n_points_in_cl, n_max, coord_in_cl, labels_in_cl = self.prep()
        mean_arr, n_rep, n_rep_max, rep_in_cl = self.rep(n_clusters, dimension, n_points_in_cl, coord_in_cl, labels_in_cl)
        middle_point, n_cl_rep = self.closest_rep(n_clusters, rep_in_cl, n_rep)
        intra_dens = self.intra_density(n_clusters, stdev, n_points_in_cl, coord_in_cl)
        inter_dens = self.inter_density(n_clusters, middle_point, n_cl_rep, stdev)
        return self.density_sep(inter_dens, intra_dens)
    
    def scattering(self):
        """
        Calculate the CDbw scattering index.
        :return: Scattering index value
        """
        n_clusters, _, _, _, _, coord_in_cl, _ = self.prep()
        mean_arr, _, _, _ = self.rep(n_clusters, self.X.shape[1], _, coord_in_cl, _)
        tot_mean = np.mean(self.X, axis=0)
        
        scat = np.sum(np.linalg.norm(mean_arr - tot_mean, axis=1)) / n_clusters
        total_std = np.sum(np.std(self.X, axis=0) ** 2)
        within_std = np.sum(np.std(mean_arr, axis=0) ** 2)
        
        return scat * (within_std / total_std)
    
    def d(self, density, scattering):
        """
        Compute the combined CDbw index.
        :param density: Density index value
        :param scattering: Scattering index value
        :return: Final CDbw index score
        """
        return density * scattering
    
    def run(self):
        """
        Compute the full CDbw index score.
        :return: CDbw index value
        """
        return self.d(self.density(), self.scattering())