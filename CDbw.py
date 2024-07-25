import importlib
import math
from collections import defaultdict, Counter
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist



class CDbw_Index:
    def __init__(self, df, metric="euclidean", alg_noise='comb', intra_dens_inf=False, s=3, multipliers=False):
        """
        Initialize the CDbwIndex class with the given parameters.
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

        if len(set(self.labels)) < 2 or len(set(self.labels)) > len(self.X) - 1:
            raise ValueError("No. of unique labels must be > 1 and < n_samples")
        if s < 2:
            raise ValueError("Parameter s must be > 2")
        
        if alg_noise == 'bind':
            self.labels = self.bind_noise_lab()
        elif alg_noise == 'comb':
            self.labels = self.comb_noise_lab()
        elif alg_noise == 'filter':
            self.labels, self.X = self.filter_noise_lab()


    def gen_dist_func(self, metric):
        """
        Generate the distance function based on the given metric.
        """
        mod = importlib.import_module("scipy.spatial.distance")
        func = getattr(mod, metric)
        return func


    def filter_noise_lab(self):
        """
        Filter out the noise points from the dataset.
        """
        filterLabel = self.labels[self.labels != -1]
        filterXYZ = self.X[self.labels != -1]
        return filterLabel, filterXYZ


    def bind_noise_lab(self):
        """
        Bind noise points to the nearest clusters.
        """
        labels = self.labels.copy()
        if -1 not in set(labels):
            return labels
        if len(set(labels)) == 1 and -1 in set(labels):
            raise ValueError('Labels contains noise point only')
        label_id = []
        label_new = []
        for i in range(len(labels)):
            if labels[i] == -1:
                point = np.array([self.X[i]])
                dist = cdist(self.X[labels != -1], point, metric=self.metric)
                lid = np.where(np.all(self.X == self.X[labels != -1][np.argmin(dist), :], axis=1))[0][0]
                label_id.append(i)
                label_new.append(labels[lid])
        labels[np.array(label_id)] = np.array(label_new)
        return labels


    def comb_noise_lab(self):
        """
        Combine noise points into a new cluster.
        """
        labels = self.labels.copy()
        max_label = np.max(labels)
        j = max_label + 1
        for i in range(len(labels)):
            if labels[i] == -1:
                labels[i] = j
        return labels


    def prep(self):
        """
        Prepare data structures for cluster validity index calculation.
        """
        dimension = self.X.shape[1]
        n_clusters = self.labels.max() + 1
        n_points_in_cl = np.zeros(n_clusters, dtype=int)
        stdv1_cl = np.zeros(shape=(n_clusters, dimension), dtype=float)
        std1_cl = np.zeros(shape=n_clusters, dtype=float)
        for i in range(n_clusters):
            n_points_in_cl[i] = Counter(self.labels).get(i)
            stdv1_cl[i] = np.std(self.X[self.labels == i], axis=0)
            std1_cl[i] = np.dot(stdv1_cl[i].T, stdv1_cl[i])
            std1_cl[i] = math.sqrt(std1_cl[i] / dimension)
        n_max = max(n_points_in_cl)
        coord_in_cl = np.full((n_clusters, n_max, dimension), np.nan)
        labels_in_cl = np.full((n_clusters, n_max), -1)
        for i in range(n_clusters):
            for j in range(n_clusters):
                stdev = np.power(np.mean([std1_cl[i] ** 2, std1_cl[j] ** 2]), 0.5)
        for i in range(n_clusters):
            coord_in_cl[i, 0:n_points_in_cl[i], 0:dimension] = self.X[self.labels == i]
            labels_in_cl[i, 0:n_points_in_cl[i]] = np.where(self.labels == i)[0]
        return n_clusters, stdev, dimension, n_points_in_cl, n_max, coord_in_cl, labels_in_cl


    def rep(self, n_clusters, dimension, n_points_in_cl, coord_in_cl, labels_in_cl):
        """
        Calculate representatives and mean for each cluster.
        """
        mean_arr = np.zeros(shape=(n_clusters, dimension), dtype=float)
        n_rep = np.zeros(shape=(n_clusters), dtype=int)
        for i in range(n_clusters):
            if n_points_in_cl[i] >= 4:
                ch = ConvexHull(coord_in_cl[i, 0:n_points_in_cl[i], 0:dimension])
                n_rep[i] = ch.vertices.size
            else:
                n_rep[i] = n_points_in_cl[i]
        n_rep_max = np.max(n_rep)
        rep_in_cl = np.full((n_clusters, n_rep_max), -1)
        for i in range(n_clusters):
            if n_points_in_cl[i] >= 4:
                ch = ConvexHull(coord_in_cl[i, 0:n_points_in_cl[i], 0:dimension])
                rep_in_cl[i, 0:n_rep[i]] = labels_in_cl[i, 0:n_points_in_cl[i]][ch.vertices]
            else:
                rep_in_cl[i, 0:n_rep[i]] = labels_in_cl[i, 0:n_points_in_cl[i]]
            mean_arr[i] = np.mean(coord_in_cl[i, 0:n_points_in_cl[i], 0:dimension], axis=0)
        return mean_arr, n_rep, n_rep_max, rep_in_cl


    def closest_rep(self, n_clusters, rep_in_cl, n_rep):
        """
        Find the closest representatives between clusters.
        """
        b1 = {}
        b2 = {}
        dist_arr = {}
        min_value1 = {}
        min_value0 = {}
        min_index0 = {}
        min_index1 = {}
        min_index2 = {}
        min_index_r = {}
        min_index_c = {}
        cl_r = {}
        s1 = []
        s2 = []
        s2_t = []
        t1 = []
        t2 = []
        v1 = []
        v2 = []
        dist_min = defaultdict(list)
        middle_point = defaultdict(list)
        n_cl_rep = {}
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i > j:
                    dist_arr[(i, j)] = cdist(self.X[rep_in_cl[i, 0:n_rep[i]]], self.X[rep_in_cl[j, 0:n_rep[j]]], metric=self.metric)
                    min_value1[(i, j)] = dist_arr[(i, j)].min(axis=1)
                    min_value0[(i, j)] = dist_arr[(i, j)].min(axis=0)
                    min_index1[(i, j)] = dist_arr[(i, j)].argmin(axis=1)
                    min_index0[(i, j)] = dist_arr[(i, j)].argmin(axis=0)
                    min_index_r[(i, j)] = np.add(np.arange(0, n_rep[i] * n_rep[j], n_rep[j]), min_index1[(i, j)])
                    min_index_c[(i, j)] = np.add(np.arange(0, n_rep[i] * n_rep[j], n_rep[i]), min_index0[(i, j)])
                    t1 += [n_rep[i]]
                    t2 += [n_rep[j]]
                    for k in range(n_rep[i]):
                        s1.append(np.unravel_index(min_index_r[(i, j)][k], (n_rep[i], n_rep[j])))
                    for n in range(n_rep[j]):
                        s2.append(np.unravel_index(min_index_c[(i, j)][n], (n_rep[j], n_rep[i])))
                        s2_t = [(x[1], x[0]) for x in s2]
        p = 0
        for m in range(len(t1)):
            p += t1[m]
            v1.append(p)
        p = 0
        for m in range(len(t2)):
            p += t2[m]
            v2.append(p)
        min_index1[(1, 0)] = s1[0:v1[0]]
        min_index2[(1, 0)] = s2_t[0:v2[0]]
        m = 0
        for i in range(2, n_clusters):
            for j in range(n_clusters):
                if i > j:
                    min_index1[(i, j)] = s1[v1[m]:v1[m + 1]]
                    min_index2[(i, j)] = s2_t[v2[m]:v2[m + 1]]
                    m += 1
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i > j:
                    b1[(i, j)] = set(min_index1[(i, j)])
                    b2[(i, j)] = set(min_index2[(i, j)])
                    dist_min[(i, j)] = list(b1[(i, j)] | b2[(i, j)])
                    middle_point[(i, j)] = self.X[[rep_in_cl[i, x[0]] for x in dist_min[(i, j)]]] + self.X[[rep_in_cl[j, x[1]] for x in dist_min[(i, j)]]]
                    middle_point[(i, j)] = 0.5 * middle_point[(i, j)]
                    n_cl_rep[(i, j)] = len(dist_min[(i, j)])
        return middle_point, n_cl_rep


    def inter_density(self, n_clusters, middle_point, n_cl_rep, stdev):
        """
        Calculate inter-cluster density.
        """
        den_inter = {}
        dist_arr1 = {}
        v = {}
        z = []
        z_c = defaultdict(list)
        inter_density = []
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i > j:
                    z.append(n_cl_rep[(i, j)])
                    z_c[(i, j)] = np.arange(np.sum(z[:len(z) - 1]), np.sum(z[:len(z)]))
                    dist_arr1[(i, j)] = cdist(self.X, middle_point[(i, j)], metric=self.metric)
                    v[(i, j)] = dist_arr1[(i, j)] <= stdev
                    for k in range(len(z_c[(i, j)])):
                        den_inter[(i, j)] = [np.sum(v[(i, j)][:, k]) / len(self.X)]
                    inter_density.append(np.sum(den_inter[(i, j)]))
        inter_density = np.sum(inter_density) / (n_clusters * (n_clusters - 1) / 2)
        return inter_density


    def intra_density(self, n_clusters, stdev, n_points_in_cl, coord_in_cl):
        """
        Calculate intra-cluster density.
        """
        if self.intra_dens_inf:
            return 0
        dens_intra_cl = []
        v_intra_cl = {}
        dens_intra = {}
        dist_arr2 = {}
        for i in range(n_clusters):
            dens_intra_cl.append(n_points_in_cl[i])
            dens_intra_cl[i] = (dens_intra_cl[i]) ** 2 - dens_intra_cl[i]
            dist_arr2[(i)] = cdist(self.X, coord_in_cl[i, 0:n_points_in_cl[i], 0:self.X.shape[1]], metric=self.metric)
            v_intra_cl[(i)] = dist_arr2[(i)] <= stdev
            dens_intra[(i)] = np.sum(v_intra_cl[(i)].astype(int), axis=0) / len(self.X)
            dens_intra[(i)] = np.sum(dens_intra[(i)])
            dens_intra_cl[i] = dens_intra[(i)] / dens_intra_cl[i]
        dens_intra_cl = np.sum(dens_intra_cl) / n_clusters
        return dens_intra_cl

    def density_sep(self, inter_density, intra_density):
        """
        Calculate density separation.
        """
        density_sep = inter_density / intra_density
        return density_sep


    def density(self):
        """
        Calculate the CDbw density index.
        """
        n_clusters, stdev, dimension, n_points_in_cl, n_max, coord_in_cl, labels_in_cl = self.prep()
        mean_arr, n_rep, n_rep_max, rep_in_cl = self.rep(n_clusters, dimension, n_points_in_cl, coord_in_cl, labels_in_cl)
        middle_point, n_cl_rep = self.closest_rep(n_clusters, rep_in_cl, n_rep)
        intra_dens = self.intra_density(n_clusters, stdev, n_points_in_cl, coord_in_cl)
        inter_dens = self.inter_density(n_clusters, middle_point, n_cl_rep, stdev)
        density = self.density_sep(inter_dens, intra_dens)
        return density


    def scattering(self):
        """
        Calculate the CDbw scattering index.
        """
        n_clusters, stdev, dimension, n_points_in_cl, n_max, coord_in_cl, labels_in_cl = self.prep()
        mean_arr, n_rep, n_rep_max, rep_in_cl = self.rep(n_clusters, dimension, n_points_in_cl, coord_in_cl, labels_in_cl)
        tot_mean = np.mean(self.X, axis=0)
        distance = []
        for i in range(n_clusters):
            distance.append(np.dot((mean_arr[i] - tot_mean), (mean_arr[i] - tot_mean).T))
        scat = np.sum(np.sqrt(distance)) / n_clusters
        total = np.sum(np.std(self.X, axis=0) ** 2)
        within = np.sum(np.std(mean_arr, axis=0) ** 2)
        scat = scat * within / total
        return scat


    def d(self, density, scattering):
        """
        Calculate the combined CDbw index.
        """
        return density * scattering


    def run(self):
        """
        Compute the CDbw index.
        """
        density = self.density()
        scattering = self.scattering()
        result = self.d(density, scattering)
        return result
