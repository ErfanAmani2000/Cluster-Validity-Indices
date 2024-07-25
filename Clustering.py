from sklearn.cluster import AgglomerativeClustering, KMeans, OPTICS, DBSCAN
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.linalg import eigsh
from sklearn.cluster import k_means
import scipy.sparse as sp
import numpy as np



class RSC_Algorithm:
    def __init__(self, k, nn=15, theta=20, m=0.5, laplacian=1, n_iter=50, normalize=False, verbose=False):
        self.k = k
        self.nn = nn
        self.theta = theta
        self.m = m
        self.n_iter = n_iter
        self.normalize = normalize
        self.verbose = verbose
        self.laplacian = laplacian
        if laplacian == 0:
            if self.verbose:
                print('Using unnormalized Laplacian L')
        elif laplacian == 1:
            if self.verbose:
                print('Using random walk based normalized Laplacian L_rw')
        elif laplacian == 2:
            raise NotImplementedError('The symmetric normalized Laplacian L_sym is not implemented yet.')
        else:
            raise ValueError('Choice of graph Laplacian not valid. Please use 0, 1 or 2.')


    def __latent_decomposition(self, X):
        A = kneighbors_graph(X=X, n_neighbors=self.nn, metric='euclidean', include_self=False, mode='connectivity')
        A = A.maximum(A.T)
        N = A.shape[0]
        deg = A.sum(0).A1
        prev_trace = np.inf
        Ag = A.copy()

        for it in range(self.n_iter):
            D = sp.diags(Ag.sum(0).A1).tocsc()
            L = D - Ag
            if self.laplacian == 0:
                h, H = eigsh(L, self.k, which='SM')
            elif self.laplacian == 1:
                h, H = eigsh(L, self.k, D, which='SM')
            trace = h.sum()
            if self.verbose:
                print('Iter: {} Trace: {:.4f}'.format(it, trace))
            if self.theta == 0:
                Ac = sp.coo_matrix((N, N), [np.int64])
                break
            if prev_trace - trace < 1e-10:
                break
            allowed_to_remove_per_node = (deg * self.m).astype(np.int64)
            prev_trace = trace
            edges = sp.tril(A).nonzero()
            removed_edges = []
            if self.laplacian == 1:
                h[np.isclose(h, 0)] = 0
                p = np.linalg.norm(H[edges[0]] - H[edges[1]], axis=1) ** 2 \
                    - np.linalg.norm(H[edges[0]] * np.sqrt(h), axis=1) ** 2 \
                    - np.linalg.norm(H[edges[1]] * np.sqrt(h), axis=1) ** 2
            else:
                p = np.linalg.norm(H[edges[0]] - H[edges[1]], axis=1) ** 2
            for ind in p.argsort()[::-1]:
                e_i, e_j, p_e = edges[0][ind], edges[1][ind], p[ind]
                if allowed_to_remove_per_node[e_i] > 0 and allowed_to_remove_per_node[e_j] > 0 and p_e > 0:
                    allowed_to_remove_per_node[e_i] -= 1
                    allowed_to_remove_per_node[e_j] -= 1
                    removed_edges.append((e_i, e_j))
                    if len(removed_edges) == self.theta:
                        break
            removed_edges = np.array(removed_edges)
            Ac = sp.coo_matrix((np.ones(len(removed_edges)), (removed_edges[:, 0], removed_edges[:, 1])), shape=(N, N))
            Ac = Ac.maximum(Ac.T)
            Ag = A - Ac
        return Ag, Ac, H

    def fit_predict(self, X):
        Ag, Ac, H = self.__latent_decomposition(X)
        self.Ag = Ag
        self.Ac = Ac
        if self.normalize:
            self.H = H / np.linalg.norm(H, axis=1)[:, None]
        else:
            self.H = H
        centroids, labels, *_ = k_means(X=self.H, n_clusters=self.k, n_init='auto')
        self.centroids = centroids
        self.labels = labels
        return labels



class Conventional_Algorithm:
    def __init__(self, df):
        self.df = df


    def kmeans_clustering(self, n_clusters):
        df_copy = self.df.iloc[:, :2]
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init='auto')
        labels = kmeans.fit_predict(df_copy)
        self.df['labels'] = labels
        return self.df


    def agglomerative_clustering(self, n_clusters):
        df_copy = self.df.iloc[:, :2]
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agglomerative.fit_predict(df_copy)
        self.df['labels'] = labels    
        return self.df


    def optics_clustering(self, min_samples):
        df_copy = self.df.iloc[:, :2]
        optics = OPTICS(min_samples=min_samples)
        labels = optics.fit_predict(df_copy)
        self.df['labels'] = labels    
        return self.df


    def dbscan_clustering(self, min_samples):
        df_copy = self.df.iloc[:, :2]
        db = DBSCAN(eps=3, min_samples=min_samples)
        labels = db.fit_predict(df_copy)
        self.df['labels'] = labels 
        return self.df
