from sklearn.cluster import AgglomerativeClustering, KMeans, OPTICS
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.linalg import eigsh
from sklearn.cluster import k_means
import scipy.sparse as sp
import pandas as pd
import numpy as np
import hdbscan


class RSC_Algorithm:
    """
    Robust Spectral Clustering (RSC) Algorithm
    """
    def __init__(self, k, nn=15, theta=20, m=0.5, laplacian=1, n_iter=50, normalize=False, verbose=False):
        """
        Initialize the RSC algorithm with parameters.
        
        :param k: Number of clusters
        :param nn: Number of nearest neighbors for graph construction
        :param theta: Number of edges to remove per iteration
        :param m: Fraction of edges that can be removed per node
        :param laplacian: Type of Laplacian (0: Unnormalized, 1: Random Walk, 2: Symmetric Not Implemented)
        :param n_iter: Maximum number of iterations for refinement
        :param normalize: Whether to normalize the eigenvectors
        :param verbose: Whether to print intermediate steps
        """
        self.k = k
        self.nn = nn
        self.theta = theta
        self.m = m
        self.n_iter = n_iter
        self.normalize = normalize
        self.verbose = verbose
        self.laplacian = laplacian
        
        if laplacian not in [0, 1]:
            raise ValueError('Invalid Laplacian choice. Use 0 (unnormalized) or 1 (random walk).')
        if laplacian == 2:
            raise NotImplementedError('Symmetric normalized Laplacian (L_sym) is not implemented yet.')

    def __latent_decomposition(self, X):
        """
        Perform spectral decomposition to refine the graph structure.
        
        :param X: Input data matrix
        :return: Refined adjacency matrices (Ag, Ac) and eigenvectors (H)
        """
        A = kneighbors_graph(X, n_neighbors=self.nn, metric='euclidean', include_self=False, mode='connectivity')
        A = A.maximum(A.T)  # Ensure symmetry
        N = A.shape[0]
        prev_trace = np.inf
        Ag = A.copy()
        
        for it in range(self.n_iter):
            D = sp.diags(Ag.sum(0).A1).tocsc()
            L = D - Ag  # Compute Laplacian
            
            # Compute eigenvalues and eigenvectors
            if self.laplacian == 0:
                h, H = eigsh(L, self.k, which='SM')
            else:  # self.laplacian == 1
                h, H = eigsh(L, self.k, D, which='SM')
            
            trace = h.sum()
            if self.verbose:
                print(f'Iter: {it} | Trace: {trace:.4f}')
            
            if abs(prev_trace - trace) < 1e-10:
                break
            prev_trace = trace
            
            # Identify and remove edges based on perturbation score
            edges = sp.tril(A).nonzero()
            p = np.linalg.norm(H[edges[0]] - H[edges[1]], axis=1) ** 2
            allowed_to_remove = (A.sum(0).A1 * self.m).astype(int)
            removed_edges = []
            
            for ind in p.argsort()[::-1]:
                e_i, e_j = edges[0][ind], edges[1][ind]
                if allowed_to_remove[e_i] > 0 and allowed_to_remove[e_j] > 0 and p[ind] > 0:
                    allowed_to_remove[e_i] -= 1
                    allowed_to_remove[e_j] -= 1
                    removed_edges.append((e_i, e_j))
                    if len(removed_edges) == self.theta:
                        break
            
            Ac = sp.coo_matrix((np.ones(len(removed_edges)), (np.array(removed_edges)[:, 0], np.array(removed_edges)[:, 1])), shape=(N, N))
            Ac = Ac.maximum(Ac.T)
            Ag = A - Ac
            
        return Ag, Ac, H

    def fit_predict(self, X):
        """
        Fit the RSC model to the data and return cluster labels.
        
        :param X: Input data matrix
        :return: Cluster labels
        """
        Ag, Ac, H = self.__latent_decomposition(X)
        self.Ag, self.Ac = Ag, Ac
        self.H = H / np.linalg.norm(H, axis=1)[:, None] if self.normalize else H
        _, labels, _ = k_means(self.H, n_clusters=self.k, n_init='auto')
        self.labels = labels
        return labels


class Conventional_Algorithm:
    """
    Conventional clustering algorithms wrapper.
    """
    def __init__(self, df):
        """
        Initialize with the dataset.
        
        :param df: Input dataframe with features
        """
        self.df = df
    
    def kmeans_clustering(self, n_clusters):
        """
        Apply K-Means clustering.
        
        :param n_clusters: Number of clusters
        :return: Dataframe with cluster labels
        """
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init='auto')
        self.df['labels'] = kmeans.fit_predict(self.df.iloc[:, :2])
        return self.df
    
    def agglomerative_clustering(self, n_clusters):
        """
        Apply Agglomerative Clustering.
        
        :param n_clusters: Number of clusters
        :return: Dataframe with cluster labels
        """
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
        self.df['labels'] = agglomerative.fit_predict(self.df.iloc[:, :2])
        return self.df
    
    def optics_clustering(self, min_samples):
        """
        Apply OPTICS clustering.
        
        :param min_samples: Minimum number of samples in a cluster
        :return: Dataframe with cluster labels
        """
        optics = OPTICS(min_samples=min_samples)
        self.df['labels'] = optics.fit_predict(self.df.iloc[:, :2])
        return self.df
    
    def hdbscan_clustering(self,
                        min_cluster_size: int = 5,
                        min_samples: int = 10,
                        metric: str = 'euclidean') -> pd.DataFrame:
        """
        Run HDBSCAN clustering on the rows of df and return df with a new 'labels' column.

        Parameters
        ----------
        df : pd.DataFrame
            Input data. All columns must be numeric features.
        min_cluster_size : int, default=5
            The minimum size of clusters; corresponds to mclSize (and mpts) in the paper.
        min_samples : int or None, default=None
            The number of samples in a neighborhood for a point to be considered a core point.
            If None, defaults to the same value as min_cluster_size.
        metric : str, default='euclidean'
            Distance metric to use for the mutual reachability graph.

        Returns
        -------
        pd.DataFrame
            The original DataFrame, with an extra 'labels' column.  Noise points are labeled -1.
        """
        if min_samples is None:
            min_samples = min_cluster_size

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method='eom'  
        )
        labels = clusterer.fit_predict(self.df.values)

        result = self.df.copy()
        result['labels'] = labels
        return result
