import numpy as np
from scipy.spatial.distance import cdist, euclidean
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.special import logsumexp


class DBCV_Index:
    def __init__(self, df, noise_label=-1, dist_function=euclidean):
        """
        Parameters:
          df : pandas DataFrame
             The data, where the last column contains the clustering labels.
          noise_label : int or other type
             The label that indicates noise (default: -1).
          dist_function : function
             The distance function to use (default: Euclidean).
        """
        self.df = df
        self.X = df.iloc[:, :-1].to_numpy()
        self.labels = df.iloc[:, -1].to_numpy()
        self.dist_function = dist_function
        self.n, self.dim = self.X.shape
        self.noise_label = noise_label

    def run(self):
        """
        Compute the overall DBCV validity index according to the paper.
        Returns a float in [-1, +1] (higher values indicate better clustering).
        """
        # Separate noise from valid clusters
        noise_mask = (self.labels == self.noise_label)
        noise_count = np.sum(noise_mask)
        valid_mask = ~noise_mask

        overall_score = 0.0

        # Get unique cluster labels (excluding noise)
        clusters = np.unique(self.labels[valid_mask])
        cluster_scores = {}
        cluster_data = {}

        for cluster in clusters:
            idx = np.where(self.labels == cluster)[0]
            X_cluster = self.X[idx]
            n_cluster = len(idx)
            
            if n_cluster < 2:
                # For clusters with a single point, we store a dummy entry.
                cluster_data[cluster] = {
                    'indices': idx,
                    'X': X_cluster,
                    'n': n_cluster,
                    'D': np.array([[0]]),
                    'core_dists': np.array([0]),
                    'mrd': np.array([[0]]),
                    'mst': np.array([[0]]),
                    'internal_indices': np.array([0]),
                    'DSC': 0.0
                }
                cluster_scores[cluster] = 0.0
                continue

            # Compute pairwise distances within cluster
            D = cdist(X_cluster, X_cluster, metric=self.dist_function)
            core_dists = np.zeros(n_cluster)
            for i in range(n_cluster):
                # Exclude self-distance
                dists = np.delete(D[i], i)
                # To avoid overflow, clip inverse distances to a maximum value.
                inv_dists = 1.0 / (dists + 1e-10)
                inv_dists = np.clip(inv_dists, None, 1e10)

                log_values = self.dim * np.log(inv_dists)
                log_sum_power = logsumexp(log_values)
                core_dists[i] = np.exp(- (1.0 / self.dim) * (log_sum_power - np.log(n_cluster - 1)))

            # Compute mutual reachability distances among points in the cluster
            mrd = np.zeros((n_cluster, n_cluster))
            for i in range(n_cluster):
                for j in range(n_cluster):
                    if i == j:
                        mrd[i, j] = 0
                    else:
                        mrd[i, j] = max(core_dists[i], core_dists[j], D[i, j])
            # Build the MST from the mutual reachability distances
            mst = minimum_spanning_tree(mrd).toarray()
            # Make the MST symmetric
            mst = np.maximum(mst, mst.T)
            # Compute degree for each node (number of nonzero connections)
            degree = np.sum(mst > 0, axis=0)
            # Define internal nodes: nodes with degree > 1.
            internal_indices = np.where(degree > 1)[0]
            if len(internal_indices) == 0:
                # Fallback: if no node has degree > 1, consider all nodes as internal.
                internal_indices = np.arange(n_cluster)
            
            # Compute Density Sparseness (DSC): maximum edge among edges connecting internal nodes.
            internal_edges = []
            for i in range(n_cluster):
                for j in range(i + 1, n_cluster):
                    if (i in internal_indices) and (j in internal_indices) and (mst[i, j] > 0):
                        internal_edges.append(mst[i, j])
            if len(internal_edges) == 0:
                DSC = np.max(mst)
            else:
                DSC = np.max(internal_edges)

            cluster_data[cluster] = {
                'indices': idx,
                'X': X_cluster,
                'n': n_cluster,
                'D': D,
                'core_dists': core_dists,
                'mrd': mrd,
                'mst': mst,
                'internal_indices': internal_indices,
                'DSC': DSC
            }

        # Compute density separation (DSPC) between clusters.
        for cluster in clusters:
            data_i = cluster_data[cluster]
            X_i = data_i['X']
            core_i = data_i['core_dists']
            internal_i = data_i['internal_indices']
            # Use points in cluster i identified as internal.
            X_i_int = X_i[internal_i]
            dspc_list = []
            for other_cluster in clusters:
                if other_cluster == cluster:
                    continue
                data_j = cluster_data[other_cluster]
                X_j = data_j['X']
                core_j = data_j['core_dists']
                internal_j = data_j['internal_indices']
                X_j_int = X_j[internal_j]
                # Compute pairwise distances between internal points of clusters i and j
                D_between = cdist(X_i_int, X_j_int, metric=self.dist_function)
                # For each pair, compute the mutual reachability distance:
                core_i_int = core_i[internal_i][:, np.newaxis]
                core_j_int = core_j[internal_j][np.newaxis, :]
                mrd_between = np.maximum(np.maximum(core_i_int, core_j_int), D_between)
                dspc_val = np.min(mrd_between)
                dspc_list.append(dspc_val)
            if len(dspc_list) == 0:
                min_dspc = 0.0
            else:
                min_dspc = np.min(dspc_list)
            # Cluster validity (VC) for cluster = (min_dspc – DSC) / max(min_dspc, DSC)
            DSC = data_i['DSC']
            if max(min_dspc, DSC) == 0:
                vc = 0.0
            else:
                vc = (min_dspc - DSC) / max(min_dspc, DSC)
            cluster_scores[cluster] = vc
            overall_score += (data_i['n'] / self.n) * vc

        # Apply noise penalty: if many points are noise, reduce the overall score.
        penalty = (self.n - noise_count) / self.n
        overall_score *= penalty

        return overall_score