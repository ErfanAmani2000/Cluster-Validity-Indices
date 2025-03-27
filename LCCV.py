import numpy as np
from sklearn.metrics import pairwise_distances
import networkx as nx


class LCCV_Index:
    def __init__(self, data, k=10, eps=1e-10):
        """
        Parameters:
          data: numpy array of shape (n_samples, n_features)
          k: number of neighbors to use (a surrogate for the natural neighbor count)
          eps: small value to avoid division by zero
        """
        self.data = data
        self.k = k
        self.eps = eps
        self.n = data.shape[0]

    def compute_knn(self):
        # Compute pairwise distances
        dist_matrix = pairwise_distances(self.data)
        # For each point, sort distances and record indices (skip self at index0)
        knn_indices = np.argsort(dist_matrix, axis=1)[:, 1:self.k+1]
        knn_dists = np.sort(dist_matrix, axis=1)[:, 1:self.k+1]
        return knn_indices, knn_dists, dist_matrix

    def compute_density(self, knn_dists):
        # Define local density as the inverse of the average k-NN distance.
        avg_knn_dist = np.mean(knn_dists, axis=1) + self.eps
        density = 1.0 / avg_knn_dist
        return density

    def assign_representatives(self, knn_indices, density):
        """
        For each point p, let its representative be the neighbor among its k-nearest
        whose density is maximum. Then apply a transfer: while Rep(Rep(p)) != Rep(p),
        update Rep(p) = Rep(Rep(p)).
        """
        n = self.n
        # Initial representative: if any neighbor has higher density than p, choose the one with maximum density among neighbors; else, p is its own rep.
        rep = np.arange(n)
        for i in range(n):
            neigh = knn_indices[i]
            # Find neighbor with highest density
            max_idx = neigh[np.argmax(density[neigh])]
            # If that neighbor has higher density, set representative
            if density[max_idx] > density[i]:
                rep[i] = max_idx
        # Representative transfer: follow chain until convergence
        for i in range(n):
            while rep[rep[i]] != rep[i]:
                rep[i] = rep[rep[i]]
        return rep

    def assign_local_cores(self, rep):
        """
        Local cores are points that are their own representative.
        Also, assign each point a label corresponding to the index of its local core.
        """
        local_core_mask = (rep == np.arange(self.n))
        local_cores = np.where(local_core_mask)[0]
        # For each point, label = representative (which now is a local core)
        labels = rep.copy()
        return local_cores, labels

    def build_graph(self, knn_indices, knn_dists):
        """
        Build an undirected k-NN graph over the data.
        Edge weights are the Euclidean distances.
        """
        G = nx.Graph()
        for i in range(self.n):
            G.add_node(i)
        for i in range(self.n):
            for j_idx, j in enumerate(knn_indices[i]):
                # add edge in both directions (undirected)
                weight = knn_dists[i, j_idx]
                G.add_edge(i, j, weight=weight)
        return G

    def compute_core_graph_distances(self, local_cores, G):
        """
        Compute the graph-based distances (using Dijkstra) between every pair of local cores.
        If no path exists, we set the distance to a large number.
        Returns a distance matrix D of shape (n_cores, n_cores).
        """
        n_cores = len(local_cores)
        D = np.zeros((n_cores, n_cores))
        # Precompute all pairs shortest paths for local cores
        # Using networkx.single_source_dijkstra for each local core
        max_dist = 0
        # First, compute all distances and also record the maximum
        all_core_dists = {}
        for idx, core in enumerate(local_cores):
            dists = nx.single_source_dijkstra_path_length(G, core, weight='weight')
            all_core_dists[core] = dists
            if dists:
                max_dist = max(max_dist, max(dists.values()))
        # Set a default large distance if no path exists
        default = max_dist * 1.1 if max_dist > 0 else 1e6
        for i, core_i in enumerate(local_cores):
            for j, core_j in enumerate(local_cores):
                if i == j:
                    D[i, j] = 0
                else:
                    D[i, j] = all_core_dists[core_i].get(core_j, default)
        return D

    def compute_lccv(self, local_cores, labels, core_dist_matrix):
        """
        Compute the LCCV value.
        For each local core i, let a(i) be the average graph-based distance
        to other local cores in the same cluster, and b(i) be the smallest average
        graph–based distance to local cores in any other cluster.
        Then LCCV(i) = (b(i) - a(i)) / max(b(i), a(i)).
        The overall LCCV is the weighted average of LCCV(i) weighted by the number
        of points assigned to core i.
        """
        n_cores = len(local_cores)
        # Create mapping from core to points (their indices)
        clusters = {core: [] for core in local_cores}
        for idx, core in enumerate(labels):
            if core in clusters:
                clusters[core].append(idx)
            else:
                # This should not happen because every rep should be a local core
                clusters[core] = [idx]

        # For easier lookup, map local core to its index in core_dist_matrix:
        core_to_idx = {core: idx for idx, core in enumerate(local_cores)}

        lccv_values = []
        weights = []
        for core in local_cores:
            i = core_to_idx[core]
            # find local cores in the same cluster (same label)
            same_cluster = [c for c in local_cores if c == core]  # only the core itself
            # In our simple representative assignment, each local core defines its own cluster.
            # (In a full hierarchical approach, clusters would be merged.)
            # So we simply set a(i)=0 (if cluster has single core) by definition.
            a_i = 0
            b_i = 0
            # For sake of demonstration, if there is only one core, set LCCV=0.
            # Otherwise, for each local core, b(i) is defined as the minimum average distance
            # to any other cluster. Here we use pairwise distances between cores.
            # In this simplified version, we treat every local core as its own cluster.
            # So, b(i) is the minimum distance to any other core.
            # (Note: the paper defines a(i) as the average intra-cluster distance;
            # if a core is alone, a(i)=0.)
            if n_cores == 1:
                lccv_i = 0
            else:
                # For this simplified version, define:
                # a(i) = 0 (since there is no other core in its own “cluster”)
                # b(i) = min_{j != i} core_dist_matrix[i,j]
                b_i = np.min(np.delete(core_dist_matrix[i], i))
                lccv_i = (b_i - a_i) / (max(b_i, a_i) + self.eps)
            # Weight: number of points assigned to this local core
            weight = len(clusters[core])
            lccv_values.append(lccv_i * weight)
            weights.append(weight)
        overall_lccv = np.sum(lccv_values) / np.sum(weights)
        return overall_lccv

    def run(self):
        # Step 1: k-NN search
        knn_indices, knn_dists, dist_matrix = self.compute_knn()
        # Step 2: Compute local density (using inverse of avg k-distance)
        density = self.compute_density(knn_dists)
        # Step 3: Determine representative for each point
        rep = self.assign_representatives(knn_indices, density)
        # Step 4: Identify local cores and assign each point a label (its local core)
        local_cores, labels = self.assign_local_cores(rep)
        # Step 5: Build k-NN graph for the whole data
        G = self.build_graph(knn_indices, knn_dists)
        # Step 6: Compute graph-based distances between local cores
        core_dist_matrix = self.compute_core_graph_distances(local_cores, G)
        # Step 7: Compute overall LCCV index
        overall_lccv = self.compute_lccv(local_cores, labels, core_dist_matrix)
        return overall_lccv