import numpy as np
from sklearn.metrics import pairwise_distances
import networkx as nx

class LCCV_Index:
    def __init__(self, data, k=10, eps=1e-10):
        """
        Parameters:
          data: numpy array of shape (n_samples, n_features)
                Note: if you have a DataFrame whose last column is the label, 
                extract the feature columns before passing to this class.
          k: number of neighbors to use (as a surrogate for the natural neighbor count)
          eps: small value to avoid division by zero
        """
        self.data = data.iloc[:, :-1].to_numpy()
        self.k = k
        self.eps = eps
        self.n = data.shape[0]

    def compute_knn(self):
        # Compute pairwise distances among all points.
        dist_matrix = pairwise_distances(self.data)
        # For each point, sort distances and record indices.
        # We skip the first index (self-distance = 0).
        knn_indices = np.argsort(dist_matrix, axis=1)[:, 1:self.k+1]
        knn_dists = np.sort(dist_matrix, axis=1)[:, 1:self.k+1]
        return knn_indices, knn_dists, dist_matrix

    def compute_density(self, knn_dists):
        """
        Define local density as the inverse of the average k-NN distance.
        (This is a simplified substitute for the natural neighbor density in the paper.)
        """
        avg_knn_dist = np.mean(knn_dists, axis=1) + self.eps
        density = 1.0 / avg_knn_dist
        return density

    def assign_representatives(self, knn_indices, density):
        """
        For each point p, assign its representative as the neighbor (among its k-NN)
        with the maximum density, but only if that neighbor's density exceeds p’s.
        Then, perform representative transfer so that each point's representative is 
        a fixed point (i.e. Rep(Rep(p)) == Rep(p)).
        """
        rep = np.arange(self.n)
        for i in range(self.n):
            neigh = knn_indices[i]
            # Select the neighbor with the highest density among the k-nearest
            max_idx = neigh[np.argmax(density[neigh])]
            if density[max_idx] > density[i]:
                rep[i] = max_idx
        # Representative transfer: follow the chain until convergence.
        for i in range(self.n):
            while rep[rep[i]] != rep[i]:
                rep[i] = rep[rep[i]]
        return rep

    def assign_local_cores(self, rep):
        """
        Local cores are those points that are their own representative.
        In this simplified version, each local core defines its own cluster.
        Each point is assigned the label of its local core.
        """
        local_core_mask = (rep == np.arange(self.n))
        local_cores = np.where(local_core_mask)[0]
        labels = rep.copy()  # each point’s label is its converged representative
        return local_cores, labels

    def build_graph(self, knn_indices, knn_dists):
        """
        Build an undirected k-NN graph on the data.
        The edge weight is simply the Euclidean distance.
        """
        G = nx.Graph()
        for i in range(self.n):
            G.add_node(i)
        for i in range(self.n):
            for j_idx, j in enumerate(knn_indices[i]):
                weight = knn_dists[i, j_idx]
                # Since the graph is undirected, adding an edge once is sufficient.
                G.add_edge(i, j, weight=weight)
        return G

    def compute_core_graph_distances(self, local_cores, G):
        """
        Compute graph-based distances between each pair of local cores.
        If there is no path between a pair, use a default large distance.
        """
        n_cores = len(local_cores)
        D = np.zeros((n_cores, n_cores))
        all_core_dists = {}
        max_dist = 0
        # Compute shortest path distances from each local core using Dijkstra.
        for idx, core in enumerate(local_cores):
            dists = nx.single_source_dijkstra_path_length(G, core, weight='weight')
            all_core_dists[core] = dists
            if dists:
                max_dist = max(max_dist, max(dists.values()))
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
        Compute the LCCV index:
         For each local core i:
           - Let a(i) be the average graph-based distance to other local cores in the same cluster.
             (In this simplified version, each core is isolated so we set a(i)=0.)
           - Let b(i) be the smallest average graph-based distance to local cores in any other cluster.
         Then, LCCV(i) = (b(i) - a(i)) / (max(b(i), a(i)) + eps).
         The overall LCCV is a weighted average of the individual LCCV(i)'s,
         weighted by the number of points assigned to each local core.
        """
        n_cores = len(local_cores)
        # Create mapping from local core to its assigned points.
        clusters = {core: [] for core in local_cores}
        for idx, core in enumerate(labels):
            clusters.setdefault(core, []).append(idx)
        
        # Map each local core to its index in the core_dist_matrix.
        core_to_idx = {core: idx for idx, core in enumerate(local_cores)}
        lccv_values = []
        weights = []
        for core in local_cores:
            i = core_to_idx[core]
            # In this simplified version, each core forms its own cluster, so set a(i)=0.
            a_i = 0
            if n_cores == 1:
                lccv_i = 0
            else:
                # b(i) is the minimum distance from core i to any other core.
                b_i = np.min(np.delete(core_dist_matrix[i], i))
                lccv_i = (b_i - a_i) / (max(b_i, a_i) + self.eps)
            weight = len(clusters[core])
            lccv_values.append(lccv_i * weight)
            weights.append(weight)
        overall_lccv = np.sum(lccv_values) / np.sum(weights)
        return overall_lccv

    def run(self):
        # Step 1: k-NN search.
        knn_indices, knn_dists, _ = self.compute_knn()
        # Step 2: Compute local density.
        density = self.compute_density(knn_dists)
        # Step 3: Determine representative for each point.
        rep = self.assign_representatives(knn_indices, density)
        # Step 4: Identify local cores and assign labels.
        local_cores, labels = self.assign_local_cores(rep)
        # Step 5: Build k-NN graph.
        G = self.build_graph(knn_indices, knn_dists)
        # Step 6: Compute graph-based distances between local cores.
        core_dist_matrix = self.compute_core_graph_distances(local_cores, G)
        # Step 7: Compute overall LCCV index.
        overall_lccv = self.compute_lccv(local_cores, labels, core_dist_matrix)
        return overall_lccv