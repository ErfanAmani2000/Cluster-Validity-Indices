import numpy as np
from scipy.spatial.distance import euclidean, cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csgraph


class DBCV_Index:
    def __init__(self, df, dist_function=euclidean):
        """
        Initialize the DBCVIndex class with the given parameters.
        """
        self.df = df
        self.X = df.iloc[:, :-1].to_numpy()
        self.labels = df.iloc[:, -1].to_numpy()
        self.dist_function = dist_function


    def run(self):
        """
        Compute the DBCV index.
        """
        graph = self._mutual_reach_dist_graph()
        mst = self._mutual_reach_dist_MST(graph)
        cluster_validity = self._clustering_validity_index(mst)
        return cluster_validity


    def _core_dist(self, point, neighbors):
        """
        Calculate the core distance of a point.
        """
        n_features = np.shape(point)[0]
        n_neighbors = np.shape(neighbors)[0]

        distance_vector = cdist(point.reshape(1, -1), neighbors)
        distance_vector = distance_vector[distance_vector != 0]
        numerator = ((1/distance_vector)**n_features).sum()
        core_dist = (numerator / (n_neighbors - 1)) ** (-1/n_features)
        return core_dist


    def _mutual_reachability_dist(self, point_i, point_j, neighbors_i, neighbors_j):
        """
        Calculate the mutual reachability distance between two points.
        """
        core_dist_i = self._core_dist(point_i, neighbors_i)
        core_dist_j = self._core_dist(point_j, neighbors_j)
        dist = self.dist_function(point_i, point_j)
        mutual_reachability = np.max([core_dist_i, core_dist_j, dist])
        return mutual_reachability


    def _mutual_reach_dist_graph(self):
        """
        Create the mutual reachability distance graph.
        """
        n_samples = np.shape(self.X)[0]
        graph = []
        for row in range(n_samples):
            graph_row = []
            for col in range(n_samples):
                point_i = self.X[row]
                point_j = self.X[col]
                class_i = self.labels[row]
                class_j = self.labels[col]
                members_i = self._get_label_members(class_i)
                members_j = self._get_label_members(class_j)
                dist = self._mutual_reachability_dist(point_i, point_j, members_i, members_j)
                graph_row.append(dist)
            graph.append(graph_row)
        graph = np.array(graph)
        return graph


    def _mutual_reach_dist_MST(self, dist_tree):
        """
        Create the mutual reachability distance minimum spanning tree.
        """
        mst = minimum_spanning_tree(dist_tree).toarray()
        return mst + np.transpose(mst)


    def _cluster_density_sparseness(self, MST, cluster):
        """
        Calculate the cluster density sparseness.
        """
        indices = np.where(self.labels == cluster)[0]
        cluster_MST = MST[indices][:, indices]
        cluster_density_sparseness = np.max(cluster_MST)
        return cluster_density_sparseness


    def _cluster_density_separation(self, MST, cluster_i, cluster_j):
        """
        Calculate the cluster density separation.
        """
        indices_i = np.where(self.labels == cluster_i)[0]
        indices_j = np.where(self.labels == cluster_j)[0]
        shortest_paths = csgraph.dijkstra(MST, indices=indices_i)
        relevant_paths = shortest_paths[:, indices_j]
        density_separation = np.min(relevant_paths)
        return density_separation


    def _cluster_validity_index(self, MST, cluster):
        """
        Calculate the cluster validity index.
        """
        min_density_separation = np.inf
        for cluster_j in np.unique(self.labels):
            if cluster_j != cluster:
                cluster_density_separation = self._cluster_density_separation(MST, cluster, cluster_j)
                if cluster_density_separation < min_density_separation:
                    min_density_separation = cluster_density_separation
        cluster_density_sparseness = self._cluster_density_sparseness(MST, cluster)
        numerator = min_density_separation - cluster_density_sparseness
        denominator = np.max([min_density_separation, cluster_density_sparseness])
        cluster_validity = numerator / denominator
        return cluster_validity


    def _clustering_validity_index(self, MST):
        """
        Calculate the overall clustering validity index.
        """
        n_samples = len(self.labels)
        validity_index = 0
        for label in np.unique(self.labels):
            fraction = np.sum(self.labels == label) / float(n_samples)
            cluster_validity = self._cluster_validity_index(MST, label)
            validity_index += fraction * cluster_validity
        return validity_index


    def _get_label_members(self, cluster):
        """
        Get members of a specific cluster.
        """
        indices = np.where(self.labels == cluster)[0]
        members = self.X[indices]
        return members
