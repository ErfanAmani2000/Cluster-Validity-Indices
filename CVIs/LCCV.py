import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import networkx as nx

class LCCV_Index:
    def __init__(self, data: pd.DataFrame, eps: float = 1e-10):
        """
        Parameters:
          data: pandas DataFrame where the last column is the label;
                features are all other columns.
          eps: small value to avoid division by zero.
        """
        # Ensure DataFrame input
        if not hasattr(data, 'iloc'):
            raise TypeError("data must be a pandas DataFrame")
        # Extract features only
        self.X = data.iloc[:, :-1].to_numpy()
        self.n = self.X.shape[0]
        self.eps = eps

        # Precompute full pairwise distances and sorted neighbors
        self.dist_matrix = pairwise_distances(self.X)
        self.sorted_idx   = np.argsort(self.dist_matrix, axis=1)  # includes self at pos0
        self.sorted_dists = np.sort(self.dist_matrix, axis=1)

    def natural_neighbors(self):
        """
        Algorithm 1 (NaN-Searching) to find λ, nbλ, and each point’s local neighbors LN
        :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
        """
        n = self.n
        sorted_idx = self.sorted_idx
        nb = np.zeros(n, dtype=int)               # reverse‐neighbor counts
        count_prev = n
        LN = [set() for _ in range(n)]            # N N_r(i) for each i
        r = 1

        while True:
            new_LN = [set(neigh) for neigh in LN]
            # Expand to r‑th neighbors
            for i in range(n):
                y = sorted_idx[i][r]              # r-th nearest neighbor of i
                nb[y] += 1
                new_LN[i].add(y)
            count_curr = np.sum(nb == 0)
            # Terminate when all have reverse neighbors or count stabilizes
            if count_curr == 0 or count_curr == count_prev:
                LN = new_LN
                break
            LN = new_LN
            count_prev = count_curr
            r += 1

        lam = r
        nb_lambda = nb.copy()                     # nbλ(i)
        return lam, nb_lambda, LN

    def compute_density(self, nb_lambda):
        """
        Eq (3): ρ(i) = μ / ∑_{j∈N N_μ(i)} dist(i,j),
        where μ = max_i nbλ(i).
        :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
        """
        mu = nb_lambda.max()
        # Use μ nearest neighbors (skip self at index 0)
        neigh_dists = self.sorted_dists[:, 1:mu+1]
        sum_d = np.sum(neigh_dists, axis=1) + self.eps
        density = mu / sum_d
        return density

    def assign_representatives(self, LN, density):
        """
        Algorithm 2 (LORE): choose Rep(p) from LN(p) with maximum density,
        break ties by closest distance (RCR), then apply RTR.
        :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}
        """
        n = self.n
        rep = np.arange(n)

        # Initial assignment with RCR
        for i in range(n):
            neigh = LN[i]
            if not neigh:
                rep[i] = i
                continue

            # find max density among LN(i)
            neigh_list = list(neigh)
            neigh_dens = density[neigh_list]
            max_d = np.max(neigh_dens)

            if max_d <= density[i]:
                rep[i] = i
            else:
                # candidates with equal top density
                cands = [j for j in neigh_list if density[j] == max_d]
                if len(cands) > 1:
                    # tie-breaker: pick nearest
                    dists = [self.dist_matrix[i, j] for j in cands]
                    rep[i] = cands[int(np.argmin(dists))]
                else:
                    rep[i] = cands[0]

        # Apply RTR via path‑compression “find”
        def find(u):
            if rep[u] != u:
                rep[u] = find(rep[u])
            return rep[u]

        for i in range(n):
            rep[i] = find(i)

        return rep

    def build_saturated_graph(self, LN):
        """
        Definition 3: undirected Saturated Neighbor Graph (USNG),
        edge between i,j iff i∈LN[j] or j∈LN[i].
        :contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}
        """
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        for i in range(self.n):
            for j in LN[i]:
                G.add_edge(i, j, weight=self.dist_matrix[i, j])
        return G

    def compute_core_graph_distances(self, local_cores, G):
        """
        Eq (4): Dijkstra shortest paths between cores; unreachable→max_dist
        :contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}
        """
        m = len(local_cores)
        D = np.zeros((m, m))
        all_dists = {}
        max_dist = 0

        for idx, c in enumerate(local_cores):
            d = nx.single_source_dijkstra_path_length(G, c, weight='weight')
            all_dists[c] = d
            if d:
                max_dist = max(max_dist, max(d.values()))

        default = max_dist
        for i, ci in enumerate(local_cores):
            for j, cj in enumerate(local_cores):
                if i == j:
                    D[i, j] = 0
                else:
                    D[i, j] = all_dists[ci].get(cj, default)

        return D

    def compute_lccv(self, local_cores, rep, clusters, core_dist_matrix):
        """
        Eq (5–6): per-core LCCV and weighted average.
        :contentReference[oaicite:12]{index=12}:contentReference[oaicite:13]{index=13}
        """
        m = len(local_cores)
        # group cores by their rep‑label (allows multi‑core clusters)
        core_groups = {}
        for c in local_cores:
            lbl = rep[c]
            core_groups.setdefault(lbl, []).append(c)

        # map core → its index in local_cores
        idx_map = {c: i for i, c in enumerate(local_cores)}

        total_weight = 0.0
        weighted_sum = 0.0

        for c in local_cores:
            i = idx_map[c]
            # intra‑core distance a(i)
            same = core_groups[rep[c]]
            if len(same) <= 1:
                a_i = 0.0
            else:
                idxs = [idx_map[cc] for cc in same if cc != c]
                a_i = np.mean([core_dist_matrix[i, j] for j in idxs])

            # inter‑cluster distance b(i)
            b_vals = []
            for lbl, group in core_groups.items():
                if lbl == rep[c]:
                    continue
                jdxs = [idx_map[cc] for cc in group]
                b_vals.append(np.mean([core_dist_matrix[i, j] for j in jdxs]))
            b_i = min(b_vals) if b_vals else 0.0

            lccv_i = (b_i - a_i) / (max(b_i, a_i) + self.eps)
            weight = len(clusters[rep[c]])
            weighted_sum += lccv_i * weight
            total_weight += weight

        return (weighted_sum / total_weight) if total_weight > 0 else 0.0

    def run(self):
        # 1) Natural neighbors & λ
        lam, nb_lambda, LN = self.natural_neighbors()

        # 2) Density
        density = self.compute_density(nb_lambda)

        # 3) Representatives & local cores
        rep = self.assign_representatives(LN, density)
        local_cores = np.where(rep == np.arange(self.n))[0]

        # 4) Assign each point to its core
        clusters = {c: np.where(rep == c)[0].tolist() for c in local_cores}

        # 5) Build saturated–neighbor graph
        G = self.build_saturated_graph(LN)

        # 6) Core–to–core distances
        core_dist_matrix = self.compute_core_graph_distances(local_cores, G)

        # 7) Compute overall LCCV index
        overall_lccv = self.compute_lccv(local_cores, rep, clusters, core_dist_matrix)

        return overall_lccv
