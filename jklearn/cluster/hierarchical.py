import numpy as np


class HierarchicalClustering:
    def __init__(
        self,
        n_clusters=2,
        method="agglomerative",
        linkage="average",
        metric="euclidean",
        max_iter=25,
    ):
        self.n_clusters = n_clusters
        self.method = method
        self.linkage = linkage
        self.metric = metric
        self.max_iter = max_iter

        self.labels_ = None
        self.linkage_ = None
        self.children_ = None
        self.distances_ = None
        self.n_leaves_ = 0

        self._nodes = {}
        self._root_id = None

    def _chk(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("X must be a 2D array")
        if x.shape[0] < 2:
            raise ValueError("X must contain at least 2 samples")
        return x

    def _validate(self, x):
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")
        if self.n_clusters > x.shape[0]:
            raise ValueError("n_clusters must not exceed the number of samples")
        if self.method not in ("agglomerative", "divisive"):
            raise ValueError("method must be 'agglomerative' or 'divisive'")
        if self.linkage not in ("single", "complete", "average", "ward"):
            raise ValueError("linkage must be one of: single, complete, average, ward")
        if self.metric not in ("euclidean", "manhattan"):
            raise ValueError("metric must be 'euclidean' or 'manhattan'")
        if self.linkage == "ward" and self.metric != "euclidean":
            raise ValueError("ward linkage requires metric='euclidean'")

    def _pairwise(self, a, b):
        if self.metric == "euclidean":
            d = a[:, None, :] - b[None, :, :]
            return np.sqrt(np.sum(d * d, axis=2))
        d = np.abs(a[:, None, :] - b[None, :, :])
        return np.sum(d, axis=2)

    def _cluster_distance(self, x, ia, ib):
        a = x[np.asarray(ia, dtype=int)]
        b = x[np.asarray(ib, dtype=int)]

        if self.linkage == "ward":
            ma = np.mean(a, axis=0)
            mb = np.mean(b, axis=0)
            diff = ma - mb
            return float((len(ia) * len(ib) / (len(ia) + len(ib))) * np.dot(diff, diff))

        d = self._pairwise(a, b)
        if self.linkage == "single":
            return float(np.min(d))
        if self.linkage == "complete":
            return float(np.max(d))
        return float(np.mean(d))

    def _cluster_spread(self, x, idx):
        pts = x[np.asarray(idx, dtype=int)]
        if pts.shape[0] <= 1:
            return 0.0
        d = self._pairwise(pts, pts)
        return float(np.max(d))

    def _farthest_pair(self, x):
        d = self._pairwise(x, x)
        iu = np.triu_indices(x.shape[0], k=1)
        k = np.argmax(d[iu])
        return int(iu[0][k]), int(iu[1][k])

    def _bisect_cluster(self, x, idx):
        pts = x[np.asarray(idx, dtype=int)]
        if pts.shape[0] == 2:
            return [idx[0]], [idx[1]]

        i, j = self._farthest_pair(pts)
        c0 = pts[i].copy()
        c1 = pts[j].copy()

        for _ in range(self.max_iter):
            d0 = np.sum((pts - c0) ** 2, axis=1)
            d1 = np.sum((pts - c1) ** 2, axis=1)
            lab = (d1 < d0).astype(int)

            if np.all(lab == 0) or np.all(lab == 1):
                half = pts.shape[0] // 2
                order = np.argsort(d0 - d1)
                lab = np.zeros(pts.shape[0], dtype=int)
                lab[order[half:]] = 1

            n0 = np.mean(pts[lab == 0], axis=0)
            n1 = np.mean(pts[lab == 1], axis=0)

            if np.allclose(n0, c0) and np.allclose(n1, c1):
                break
            c0, c1 = n0, n1

        left = [idx[k] for k in np.where(lab == 0)[0]]
        right = [idx[k] for k in np.where(lab == 1)[0]]
        return left, right

    def _fit_agglomerative(self, x):
        n = x.shape[0]
        clusters = {i: [i] for i in range(n)}
        active = list(range(n))
        rows = []
        nodes = {
            i: {
                "members": [i],
                "left": None,
                "right": None,
                "height": 0.0,
                "size": 1,
            }
            for i in range(n)
        }

        nxt = n
        while len(active) > 1:
            best_d = np.inf
            best_i = None
            best_j = None

            for a in range(len(active) - 1):
                for b in range(a + 1, len(active)):
                    i = active[a]
                    j = active[b]
                    d = self._cluster_distance(x, clusters[i], clusters[j])
                    if d < best_d:
                        best_d = d
                        best_i, best_j = i, j

            m = clusters[best_i] + clusters[best_j]
            clusters[nxt] = m
            rows.append([best_i, best_j, float(best_d), len(m)])
            nodes[nxt] = {
                "members": m,
                "left": best_i,
                "right": best_j,
                "height": float(best_d),
                "size": len(m),
            }

            active.remove(best_i)
            active.remove(best_j)
            active.append(nxt)
            del clusters[best_i]
            del clusters[best_j]
            nxt += 1

        self._nodes = nodes
        self._root_id = active[0]
        return np.asarray(rows, dtype=float)

    def _fit_divisive(self, x):
        n = x.shape[0]
        nodes = {
            i: {
                "members": [i],
                "left": None,
                "right": None,
                "height": 0.0,
                "size": 1,
            }
            for i in range(n)
        }

        root_id = 2 * n - 2
        nodes[root_id] = {
            "members": list(range(n)),
            "left": None,
            "right": None,
            "height": self._cluster_spread(x, list(range(n))),
            "size": n,
        }

        leaf_nodes = [root_id]
        next_internal = root_id - 1

        while True:
            cand = [cid for cid in leaf_nodes if len(nodes[cid]["members"]) > 1]
            if not cand:
                break

            split_id = max(cand, key=lambda cid: nodes[cid]["height"])
            members = nodes[split_id]["members"]
            left_idx, right_idx = self._bisect_cluster(x, members)

            left_id = left_idx[0] if len(left_idx) == 1 else next_internal
            if len(left_idx) > 1:
                next_internal -= 1
            right_id = right_idx[0] if len(right_idx) == 1 else next_internal
            if len(right_idx) > 1:
                next_internal -= 1

            if len(left_idx) > 1:
                nodes[left_id] = {
                    "members": left_idx,
                    "left": None,
                    "right": None,
                    "height": self._cluster_spread(x, left_idx),
                    "size": len(left_idx),
                }
            if len(right_idx) > 1:
                nodes[right_id] = {
                    "members": right_idx,
                    "left": None,
                    "right": None,
                    "height": self._cluster_spread(x, right_idx),
                    "size": len(right_idx),
                }

            nodes[split_id]["left"] = left_id
            nodes[split_id]["right"] = right_id

            leaf_nodes.remove(split_id)
            leaf_nodes.append(left_id)
            leaf_nodes.append(right_id)

        rows = []
        next_merge_id = n

        def build(cid):
            nonlocal next_merge_id
            node = nodes[cid]
            if node["left"] is None:
                return cid
            lid = build(node["left"])
            rid = build(node["right"])
            rows.append([lid, rid, float(node["height"]), node["size"]])
            out = next_merge_id
            next_merge_id += 1
            return out

        build(root_id)
        self._nodes = nodes
        self._root_id = root_id
        return np.asarray(rows, dtype=float)

    def _cut_tree(self, n_clusters):
        active = [self._root_id]
        while len(active) < n_clusters:
            splittable = [
                cid
                for cid in active
                if self._nodes[cid]["left"] is not None and self._nodes[cid]["right"] is not None
            ]
            if not splittable:
                break
            sid = max(splittable, key=lambda cid: self._nodes[cid]["height"])
            active.remove(sid)
            active.append(self._nodes[sid]["left"])
            active.append(self._nodes[sid]["right"])

        labels = np.empty(self.n_leaves_, dtype=int)
        for k, cid in enumerate(active):
            labels[self._nodes[cid]["members"]] = k
        return labels

    def fit(self, x):
        x = self._chk(x)
        self._validate(x)
        self.n_leaves_ = x.shape[0]

        if self.method == "agglomerative":
            self.linkage_ = self._fit_agglomerative(x)
        else:
            self.linkage_ = self._fit_divisive(x)

        self.children_ = self.linkage_[:, :2].astype(int)
        self.distances_ = self.linkage_[:, 2].astype(float)
        self.labels_ = self._cut_tree(self.n_clusters)
        return self

    def fit_predict(self, x):
        return self.fit(x).labels_

    def dendrogram(self):
        if self.linkage_ is None:
            raise ValueError("Model is not fitted yet. Call fit(x) first.")
        return self.linkage_.copy()
