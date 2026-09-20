"""Графовые роли stable_topology из экспериментов v19/v20."""

from __future__ import annotations

import hdbscan
import networkx as nx
import numpy as np
import pandas as pd

TOPOLOGY_COLUMNS = [
    "directed_pagerank",
    "undirected_pagerank",
    "reciprocity",
    "core_number",
    "clustering",
    "average_neighbor_degree",
    "two_hop_neighbors",
]


class StableTopologyBuilder:
    """Строит семь координат лучшего graph-only пространства.

    В stable-графе остаются только направленные рёбра, наблюдавшиеся не менее
    чем в трёх разных днях интервала ``[0, 15)``. Самопетли исключаются ещё
    при агрегации рёбер.
    """

    def __init__(self, stable_min_days: int = 3) -> None:
        self.stable_min_days = stable_min_days
        self.median_: pd.Series | None = None
        self.scale_: pd.Series | None = None

    def fit_transform(
        self,
        edges: pd.DataFrame,
        host_ids: pd.Index,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Возвращает исходные и robust-scaled топологические признаки."""

        # В v19 множество узлов создавалось по полному графу, а уже затем
        # добавлялись только устойчивые рёбра. Это важно для PageRank:
        # узлы без stable-рёбер остаются в графе как изолированные.
        node_universe = pd.Index(
            pd.unique(
                pd.concat(
                    [
                        edges.src.astype(str),
                        edges.dst.astype(str),
                        pd.Series(host_ids.astype(str)),
                    ],
                    ignore_index=True,
                )
            )
        )
        stable_edges = self.select_stable_edges(edges)
        raw = self.build_features(stable_edges, host_ids, node_universe)

        self.median_ = raw.median()
        q1 = raw.quantile(0.25)
        q3 = raw.quantile(0.75)
        scale = q3 - q1
        fallback = raw.std(ddof=0)
        self.scale_ = scale.where(scale.gt(0), fallback).replace(0, 1).fillna(1)

        return raw, self.transform_features(raw)

    def transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Применяет масштабирование v19: медиана, IQR и clip ``[-8, 8]``."""

        if self.median_ is None or self.scale_ is None:
            raise RuntimeError("Сначала вызови fit_transform")

        return (
            (features[TOPOLOGY_COLUMNS] - self.median_) / self.scale_
        ).clip(-8, 8).astype(np.float32)

    def select_stable_edges(self, edges: pd.DataFrame) -> pd.DataFrame:
        """Отбирает те же устойчивые связи, что использовались в v19."""

        required = {"src", "dst", "flows", "active_days"}
        missing = required - set(edges.columns)
        if missing:
            raise ValueError(f"В таблице рёбер отсутствуют столбцы: {sorted(missing)}")

        selected = edges.loc[
            edges.active_days.ge(self.stable_min_days)
            & edges.src.notna()
            & edges.dst.notna()
            & edges.src.ne(edges.dst)
        ].copy()
        selected["src"] = selected.src.astype(str)
        selected["dst"] = selected.dst.astype(str)
        return selected

    @staticmethod
    def build_features(
        edges: pd.DataFrame,
        host_ids: pd.Index,
        node_universe: pd.Index,
    ) -> pd.DataFrame:
        """Вычисляет семь топологических координат stable_topology."""

        host_ids = pd.Index(host_ids.astype(str), name="host")
        node_universe = pd.Index(node_universe.astype(str))

        directed = nx.DiGraph()
        directed.add_nodes_from(node_universe)
        for row in edges.itertuples(index=False):
            directed.add_edge(str(row.src), str(row.dst), flows=float(row.flows))

        undirected = nx.Graph()
        undirected.add_nodes_from(node_universe)
        for left, right, data in directed.edges(data=True):
            if undirected.has_edge(left, right):
                undirected[left][right]["flows"] += data["flows"]
            else:
                undirected.add_edge(left, right, flows=data["flows"])

        incoming = {host: set(directed.predecessors(host)) for host in host_ids}
        outgoing = {host: set(directed.successors(host)) for host in host_ids}

        result = pd.DataFrame(index=host_ids)
        result["reciprocity"] = [
            len(incoming[host] & outgoing[host]) / len(incoming[host] | outgoing[host])
            if incoming[host] | outgoing[host]
            else 0.0
            for host in host_ids
        ]
        result["directed_pagerank"] = pd.Series(
            nx.pagerank(directed, weight="flows")
        ).reindex(host_ids, fill_value=0)
        result["undirected_pagerank"] = pd.Series(
            nx.pagerank(undirected, weight="flows")
        ).reindex(host_ids, fill_value=0)
        result["core_number"] = pd.Series(nx.core_number(undirected)).reindex(
            host_ids, fill_value=0
        )
        result["clustering"] = pd.Series(
            nx.clustering(undirected, weight=None)
        ).reindex(host_ids, fill_value=0)
        result["average_neighbor_degree"] = pd.Series(
            nx.average_neighbor_degree(undirected, weight=None)
        ).reindex(host_ids, fill_value=0)

        nodes = list(undirected.nodes)
        positions = {host: position for position, host in enumerate(nodes)}
        adjacency = nx.to_scipy_sparse_array(
            undirected,
            nodelist=nodes,
            weight=None,
            format="csr",
            dtype=np.int32,
        )
        target_positions = [positions[host] for host in host_ids]
        two_hop = adjacency[target_positions] @ adjacency
        two_hop.data[:] = 1
        reached = (adjacency[target_positions] + two_hop).astype(bool)
        result["two_hop_neighbors"] = (
            np.asarray(reached.sum(axis=1)).ravel() - 1
        ).clip(min=0)

        return result[TOPOLOGY_COLUMNS].fillna(0).astype(float)


class GraphRoleModel:
    """HDBSCAN-модель лучшего stable_topology решения v19."""

    def __init__(self) -> None:
        self.clusterer = hdbscan.HDBSCAN(
            metric="euclidean",
            min_cluster_size=120,
            min_samples=10,
            cluster_selection_method="eom",
            gen_min_span_tree=True,
        )
        self.features_: pd.DataFrame | None = None
        self.labels_: pd.Series | None = None
        self.probabilities_: pd.Series | None = None

    def fit(self, scaled_features: pd.DataFrame) -> GraphRoleModel:
        """Обучает графовые роли на robust-scaled семи координатах."""

        # В v19 HDBSCAN получал именно float32-массив. Для этого пространства
        # много близких и совпадающих точек, поэтому смена dtype способна
        # изменить минимальное остовное дерево и судьбу пограничного хоста.
        values = scaled_features[TOPOLOGY_COLUMNS].to_numpy(dtype=np.float32)
        labels = self.clusterer.fit_predict(values)
        self.features_ = scaled_features[TOPOLOGY_COLUMNS].copy()
        self.labels_ = pd.Series(labels, index=scaled_features.index, name="graph_role")
        self.probabilities_ = pd.Series(
            self.clusterer.probabilities_,
            index=scaled_features.index,
            name="graph_role_strength",
        )
        return self

    def explain_pair(self, left: str, right: str) -> pd.DataFrame:
        """Разлагает квадрат евклидова расстояния по семи координатам."""

        if self.features_ is None:
            raise RuntimeError("Графовая модель ещё не обучена")

        delta = self.features_.loc[left] - self.features_.loc[right]
        squared = delta.pow(2)
        total = squared.sum()
        result = pd.DataFrame(
            {
                "left_value": self.features_.loc[left],
                "right_value": self.features_.loc[right],
                "squared_contribution": squared,
                "share": squared / total if total > 0 else 0.0,
            }
        )
        return result.sort_values("squared_contribution", ascending=False)

    def role_profiles(self, raw_features: pd.DataFrame) -> pd.DataFrame:
        """Возвращает медианный исходный профиль каждой найденной роли."""

        if self.labels_ is None:
            raise RuntimeError("Графовая модель ещё не обучена")
        return raw_features.assign(graph_role=self.labels_).groupby("graph_role").median()
