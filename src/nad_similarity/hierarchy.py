"""Поведенческие подтипы внутри stable-topology ролей из эксперимента v20."""

from __future__ import annotations

from typing import Any

import hdbscan
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


class HierarchicalSubtypeModel:
    """Обучает отдельный HDBSCAN внутри каждой non-noise графовой роли."""

    def __init__(self, min_cluster_size: int = 30, min_samples: int = 8) -> None:
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.models_: dict[int, hdbscan.HDBSCAN] = {}
        self.behavior_space_: pd.DataFrame | None = None
        self.labels_: pd.Series | None = None
        self.hierarchical_labels_: pd.Series | None = None
        self.report_: dict[int, dict[str, Any]] = {}

    def fit(
        self,
        behavior_space: pd.DataFrame,
        graph_roles: pd.Series,
    ) -> "HierarchicalSubtypeModel":
        """Повторяет второй уровень v20 с параметрами HDBSCAN ``30/8``."""

        graph_roles = graph_roles.reindex(behavior_space.index)
        if graph_roles.isna().any():
            raise ValueError(
                "Индексы поведенческого пространства и графовых ролей не совпали"
            )
        if behavior_space.index.has_duplicates:
            raise ValueError("Индекс поведенческого пространства содержит повторы")
        if not np.isfinite(behavior_space.to_numpy()).all():
            raise ValueError("Поведенческое пространство содержит NaN или inf")

        self.behavior_space_ = behavior_space.astype(np.float32).copy()
        labels = pd.Series(
            -1,
            index=behavior_space.index,
            name="behavior_subtype",
            dtype=int,
        )
        self.models_ = {}
        self.report_ = {}

        role_values = graph_roles.astype(int).to_numpy()
        for role in sorted(graph_roles.loc[graph_roles.ne(-1)].astype(int).unique()):
            positions = np.flatnonzero(role_values == role)
            values = self.behavior_space_.iloc[positions].to_numpy(dtype=np.float32)

            model = hdbscan.HDBSCAN(
                metric="euclidean",
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_method="eom",
                gen_min_span_tree=True,
            ).fit(values)
            raw_labels = model.labels_.astype(int)
            clusters = sorted(set(raw_labels) - {-1})

            # В v20 роль без хотя бы двух найденных кластеров оставалась единой.
            if len(clusters) < 2:
                final_labels = np.zeros(len(positions), dtype=int)
                status = "unsplit"
            else:
                final_labels = raw_labels
                status = "split"

            labels.iloc[positions] = final_labels
            self.models_[int(role)] = model
            self.report_[int(role)] = self._role_report(
                role=int(role),
                values=values,
                labels=final_labels,
                status=status,
                model=model,
            )

        hierarchical = pd.Series(
            "graph_noise",
            index=behavior_space.index,
            name="hierarchical_group",
            dtype=object,
        )
        for host in behavior_space.index:
            role = int(graph_roles.loc[host])
            subtype = int(labels.loc[host])
            if role == -1:
                group = "graph_noise"
            elif subtype == -1:
                group = f"G{role}.noise"
            else:
                group = f"G{role}.B{subtype}"
            hierarchical.loc[host] = group

        self.labels_ = labels
        self.hierarchical_labels_ = hierarchical
        return self

    @staticmethod
    def _role_report(
        role: int,
        values: np.ndarray,
        labels: np.ndarray,
        status: str,
        model: hdbscan.HDBSCAN,
    ) -> dict[str, Any]:
        clustered = labels != -1
        sizes = (
            pd.Series(labels[clustered])
            .value_counts()
            .sort_values(ascending=False)
            .astype(int)
        )
        silhouette: float | None = None
        if status == "split" and clustered.sum() >= 2:
            clustered_labels = labels[clustered]
            cluster_count = len(np.unique(clustered_labels))
            if 2 <= cluster_count < clustered.sum():
                silhouette = float(
                    silhouette_score(values[clustered], clustered_labels)
                )

        try:
            relative_validity = float(model.relative_validity_)
        except (AttributeError, ValueError):
            relative_validity = float("nan")

        return {
            "graph_role": role,
            "hosts": int(len(labels)),
            "status": status,
            "behavior_clusters": int(len(sizes)),
            "behavior_coverage": float(clustered.mean()),
            "behavior_noise_share": float(1 - clustered.mean()),
            "noise_hosts": int((~clustered).sum()),
            "largest_subtype_share": (
                float(sizes.iloc[0] / sizes.sum()) if len(sizes) else None
            ),
            "silhouette": silhouette,
            "relative_validity": relative_validity,
            "subtype_sizes": [int(value) for value in sizes.tolist()],
        }
