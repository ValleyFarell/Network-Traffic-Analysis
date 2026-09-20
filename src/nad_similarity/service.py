"""Прикладной слой двухуровневого поиска похожих хостов."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nad_similarity.artifacts import ModelArtifact, load_artifact
from nad_similarity.inference import build_new_host_tables
from nad_similarity.schemas import (
    FeatureContribution,
    HostPredictionRequest,
    HostResponse,
    NeighborResponse,
    PairComparisonResponse,
    PredictionResponse,
    SimilarHostsResponse,
)

# Именно эти четыре семейства использовались на втором уровне в v20.
BEHAVIOR_FAMILIES = ("time", "services", "flow", "direction")


class KnownHostConflictError(ValueError):
    """Новый inference нельзя запускать с ID обучающего хоста."""


class SimilarityService:
    """Ищет соседей внутри графовой роли и объясняет найденное расстояние.

    Графовая роль задаёт множество кандидатов. Внутри него хосты ранжируются
    евклидовым расстоянием в поведенческом пространстве v20. Графовое и
    поведенческое расстояния не складываются.
    """

    def __init__(self, artifact: ModelArtifact) -> None:
        self.artifact = artifact
        model = artifact.similarity_model
        graph_model = artifact.graph_role_model
        hierarchy_model = artifact.hierarchy_model

        if model.embedding_ is None:
            raise ValueError("В артефакте отсутствует поведенческий embedding")
        if model.cluster_labels_ is None or model.cluster_strengths_ is None:
            raise ValueError("В артефакте отсутствуют результаты поведенческой модели")
        if (
            graph_model.labels_ is None
            or graph_model.probabilities_ is None
            or graph_model.features_ is None
        ):
            raise ValueError("В артефакте отсутствуют графовые роли или признаки")
        if (
            hierarchy_model.labels_ is None
            or hierarchy_model.hierarchical_labels_ is None
            or hierarchy_model.behavior_space_ is None
        ):
            raise ValueError("В артефакте отсутствуют поведенческие подтипы v20")

        self.host_ids = pd.Index(model.embedding_.index.astype(str), name="host")
        self.cluster_labels = model.cluster_labels_.reindex(self.host_ids)
        self.cluster_strengths = model.cluster_strengths_.reindex(self.host_ids)
        self.graph_roles = graph_model.labels_.reindex(self.host_ids)
        self.graph_role_strengths = graph_model.probabilities_.reindex(self.host_ids)
        self.graph_features = graph_model.features_.reindex(self.host_ids)
        self.behavior_subtypes = hierarchy_model.labels_.reindex(self.host_ids)
        self.hierarchical_groups = hierarchy_model.hierarchical_labels_.reindex(
            self.host_ids
        )
        self.behavior = hierarchy_model.behavior_space_.reindex(self.host_ids)

        if any(frame.isna().any().any() for frame in [
            self.cluster_labels.to_frame(),
            self.cluster_strengths.to_frame(),
            self.graph_roles.to_frame(),
            self.graph_role_strengths.to_frame(),
            self.graph_features,
            self.behavior_subtypes.to_frame(),
            self.hierarchical_groups.to_frame(),
            self.behavior,
        ]):
            raise ValueError("Индексы компонентов артефакта не совпадают")

        family_columns = model.family_columns_
        missing = [family for family in BEHAVIOR_FAMILIES if family not in family_columns]
        if missing:
            raise ValueError(f"В embedding отсутствуют семейства v20: {missing}")

        self.behavior_columns = {
            family: list(family_columns[family]) for family in BEHAVIOR_FAMILIES
        }
        behavior_columns = [
            column
            for family in BEHAVIOR_FAMILIES
            for column in self.behavior_columns[family]
        ]
        if list(self.behavior.columns) != behavior_columns:
            raise ValueError("Координаты второго уровня не совпадают с embedding")

    @classmethod
    def from_path(cls, path: Path) -> SimilarityService:
        """Загружает проверенный артефакт обучения."""

        return cls(load_artifact(path))

    def host(self, host_id: str) -> HostResponse:
        """Возвращает сохранённые модельные метки известного хоста."""

        host_id = self._require_host(host_id)
        cluster_id = int(self.cluster_labels.loc[host_id])
        graph_role = int(self.graph_roles.loc[host_id])
        behavior_subtype = int(self.behavior_subtypes.loc[host_id])
        return HostResponse(
            host_id=host_id,
            cluster_id=cluster_id,
            cluster_strength=float(self.cluster_strengths.loc[host_id]),
            is_noise=cluster_id == -1,
            graph_role=graph_role,
            graph_role_strength=float(self.graph_role_strengths.loc[host_id]),
            is_graph_noise=graph_role == -1,
            behavior_subtype=behavior_subtype,
            is_subtype_noise=graph_role != -1 and behavior_subtype == -1,
            hierarchical_group=str(self.hierarchical_groups.loc[host_id]),
            model_version=self.artifact.version,
        )

    def similar_hosts(
        self,
        host_id: str,
        limit: int = 10,
        within_graph_role: bool = True,
    ) -> SimilarHostsResponse:
        """Реализует поиск v20: графовая роль, затем поведенческая дистанция."""

        host_id = self._require_host(host_id)
        query_position = int(self.host_ids.get_loc(host_id))

        if within_graph_role:
            query_role = int(self.graph_roles.loc[host_id])
            candidate_mask = self.graph_roles.eq(query_role).to_numpy(copy=True)
            search_scope = "within_graph_role"
        else:
            candidate_mask = np.ones(len(self.host_ids), dtype=bool)
            search_scope = "global_behavior"

        candidate_mask[query_position] = False
        candidate_positions = np.flatnonzero(candidate_mask)
        if len(candidate_positions) == 0:
            selected_positions = np.array([], dtype=int)
            selected_distances = np.array([], dtype=float)
        else:
            query = self.behavior.iloc[query_position].to_numpy(dtype=float)
            candidates = self.behavior.iloc[candidate_positions].to_numpy(dtype=float)
            distances = np.linalg.norm(candidates - query, axis=1)
            order = np.argsort(distances, kind="stable")[: min(limit, len(distances))]
            selected_positions = candidate_positions[order]
            selected_distances = distances[order]

        neighbors = []
        for position, distance in zip(
            selected_positions,
            selected_distances,
            strict=True,
        ):
            candidate = str(self.host_ids[position])
            neighbors.append(
                NeighborResponse(
                    host_id=candidate,
                    distance=float(distance),
                    similarity_score=float(np.exp(-distance)),
                    cluster_id=int(self.cluster_labels.loc[candidate]),
                    cluster_strength=float(self.cluster_strengths.loc[candidate]),
                    graph_role=int(self.graph_roles.loc[candidate]),
                    graph_role_strength=float(
                        self.graph_role_strengths.loc[candidate]
                    ),
                    behavior_subtype=int(self.behavior_subtypes.loc[candidate]),
                    hierarchical_group=str(self.hierarchical_groups.loc[candidate]),
                    contributions=self._behavior_contributions(host_id, candidate),
                )
            )

        return SimilarHostsResponse(
            query=self.host(host_id),
            search_scope=search_scope,
            neighbors=neighbors,
        )

    def compare(self, left: str, right: str) -> PairComparisonResponse:
        """Сравнивает два хоста в поведенческом и графовом пространствах."""

        left = self._require_host(left)
        right = self._require_host(right)
        behavior_contributions = self._behavior_contributions(left, right)
        graph_contributions = self._graph_contributions(left, right)

        behavior_distance = float(
            np.linalg.norm(self.behavior.loc[left] - self.behavior.loc[right])
        )
        graph_distance = float(
            np.linalg.norm(self.graph_features.loc[left] - self.graph_features.loc[right])
        )
        return PairComparisonResponse(
            left=self.host(left),
            right=self.host(right),
            same_graph_role=(
                int(self.graph_roles.loc[left]) != -1
                and int(self.graph_roles.loc[left]) == int(self.graph_roles.loc[right])
            ),
            same_hierarchical_group=(
                str(self.hierarchical_groups.loc[left])
                == str(self.hierarchical_groups.loc[right])
            ),
            behavior_distance=behavior_distance,
            graph_distance=graph_distance,
            behavior_contributions=behavior_contributions,
            graph_contributions=graph_contributions,
        )

    def predict(self, request: HostPredictionRequest) -> PredictionResponse:
        """Применяет глобальную модель к полной flow-истории нового хоста."""

        if request.host_id in self.host_ids:
            raise KnownHostConflictError(
                f"Хост {request.host_id!r} уже присутствует в модели; используй GET /hosts"
            )

        builder = self.artifact.behavior_builder
        tables = build_new_host_tables(
            request=request,
            settings=builder.settings,
            port_categories=builder.categories_.get("port", []),
        )
        vector = builder.transform(tables, pd.Index([request.host_id], name="host"))
        cluster_id, cluster_strength = self.artifact.similarity_model.predict_cluster(vector)
        nearest = self.artifact.similarity_model.nearest_vector(
            vector,
            limit=request.neighbors,
        )

        neighbors = []
        for row in nearest.itertuples(index=False):
            candidate = str(row.host_id)
            neighbors.append(
                NeighborResponse(
                    host_id=candidate,
                    distance=float(row.distance),
                    similarity_score=float(row.similarity_score),
                    cluster_id=int(row.cluster_id),
                    cluster_strength=float(row.cluster_strength),
                    graph_role=int(self.graph_roles.loc[candidate]),
                    graph_role_strength=float(
                        self.graph_role_strengths.loc[candidate]
                    ),
                    behavior_subtype=int(self.behavior_subtypes.loc[candidate]),
                    hierarchical_group=str(self.hierarchical_groups.loc[candidate]),
                    contributions=self._vector_contributions(vector, candidate),
                )
            )

        return PredictionResponse(
            host_id=request.host_id,
            cluster_id=cluster_id,
            cluster_strength=cluster_strength,
            is_noise=cluster_id == -1,
            observations=len(request.flows),
            limitations=[
                "Графовая роль и локальный поведенческий подтип не вычисляются: "
                "для них нужен контекст полного stable-графа.",
                "Поиск выполняется глобально в 454-мерном embedding; входные flows "
                "должны представлять весь период наблюдения [0, 15).",
            ],
            model_version=self.artifact.version,
            neighbors=neighbors,
        )

    def _behavior_contributions(
        self,
        left: str,
        right: str,
    ) -> list[FeatureContribution]:
        delta = self.behavior.loc[left] - self.behavior.loc[right]
        squared_total = float(np.dot(delta, delta))
        rows = []
        for family, columns in self.behavior_columns.items():
            values = delta[columns].to_numpy(dtype=float)
            squared = float(np.dot(values, values))
            rows.append(
                FeatureContribution(
                    family=family,
                    distance=float(np.sqrt(squared)),
                    share=squared / squared_total if squared_total else 0.0,
                )
            )
        return sorted(rows, key=lambda row: row.share, reverse=True)

    def _graph_contributions(
        self,
        left: str,
        right: str,
    ) -> list[FeatureContribution]:
        delta = self.graph_features.loc[left] - self.graph_features.loc[right]
        squared = delta.pow(2)
        total = float(squared.sum())
        return [
            FeatureContribution(
                family=str(column),
                distance=float(np.sqrt(value)),
                share=float(value / total) if total else 0.0,
            )
            for column, value in squared.sort_values(ascending=False).items()
        ]

    def _vector_contributions(
        self,
        vector: pd.DataFrame,
        candidate: str,
    ) -> list[FeatureContribution]:
        model = self.artifact.similarity_model
        query = vector.iloc[0].reindex(model.embedding_.columns)
        delta = query - model.embedding_.loc[candidate]
        squared_total = float(np.dot(delta, delta))
        rows = []
        for family, columns in model.family_columns_.items():
            values = delta[columns].to_numpy(dtype=float)
            squared = float(np.dot(values, values))
            rows.append(
                FeatureContribution(
                    family=family,
                    distance=float(np.sqrt(squared)),
                    share=squared / squared_total if squared_total else 0.0,
                )
            )
        return sorted(rows, key=lambda row: row.share, reverse=True)

    def _require_host(self, host_id: str) -> str:
        host_id = str(host_id).strip()
        if host_id not in self.host_ids:
            raise KeyError(f"Хост {host_id!r} отсутствует в модели")
        return host_id
