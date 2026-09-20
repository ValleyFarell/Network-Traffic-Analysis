"""Поиск похожих хостов и поведенческая HDBSCAN-кластеризация."""

from __future__ import annotations

import hdbscan
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class HostSimilarityModel:
    """Основная модель на family-balanced embedding из v19.

    Поиск выполняется по евклидову расстоянию в исходном 454-мерном
    пространстве. UMAP в модель не входит.
    """

    def __init__(self) -> None:
        self.clusterer = hdbscan.HDBSCAN(
            metric="euclidean",
            min_cluster_size=80,
            min_samples=5,
            cluster_selection_method="eom",
            gen_min_span_tree=True,
            prediction_data=True,
        )
        self.neighbor_model = NearestNeighbors(metric="euclidean")
        self.embedding_: pd.DataFrame | None = None
        self.family_columns_: dict[str, list[str]] = {}
        self.cluster_labels_: pd.Series | None = None
        self.cluster_strengths_: pd.Series | None = None

    def fit(
        self,
        embedding: pd.DataFrame,
        family_columns: dict[str, list[str]],
    ) -> HostSimilarityModel:
        """Обучает kNN и HDBSCAN с параметрами выбранного решения v19."""

        if embedding.index.has_duplicates:
            raise ValueError("Индекс embedding содержит повторяющиеся хосты")
        if not np.isfinite(embedding.to_numpy()).all():
            raise ValueError("Embedding содержит NaN или inf")

        self.embedding_ = embedding.astype(np.float32).copy()
        self.family_columns_ = {
            family: list(columns) for family, columns in family_columns.items()
        }

        values = self.embedding_.to_numpy(dtype=np.float64)
        self.neighbor_model.fit(values)
        labels = self.clusterer.fit_predict(values)
        self.cluster_labels_ = pd.Series(labels, index=embedding.index, name="cluster")
        self.cluster_strengths_ = pd.Series(
            self.clusterer.probabilities_,
            index=embedding.index,
            name="cluster_strength",
        )
        return self

    def nearest(self, host_id: str, limit: int = 10) -> pd.DataFrame:
        """Возвращает ближайшие известные хосты, исключая сам запрос."""

        embedding = self._require_embedding()
        if host_id not in embedding.index:
            raise KeyError(f"Хост {host_id!r} отсутствует в модели")

        query = embedding.loc[[host_id]].to_numpy(dtype=np.float64)
        count = min(limit + 1, len(embedding))
        distances, positions = self.neighbor_model.kneighbors(query, n_neighbors=count)

        rows = []
        for distance, position in zip(distances[0], positions[0], strict=True):
            candidate = str(embedding.index[position])
            if candidate == host_id:
                continue
            rows.append(
                {
                    "host_id": candidate,
                    "distance": float(distance),
                    # Это отображаемая близость из v14, а не вероятность.
                    "similarity_score": float(np.exp(-distance)),
                    "cluster_id": int(self.cluster_labels_.loc[candidate]),
                    "cluster_strength": float(self.cluster_strengths_.loc[candidate]),
                }
            )
            if len(rows) == limit:
                break

        return pd.DataFrame(rows)

    def nearest_vector(self, vector: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
        """Ищет соседей для одного нового вектора в тех же координатах."""

        embedding = self._require_embedding()
        vector = vector.reindex(columns=embedding.columns)
        if len(vector) != 1:
            raise ValueError("Для поиска должен быть передан ровно один вектор")
        if vector.isna().any().any():
            raise ValueError("Координаты нового вектора не совпадают с обучающими")

        distances, positions = self.neighbor_model.kneighbors(
            vector.to_numpy(dtype=np.float64),
            n_neighbors=min(limit, len(embedding)),
        )
        rows = []
        for distance, position in zip(distances[0], positions[0], strict=True):
            candidate = str(embedding.index[position])
            rows.append(
                {
                    "host_id": candidate,
                    "distance": float(distance),
                    "similarity_score": float(np.exp(-distance)),
                    "cluster_id": int(self.cluster_labels_.loc[candidate]),
                    "cluster_strength": float(self.cluster_strengths_.loc[candidate]),
                }
            )
        return pd.DataFrame(rows)

    def predict_cluster(self, vector: pd.DataFrame) -> tuple[int, float]:
        """Применяет HDBSCAN approximate_predict к одному новому хосту."""

        embedding = self._require_embedding()
        vector = vector.reindex(columns=embedding.columns)
        if len(vector) != 1 or vector.isna().any().any():
            raise ValueError("Ожидается один вектор с обучающим набором координат")

        labels, strengths = hdbscan.approximate_predict(
            self.clusterer,
            vector.to_numpy(dtype=np.float64),
        )
        return int(labels[0]), float(strengths[0])

    def explain_pair(self, left: str, right: str) -> pd.DataFrame:
        """Разлагает квадрат итогового расстояния по семействам признаков."""

        embedding = self._require_embedding()
        delta = embedding.loc[left] - embedding.loc[right]
        total = float(np.dot(delta, delta))

        rows = []
        for family, columns in self.family_columns_.items():
            family_delta = delta[columns].to_numpy(dtype=float)
            squared = float(np.dot(family_delta, family_delta))
            rows.append(
                {
                    "family": family,
                    "distance": float(np.sqrt(squared)),
                    "squared_contribution": squared,
                    "share": squared / total if total > 0 else 0.0,
                }
            )
        return pd.DataFrame(rows).sort_values("squared_contribution", ascending=False)

    def _require_embedding(self) -> pd.DataFrame:
        if self.embedding_ is None:
            raise RuntimeError("Модель ещё не обучена")
        if self.cluster_labels_ is None or self.cluster_strengths_ is None:
            raise RuntimeError("Не сохранены результаты HDBSCAN")
        return self.embedding_
