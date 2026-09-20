"""Версионированное сохранение и загрузка обученной модели."""

from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nad_similarity.features import BehaviorFeatureBuilder
from nad_similarity.graph import GraphRoleModel, StableTopologyBuilder
from nad_similarity.hierarchy import HierarchicalSubtypeModel
from nad_similarity.model import HostSimilarityModel


ARTIFACT_VERSION = "v2_v20_hierarchical"


@dataclass
class ModelArtifact:
    """Единый артефакт, который загружает API."""

    version: str
    created_at: str
    behavior_builder: BehaviorFeatureBuilder
    similarity_model: HostSimilarityModel
    topology_builder: StableTopologyBuilder
    graph_role_model: GraphRoleModel
    hierarchy_model: HierarchicalSubtypeModel
    behavior_report: dict[str, Any]
    graph_report: dict[str, Any]
    hierarchy_report: dict[int, dict[str, Any]]
    selection_report: dict[str, int]

    @classmethod
    def create(
        cls,
        behavior_builder: BehaviorFeatureBuilder,
        similarity_model: HostSimilarityModel,
        topology_builder: StableTopologyBuilder,
        graph_role_model: GraphRoleModel,
        hierarchy_model: HierarchicalSubtypeModel,
        behavior_report: dict[str, Any],
        graph_report: dict[str, Any],
        hierarchy_report: dict[int, dict[str, Any]],
        selection_report: dict[str, int],
    ) -> "ModelArtifact":
        return cls(
            version=ARTIFACT_VERSION,
            created_at=datetime.now(UTC).isoformat(),
            behavior_builder=behavior_builder,
            similarity_model=similarity_model,
            topology_builder=topology_builder,
            graph_role_model=graph_role_model,
            hierarchy_model=hierarchy_model,
            behavior_report=behavior_report,
            graph_report=graph_report,
            hierarchy_report=hierarchy_report,
            selection_report=selection_report,
        )


def save_artifact(artifact: ModelArtifact, path: Path) -> str:
    """Атомарно сохраняет pickle и возвращает SHA-256 готового файла."""

    if artifact.version != ARTIFACT_VERSION:
        raise ValueError(f"Неизвестная версия артефакта: {artifact.version}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".building")
    with temporary.open("wb") as file:
        pickle.dump(artifact, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)
    return _sha256(path)


def load_artifact(path: Path) -> ModelArtifact:
    """Загружает артефакт и проверяет его тип и версию."""

    path = Path(path)
    with path.open("rb") as file:
        artifact = pickle.load(file)

    if not isinstance(artifact, ModelArtifact):
        raise TypeError("Файл не содержит ModelArtifact")
    if artifact.version != ARTIFACT_VERSION:
        raise ValueError(
            f"Версия {artifact.version!r} несовместима с {ARTIFACT_VERSION!r}"
        )
    return artifact


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()