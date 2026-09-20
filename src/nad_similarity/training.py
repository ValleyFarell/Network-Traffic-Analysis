"""CLI обучения финальной модели на агрегатах LANL flows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from nad_similarity.aggregation import TrainingDataBuilder
from nad_similarity.artifacts import ModelArtifact, save_artifact
from nad_similarity.features import BehaviorFeatureBuilder
from nad_similarity.graph import GraphRoleModel, StableTopologyBuilder
from nad_similarity.hierarchy import HierarchicalSubtypeModel
from nad_similarity.model import HostSimilarityModel


EXPECTED_BEHAVIOR = {
    "hosts": 4_350,
    "dimensions": 454,
    "clusters": 5,
    "noise_hosts": 1_019,
    "cluster_sizes": [1_271, 965, 537, 317, 241],
    "silhouette": 0.243980,
    "dbcv": 0.042611,
}

EXPECTED_GRAPH = {
    "hosts": 4_350,
    "dimensions": 7,
    "clusters": 8,
    "noise_hosts": 945,
    "cluster_sizes": [1_576, 457, 430, 281, 182, 163, 161, 155],
    "silhouette": 0.370176,
    "dbcv": 0.447839,
}

EXPECTED_HIERARCHY = {
    0: {"hosts": 182, "status": "unsplit", "noise_hosts": 0, "subtype_sizes": [182]},
    1: {"hosts": 163, "status": "unsplit", "noise_hosts": 0, "subtype_sizes": [163]},
    2: {"hosts": 161, "status": "unsplit", "noise_hosts": 0, "subtype_sizes": [161]},
    3: {"hosts": 155, "status": "unsplit", "noise_hosts": 0, "subtype_sizes": [155]},
    4: {"hosts": 430, "status": "unsplit", "noise_hosts": 0, "subtype_sizes": [430]},
    5: {
        "hosts": 1_576,
        "status": "split",
        "noise_hosts": 220,
        "subtype_sizes": [1_150, 206],
        "silhouette": 0.382714,
        "relative_validity": 0.006282,
    },
    6: {
        "hosts": 457,
        "status": "split",
        "noise_hosts": 149,
        "subtype_sizes": [176, 77, 55],
        "silhouette": 0.198033,
        "relative_validity": 0.135332,
    },
    7: {
        "hosts": 281,
        "status": "split",
        "noise_hosts": 147,
        "subtype_sizes": [71, 63],
        "silhouette": 0.311874,
        "relative_validity": 0.005890,
    },
}


def train(
    flows_path: Path,
    cache_path: Path,
    temp_dir: Path,
    artifact_path: Path,
    rebuild_cache: bool = False,
) -> ModelArtifact:
    """Строит обе модели, проверяет контрольные результаты и сохраняет артефакт."""

    data_builder = TrainingDataBuilder(cache_path=cache_path, temp_dir=temp_dir)
    data = data_builder.load_or_build(flows_path, rebuild=rebuild_cache)

    print("Строим поведенческий embedding и обучаем основную HDBSCAN-модель...")
    behavior_builder = BehaviorFeatureBuilder()
    embedding = behavior_builder.fit_transform(data.behavior, data.host_ids)
    similarity_model = HostSimilarityModel().fit(
        embedding,
        behavior_builder.family_columns_,
    )
    behavior_report = clustering_report(
        embedding,
        similarity_model.cluster_labels_,
        similarity_model.clusterer.relative_validity_,
    )
    verify_report("behavior", behavior_report, EXPECTED_BEHAVIOR)

    print("Строим stable-граф и обучаем модель графовых ролей...")
    topology_builder = StableTopologyBuilder(stable_min_days=3)
    raw_topology, scaled_topology = topology_builder.fit_transform(
        data.graph_edges,
        data.host_ids,
    )
    graph_role_model = GraphRoleModel().fit(scaled_topology)
    graph_report = clustering_report(
        scaled_topology,
        graph_role_model.labels_,
        graph_role_model.clusterer.relative_validity_,
    )
    verify_report("graph", graph_report, EXPECTED_GRAPH)
    print("Структура кластеров совпала с экспериментами v14/v19.")

    # Исходные медианные профили нужны для содержательного описания ролей.
    graph_report["role_profiles"] = (
        graph_role_model.role_profiles(raw_topology).to_dict(orient="index")
    )

    print("Строим поведенческие подтипы внутри графовых ролей по v20...")
    behavior_space = behavior_builder.v20_behavior_space_
    if behavior_space is None:
        raise RuntimeError("Не сохранено поведенческое пространство второго уровня v20")
    hierarchy_model = HierarchicalSubtypeModel().fit(
        behavior_space,
        graph_role_model.labels_,
    )
    verify_hierarchy_report(hierarchy_model.report_, EXPECTED_HIERARCHY)
    print("Поведенческие подтипы совпали с контрольным экспериментом v20.")

    artifact = ModelArtifact.create(
        behavior_builder=behavior_builder,
        similarity_model=similarity_model,
        topology_builder=topology_builder,
        graph_role_model=graph_role_model,
        hierarchy_model=hierarchy_model,
        behavior_report=behavior_report,
        graph_report=graph_report,
        hierarchy_report=hierarchy_model.report_,
        selection_report=data.selection_report,
    )
    checksum = save_artifact(artifact, artifact_path)

    summary_path = artifact_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "version": artifact.version,
                "created_at": artifact.created_at,
                "sha256": checksum,
                "selection": artifact.selection_report,
                "behavior": artifact.behavior_report,
                "graph": artifact.graph_report,
                "hierarchy": artifact.hierarchy_report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return artifact


def clustering_report(
    values: pd.DataFrame,
    labels: pd.Series,
    dbcv: float,
) -> dict[str, Any]:
    """Считает метрики тем же способом, что evaluate_space в v19."""

    label_values = labels.to_numpy()
    clustered_positions = np.flatnonzero(label_values != -1)
    sizes = (
        pd.Series(label_values[clustered_positions])
        .value_counts()
        .sort_values(ascending=False)
    )

    sampled = clustered_positions
    if len(sampled) > 1_500:
        rng = np.random.default_rng(42)
        sampled = rng.choice(sampled, 1_500, replace=False)

    silhouette = float(
        silhouette_score(
            values.to_numpy(dtype=float)[sampled],
            label_values[sampled],
            metric="euclidean",
        )
    )
    return {
        "hosts": int(len(values)),
        "dimensions": int(values.shape[1]),
        "clusters": int(len(sizes)),
        "noise_hosts": int(np.sum(label_values == -1)),
        "coverage": float(np.mean(label_values != -1)),
        "noise_share": float(np.mean(label_values == -1)),
        "largest_cluster_share": float(sizes.iloc[0] / sizes.sum()),
        "cluster_sizes": [int(value) for value in sizes.tolist()],
        "silhouette": silhouette,
        "dbcv": float(dbcv),
    }


def verify_report(
    name: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    """Проверяет структуру кластеров и сопоставляет диагностические метрики."""

    exact_keys = ["hosts", "dimensions", "clusters", "noise_hosts", "cluster_sizes"]
    differences = {
        key: {"actual": actual[key], "expected": expected[key]}
        for key in exact_keys
        if actual[key] != expected[key]
    }

    # В v19 silhouette считался на случайной подвыборке, а состояние RNG
    # зависело от предыдущих строк sweep, поэтому для неё допустим малый разброс.
    if not np.isclose(actual["silhouette"], expected["silhouette"], atol=0.01, rtol=0):
        differences["silhouette"] = {
            "actual": actual["silhouette"],
            "expected": expected["silhouette"],
        }

    # relative_validity_ вычисляется на полном MST модели и должен
    # воспроизводить контрольный запуск. Большое расхождение здесь скрывать нельзя.
    if not np.isclose(actual["dbcv"], expected["dbcv"], atol=0.002, rtol=0):
        differences["dbcv"] = {
            "actual": actual["dbcv"],
            "expected": expected["dbcv"],
        }

    if differences:
        raise RuntimeError(
            f"Результат {name} не совпал с контрольным экспериментом: "
            f"{json.dumps(differences, ensure_ascii=False)}"
        )


def verify_hierarchy_report(
    actual: dict[int, dict[str, Any]],
    expected: dict[int, dict[str, Any]],
) -> None:
    """Строго сверяет второй уровень с сохранённым результатом v20."""

    differences: dict[str, Any] = {}
    if set(actual) != set(expected):
        differences["graph_roles"] = {
            "actual": sorted(actual),
            "expected": sorted(expected),
        }

    exact_keys = ["hosts", "status", "noise_hosts", "subtype_sizes"]
    for role in sorted(set(actual) & set(expected)):
        role_differences = {
            key: {"actual": actual[role][key], "expected": expected[role][key]}
            for key in exact_keys
            if actual[role][key] != expected[role][key]
        }
        for metric, tolerance in [("silhouette", 0.01), ("relative_validity", 0.002)]:
            if metric not in expected[role]:
                continue
            if not np.isclose(
                actual[role][metric],
                expected[role][metric],
                atol=tolerance,
                rtol=0,
            ):
                role_differences[metric] = {
                    "actual": actual[role][metric],
                    "expected": expected[role][metric],
                }
        if role_differences:
            differences[str(role)] = role_differences

    if differences:
        raise RuntimeError(
            "Поведенческие подтипы не совпали с v20: "
            f"{json.dumps(differences, ensure_ascii=False)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение NAD Host Similarity")
    parser.add_argument("--flows", type=Path, default=Path("data/raw/flows.txt.gz"))
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/processed/training_tables.duckdb"),
    )
    parser.add_argument("--temp-dir", type=Path, default=Path("data/processed/duckdb_tmp"))
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/similarity_model.pkl"),
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        flows_path=args.flows,
        cache_path=args.cache,
        temp_dir=args.temp_dir,
        artifact_path=args.artifact,
        rebuild_cache=args.rebuild_cache,
    )
    print(f"Модель сохранена: {args.artifact}")


if __name__ == "__main__":
    main()
