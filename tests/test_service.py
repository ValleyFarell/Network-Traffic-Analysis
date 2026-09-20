"""Проверки двухуровневого поиска без тяжёлого обучения."""

from types import SimpleNamespace

import pandas as pd

from nad_similarity.features import BehaviorFeatureSettings
from nad_similarity.schemas import HostPredictionRequest
from nad_similarity.service import SimilarityService


def make_service() -> SimilarityService:
    hosts = pd.Index(["A", "B", "C", "D"], name="host")
    embedding = pd.DataFrame(
        {
            "time": [1.0, 0.8, 0.99, 0.0],
            "services": [0.0, 0.2, 0.01, 1.0],
            "flow": [0.0, 0.0, 0.0, 0.0],
            "direction": [0.0, 0.0, 0.0, 0.0],
            "graph": [0.0, 0.0, 0.0, 0.0],
        },
        index=hosts,
    )
    similarity_model = SimpleNamespace(
        embedding_=embedding,
        family_columns_={
            "time": ["time"],
            "services": ["services"],
            "flow": ["flow"],
            "direction": ["direction"],
            "graph": ["graph"],
        },
        cluster_labels_=pd.Series([0, 0, 0, 1], index=hosts),
        cluster_strengths_=pd.Series([1.0, 0.9, 0.8, 0.7], index=hosts),
    )
    similarity_model.predict_cluster = lambda vector: (4, 0.75)
    similarity_model.nearest_vector = lambda vector, limit: pd.DataFrame(
        [
            {
                "host_id": "B",
                "distance": 0.2,
                "similarity_score": 0.818730753,
                "cluster_id": 0,
                "cluster_strength": 0.9,
            }
        ]
    ).head(limit)
    graph_model = SimpleNamespace(
        labels_=pd.Series([2, 2, -1, 2], index=hosts),
        probabilities_=pd.Series([1.0, 0.9, 0.0, 0.8], index=hosts),
        features_=pd.DataFrame(
            {"pagerank": [0.1, 0.2, 0.9, 0.3], "core": [1.0, 1.0, 3.0, 2.0]},
            index=hosts,
        ),
    )
    hierarchy_model = SimpleNamespace(
        labels_=pd.Series([0, 0, -1, 0], index=hosts),
        hierarchical_labels_=pd.Series(
            ["G2.B0", "G2.B0", "graph_noise", "G2.B0"],
            index=hosts,
        ),
        behavior_space_=embedding[["time", "services", "flow", "direction"]],
    )
    artifact = SimpleNamespace(
        version="test",
        behavior_builder=SimpleNamespace(
            settings=BehaviorFeatureSettings(),
            categories_={"port": ["80", "OTHER", "N_OTHER"]},
            transform=lambda tables, host_ids: pd.DataFrame(
                [[0.9, 0.1, 0.0, 0.0, 0.0]],
                index=host_ids,
                columns=embedding.columns,
            ),
        ),
        similarity_model=similarity_model,
        graph_role_model=graph_model,
        hierarchy_model=hierarchy_model,
    )
    return SimilarityService(artifact)


def test_role_is_a_filter_not_a_distance_weight() -> None:
    service = make_service()

    local = service.similar_hosts("A", limit=3, within_graph_role=True)
    global_result = service.similar_hosts("A", limit=3, within_graph_role=False)

    assert [row.host_id for row in local.neighbors] == ["B", "D"]
    assert global_result.neighbors[0].host_id == "C"
    assert all(row.graph_role == 2 for row in local.neighbors)
    assert all(row.hierarchical_group == "G2.B0" for row in local.neighbors)


def test_pair_contributions_sum_to_one() -> None:
    result = make_service().compare("A", "B")

    assert result.same_graph_role is True
    assert abs(sum(row.share for row in result.behavior_contributions) - 1.0) < 1e-6
    assert abs(sum(row.share for row in result.graph_contributions) - 1.0) < 1e-6


def test_host_exposes_hierarchical_membership() -> None:
    result = make_service().host("A")

    assert result.graph_role == 2
    assert result.behavior_subtype == 0
    assert result.hierarchical_group == "G2.B0"
    assert result.is_graph_noise is False
    assert result.is_subtype_noise is False


def test_predict_new_host_uses_global_model_without_graph_role() -> None:
    request = HostPredictionRequest.model_validate(
        {
            "host_id": "X",
            "neighbors": 1,
            "flows": [
                {
                    "time": 1,
                    "duration": 2,
                    "src": "X",
                    "src_port": "12345",
                    "dst": "B",
                    "dst_port": "80",
                    "protocol": "6",
                    "packets": 3,
                    "bytes": 100,
                }
            ],
        }
    )

    result = make_service().predict(request)

    assert result.cluster_id == 4
    assert result.cluster_strength == 0.75
    assert result.graph_role is None
    assert result.behavior_subtype is None
    assert result.hierarchical_group is None
    assert result.neighbors[0].host_id == "B"
    assert abs(sum(row.share for row in result.neighbors[0].contributions) - 1.0) < 1e-6
