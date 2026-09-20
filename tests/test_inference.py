"""Проверки преобразования входной flow-истории в обучающие агрегаты."""

from nad_similarity.features import BehaviorFeatureSettings
from nad_similarity.inference import build_new_host_tables
from nad_similarity.schemas import HostPredictionRequest


def test_build_new_host_tables_preserves_direction_and_port_semantics() -> None:
    request = HostPredictionRequest.model_validate(
        {
            "host_id": "X",
            "flows": [
                {
                    "time": 1,
                    "duration": 3,
                    "src": "X",
                    "src_port": "50000",
                    "dst": "A",
                    "dst_port": "80",
                    "protocol": "6",
                    "packets": 4,
                    "bytes": 100,
                },
                {
                    "time": 2,
                    "duration": 7,
                    "src": "B",
                    "src_port": "N123",
                    "dst": "X",
                    "dst_port": "443",
                    "protocol": "17",
                    "packets": 6,
                    "bytes": 200,
                },
            ],
        }
    )

    tables = build_new_host_tables(
        request,
        settings=BehaviorFeatureSettings(),
        port_categories=["80", "443", "OTHER", "N_OTHER"],
    )

    assert tables.host_hour.flows.sum() == 2
    assert tables.host_hour.bytes.sum() == 300
    assert tables.direction.iloc[0].out_flows == 1
    assert tables.direction.iloc[0].in_flows == 1
    assert set(tables.edge.peer) == {"A", "B"}
    assert set(tables.port.port) == {"80", "443", "OTHER", "N_OTHER"}
    assert tables.flow_hist.groupby("metric").observations.sum().to_dict() == {
        "bytes": 2,
        "duration": 2,
        "packets": 2,
    }
