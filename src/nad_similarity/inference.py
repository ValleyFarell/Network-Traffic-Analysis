"""Агрегация переданной истории flows для применения обученного feature builder."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import pandas as pd

from nad_similarity.features import BehaviorFeatureSettings, BehaviorTables
from nad_similarity.schemas import HostPredictionRequest


def build_new_host_tables(
    request: HostPredictionRequest,
    settings: BehaviorFeatureSettings,
    port_categories: Collection[object],
) -> BehaviorTables:
    """Строит те же агрегаты v14 для одного хоста из его полной истории flows."""

    host_id = request.host_id
    events: list[dict[str, object]] = []
    port_events: list[dict[str, object]] = []

    known_ports = {str(value) for value in port_categories}
    ordinary_ports = known_ports - {"OTHER", "N_OTHER"}

    for flow in request.flows:
        base = {
            "time": flow.time,
            "day": (flow.time - 1) // 86_400,
            "absolute_hour": (flow.time - 1) // 3_600,
            "hour_of_day": ((flow.time - 1) // 3_600) % 24,
            "duration": flow.duration,
            "protocol": flow.protocol,
            "packets": flow.packets,
            "bytes": flow.bytes,
        }
        if flow.src == host_id:
            events.append({**base, "direction": "out", "peer": flow.dst})
            if flow.protocol in {"6", "17"}:
                port_events.extend(
                    [
                        {
                            "host": host_id,
                            "side": "local",
                            "port": _port_bucket(flow.src_port, ordinary_ports),
                            "bytes": flow.bytes,
                        },
                        {
                            "host": host_id,
                            "side": "remote",
                            "port": _port_bucket(flow.dst_port, ordinary_ports),
                            "bytes": flow.bytes,
                        },
                    ]
                )
        if flow.dst == host_id:
            events.append({**base, "direction": "in", "peer": flow.src})
            if flow.protocol in {"6", "17"}:
                port_events.extend(
                    [
                        {
                            "host": host_id,
                            "side": "local",
                            "port": _port_bucket(flow.dst_port, ordinary_ports),
                            "bytes": flow.bytes,
                        },
                        {
                            "host": host_id,
                            "side": "remote",
                            "port": _port_bucket(flow.src_port, ordinary_ports),
                            "bytes": flow.bytes,
                        },
                    ]
                )

    frame = pd.DataFrame(events)
    if frame.empty:
        raise ValueError("История нового хоста не содержит событий")
    frame.insert(0, "host", host_id)

    host_hour = (
        frame.groupby(
            ["host", "day", "absolute_hour", "hour_of_day"],
            as_index=False,
        )
        .agg(flows=("time", "size"), bytes=("bytes", "sum"), packets=("packets", "sum"))
    )

    direction_counts = frame.direction.value_counts()
    direction = pd.DataFrame(
        {
            "host": [host_id],
            "out_flows": [int(direction_counts.get("out", 0))],
            "in_flows": [int(direction_counts.get("in", 0))],
        }
    )

    protocol = (
        frame.groupby(["host", "protocol"], as_index=False)
        .agg(flows=("time", "size"), bytes=("bytes", "sum"))
    )

    if port_events:
        port = (
            pd.DataFrame(port_events)
            .groupby(["host", "side", "port"], as_index=False)
            .agg(flows=("port", "size"), bytes=("bytes", "sum"))
        )
    else:
        port = pd.DataFrame(
            columns=["host", "side", "port", "flows", "bytes"]
        )

    histogram_rows = []
    for metric, max_bin in {
        "bytes": settings.byte_bin_max,
        "packets": settings.packet_bin_max,
        "duration": settings.duration_bin_max,
    }.items():
        bins = np.floor(np.log2(frame[metric].to_numpy(dtype=float) + 1)).astype(int)
        bins = np.clip(bins, 0, max_bin)
        counts = pd.Series(bins).value_counts().sort_index()
        histogram_rows.extend(
            {
                "host": host_id,
                "metric": metric,
                "bin": int(bin_number),
                "observations": int(count),
            }
            for bin_number, count in counts.items()
        )
    flow_hist = pd.DataFrame(
        histogram_rows,
        columns=["host", "metric", "bin", "observations"],
    )

    edge = (
        frame.groupby(["host", "peer"], as_index=False)
        .agg(flows=("time", "size"), bytes=("bytes", "sum"), packets=("packets", "sum"))
    )

    return BehaviorTables(
        host_hour=host_hour,
        direction=direction,
        protocol=protocol,
        port=port,
        flow_hist=flow_hist,
        edge=edge,
    )


def _port_bucket(port: str, ordinary_ports: set[str]) -> str:
    port = str(port)
    if port in ordinary_ports:
        return port
    return "N_OTHER" if port.startswith("N") else "OTHER"
