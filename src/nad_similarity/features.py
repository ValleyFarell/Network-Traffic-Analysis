"""Поведенческое представление хостов из экспериментов v14/v19."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


V20_BEHAVIOR_FAMILIES = ("time", "services", "flow", "direction")
V20_BEHAVIOR_DIMENSIONS = 372


@dataclass(frozen=True)
class BehaviorFeatureSettings:
    """Параметры, зафиксированные в выполненных экспериментах."""

    start_day: int = 0
    end_day: int = 15
    day_block_days: int = 2
    byte_bin_max: int = 30
    packet_bin_max: int = 20
    duration_bin_max: int = 20
    intensity_bin_max: int = 30
    edge_bin_max: int = 30
    degree_bin_max: int = 16
    shrinkage: float = 1.0


@dataclass
class BehaviorTables:
    """Компактные агрегаты, которые строились в DuckDB-кэше v14."""

    host_hour: pd.DataFrame
    direction: pd.DataFrame
    protocol: pd.DataFrame
    port: pd.DataFrame
    flow_hist: pd.DataFrame
    edge: pd.DataFrame


def feature_family(block_name: str) -> str:
    """Возвращает семейство блока по тем же правилам, что в v15/v19."""

    if block_name.startswith(("hour_of_day_", "day_block_", "hourly_intensity_")):
        return "time"
    if block_name in {"protocol_flows", "port_local", "port_remote"}:
        return "services"
    if block_name in {"flow_bytes", "flow_packets", "flow_duration"}:
        return "flow"
    if block_name.startswith("graph_"):
        return "graph"
    if block_name == "directionality":
        return "direction"
    raise ValueError(f"Неизвестное семейство блока: {block_name}")


class BehaviorFeatureBuilder:
    """Строит family-balanced и L2-нормированный embedding.

    Последовательность полностью повторяет выбранное ручное пространство v19:

    1. распределения v14 переводятся в ``sqrt(p)``;
    2. циклическое время заменяется модулем Fourier-спектра;
    3. каждый блок делится на корень из числа блоков своего семейства;
    4. итоговая строка нормируется до единичной L2-нормы.
    """

    def __init__(self, settings: BehaviorFeatureSettings | None = None) -> None:
        self.settings = settings or BehaviorFeatureSettings()
        self.priors_: dict[str, pd.Series] = {}
        self.categories_: dict[str, list[object]] = {}
        self.block_columns_: dict[str, list[str]] = {}
        self.family_columns_: dict[str, list[str]] = {}
        self.neighbor_degrees_: pd.Series | None = None
        self.v20_behavior_space_: pd.DataFrame | None = None
        self.fitted_: bool = False

    def fit_transform(
        self,
        tables: BehaviorTables,
        host_ids: pd.Index,
    ) -> pd.DataFrame:
        """Обучает словари и priors на train-периоде и возвращает embedding."""

        host_ids = self._host_index(host_ids)
        blocks = self._build_blocks(tables, host_ids, fit=True)
        embedding = self._join_and_balance(blocks)
        self.v20_behavior_space_ = self._build_v20_behavior_space(blocks)
        self.fitted_ = True
        return embedding

    def transform(
        self,
        tables: BehaviorTables,
        host_ids: pd.Index,
    ) -> pd.DataFrame:
        """Применяет обученные priors и наборы категорий к новым агрегатам."""

        if not self.fitted_:
            raise RuntimeError("Сначала вызови fit_transform")

        host_ids = self._host_index(host_ids)
        blocks = self._build_blocks(tables, host_ids, fit=False)
        return self._join_and_balance(blocks)

    @staticmethod
    def _host_index(host_ids: pd.Index) -> pd.Index:
        return pd.Index(host_ids.astype(str), name="host")

    @staticmethod
    def _pivot_counts(
        frame: pd.DataFrame,
        host_ids: pd.Index,
        category: str,
        value: str,
        categories: list[object],
    ) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(0.0, index=host_ids, columns=categories)

        result = frame.pivot_table(
            index="host",
            columns=category,
            values=value,
            aggfunc="sum",
            fill_value=0,
        )
        result.index = result.index.astype(str)
        return result.reindex(index=host_ids, columns=categories, fill_value=0).astype(float)

    def _remember_categories(
        self,
        name: str,
        values: pd.Series,
        fit: bool,
    ) -> list[object]:
        if fit:
            self.categories_[name] = sorted(values.dropna().unique().tolist(), key=str)
        return self.categories_[name]

    def _hellinger(
        self,
        counts: pd.DataFrame,
        name: str,
        fit: bool,
    ) -> pd.DataFrame:
        counts = counts.astype(float).clip(lower=0)
        mass = counts.sum(axis=1)

        if fit:
            global_mass = counts.sum(axis=0)
            if global_mass.sum() > 0:
                prior = global_mass / global_mass.sum()
            else:
                prior = pd.Series(1 / max(1, counts.shape[1]), index=counts.columns)
            self.priors_[name] = prior

        prior = self.priors_[name].reindex(counts.columns, fill_value=0)
        probabilities = counts.add(self.settings.shrinkage * prior, axis=1).div(
            mass + self.settings.shrinkage,
            axis=0,
        )
        probabilities["NO_MASS"] = 0.0

        empty = mass.eq(0)
        probabilities.loc[empty, :] = 0.0
        probabilities.loc[empty, "NO_MASS"] = 1.0

        embedded = np.sqrt(probabilities)
        embedded.columns = [f"{name}__{column}" for column in embedded.columns]
        return embedded

    def _cyclic_spectrum(
        self,
        counts: pd.DataFrame,
        name: str,
        fit: bool,
    ) -> pd.DataFrame:
        counts = counts.astype(float).clip(lower=0)
        mass = counts.sum(axis=1)

        if fit:
            global_mass = counts.sum(axis=0)
            self.priors_[name] = global_mass / global_mass.sum()

        prior = self.priors_[name].reindex(counts.columns, fill_value=0)
        probabilities = counts.add(self.settings.shrinkage * prior, axis=1).div(
            mass + self.settings.shrinkage,
            axis=0,
        )

        spectrum = np.abs(np.fft.rfft(probabilities.to_numpy(dtype=float), axis=1))
        empty = mass.eq(0).to_numpy()
        spectrum[empty, :] = 0.0
        spectrum = np.concatenate([spectrum, empty.astype(float)[:, None]], axis=1)
        norms = np.linalg.norm(spectrum, axis=1, keepdims=True)
        spectrum = spectrum / np.where(norms > 0, norms, 1.0)

        columns = [f"{name}__freq_{number}" for number in range(spectrum.shape[1] - 1)]
        columns.append(f"{name}__NO_MASS")
        return pd.DataFrame(spectrum, index=counts.index, columns=columns)

    def _log_histogram(
        self,
        frame: pd.DataFrame,
        host_ids: pd.Index,
        value_column: str,
        max_bin: int,
    ) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(0.0, index=host_ids, columns=range(max_bin + 1))

        values = frame[value_column].to_numpy(dtype=float)
        bins = np.floor(np.log2(values + 1)).astype(int)
        bins = np.clip(bins, 0, max_bin)
        events = pd.DataFrame(
            {
                "host": frame.host.astype(str).to_numpy(),
                "bin": bins,
                "observations": 1,
            }
        )
        return self._pivot_counts(
            events,
            host_ids,
            "bin",
            "observations",
            list(range(max_bin + 1)),
        )

    def _build_blocks(
        self,
        tables: BehaviorTables,
        host_ids: pd.Index,
        fit: bool,
    ) -> dict[str, pd.DataFrame]:
        settings = self.settings
        blocks: dict[str, pd.DataFrame] = {}

        host_hour = tables.host_hour.copy()
        if not host_hour.empty:
            host_hour["host"] = host_hour.host.astype(str)

        for metric in ["flows", "bytes", "packets"]:
            name = f"hour_of_day_{metric}"
            counts = self._pivot_counts(
                host_hour, host_ids, "hour_of_day", metric, list(range(24))
            )
            blocks[name] = self._cyclic_spectrum(counts, name, fit)

        host_hour["day_block"] = (
            (host_hour.day.astype(int) - settings.start_day) // settings.day_block_days
        )
        day_blocks = list(
            range(int(np.ceil((settings.end_day - settings.start_day) / settings.day_block_days)))
        )
        for metric in ["flows", "bytes", "packets"]:
            name = f"day_block_{metric}"
            counts = self._pivot_counts(host_hour, host_ids, "day_block", metric, day_blocks)
            blocks[name] = self._cyclic_spectrum(counts, name, fit)

        absolute_hours = pd.Index(
            range(settings.start_day * 24, settings.end_day * 24),
            name="absolute_hour",
        )
        full_grid = pd.MultiIndex.from_product(
            [host_ids, absolute_hours], names=["host", "absolute_hour"]
        )
        hourly_values = (
            host_hour.groupby(["host", "absolute_hour"])[["flows", "bytes", "packets"]]
            .sum()
            .reindex(full_grid, fill_value=0)
            .reset_index()
        )
        for metric in ["flows", "bytes", "packets"]:
            values = hourly_values[metric].to_numpy(dtype=float)
            bins = np.floor(np.log2(values + 1)).astype(int)
            bins = np.clip(bins, 0, settings.intensity_bin_max)
            events = pd.DataFrame(
                {
                    "host": hourly_values.host.to_numpy(),
                    "bin": bins,
                    "observations": 1,
                }
            )
            name = f"hourly_intensity_{metric}"
            counts = self._pivot_counts(
                events,
                host_ids,
                "bin",
                "observations",
                list(range(settings.intensity_bin_max + 1)),
            )
            blocks[name] = self._hellinger(counts, name, fit)

        direction = tables.direction.copy()
        if not direction.empty:
            direction["host"] = direction.host.astype(str)
        direction_profile = (
            direction.set_index("host")[["out_flows", "in_flows"]]
            .reindex(host_ids, fill_value=0)
            .astype(float)
        )
        direction_total = direction_profile.out_flows + direction_profile.in_flows
        flow_balance = (
            (direction_profile.out_flows - direction_profile.in_flows)
            .div(direction_total.replace(0, np.nan))
            .fillna(0.0)
        )
        blocks["directionality"] = pd.DataFrame(
            {"directionality__flow_balance": flow_balance / np.sqrt(2)},
            index=host_ids,
        )

        protocol = tables.protocol.copy()
        if not protocol.empty:
            protocol["host"] = protocol.host.astype(str)
        protocol_categories = self._remember_categories(
            "protocol_flows", protocol.protocol, fit
        )
        protocol_counts = self._pivot_counts(
            protocol, host_ids, "protocol", "flows", protocol_categories
        )
        blocks["protocol_flows"] = self._hellinger(
            protocol_counts, "protocol_flows", fit
        )

        port = tables.port.copy()
        if not port.empty:
            port["host"] = port.host.astype(str)
            port["port"] = port.port.astype(str)
        port_categories = self._remember_categories("port", port.port, fit)
        for side in ["local", "remote"]:
            name = f"port_{side}"
            selected = port.loc[port.side.eq(side)]
            counts = self._pivot_counts(selected, host_ids, "port", "flows", port_categories)
            blocks[name] = self._hellinger(counts, name, fit)

        metric_bins = {
            "bytes": settings.byte_bin_max,
            "packets": settings.packet_bin_max,
            "duration": settings.duration_bin_max,
        }
        flow_hist = tables.flow_hist.copy()
        if not flow_hist.empty:
            flow_hist["host"] = flow_hist.host.astype(str)
        for metric, max_bin in metric_bins.items():
            name = f"flow_{metric}"
            selected = flow_hist.loc[flow_hist.metric.eq(metric)]
            counts = self._pivot_counts(
                selected,
                host_ids,
                "bin",
                "observations",
                list(range(max_bin + 1)),
            )
            blocks[name] = self._hellinger(counts, name, fit)

        edge = tables.edge.copy()
        if not edge.empty:
            edge["host"] = edge.host.astype(str)
            edge["peer"] = edge.peer.astype(str)

        edge_flows = self._log_histogram(edge, host_ids, "flows", settings.edge_bin_max)
        edge_bytes = self._log_histogram(edge, host_ids, "bytes", settings.edge_bin_max)
        blocks["graph_edge_flows"] = self._hellinger(
            edge_flows, "graph_edge_flows", fit
        )
        blocks["graph_edge_bytes"] = self._hellinger(
            edge_bytes, "graph_edge_bytes", fit
        )

        if fit:
            self.neighbor_degrees_ = edge.groupby("host").peer.nunique().astype(float)
        if self.neighbor_degrees_ is None:
            raise RuntimeError("Не сохранены степени соседей обучающего графа")

        neighbor_degree_events = edge[["host", "peer"]].copy()
        neighbor_degree_events["neighbor_degree"] = (
            neighbor_degree_events.peer.map(self.neighbor_degrees_).fillna(0)
        )
        neighbor_degree_counts = self._log_histogram(
            neighbor_degree_events,
            host_ids,
            "neighbor_degree",
            settings.degree_bin_max,
        )
        blocks["graph_neighbor_degree"] = self._hellinger(
            neighbor_degree_counts, "graph_neighbor_degree", fit
        )

        if fit:
            self.block_columns_ = {name: list(block.columns) for name, block in blocks.items()}
        else:
            for name, columns in self.block_columns_.items():
                blocks[name] = blocks[name].reindex(columns=columns, fill_value=0)

        return blocks

    def _join_and_balance(self, blocks: dict[str, pd.DataFrame]) -> pd.DataFrame:
        family_blocks: dict[str, list[str]] = {}
        for block_name in blocks:
            family_blocks.setdefault(feature_family(block_name), []).append(block_name)

        parts = []
        family_columns: dict[str, list[str]] = {}
        for family, names in family_blocks.items():
            columns = [column for name in names for column in blocks[name].columns]
            part = pd.concat([blocks[name] for name in names], axis=1) / np.sqrt(len(names))
            parts.append(part)
            family_columns[family] = columns

        frame = pd.concat(parts, axis=1).astype(np.float32)
        values = normalize(frame.to_numpy(dtype=float)).astype(np.float32)
        result = pd.DataFrame(values, index=frame.index, columns=frame.columns)

        if not np.isfinite(result.to_numpy()).all():
            raise ValueError("В поведенческом embedding появились NaN или inf")

        self.family_columns_ = family_columns
        return result

    @staticmethod
    def _build_v20_behavior_space(
        blocks: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Строит второй уровень ровно из исходных блоков, как в v20."""

        family_blocks = {family: [] for family in V20_BEHAVIOR_FAMILIES}
        for block_name in blocks:
            family = feature_family(block_name)
            if family in family_blocks:
                family_blocks[family].append(block_name)

        parts = []
        for family in V20_BEHAVIOR_FAMILIES:
            names = family_blocks[family]
            if not names:
                raise ValueError(f"Для второго уровня отсутствует семейство {family!r}")
            part = pd.concat([blocks[name] for name in names], axis=1) / np.sqrt(
                len(names)
            )
            parts.append(part)

        frame = pd.concat(parts, axis=1).astype(np.float32)
        if frame.shape[1] != V20_BEHAVIOR_DIMENSIONS:
            raise ValueError(
                "Размерность пространства второго уровня не совпала с v20: "
                f"{frame.shape[1]} вместо {V20_BEHAVIOR_DIMENSIONS}"
            )
        values = normalize(frame.to_numpy(dtype=float)).astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Поведенческое пространство v20 содержит NaN или inf")
        return pd.DataFrame(values, index=frame.index, columns=frame.columns)
