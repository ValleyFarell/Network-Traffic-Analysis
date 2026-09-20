"""Построение агрегатов из LANL flows без загрузки исходного файла в RAM."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from nad_similarity.features import BehaviorTables

FLOW_SCHEMA = """
{
    'time': 'BIGINT', 'duration': 'BIGINT',
    'src': 'VARCHAR', 'src_port': 'VARCHAR',
    'dst': 'VARCHAR', 'dst_port': 'VARCHAR',
    'protocol': 'VARCHAR', 'packets': 'BIGINT', 'bytes': 'BIGINT'
}
"""


@dataclass(frozen=True)
class AggregationSettings:
    """Параметры, использованные для выборки v4 и агрегатов v14/v19."""

    build_start_day: int = 0
    build_end_day: int = 15
    selection_start_day: int = 1
    selection_end_day: int = 15
    selection_exclude_left: int = 14
    selection_exclude_right: int = 16
    min_network_hour_share: float = 1.0
    require_activity_span: bool = True
    min_flows: int = 100
    sample_fraction: float = 0.01
    sample_seed: int = 42
    min_sample_per_direction: int = 20
    min_partner_transitions: int = 2
    port_top_n: int = 64
    byte_bin_max: int = 30
    packet_bin_max: int = 20
    duration_bin_max: int = 20
    cache_version: int = 1


@dataclass
class AggregatedTrainingData:
    """Все таблицы, необходимые для повторения финальных экспериментов."""

    host_ids: pd.Index
    behavior: BehaviorTables
    graph_edges: pd.DataFrame
    selection_report: dict[str, int]


class TrainingDataBuilder:
    """Одним чтением gzip строит host universe и постоянный DuckDB-кэш."""

    def __init__(
        self,
        cache_path: Path,
        temp_dir: Path,
        settings: AggregationSettings | None = None,
        threads: int | None = None,
        memory_limit: str = "20GB",
    ) -> None:
        self.cache_path = Path(cache_path)
        self.temp_dir = Path(temp_dir)
        self.settings = settings or AggregationSettings()
        self.threads = threads or min(8, os.cpu_count() or 4)
        self.memory_limit = memory_limit

    def load_or_build(
        self,
        flows_path: Path,
        rebuild: bool = False,
    ) -> AggregatedTrainingData:
        """Использует совместимый кэш или атомарно строит его заново."""

        if rebuild or not self._cache_is_compatible():
            self.build(flows_path)
        else:
            print(f"Используем готовый кэш: {self.cache_path}")
        return self.load()

    def build(self, flows_path: Path) -> None:
        """Строит агрегаты v4, v14 и v19 в одном DuckDB-файле."""

        flows_path = Path(flows_path)
        if not flows_path.exists():
            raise FileNotFoundError(f"Не найден файл flows: {flows_path}")

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        building_path = self.cache_path.with_name(self.cache_path.stem + ".building.duckdb")
        building_path.unlink(missing_ok=True)
        Path(str(building_path) + ".wal").unlink(missing_ok=True)

        connection = duckdb.connect(str(building_path))
        try:
            self._configure(connection)
            print("1/4 Читаем flows за дни [0, 15) в DuckDB...")
            self._read_flows(connection, flows_path)
            self._validate_flows(connection)
            print("2/4 Воспроизводим отбор хостов из v4...")
            self._build_selection_tables(connection)

            host_ids, report = self._select_hosts(connection)
            if len(host_ids) != 4_350:
                raise ValueError(
                    "Выборка хостов не совпала с экспериментом: "
                    f"получено {len(host_ids):,}, ожидалось 4 350"
                )
            print(f"Отобрано хостов: {len(host_ids):,}")

            target_hosts = pd.DataFrame({"host": host_ids})
            connection.register("target_hosts_frame", target_hosts)
            connection.execute(
                "CREATE TABLE target_hosts AS SELECT host FROM target_hosts_frame ORDER BY host"
            )
            print("3/4 Строим поведенческие агрегаты v14...")
            self._build_behavior_tables(connection)
            print("4/4 Строим граф рёбер v19...")
            self._build_graph_edges(connection)

            metadata = {
                "settings": asdict(self.settings),
                "selected_hosts": len(host_ids),
                "selection_report": report,
            }
            connection.execute("CREATE TABLE cache_meta(metadata_json VARCHAR)")
            connection.execute(
                "INSERT INTO cache_meta VALUES (?)",
                [json.dumps(metadata, ensure_ascii=False, sort_keys=True)],
            )
            connection.execute("CHECKPOINT")
        finally:
            connection.close()

        building_path.replace(self.cache_path)
        print(f"Кэш сохранён: {self.cache_path}")

    def load(self) -> AggregatedTrainingData:
        """Загружает компактные таблицы из готового кэша."""

        if not self._cache_is_compatible():
            raise ValueError("Совместимый кэш агрегатов не найден")

        connection = duckdb.connect(str(self.cache_path), read_only=True)
        try:
            metadata = json.loads(
                connection.execute("SELECT metadata_json FROM cache_meta").fetchone()[0]
            )
            host_ids = pd.Index(
                connection.execute("SELECT host FROM target_hosts ORDER BY host").df().host,
                name="host",
            )
            behavior = BehaviorTables(
                host_hour=connection.execute("SELECT * FROM agg_host_hour").df(),
                direction=connection.execute("SELECT * FROM agg_direction").df(),
                protocol=connection.execute("SELECT * FROM agg_protocol").df(),
                port=connection.execute("SELECT * FROM agg_port").df(),
                flow_hist=connection.execute("SELECT * FROM agg_flow_hist").df(),
                edge=connection.execute("SELECT * FROM agg_edge").df(),
            )
            graph_edges = connection.execute("SELECT * FROM graph_edges").df()
        finally:
            connection.close()

        return AggregatedTrainingData(
            host_ids=host_ids,
            behavior=behavior,
            graph_edges=graph_edges,
            selection_report=metadata["selection_report"],
        )

    def _configure(self, connection: duckdb.DuckDBPyConnection) -> None:
        connection.execute(f"PRAGMA threads={self.threads}")
        connection.execute(f"PRAGMA memory_limit='{self.memory_limit}'")
        connection.execute(f"PRAGMA temp_directory='{self._sql_path(self.temp_dir)}'")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute("SET enable_progress_bar=false")

    def _read_flows(
        self,
        connection: duckdb.DuckDBPyConnection,
        flows_path: Path,
    ) -> None:
        settings = self.settings
        start_time = settings.build_start_day * 86_400 + 1
        end_time = settings.build_end_day * 86_400 + 1
        connection.execute(
            f"""
            CREATE TEMP TABLE flows AS
            SELECT
                *,
                ((time - 1) // 86400)::INTEGER AS day,
                ((time - 1) // 3600)::BIGINT AS absolute_hour,
                (((time - 1) // 3600) % 24)::INTEGER AS hour_of_day
            FROM read_csv(
                '{self._sql_path(flows_path)}',
                header=false,
                auto_detect=false,
                columns={FLOW_SCHEMA},
                nullstr='?',
                compression='gzip',
                parallel=true
            )
            WHERE time >= {start_time} AND time < {end_time}
            """
        )

    @staticmethod
    def _validate_flows(connection: duckdb.DuckDBPyConnection) -> None:
        invalid = connection.execute(
            """
            SELECT count(*) FROM flows
            WHERE time IS NULL OR duration IS NULL OR packets IS NULL OR bytes IS NULL
               OR src IS NULL OR dst IS NULL OR protocol IS NULL
               OR time < 1 OR duration < 0 OR packets < 0 OR bytes < 0
            """
        ).fetchone()[0]
        invalid_ports = connection.execute(
            """
            SELECT count(*) FROM flows
            WHERE protocol IN ('6', '17') AND (src_port IS NULL OR dst_port IS NULL)
            """
        ).fetchone()[0]
        if invalid:
            raise ValueError(f"Найдено {invalid:,} строк с невалидными значениями")
        if invalid_ports:
            raise ValueError(f"Найдено {invalid_ports:,} TCP/UDP-строк без портов")

    def _build_selection_tables(self, connection: duckdb.DuckDBPyConnection) -> None:
        """Строит только таблицы, участвовавшие в выборе 4 350 хостов."""

        connection.execute(
            """
            CREATE TABLE selection_window AS
            SELECT host, day, direction, count(*)::BIGINT AS flows
            FROM (
                SELECT src AS host, day, 'out' AS direction FROM flows
                UNION ALL
                SELECT dst AS host, day, 'in' AS direction FROM flows
            )
            GROUP BY ALL
            """
        )
        connection.execute(
            """
            CREATE TABLE selection_peer AS
            SELECT host, day, direction, peer
            FROM (
                SELECT src AS host, day, 'out' AS direction, dst AS peer FROM flows
                UNION ALL
                SELECT dst AS host, day, 'in' AS direction, src AS peer FROM flows
            )
            GROUP BY ALL
            """
        )
        connection.execute(
            """
            CREATE TABLE selection_bounds AS
            SELECT host, min(time) AS first_time, max(time) AS last_time
            FROM (
                SELECT src AS host, time FROM flows
                UNION ALL
                SELECT dst AS host, time FROM flows
            )
            GROUP BY host
            """
        )
        connection.execute(
            """
            CREATE TABLE selection_network AS
            SELECT absolute_hour AS hour, count(*)::BIGINT AS flows
            FROM flows
            GROUP BY absolute_hour
            """
        )

        threshold = int(round(self.settings.sample_fraction * 1_000_000))
        seed = self.settings.sample_seed
        connection.execute(
            f"""
            CREATE TABLE selection_sample AS
            SELECT src, dst, day
            FROM flows
            WHERE hash(
                time, duration, src, src_port, dst, dst_port,
                protocol, packets, bytes, {seed}
            ) % 1000000 < {threshold}
            """
        )

    def _select_hosts(
        self,
        connection: duckdb.DuckDBPyConnection,
    ) -> tuple[pd.Index, dict[str, int]]:
        """Повторяет два фильтра host universe из v4."""

        settings = self.settings
        network = (
            connection.execute("SELECT * FROM selection_network")
            .df()
            .set_index("hour")
            .flows
        )
        bins = []
        for day in range(settings.selection_start_day, settings.selection_end_day):
            excluded = (
                day < settings.selection_exclude_right
                and day + 1 > settings.selection_exclude_left
            )
            hours = np.arange(day * 24, (day + 1) * 24)
            coverage = network.reindex(hours, fill_value=0).gt(0).mean()
            complete = (
                day >= settings.build_start_day
                and day + 1 <= settings.build_end_day
            )
            if not excluded and complete and coverage >= settings.min_network_hour_share:
                bins.append(day)
        if bins != list(range(1, 14)):
            raise ValueError(f"Неожиданная сетка выбора хостов: {bins}")

        window = connection.execute(
            f"SELECT * FROM selection_window WHERE day IN ({','.join(map(str, bins))})"
        ).df()
        bounds = connection.execute("SELECT * FROM selection_bounds").df().set_index("host")

        counts = window.groupby("host").flows.sum()
        report = counts.rename("flows").to_frame()
        host_bounds = bounds.reindex(report.index)
        first_hour = (host_bounds.first_time - 1) // 3_600
        last_hour_end = (host_bounds.last_time - 1) // 3_600 + 1
        report["enough_flows"] = report.flows.ge(settings.min_flows)
        report["covers_grid"] = first_hour.le(bins[0] * 24) & last_hour_end.ge(
            (bins[-1] + 1) * 24
        )
        span_ok = report.covers_grid if settings.require_activity_span else True
        first_stage = report.index[report.enough_flows & span_ok].sort_values()

        peer = connection.execute(
            f"SELECT * FROM selection_peer WHERE day IN ({','.join(map(str, bins))})"
        ).df()
        peer = peer.loc[peer.host.isin(first_stage)]
        partner_defined = pd.Series(True, index=first_stage)
        allowed = set(bins)
        for (host, _direction), group in peer.groupby(["host", "direction"], sort=False):
            sets = {day: set(part.peer) for day, part in group.groupby("day")}
            transitions = 0
            for day in bins:
                if day - 1 not in allowed:
                    continue
                if sets.get(day - 1, set()) | sets.get(day, set()):
                    transitions += 1
            if transitions < settings.min_partner_transitions:
                partner_defined.loc[host] = False

        sample = connection.execute(
            f"SELECT src, dst FROM selection_sample "
            f"WHERE day IN ({','.join(map(str, bins))})"
        ).df()
        active = window.groupby(["host", "direction"]).flows.sum()
        sample_out = sample.groupby("src").size()
        sample_in = sample.groupby("dst").size()
        flow_defined = pd.Series(True, index=first_stage)
        for host in first_stage:
            out_active = active.get((host, "out"), 0) > 0
            in_active = active.get((host, "in"), 0) > 0
            out_enough = (
                not out_active
                or sample_out.get(host, 0) >= settings.min_sample_per_direction
            )
            in_enough = (
                not in_active
                or sample_in.get(host, 0) >= settings.min_sample_per_direction
            )
            flow_defined.loc[host] = out_enough and in_enough

        selected = first_stage[partner_defined & flow_defined].sort_values()
        selection_report = {
            "hosts_in_selection_period": int(len(report)),
            "after_flow_and_span_filter": int(len(first_stage)),
            "undefined_partner_statistics": int((~partner_defined).sum()),
            "undefined_flow_statistics": int((~flow_defined).sum()),
            "selected_hosts": int(len(selected)),
        }
        return pd.Index(selected.astype(str), name="host"), selection_report

    def _build_behavior_tables(self, connection: duckdb.DuckDBPyConnection) -> None:
        """Повторяет восемь компактных агрегатов distribution cache v14."""

        connection.execute(
            """
            CREATE TABLE agg_host_hour AS
            WITH events AS (
                SELECT src AS host, day, absolute_hour, hour_of_day, bytes, packets
                FROM flows WHERE src IN (SELECT host FROM target_hosts)
                UNION ALL
                SELECT dst AS host, day, absolute_hour, hour_of_day, bytes, packets
                FROM flows WHERE dst IN (SELECT host FROM target_hosts)
            )
            SELECT host, day, absolute_hour, hour_of_day,
                   count(*)::BIGINT AS flows,
                   CAST(sum(bytes) AS BIGINT) AS bytes,
                   CAST(sum(packets) AS BIGINT) AS packets
            FROM events GROUP BY ALL
            """
        )
        connection.execute(
            """
            CREATE TABLE agg_direction AS
            WITH counts AS (
                SELECT src AS host, count(*)::BIGINT AS out_flows, 0::BIGINT AS in_flows
                FROM flows WHERE src IN (SELECT host FROM target_hosts) GROUP BY src
                UNION ALL
                SELECT dst AS host, 0::BIGINT, count(*)::BIGINT
                FROM flows WHERE dst IN (SELECT host FROM target_hosts) GROUP BY dst
            )
            SELECT host, CAST(sum(out_flows) AS BIGINT) AS out_flows,
                         CAST(sum(in_flows) AS BIGINT) AS in_flows
            FROM counts GROUP BY host
            """
        )
        connection.execute(
            """
            CREATE TABLE agg_protocol AS
            WITH events AS (
                SELECT src AS host, protocol, bytes
                FROM flows WHERE src IN (SELECT host FROM target_hosts)
                UNION ALL
                SELECT dst AS host, protocol, bytes
                FROM flows WHERE dst IN (SELECT host FROM target_hosts)
            )
            SELECT host, protocol, count(*)::BIGINT AS flows,
                   CAST(sum(bytes) AS BIGINT) AS bytes
            FROM events GROUP BY ALL
            """
        )
        connection.execute(
            """
            CREATE TEMP TABLE port_events AS
            SELECT src AS host, 'local' AS side, src_port AS port, bytes
            FROM flows WHERE src IN (SELECT host FROM target_hosts) AND protocol IN ('6', '17')
            UNION ALL
            SELECT src, 'remote', dst_port, bytes
            FROM flows WHERE src IN (SELECT host FROM target_hosts) AND protocol IN ('6', '17')
            UNION ALL
            SELECT dst, 'local', dst_port, bytes
            FROM flows WHERE dst IN (SELECT host FROM target_hosts) AND protocol IN ('6', '17')
            UNION ALL
            SELECT dst, 'remote', src_port, bytes
            FROM flows WHERE dst IN (SELECT host FROM target_hosts) AND protocol IN ('6', '17')
            """
        )
        connection.execute(
            f"""
            CREATE TEMP TABLE port_vocab AS
            SELECT port FROM port_events WHERE port IS NOT NULL
            GROUP BY port ORDER BY count(*) DESC, port
            LIMIT {self.settings.port_top_n}
            """
        )
        connection.execute(
            """
            CREATE TABLE agg_port AS
            SELECT host, side,
                   CASE WHEN port IN (SELECT port FROM port_vocab) THEN port
                        WHEN starts_with(port, 'N') THEN 'N_OTHER'
                        ELSE 'OTHER' END AS port,
                   count(*)::BIGINT AS flows,
                   CAST(sum(bytes) AS BIGINT) AS bytes
            FROM port_events GROUP BY ALL
            """
        )

        connection.execute(
            f"""
            CREATE TABLE agg_flow_hist AS
            WITH events AS (
                SELECT src AS host, bytes, packets, duration
                FROM flows WHERE src IN (SELECT host FROM target_hosts)
                UNION ALL
                SELECT dst AS host, bytes, packets, duration
                FROM flows WHERE dst IN (SELECT host FROM target_hosts)
            ), long_form AS (
                SELECT host, 'bytes' AS metric,
                       least(
                           {self.settings.byte_bin_max},
                           floor(ln(bytes + 1) / ln(2))
                       )::INTEGER AS bin
                FROM events
                UNION ALL
                SELECT host, 'packets',
                       least(
                           {self.settings.packet_bin_max},
                           floor(ln(packets + 1) / ln(2))
                       )::INTEGER
                FROM events
                UNION ALL
                SELECT host, 'duration',
                       least(
                           {self.settings.duration_bin_max},
                           floor(ln(duration + 1) / ln(2))
                       )::INTEGER
                FROM events
            )
            SELECT host, metric, bin, count(*)::BIGINT AS observations
            FROM long_form GROUP BY ALL
            """
        )
        connection.execute(
            """
            CREATE TABLE agg_edge AS
            WITH events AS (
                SELECT src AS host, dst AS peer, bytes, packets
                FROM flows WHERE src IN (SELECT host FROM target_hosts)
                UNION ALL
                SELECT dst AS host, src AS peer, bytes, packets
                FROM flows WHERE dst IN (SELECT host FROM target_hosts)
            )
            SELECT host, peer, count(*)::BIGINT AS flows,
                   CAST(sum(bytes) AS BIGINT) AS bytes,
                   CAST(sum(packets) AS BIGINT) AS packets
            FROM events GROUP BY ALL
            """
        )

    @staticmethod
    def _build_graph_edges(connection: duckdb.DuckDBPyConnection) -> None:
        """Повторяет edge cache v19 на полном периоде ``[0, 15)``."""

        connection.execute(
            """
            CREATE TABLE graph_edges AS
            SELECT src, dst, count(*)::BIGINT AS flows,
                   CAST(sum(bytes) AS BIGINT) AS bytes,
                   CAST(sum(packets) AS BIGINT) AS packets,
                   count(DISTINCT day)::INTEGER AS active_days
            FROM flows
            WHERE src <> dst
            GROUP BY src, dst
            """
        )

    def _cache_is_compatible(self) -> bool:
        if not self.cache_path.exists():
            return False
        try:
            connection = duckdb.connect(str(self.cache_path), read_only=True)
            row = connection.execute("SELECT metadata_json FROM cache_meta").fetchone()
            connection.close()
            if row is None:
                return False
            metadata = json.loads(row[0])
            return metadata.get("settings") == asdict(self.settings)
        except Exception:
            return False

    @staticmethod
    def _sql_path(path: Path) -> str:
        return str(Path(path).resolve()).replace("'", "''")
