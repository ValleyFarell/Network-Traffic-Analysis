"""Pydantic-модели входных данных и ответов API."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


HostId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
]
PortValue = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=64),
]
ProtocolValue = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=32),
]


class ApiModel(BaseModel):
    """Строгая базовая модель: неожиданные поля считаются ошибкой клиента."""

    model_config = ConfigDict(extra="forbid")


class FlowInput(ApiModel):
    """Одна запись сетевого потока из истории нового хоста."""

    time: int = Field(ge=1, description="Относительное время в секундах")
    duration: float = Field(ge=0)
    src: HostId
    src_port: PortValue
    dst: HostId
    dst_port: PortValue
    protocol: ProtocolValue
    packets: int = Field(ge=0)
    bytes: int = Field(ge=0)


class HostPredictionRequest(ApiModel):
    """История хоста, которого может не быть в обучающей выборке."""

    host_id: HostId
    flows: list[FlowInput] = Field(min_length=1)
    neighbors: int = Field(default=10, ge=1, le=100)

    @model_validator(mode="after")
    def host_must_participate_in_every_flow(self) -> "HostPredictionRequest":
        unrelated_rows = [
            index
            for index, flow in enumerate(self.flows)
            if self.host_id not in {flow.src, flow.dst}
        ]
        if unrelated_rows:
            preview = unrelated_rows[:5]
            raise ValueError(
                "Каждый flow должен содержать host_id в src или dst. "
                f"Индексы некорректных строк: {preview}"
            )
        return self


class FeatureContribution(ApiModel):
    """Вклад одного семейства признаков в итоговое расстояние."""

    family: str
    distance: float = Field(ge=0)
    share: float = Field(ge=0, le=1)


class NeighborResponse(ApiModel):
    """Один известный хост из результатов поиска похожих узлов."""

    host_id: HostId
    distance: float = Field(ge=0)
    similarity_score: float = Field(ge=0, le=1)
    cluster_id: int
    cluster_strength: float = Field(ge=0, le=1)
    graph_role: int | None = None
    graph_role_strength: float = Field(ge=0, le=1)
    behavior_subtype: int
    hierarchical_group: str
    contributions: list[FeatureContribution] = Field(default_factory=list)


class HostResponse(ApiModel):
    """Информация модели об известном хосте."""

    host_id: HostId
    cluster_id: int
    cluster_strength: float = Field(ge=0, le=1)
    is_noise: bool
    graph_role: int | None = None
    graph_role_strength: float = Field(ge=0, le=1)
    is_graph_noise: bool
    behavior_subtype: int
    is_subtype_noise: bool
    hierarchical_group: str
    model_version: str


class SimilarHostsResponse(ApiModel):
    """Ближайшие известные хосты для узла из обучающей выборки."""

    query: HostResponse
    search_scope: Literal["within_graph_role", "global_behavior"]
    neighbors: list[NeighborResponse]


class PairComparisonResponse(ApiModel):
    """Сравнение двух известных хостов в обоих пространствах."""

    left: HostResponse
    right: HostResponse
    same_graph_role: bool
    same_hierarchical_group: bool
    behavior_distance: float = Field(ge=0)
    graph_distance: float = Field(ge=0)
    behavior_contributions: list[FeatureContribution]
    graph_contributions: list[FeatureContribution]


class PredictionResponse(ApiModel):
    """Кластер и ближайшие хосты для переданной сетевой истории."""

    host_id: HostId
    cluster_id: int
    cluster_strength: float = Field(ge=0, le=1)
    is_noise: bool
    model_version: str
    neighbors: list[NeighborResponse]


class ClusterResponse(ApiModel):
    """Краткое описание одного поведенческого кластера."""

    cluster_id: int
    size: int = Field(ge=0)
    medoid_host_id: HostId | None = None
    sample_hosts: list[HostId] = Field(default_factory=list)
    profile: dict[str, float] = Field(default_factory=dict)
    model_version: str


class HealthResponse(ApiModel):
    """Информация о готовности сервиса."""

    status: str = "ok"
    model_loaded: bool
    model_version: str | None = None


class ErrorResponse(ApiModel):
    """Единый формат ошибки в документированных ответах API."""

    detail: str
