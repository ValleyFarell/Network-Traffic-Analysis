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

    time: int = Field(
        ge=1,
        le=15 * 86_400,
        description="Относительное время в секундах внутри периода [0, 15)",
    )
    duration: float = Field(ge=0, description="Длительность потока в секундах")
    src: HostId = Field(description="Идентификатор компьютера-источника")
    src_port: PortValue = Field(description="Порт источника как строка из LANL")
    dst: HostId = Field(description="Идентификатор компьютера-получателя")
    dst_port: PortValue = Field(description="Порт назначения как строка из LANL")
    protocol: ProtocolValue = Field(
        description="Код протокола как строка, например 6 для TCP или 17 для UDP"
    )
    packets: int = Field(ge=0, description="Количество пакетов")
    bytes: int = Field(ge=0, description="Количество переданных байт")


class HostPredictionRequest(ApiModel):
    """История хоста, которого может не быть в обучающей выборке."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "host_id": "NEW_HOST",
                    "neighbors": 5,
                    "flows": [
                        {
                            "time": 3601,
                            "duration": 2.4,
                            "src": "NEW_HOST",
                            "src_port": "49152",
                            "dst": "C1000",
                            "dst_port": "443",
                            "protocol": "6",
                            "packets": 12,
                            "bytes": 8400,
                        }
                    ],
                }
            ]
        },
    )

    host_id: HostId = Field(description="ID нового хоста, отсутствующий в модели")
    flows: list[FlowInput] = Field(
        min_length=1,
        description="Полная доступная история flows этого хоста за 15 суток",
    )
    neighbors: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Сколько ближайших известных хостов вернуть",
    )

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

    family: str = Field(description="Группа признаков")
    distance: float = Field(ge=0, description="Расстояние внутри этой группы")
    share: float = Field(
        ge=0,
        le=1,
        description="Доля группы в квадрате полного расстояния",
    )


class NeighborResponse(ApiModel):
    """Один известный хост из результатов поиска похожих узлов."""

    host_id: HostId
    distance: float = Field(ge=0, description="Евклидово расстояние; меньше — ближе")
    similarity_score: float = Field(
        ge=0,
        le=1,
        description="exp(-distance); удобный балл близости, но не вероятность",
    )
    cluster_id: int = Field(description="Глобальный поведенческий кластер")
    cluster_strength: float = Field(
        ge=0,
        le=1,
        description="Сила принадлежности глобальному кластеру по HDBSCAN",
    )
    graph_role: int | None = Field(description="Кластер по положению узла в графе")
    graph_role_strength: float = Field(
        ge=0,
        le=1,
        description="Сила принадлежности графовой роли",
    )
    behavior_subtype: int = Field(description="Подтип поведения внутри графовой роли")
    hierarchical_group: str = Field(
        description="Составная метка вида G6.B1 или G6.noise"
    )
    contributions: list[FeatureContribution] = Field(default_factory=list)


class HostResponse(ApiModel):
    """Информация модели об известном хосте."""

    host_id: HostId
    cluster_id: int = Field(description="Глобальный поведенческий кластер; -1 — шум")
    cluster_strength: float = Field(ge=0, le=1)
    is_noise: bool = Field(description="Не найден плотный глобальный кластер")
    graph_role: int | None = Field(description="Кластер по положению узла в графе")
    graph_role_strength: float = Field(ge=0, le=1)
    is_graph_noise: bool = Field(description="Не найдена плотная графовая роль")
    behavior_subtype: int = Field(description="Подтип внутри графовой роли")
    is_subtype_noise: bool = Field(description="Не найден плотный локальный подтип")
    hierarchical_group: str = Field(description="Итоговая составная метка")
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
    """Глобальный кластер и соседи для переданной истории нового хоста."""

    host_id: HostId
    cluster_id: int
    cluster_strength: float = Field(ge=0, le=1)
    is_noise: bool
    graph_role: None = Field(
        default=None,
        description="Не вычисляется без полного графа сети",
    )
    behavior_subtype: None = Field(
        default=None,
        description="Не вычисляется без графовой роли",
    )
    hierarchical_group: None = Field(
        default=None,
        description="Не вычисляется без графовой роли и подтипа",
    )
    observations: int = Field(ge=1)
    search_scope: Literal["global_embedding"] = "global_embedding"
    limitations: list[str]
    model_version: str
    neighbors: list[NeighborResponse]


class HealthResponse(ApiModel):
    """Информация о готовности сервиса."""

    status: str = "ok"
    model_loaded: bool
    model_version: str | None = None


class ErrorResponse(ApiModel):
    """Единый формат ошибки в документированных ответах API."""

    detail: str
