"""HTTP API поиска похожих хостов."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request

from nad_similarity.config import get_settings
from nad_similarity.schemas import (
    ErrorResponse,
    HealthResponse,
    HostPredictionRequest,
    HostResponse,
    PairComparisonResponse,
    PredictionResponse,
    SimilarHostsResponse,
)
from nad_similarity.service import KnownHostConflictError, SimilarityService

API_DESCRIPTION = """
Сервис решает две задачи:

1. для хоста из обучающей выборки возвращает его кластер, графовую роль и
   наиболее похожие известные узлы;
2. для нового хоста принимает историю сетевых потоков за 15 суток и назначает
   глобальный поведенческий кластер.

Число `similarity_score` — преобразованное расстояние, а не вероятность.
Для нового хоста графовая роль не вычисляется: для неё нужен полный граф сети.
"""

API_TAGS = [
    {
        "name": "Состояние",
        "description": "Проверка загрузки артефакта модели.",
    },
    {
        "name": "Известные хосты",
        "description": "Запросы для узлов, входивших в обучающую выборку.",
    },
    {
        "name": "Новый хост",
        "description": "Классификация ранее неизвестного узла по его flow-истории.",
    },
]


def create_app(
    artifact_path: Path | None = None,
    service: SimilarityService | None = None,
) -> FastAPI:
    """Создаёт приложение; готовый service используется в тестах."""

    settings = get_settings()
    model_path = artifact_path or settings.model_artifact_path

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.similarity_service = service or SimilarityService.from_path(model_path)
        yield

    application = FastAPI(
        title=settings.app_name,
        description=API_DESCRIPTION,
        version="0.1.0",
        openapi_tags=API_TAGS,
        lifespan=lifespan,
    )

    def current_service(request: Request) -> SimilarityService:
        return request.app.state.similarity_service

    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["Состояние"],
        summary="Проверить готовность сервиса",
        description="Возвращает версию загруженного модельного артефакта.",
    )
    def health(request: Request) -> HealthResponse:
        loaded = current_service(request)
        return HealthResponse(model_loaded=True, model_version=loaded.artifact.version)

    # Статический маршрут объявлен раньше /hosts/{host_id}.
    @application.get(
        "/hosts/compare",
        response_model=PairComparisonResponse,
        responses={404: {"model": ErrorResponse, "description": "Хост не найден"}},
        tags=["Известные хосты"],
        summary="Сравнить два известных хоста",
        description=(
            "Считает поведенческое и графовое расстояния отдельно и показывает, "
            "какие группы признаков внесли наибольший вклад."
        ),
    )
    def compare_hosts(request: Request, left: str, right: str) -> PairComparisonResponse:
        try:
            return current_service(request).compare(left, right)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @application.post(
        "/hosts/predict",
        response_model=PredictionResponse,
        responses={
            409: {
                "model": ErrorResponse,
                "description": "Переданный host_id уже присутствует в модели",
            }
        },
        tags=["Новый хост"],
        summary="Определить кластер нового хоста",
        description=(
            "Принимает полную историю сетевых потоков нового хоста за период "
            "обучения, строит 454 признака и применяет глобальную HDBSCAN-модель. "
            "Дополнительно возвращает ближайшие известные узлы и объяснение "
            "расстояния. Графовая роль и локальный подтип остаются null."
        ),
    )
    def predict_host(
        request: Request,
        payload: HostPredictionRequest,
    ) -> PredictionResponse:
        try:
            return current_service(request).predict(payload)
        except KnownHostConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @application.get(
        "/hosts/{host_id}",
        response_model=HostResponse,
        responses={404: {"model": ErrorResponse, "description": "Хост не найден"}},
        tags=["Известные хосты"],
        summary="Получить модельные метки хоста",
        description=(
            "Возвращает глобальный кластер поведения, графовую роль и локальный "
            "поведенческий подтип известного хоста."
        ),
    )
    def host(request: Request, host_id: str) -> HostResponse:
        try:
            return current_service(request).host(host_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @application.get(
        "/hosts/{host_id}/similar",
        response_model=SimilarHostsResponse,
        responses={404: {"model": ErrorResponse, "description": "Хост не найден"}},
        tags=["Известные хосты"],
        summary="Найти похожие известные хосты",
        description=(
            "По умолчанию ищет только среди узлов с той же графовой ролью, затем "
            "ранжирует кандидатов по поведенческому расстоянию. Для каждого соседа "
            "возвращает вклад времени, сервисов, размеров flows и направления."
        ),
    )
    def similar_hosts(
        request: Request,
        host_id: str,
        limit: int = Query(
            default=settings.default_neighbors,
            ge=1,
            le=settings.max_neighbors,
            description="Максимальное число соседей в ответе",
        ),
        within_graph_role: bool = Query(
            default=True,
            description=(
                "true — искать внутри графовой роли; false — искать глобально"
            ),
        ),
    ) -> SimilarHostsResponse:
        try:
            return current_service(request).similar_hosts(
                host_id,
                limit=limit,
                within_graph_role=within_graph_role,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    return application


app = create_app()
