"""HTTP API поиска похожих хостов."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request

from nad_similarity.config import get_settings
from nad_similarity.schemas import (
    HealthResponse,
    HostResponse,
    PairComparisonResponse,
    SimilarHostsResponse,
)
from nad_similarity.service import SimilarityService


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

    application = FastAPI(title=settings.app_name, lifespan=lifespan)

    def current_service(request: Request) -> SimilarityService:
        return request.app.state.similarity_service

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        loaded = current_service(request)
        return HealthResponse(model_loaded=True, model_version=loaded.artifact.version)

    # Статический маршрут объявлен раньше /hosts/{host_id}.
    @application.get("/hosts/compare", response_model=PairComparisonResponse)
    def compare_hosts(request: Request, left: str, right: str) -> PairComparisonResponse:
        try:
            return current_service(request).compare(left, right)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @application.get("/hosts/{host_id}", response_model=HostResponse)
    def host(request: Request, host_id: str) -> HostResponse:
        try:
            return current_service(request).host(host_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @application.get("/hosts/{host_id}/similar", response_model=SimilarHostsResponse)
    def similar_hosts(
        request: Request,
        host_id: str,
        limit: int = Query(default=settings.default_neighbors, ge=1, le=settings.max_neighbors),
        within_graph_role: bool = True,
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
