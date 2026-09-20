# NAD Host Similarity

Сервис поиска похожих компьютеров по истории сетевых взаимодействий LANL flows.

Основной поиск повторяет проверенную двухуровневую схему v20:

1. stable-topology определяет структурную роль хоста;
2. кандидатами становятся хосты той же роли;
3. кандидаты ранжируются по поведенческому расстоянию между семействами
   `time`, `services`, `flow` и `direction`.

Графовое и поведенческое расстояния не складываются в один произвольный score.

## Локальная установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

На Windows активация окружения выполняется командой:

```powershell
.venv\Scripts\activate
```

## Обучение

```powershell
python -m nad_similarity.training --flows data/raw/flows.txt.gz
```

При повторном запуске используется `data/processed/training_tables.duckdb`.
Флаг `--rebuild-cache` нужен только при изменении исходных данных или агрегации.

## API

После успешного обучения:

```powershell
uvicorn nad_similarity.api:app --reload
```

Основные запросы:

```text
GET /health
GET /hosts/C1707
GET /hosts/C1707/similar?limit=10
GET /hosts/C1707/similar?limit=10&within_graph_role=false
GET /hosts/compare?left=C1707&right=C12598
```

Первый вариант поиска ограничивает кандидатов графовой ролью и является основным.
Глобальный вариант оставлен как диагностический контроль.

## Тесты

```powershell
pytest
```
