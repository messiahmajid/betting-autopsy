.PHONY: install dev test lint ingest serve notebook clean

install:
	uv sync

dev:
	uv sync --all-extras

test:
	uv run pytest -v

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

# Data ingestion
ingest-polymarket:
	uv run python -m src.ingest.polymarket

ingest-metaculus:
	uv run python -m src.ingest.metaculus

ingest-manifold:
	uv run python -m src.ingest.manifold

ingest-all: ingest-polymarket ingest-metaculus ingest-manifold

# Database
init-db:
	uv run python -m src.models.database --init

# Server
serve:
	uv run uvicorn src.api.main:app --reload --port 8000

# Notebooks
notebook:
	uv run marimo edit notebooks/

# Clean
clean:
	rm -rf data/*.duckdb
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
