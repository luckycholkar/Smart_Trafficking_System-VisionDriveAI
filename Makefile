.PHONY: help install install-dev run pipeline test lint format typecheck \
        docker-build docker-run docker-stop weights clean

PYTHON ?= python3
HOST ?= 127.0.0.1
PORT ?= 5000

help:
	@echo "Common targets:"
	@echo "  install       - install runtime deps"
	@echo "  install-dev   - install runtime + dev deps"
	@echo "  weights       - download default YOLO weights"
	@echo "  run           - start uvicorn + pipeline locally"
	@echo "  pipeline      - run vision pipeline only (no API)"
	@echo "  test          - run pytest"
	@echo "  lint          - ruff check"
	@echo "  format        - ruff format"
	@echo "  typecheck     - mypy"
	@echo "  docker-build  - build the docker image"
	@echo "  docker-run    - docker compose up -d"
	@echo "  docker-stop   - docker compose down"
	@echo "  clean         - remove caches + generated artifacts"

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,notifications]"

weights:
	bash scripts/download_weights.sh

run:
	uvicorn visiondrive.api.app:app --host $(HOST) --port $(PORT)

pipeline:
	$(PYTHON) -m visiondrive pipeline

test:
	pytest

lint:
	ruff check src tests

format:
	ruff format src tests

typecheck:
	mypy

docker-build:
	docker compose build

docker-run:
	docker compose up -d

docker-stop:
	docker compose down

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
