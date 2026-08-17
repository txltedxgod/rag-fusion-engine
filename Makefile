PYTHON := python
PIP := pip
PYTEST := pytest
RUFF := ruff

.PHONY: help install test lint format run docker-build docker-up clean

help:
	@echo "Available commands:"
	@echo "  make install      Install production and dev dependencies"
	@echo "  make test         Run test suite with pytest"
	@echo "  make lint         Run ruff and mypy linters"
	@echo "  make format       Auto-format code with ruff"
	@echo "  make run          Start development server"
	@echo "  make docker-build Build Docker container image"
	@echo "  make docker-up    Run application via docker-compose"
	@echo "  make clean        Remove temporary files and caches"

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTEST) -v --durations=5 tests/

lint:
	$(RUFF) check src/ tests/
	$(RUFF) format --check src/ tests/

format:
	$(RUFF) format src/ tests/
	$(RUFF) check --fix src/ tests/

run:
	uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t rag-fusion-engine:latest .

docker-up:
	docker compose up -d --build

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf dist/ build/ *.egg-info
