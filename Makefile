.PHONY: all install test run lint format clean docker-build

all: lint test

install:
	python -m pip install -r requirements.txt ruff

test:
	pytest -v tests/

run:
	uvicorn src.main:app --reload

lint:
	ruff check .

format:
	ruff format .

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__

docker-build:
	docker build -t rag-fusion-engine:latest .
