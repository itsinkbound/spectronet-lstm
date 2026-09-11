.PHONY: install test lint format run docker-build docker-run clean

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest -q

lint:
	ruff check src tests

format:
	ruff format src tests

run:
	python -m spectronet.pipeline --config configs/config.yaml

docker-build:
	docker build -t spectronet-lstm:latest .

docker-run:
	docker compose up --build

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache artifacts/checkpoints
