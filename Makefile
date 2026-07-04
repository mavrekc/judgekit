.PHONY: install lint format format-check typecheck test check docker-build

install:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy src

test:
	uv run pytest -q

check: lint format-check typecheck test

docker-build:
	docker build -t judgekit:dev .
