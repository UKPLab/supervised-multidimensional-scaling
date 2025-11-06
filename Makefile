.PHONY: test format format-check lint

test:
	uv run pytest ./tests

format:
	uv run ruff format .
	uv run ruff check . --fix

format-check:
	uv run ruff format .
	uv run ruff check . --fix
	uv run mypy . --config=pyproject.toml