.PHONY: all install test test-cov coverage format format-check lint clean help

PM := uv
RUN := $(PM) run

install: ## Install/sync all dependencies (including dev)
	$(PM) sync --all-extras

test: ## Run tests using pytest
	$(RUN) pytest ./tests

coverage: ## Run tests with coverage and generate HTML report
	$(RUN) pytest ./tests --cov=. --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

format: ## Format code and apply auto-fixes using Ruff
	$(RUN) ruff format .
	$(RUN) ruff check . --fix

check:
	$(RUN) ruff check .
	$(RUN) mypy . --config=pyproject.toml


format-check: ## Check code for formatting, linting, and type errors
	$(RUN) ruff check .
	$(RUN) ruff format . --check
	$(RUN) mypy . --config=pyproject.toml

lint: ## Alias for format-check (often used interchangeably)
	$(MAKE) format-check

clean: ## Remove coverage reports and cache files
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'


