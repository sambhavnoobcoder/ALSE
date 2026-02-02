# ALSE Makefile
# Common development tasks

.PHONY: help install test lint format clean figures

help:
	@echo "ALSE Development Commands:"
	@echo "  make install    - Install package in development mode"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linting"
	@echo "  make format     - Format code with Black"
	@echo "  make clean      - Clean temporary files"
	@echo "  make figures    - Generate publication figures"

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	flake8 alse/ tests/
	mypy alse/

format:
	black alse/ tests/ examples/ scripts/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/

figures:
	python scripts/generate_figures.py
