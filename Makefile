.PHONY: install dev test lint type docs clean

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .

dev: install
	pre-commit install

test:
	pytest -q --maxfail=1 --disable-warnings --cov=src

lint:
	black --check src tests
	isort --check-only src tests
	flake8 src tests

type:
	mypy src

docs:
	cd docs && mkdocs build

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -rf build dist *.egg-info .pytest_cache htmlcov
