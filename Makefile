LINT_PATHS=capof/ examples/ bbo/ tests/


install:
	python -m pip install -e ".[dev]"

pytest:
	python3 -m pytest tests/ -v --color=yes

pytest_cov:
	python3 -m pytest tests/ --cov-report html --cov-report term --cov=. -v --color=yes

mypy:
	python -m mypy ${LINT_PATHS}

type: mypy

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://www.flake8rules.com/
	ruff check ${LINT_PATHS} --select=E9,F63,F7,F82
	# exit-zero treats all errors as warnings.
	ruff check ${LINT_PATHS} --exit-zero

format:
	# Sort imports
	isort ${LINT_PATHS}
	# Reformat using black
	black ${LINT_PATHS}

check-codestyle:
	# Sort imports
	isort --check ${LINT_PATHS}
	# Reformat using black
	black --check ${LINT_PATHS}

commit-checks: format mypy lint

.PHONY: clean spelling doc gh-pages lint format check-codestyle commit-checks install pytest pytest_cov mypy type
