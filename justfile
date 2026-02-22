# Cleans the repo.
clean:
	@find . | grep -E "(__pycache__|\.pyc|\.pyo|build|generated$)" | xargs rm -rf
	@rm -rf src/*.egg-info/ build/ dist/ .coverage .pytest_cache/

# Applies formatting to all files.
format:
	isort --profile black .
	black .
	blacken-docs

# Lints all files.
lint:
	ruff check

# Runs all non-MPI tests and determines coverage.
test-cov workers="4":
	pytest -n {{workers}} --cov=src/dalia --cov-report=term --cov-report=xml tests/

# Runs all tests.
test: test-cov
