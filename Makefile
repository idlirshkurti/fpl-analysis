# Makefile for FPL Transfer Recommender project

# Python command
PYTHON := python

# Directories
SRC_DIR := fpl_transfer_recommender
TEST_DIR := tests

# Tools
RUFF := ruff
MYPY := mypy
PYTEST := pytest

# Tool options
RUFF_OPTS ?= 
MYPY_OPTS ?= --ignore-missing-imports
PYTEST_OPTS ?= -xvs

# Target: help - Display available targets
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  help  - Display this help"
	@echo "  check - Run all checks (lint, type, test)"
	@echo "  lint  - Run ruff linting"
	@echo "  type  - Run mypy type checking"
	@echo "  test  - Run pytest tests"
	@echo "  clean - Remove Python cache files and build artifacts"

# Target: check - Run all checks
.PHONY: check
check: lint type test

# Target: lint - Run ruff linting
.PHONY: lint
lint:
	$(RUFF) check $(RUFF_OPTS) $(SRC_DIR) $(TEST_DIR)

# Target: type - Run mypy type checking
.PHONY: type
type:
	$(MYPY) $(MYPY_OPTS) $(SRC_DIR)

# Target: test - Run pytest tests
.PHONY: test
test:
	$(PYTEST) $(PYTEST_OPTS) $(TEST_DIR)

# Target: clean - Remove Python cache files and build artifacts
.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

