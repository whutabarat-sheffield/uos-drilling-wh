# Makefile for abyss package

.PHONY: help clean build install dev-install test wheel check lint format

# Default target
help:
	@echo "Available targets:"
	@echo "  help         Show this help message"
	@echo "  clean        Clean build artifacts and caches"
	@echo "  build        Build wheel package"
	@echo "  install      Build and install wheel"
	@echo "  dev-install  Build and install in editable/development mode"
	@echo "  test         Run tests"
	@echo "  wheel        Build wheel (alias for build)"
	@echo "  check        Validate package without installing"
	@echo "  lint         Run linting (if available)"
	@echo "  format       Format code (if available)"
	@echo ""
	@echo "Examples:"
	@echo "  make build            # Build wheel"
	@echo "  make clean build      # Clean and build"
	@echo "  make dev-install      # Install in development mode"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	python3 build_wheel.py --clean

# Build wheel
build wheel:
	@echo "Building wheel..."
	python3 build_wheel.py

# Build and install wheel
install:
	@echo "Building and installing wheel..."
	python3 build_wheel.py --clean --install

# Build and install in editable mode
dev-install:
	@echo "Installing in development mode..."
	python3 build_wheel.py --clean --editable

# Run tests
test:
	@echo "Running tests..."
	@if command -v pytest >/dev/null 2>&1; then \
		python3 -m pytest tests/ -v; \
	else \
		echo "pytest not found, running with unittest..."; \
		python3 -m unittest discover -s tests -p "test_*.py" -v; \
	fi

# Validate package
check:
	@echo "Validating package..."
	python3 build_wheel.py --clean --skip-validation
	@if command -v twine >/dev/null 2>&1; then \
		echo "Running twine check..."; \
		twine check dist/*.whl; \
	else \
		echo "twine not available for additional validation"; \
	fi

# Lint code (if tools available)
lint:
	@echo "Running linting..."
	@if command -v flake8 >/dev/null 2>&1; then \
		echo "Running flake8..."; \
		flake8 src/abyss --max-line-length=120 --ignore=E501,W503; \
	fi
	@if command -v pylint >/dev/null 2>&1; then \
		echo "Running pylint..."; \
		pylint src/abyss --disable=C0103,R0903,R0913; \
	fi
	@if command -v mypy >/dev/null 2>&1; then \
		echo "Running mypy..."; \
		mypy src/abyss --ignore-missing-imports; \
	fi

# Format code (if tools available)
format:
	@echo "Formatting code..."
	@if command -v black >/dev/null 2>&1; then \
		echo "Running black..."; \
		black src/abyss tests/; \
	fi
	@if command -v isort >/dev/null 2>&1; then \
		echo "Running isort..."; \
		isort src/abyss tests/; \
	fi

# Combined clean and build
clean-build: clean build

# Combined clean, build, and install
clean-install: clean install

# Show package info
info:
	@echo "Package Information:"
	@echo "==================="
	@grep -E "^(name|version)" pyproject.toml || echo "Could not read pyproject.toml"
	@echo ""
	@echo "Python version:"
	@python3 --version
	@echo ""
	@echo "Build tools:"
	@python3 -c "import build; print(f'build: {build.__version__}')" 2>/dev/null || echo "build: not installed"
	@python3 -c "import wheel; print(f'wheel: {wheel.__version__}')" 2>/dev/null || echo "wheel: not installed"
	@python3 -c "import setuptools; print(f'setuptools: {setuptools.__version__}')" 2>/dev/null || echo "setuptools: not installed"