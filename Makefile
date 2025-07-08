# Makefile for UOS Drilling Depth Estimation System
#
# This Makefile provides convenient targets for building, testing, and managing
# the wheel distribution of the abyss package.
#
# Usage:
#   make build          # Build wheel with clean and validation
#   make build-simple   # Build wheel without clean or validation
#   make clean          # Clean build artifacts
#   make validate       # Build and validate wheel
#   make install        # Install the built wheel locally
#   make test           # Run tests (if available)
#   make help           # Show this help message

.PHONY: build build-simple clean validate install test help
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
BUILD_SCRIPT := build_wheel.py
DIST_DIR := abyss/dist
ABYSS_DIR := abyss

# Default build target - clean build with validation
build: ## Build wheel with clean and validation
	@echo "Building wheel with clean and validation..."
	$(PYTHON) $(BUILD_SCRIPT) --clean --validate --verbose

# Simple build without clean or validation
build-simple: ## Build wheel without clean or validation
	@echo "Building wheel (simple)..."
	$(PYTHON) $(BUILD_SCRIPT) --verbose

# Clean build artifacts
clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	$(PYTHON) $(BUILD_SCRIPT) --clean --verbose
	@echo "Build artifacts cleaned."

# Build and validate wheel
validate: ## Build and validate wheel
	@echo "Building and validating wheel..."
	$(PYTHON) $(BUILD_SCRIPT) --validate --verbose

# Install the built wheel locally
install: build ## Install the built wheel locally
	@echo "Installing built wheel..."
	@if [ -f "$(DIST_DIR)/abyss-"*.whl ]; then \
		pip install --force-reinstall "$(DIST_DIR)/abyss-"*.whl; \
		echo "Installation completed."; \
	else \
		echo "No wheel file found. Run 'make build' first."; \
		exit 1; \
	fi

# Run tests (placeholder for future test implementation)
test: ## Run tests
	@echo "Running tests..."
	@if [ -d "$(ABYSS_DIR)/tests" ]; then \
		cd $(ABYSS_DIR) && $(PYTHON) -m pytest tests/ -v; \
	else \
		echo "No tests directory found in $(ABYSS_DIR)/"; \
		echo "Skipping tests..."; \
	fi

# Development targets
dev-install: ## Install package in development mode
	@echo "Installing package in development mode..."
	cd $(ABYSS_DIR) && pip install -e .

# Lint and format code (if tools are available)
lint: ## Lint the code
	@echo "Linting code..."
	@if command -v flake8 >/dev/null 2>&1; then \
		cd $(ABYSS_DIR) && flake8 src/; \
	else \
		echo "flake8 not found, skipping linting"; \
	fi

format: ## Format the code
	@echo "Formatting code..."
	@if command -v black >/dev/null 2>&1; then \
		cd $(ABYSS_DIR) && black src/; \
	else \
		echo "black not found, skipping formatting"; \
	fi

# Documentation
docs: ## Generate documentation (placeholder)
	@echo "Documentation generation not yet implemented"

# Release targets
release-check: build ## Check if ready for release
	@echo "Checking release readiness..."
	@echo "✓ Wheel builds successfully"
	@echo "✓ Validation passes"
	@echo "Release checks completed."

# Utility targets
list-wheels: ## List built wheel files
	@echo "Built wheel files:"
	@ls -la $(DIST_DIR)/*.whl 2>/dev/null || echo "No wheel files found"

wheel-info: ## Show information about built wheels
	@echo "Wheel information:"
	@for wheel in $(DIST_DIR)/*.whl; do \
		if [ -f "$$wheel" ]; then \
			echo "File: $$wheel"; \
			echo "Size: $$(du -h "$$wheel" | cut -f1)"; \
			echo "Modified: $$(stat -c %y "$$wheel" | cut -d' ' -f1)"; \
			echo "---"; \
		fi; \
	done

# Help target
help: ## Show this help message
	@echo "UOS Drilling Depth Estimation System - Build Targets"
	@echo "=================================================="
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Examples:"
	@echo "  make build          # Build wheel with clean and validation"
	@echo "  make install        # Build and install wheel locally"
	@echo "  make clean          # Clean build artifacts"
	@echo "  make wheel-info     # Show information about built wheels"