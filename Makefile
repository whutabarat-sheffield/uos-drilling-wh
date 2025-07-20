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
clean: ## Clean build artifacts (preserves caches)
	@echo "Cleaning build artifacts..."
	$(PYTHON) $(BUILD_SCRIPT) --clean --verbose
	@echo "Build artifacts cleaned."

# Clean everything including caches
clean-all: ## Clean all artifacts including pip and Docker caches
	@echo "Cleaning all build artifacts and caches..."
	$(PYTHON) $(BUILD_SCRIPT) --clean --verbose
	@echo "Cleaning Docker build cache..."
	@docker builder prune -af 2>/dev/null || true
	@echo "Cleaning pip cache..."
	@pip cache purge 2>/dev/null || true
	@echo "All artifacts and caches cleaned."

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

# Run tests
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

# Docker build targets
# Enable BuildKit for better caching
export DOCKER_BUILDKIT=1

# Check if Docker is installed
check-docker:
	@command -v docker >/dev/null 2>&1 || { echo "Error: Docker is not installed. Please install Docker first."; exit 1; }
	@docker info >/dev/null 2>&1 || { echo "Error: Docker daemon is not running. Please start Docker."; exit 1; }

docker-all: check-docker ## Build all Docker images (with caching)
	@echo "Building all Docker images with caching enabled..."
	@./build-all.sh

docker-all-fresh: check-docker ## Build all Docker images (no cache, forces fresh downloads)
	@echo "Building all Docker images without cache (fresh build)..."
	@./build-all.sh --no-cache

docker-main: check-docker ## Build main Docker image (CPU with PyTorch)
	@echo "Building main Docker image with caching..."
	@./build-main.sh

docker-main-fresh: check-docker ## Build main Docker image (no cache)
	@echo "Building main Docker image without cache..."
	@./build-main.sh --no-cache

docker-cpu: check-docker ## Build CPU-optimized Docker image
	@echo "Building CPU-optimized Docker image with caching..."
	@./build-cpu.sh

docker-cpu-fresh: check-docker ## Build CPU-optimized Docker image (no cache)
	@echo "Building CPU-optimized Docker image without cache..."
	@./build-cpu.sh --no-cache

docker-runtime: check-docker ## Build GPU-enabled runtime Docker image
	@echo "Building GPU runtime Docker image with caching..."
	@./build-runtime.sh

docker-runtime-fresh: check-docker ## Build GPU runtime Docker image (no cache)
	@echo "Building GPU runtime Docker image without cache..."
	@./build-runtime.sh --no-cache

docker-devel: check-docker ## Build development Docker image
	@echo "Building development Docker image with caching..."
	@./build-devel.sh

docker-devel-fresh: check-docker ## Build development Docker image (no cache)
	@echo "Building development Docker image without cache..."
	@./build-devel.sh --no-cache

docker-publish: check-docker ## Build publisher Docker image (full system)
	@echo "Building publisher Docker image with caching..."
	@./build-publish.sh

docker-publish-fresh: check-docker ## Build publisher Docker image (full system, no cache)
	@echo "Building publisher Docker image without cache..."
	@./build-publish.sh --no-cache

docker-publisher: check-docker ## Build lightweight publisher Docker image
	@echo "Building lightweight publisher Docker image with caching..."
	@./build-publisher.sh

docker-publisher-fresh: check-docker ## Build lightweight publisher Docker image (no cache)
	@echo "Building lightweight publisher Docker image without cache..."
	@./build-publisher.sh --no-cache

# Publisher testing targets
test-publisher: docker-publish ## Test the publisher Docker image (full system)
	@echo "Testing publisher module..."
	@docker run --rm uos-publish-json:publisher python -m abyss.mqtt.publishers --help

test-publisher-lightweight: docker-publisher ## Test the lightweight publisher Docker image
	@echo "Testing lightweight publisher module..."
	@docker run --rm abyss-publisher:lightweight python -m abyss.mqtt.publishers --help

publisher-modes: docker-publish ## Show publisher operation modes
	@echo "Publisher Operation Modes:"
	@echo "========================="
	@echo ""
	@echo "1. Standard Mode (with realistic patterns):"
	@echo "   docker run --rm -v \$$(pwd)/test_data:/data uos-publish-json:publisher \\"
	@echo "     python -m abyss.mqtt.publishers /data"
	@echo ""
	@echo "2. Stress Test Mode (high performance):"
	@echo "   docker run --rm -v \$$(pwd)/test_data:/data uos-publish-json:publisher \\"
	@echo "     python -m abyss.mqtt.publishers /data --stress-test --rate 1000 --duration 60"
	@echo ""
	@echo "3. Standard Mode (without patterns):"
	@echo "   docker run --rm -v \$$(pwd)/test_data:/data uos-publish-json:publisher \\"
	@echo "     python -m abyss.mqtt.publishers /data --no-patterns"
	@echo ""
	@echo "4. With Signal Tracking:"
	@echo "   docker run --rm -v \$$(pwd)/test_data:/data -v \$$(pwd)/tracking:/tracking uos-publish-json:publisher \\"
	@echo "     python -m abyss.mqtt.publishers /data --track-signals --signal-log /tracking/signals.csv"

publisher-modes-lightweight: docker-publisher ## Show lightweight publisher operation modes
	@echo "Lightweight Publisher Operation Modes:"
	@echo "====================================="
	@echo ""
	@echo "⚡ Optimized for edge devices and minimal footprint (217MB vs 2GB)"
	@echo ""
	@echo "1. Standard Mode:"
	@echo "   docker run --rm -v \$$(pwd)/test_data:/data abyss-publisher:lightweight /data"
	@echo ""
	@echo "2. With custom configuration:"
	@echo "   docker run --rm -v \$$(pwd)/test_data:/data -v \$$(pwd)/config.yaml:/app/config/custom.yaml \\"
	@echo "     abyss-publisher:lightweight /data -c /app/config/custom.yaml"
	@echo ""
	@echo "3. With signal tracking:"
	@echo "   docker run --rm -v \$$(pwd)/test_data:/data -v \$$(pwd)/tracking:/tracking \\"
	@echo "     abyss-publisher:lightweight /data --track-signals --signal-log /tracking/signals.csv"
	@echo ""
	@echo "Dependencies: Only paho-mqtt and pyyaml (no deep learning libraries)"

# Docker utility targets
docker-list: check-docker ## List all project Docker images
	@echo "Project Docker images:"
	@docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" | grep -E "(REPOSITORY|uos-depthest-listener|uos-publish-json|abyss-publisher)" || echo "No project images found"

docker-clean: check-docker ## Remove all project Docker images
	@echo "Removing project Docker images..."
	@docker images --format "{{.Repository}}:{{.Tag}}" | grep -E "(uos-depthest-listener|uos-publish-json|abyss-publisher)" | xargs -r docker rmi -f || echo "No images to remove"
	@echo "Docker images cleaned."

docker-cache-info: check-docker ## Show Docker build cache usage
	@echo "Docker build cache information:"
	@docker system df -v | grep -A 10 "Build Cache" || docker buildx du

docker-cache-clean: check-docker ## Clean Docker build cache
	@echo "Cleaning Docker build cache..."
	@docker builder prune -f
	@echo "Docker build cache cleaned."

# Combined build targets
full-build: build docker-all ## Build both Python wheel and all Docker images
	@echo "Full build completed!"

quick-start: build docker-main ## Build wheel and main Docker image
	@echo "Quick start build completed!"

# Documentation and utility targets
docs-build: ## Build project documentation
	@echo "Building project documentation..."
	@if [ -d ".devnotes" ]; then \
		echo "✓ .devnotes/ documentation found"; \
	else \
		echo "⚠ .devnotes/ directory not found"; \
	fi
	@if [ -f "GETTING_STARTED.md" ]; then \
		echo "✓ GETTING_STARTED.md found"; \
	else \
		echo "⚠ GETTING_STARTED.md not found"; \
	fi
	@if [ -f "REPOSITORY_LAYOUT.md" ]; then \
		echo "✓ REPOSITORY_LAYOUT.md found"; \
	else \
		echo "⚠ REPOSITORY_LAYOUT.md not found"; \
	fi
	@if [ -f "DEVELOPERS.md" ]; then \
		echo "✓ DEVELOPERS.md found"; \
	else \
		echo "⚠ DEVELOPERS.md not found"; \
	fi
	@echo "Documentation build completed!"

security-scan: ## Run security scans on Docker images
	@echo "Running security scans..."
	@command -v docker >/dev/null 2>&1 || { echo "Error: Docker not installed"; exit 1; }
	@echo "Scanning project Docker images..."
	@for image in $$(docker images --format "{{.Repository}}:{{.Tag}}" | grep -E "(uos-depthest-listener|uos-publish-json|abyss-publisher)"); do \
		echo "Scanning $$image..."; \
		if command -v trivy >/dev/null 2>&1; then \
			trivy image --exit-code 1 --severity HIGH,CRITICAL $$image || echo "⚠ Vulnerabilities found in $$image"; \
		else \
			echo "⚠ trivy not installed - install with: curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh"; \
		fi; \
	done
	@echo "Security scan completed!"

validate-config: ## Validate all configuration files
	@echo "Validating configuration files..."
	@echo "Checking YAML configurations..."
	@for file in $$(find . -name "*.yml" -o -name "*.yaml" | grep -v node_modules | grep -v .git); do \
		echo "Validating $$file..."; \
		python3 -c "import yaml; yaml.safe_load(open('$$file'))" && echo "✓ $$file" || echo "❌ $$file"; \
	done
	@echo "Checking Docker Compose files..."
	@for file in $$(find . -name "docker-compose*.yml" | head -3); do \
		if [ -f "$$file" ]; then \
			echo "Validating $$file..."; \
			docker-compose -f "$$file" config >/dev/null 2>&1 && echo "✓ $$file" || echo "❌ $$file"; \
		fi; \
	done
	@echo "Configuration validation completed!"

clean-dangling: ## Clean dangling Docker resources
	@echo "Cleaning dangling Docker resources..."
	@command -v docker >/dev/null 2>&1 || { echo "Error: Docker not installed"; exit 1; }
	@echo "Removing dangling images..."
	@docker image prune -f >/dev/null 2>&1 || true
	@echo "Removing dangling volumes..."
	@docker volume prune -f >/dev/null 2>&1 || true
	@echo "Removing dangling networks..."
	@docker network prune -f >/dev/null 2>&1 || true
	@echo "Removing stopped containers..."
	@docker container prune -f >/dev/null 2>&1 || true
	@echo "Dangling resource cleanup completed!"

# Caching help
cache-help: ## Show caching best practices and tips
	@echo "Docker Build Caching Best Practices"
	@echo "==================================="
	@echo ""
	@echo "🚀 Caching saves 90% of build time by reusing layers!"
	@echo ""
	@echo "Cached builds (default - fast):"
	@echo "  make docker-all      # Build all images with cache"
	@echo "  make docker-main     # Build main image with cache"
	@echo ""
	@echo "Fresh builds (slow - only when needed):"
	@echo "  make docker-all-fresh   # Rebuild all from scratch"
	@echo "  make docker-main-fresh  # Rebuild main from scratch"
	@echo ""
	@echo "Cache management:"
	@echo "  make docker-cache-info  # Show cache usage"
	@echo "  make docker-cache-clean # Clean build cache"
	@echo ""
	@echo "Tips:"
	@echo "- Use cached builds for normal development"
	@echo "- Use fresh builds only when dependencies change"
	@echo "- BuildKit cache mounts preserve pip downloads"

# Help target
help: ## Show this help message
	@echo "UOS Drilling Depth Estimation System - Build Targets"
	@echo "=================================================="
	@echo
	@echo "Python Wheel Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -v "docker-" | grep -v "docs-build\|security-scan\|validate-config\|clean-dangling" | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Docker Build Targets:"
	@grep -E '^docker-[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Utility Targets:"
	@grep -E '^(docs-build|security-scan|validate-config|clean-dangling):.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Examples:"
	@echo "  make build          # Build Python wheel"
	@echo "  make docker-all     # Build all Docker images (cached)"
	@echo "  make full-build     # Build wheel + all Docker images"
	@echo "  make cache-help     # Learn about caching"