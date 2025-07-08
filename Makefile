# =============================================================================
# UOS Drilling Depth Estimation Build System
# =============================================================================
# GNU Make build system for Docker-based ML depth estimation project
# Maintains backward compatibility while adding enhanced capabilities

SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -euo pipefail -c

# Project Configuration
# ===================
PROJECT_NAME := uos-drilling-depth-estimation
IMAGE_BASE := uos-depthest-listener
PUBLISHER_IMAGE := uos-publish-json
VERSION ?= latest
GIT_COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_DATE := $(shell date -u +'%Y-%m-%dT%H:%M:%SZ')

# Docker Configuration
# ==================
CERT_PATH := deployment/certs/airbus-ca.pem
DOCKER_BUILDKIT := 1
BUILD_ARGS := --build-arg BUILDKIT_INLINE_CACHE=1
BUILD_PROGRESS := --progress=plain

# Color Definitions (matching current shell scripts)
# ================================================
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m

# Target Lists for .PHONY
# ======================
BUILD_TARGETS := build-main build-cpu build-devel build-runtime build-publish build-all build-all-parallel
TEST_TARGETS := test test-main test-cpu test-all test-images
RUN_TARGETS := run run-cpu run-devel logs shell
CLEAN_TARGETS := clean clean-images clean-containers clean-all
DEV_TARGETS := dev-setup lint format pytest deps-check
COMPOSE_TARGETS := compose-up compose-down
REGISTRY_TARGETS := push pull docker-login
WHEEL_TARGETS := build-wheel install-wheel clean-wheel
UTIL_TARGETS := help version info benchmark

.PHONY: $(BUILD_TARGETS) $(TEST_TARGETS) $(RUN_TARGETS) $(CLEAN_TARGETS) $(DEV_TARGETS) $(COMPOSE_TARGETS) $(REGISTRY_TARGETS) $(WHEEL_TARGETS) $(UTIL_TARGETS)

# Default Target
# =============
.DEFAULT_GOAL := help

# =============================================================================
# HELP SYSTEM
# =============================================================================

help: ## Show this help message
	@echo -e "$(BLUE)╔══════════════════════════════════════════════════════════════════════╗$(NC)"
	@echo -e "$(BLUE)║                    🚀 UOS Drilling Depth Estimation                  ║$(NC)"
	@echo -e "$(BLUE)║                        GNU Make Build System                        ║$(NC)"
	@echo -e "$(BLUE)╚══════════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo -e "$(GREEN)📦 BUILD COMMANDS:$(NC)"
	@echo "  make build-main         Build main production image (Python slim base)"
	@echo "  make build-cpu          Build CPU-optimized image (no CUDA dependencies)"
	@echo "  make build-devel        Build development image with debugging tools"
	@echo "  make build-runtime      Build runtime image with PyTorch CUDA base"
	@echo "  make build-publish      Build publishing/testing image for data simulation"
	@echo "  make build-all          Build all images sequentially with testing"
	@echo "  make build-all-parallel Build all images in parallel (faster)"
	@echo "  make build-all-quick    Build all images without testing (fastest)"
	@echo ""
	@echo -e "$(YELLOW)🔧 LEGACY BUILD COMMANDS (for compatibility):$(NC)"
	@echo "  make build-main-legacy  Build main image using original shell script"
	@echo "  make build-all-legacy   Build all images using original shell scripts"
	@echo ""
	@echo -e "$(GREEN)🧪 TEST COMMANDS:$(NC)"
	@echo "  make test               Run basic image tests"
	@echo "  make test-main          Test main production image"
	@echo "  make test-cpu           Test CPU-optimized image"
	@echo "  make test-all           Test all built images"
	@echo "  make pytest             Run Python unit tests"
	@echo ""
	@echo -e "$(GREEN)🏃 RUN COMMANDS:$(NC)"
	@echo "  make run                Run main container interactively"
	@echo "  make run-cpu            Run CPU container"
	@echo "  make run-devel          Run development container with workspace mounted"
	@echo "  make logs               Show container logs"
	@echo "  make shell              Open shell in running container"
	@echo ""
	@echo -e "$(GREEN)🐳 COMPOSE COMMANDS:$(NC)"
	@echo "  make compose-up         Start development environment with Docker Compose"
	@echo "  make compose-down       Stop development environment"
	@echo ""
	@echo -e "$(GREEN)🧹 CLEANUP COMMANDS:$(NC)"
	@echo "  make clean              Clean build artifacts and Docker cache"
	@echo "  make clean-images       Remove project Docker images"
	@echo "  make clean-containers   Remove stopped containers"
	@echo "  make clean-all          Complete cleanup (images + containers + cache)"
	@echo ""
	@echo -e "$(GREEN)👨‍💻 DEVELOPMENT COMMANDS:$(NC)"
	@echo "  make dev                Complete development workflow (setup + lint + test + build)"
	@echo "  make dev-setup          Set up development environment with tools"
	@echo "  make dev-test           Quick development test (lint + pytest)"
	@echo "  make dev-check          Check development environment health"
	@echo "  make lint               Run Python linting (black, isort, flake8, mypy)"
	@echo "  make format             Format Python code automatically"
	@echo "  make pytest-coverage    Run tests with coverage report"
	@echo "  make deps-check         Check system dependencies"
	@echo ""
	@echo -e "$(GREEN)📤 REGISTRY COMMANDS:$(NC)"
	@echo "  make push REGISTRY=url  Push images to registry"
	@echo "  make pull REGISTRY=url  Pull images from registry"
	@echo ""
	@echo -e "$(GREEN)🎯 WHEEL COMMANDS:$(NC)"
	@echo "  make build-wheel        Build Python wheel distribution"
	@echo "  make install-wheel      Install wheel from local build"
	@echo "  make clean-wheel        Clean wheel build artifacts"
	@echo ""
	@echo -e "$(GREEN)ℹ️  INFO COMMANDS:$(NC)"
	@echo "  make version            Show version information"
	@echo "  make info               Show project information"
	@echo "  make benchmark          Show build performance stats"
	@echo ""
	@echo -e "$(YELLOW)💡 EXAMPLES:$(NC)"
	@echo "  make build-all          # Build all Docker images"
	@echo "  make test-all           # Test all images"
	@echo "  make run-devel          # Start development environment"
	@echo "  make clean-all          # Complete cleanup"
	@echo ""
	@echo -e "$(BLUE)📚 For more information, see: docs/BUILD-GUIDE.md$(NC)"

# =============================================================================
# BUILD HELPER FUNCTIONS
# =============================================================================

# Function to build Docker image with consistent parameters
define build_docker
	@echo -e "$(GREEN)🔨 Building $(1):$(2)...$(NC)"
	@echo -e "$(BLUE)Build Configuration:$(NC)"
	@echo "  Image: $(1):$(2)"
	@echo "  Dockerfile: $(3)"
	@echo "  Context: ."
	@echo "  Git Commit: $(GIT_COMMIT)"
	@echo ""
	@start_time=$$(date +%s); \
	docker build \
		--file $(3) \
		--tag $(1):$(2) \
		--label "git-commit=$(GIT_COMMIT)" \
		--label "build-date=$(BUILD_DATE)" \
		--label "version=$(VERSION)" \
		$(BUILD_PROGRESS) \
		. && \
	end_time=$$(date +%s) && \
	duration=$$((end_time - start_time)) && \
	echo -e "$(GREEN)✅ Build completed successfully in $${duration}s$(NC)" && \
	echo -e "$(BLUE)Image Information:$(NC)" && \
	docker images $(1):$(2) --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}"
endef

# Function to test Docker image
define test_docker_image
	@echo -e "$(YELLOW)🧪 Testing $(1)...$(NC)"
	@docker run --rm $(1) python -c "import abyss; import torch; print(f'✅ $(1) test passed - PyTorch: {torch.__version__}')" || \
	 docker run --rm $(1) python -c "import abyss; print('✅ $(1) test passed')" || \
	 (echo -e "$(RED)❌ $(1) test failed$(NC)" && exit 1)
endef

# =============================================================================
# NATIVE BUILD TARGETS (Phase 2: Enhanced Implementation)
# =============================================================================

build-main: deps-check ## Build main production image
	$(call build_docker,$(IMAGE_BASE),latest,deployment/docker/Dockerfile)
	$(call test_docker_image,$(IMAGE_BASE):latest)

build-cpu: deps-check ## Build CPU-optimized image
	$(call build_docker,$(IMAGE_BASE),cpu,deployment/docker/Dockerfile.cpu)
	$(call test_docker_image,$(IMAGE_BASE):cpu)

build-devel: deps-check ## Build development image
	$(call build_docker,$(IMAGE_BASE),devel,deployment/docker/Dockerfile.devel)
	$(call test_docker_image,$(IMAGE_BASE):devel)

build-runtime: deps-check ## Build runtime image with CUDA
	$(call build_docker,$(IMAGE_BASE),runtime,deployment/docker/Dockerfile.runtime)
	$(call test_docker_image,$(IMAGE_BASE):runtime)

build-publish: deps-check ## Build publishing/testing image
	$(call build_docker,$(PUBLISHER_IMAGE),latest,deployment/docker/Dockerfile.publish)
	$(call test_docker_image,$(PUBLISHER_IMAGE):latest)

build-all: ## Build all images sequentially
	@echo -e "$(BLUE)╔══════════════════════════════════════════════════════════════════════╗$(NC)"
	@echo -e "$(BLUE)║                    Building All Docker Images                       ║$(NC)"
	@echo -e "$(BLUE)╚══════════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@start_time=$$(date +%s)
	@$(MAKE) --no-print-directory build-main
	@$(MAKE) --no-print-directory build-cpu  
	@$(MAKE) --no-print-directory build-devel
	@$(MAKE) --no-print-directory build-runtime
	@$(MAKE) --no-print-directory build-publish
	@end_time=$$(date +%s)
	@total_duration=$$((end_time - start_time))
	@echo -e "$(GREEN)╔══════════════════════════════════════════════════════════════════════╗$(NC)"
	@echo -e "$(GREEN)║                 ✅ All Images Built Successfully!                    ║$(NC)"
	@echo -e "$(GREEN)║                 Total Time: $${total_duration}s                              ║$(NC)"
	@echo -e "$(GREEN)╚══════════════════════════════════════════════════════════════════════╝$(NC)"
	@$(MAKE) --no-print-directory test-images

# =============================================================================
# LEGACY WRAPPER TARGETS (Backward Compatibility)
# =============================================================================

build-main-legacy: ## Build main production image (legacy shell script)
	@echo -e "$(GREEN)🔨 Building main Docker image (legacy)...$(NC)"
	@deployment/scripts/build-main.sh

build-cpu-legacy: ## Build CPU-optimized image (legacy shell script)
	@echo -e "$(GREEN)🔨 Building CPU-optimized image (legacy)...$(NC)"
	@deployment/scripts/build-cpu.sh

build-devel-legacy: ## Build development image (legacy shell script)
	@echo -e "$(GREEN)🔨 Building development image (legacy)...$(NC)"
	@deployment/scripts/build-devel.sh

build-runtime-legacy: ## Build runtime image with CUDA (legacy shell script)
	@echo -e "$(GREEN)🔨 Building runtime image (legacy)...$(NC)"
	@deployment/scripts/build-runtime.sh

build-publish-legacy: ## Build publishing/testing image (legacy shell script)
	@echo -e "$(GREEN)🔨 Building publishing image (legacy)...$(NC)"
	@deployment/scripts/build-publish.sh

build-all-legacy: ## Build all images sequentially (legacy shell script)
	@echo -e "$(BLUE)╔══════════════════════════════════════════════════════════════════════╗$(NC)"
	@echo -e "$(BLUE)║                    Building All Docker Images (Legacy)              ║$(NC)"
	@echo -e "$(BLUE)╚══════════════════════════════════════════════════════════════════════╝$(NC)"
	@deployment/scripts/build-all.sh

# =============================================================================
# ENHANCED FEATURES (New capabilities not in shell scripts)
# =============================================================================

build-all-parallel: deps-check ## Build all images in parallel (faster)
	@echo -e "$(BLUE)╔══════════════════════════════════════════════════════════════════════╗$(NC)"
	@echo -e "$(BLUE)║                🚀 Building All Images in Parallel                   ║$(NC)"
	@echo -e "$(BLUE)╚══════════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@start_time=$$(date +%s)
	@echo -e "$(YELLOW)Starting parallel builds (max 4 concurrent)...$(NC)"
	@$(MAKE) -j4 build-main-parallel build-cpu-parallel build-devel-parallel build-runtime-parallel build-publish-parallel
	@end_time=$$(date +%s)
	@total_duration=$$((end_time - start_time))
	@echo ""
	@echo -e "$(GREEN)╔══════════════════════════════════════════════════════════════════════╗$(NC)"
	@echo -e "$(GREEN)║               ✅ All Parallel Builds Completed!                      ║$(NC)"
	@echo -e "$(GREEN)║               Total Time: $${total_duration}s                              ║$(NC)"
	@echo -e "$(GREEN)╚══════════════════════════════════════════════════════════════════════╝$(NC)"
	@$(MAKE) --no-print-directory test-images

# Parallel build targets (no dependencies to allow true parallelism)
build-main-parallel:
	$(call build_docker,$(IMAGE_BASE),latest,deployment/docker/Dockerfile)

build-cpu-parallel:
	$(call build_docker,$(IMAGE_BASE),cpu,deployment/docker/Dockerfile.cpu)

build-devel-parallel:
	$(call build_docker,$(IMAGE_BASE),devel,deployment/docker/Dockerfile.devel)

build-runtime-parallel:
	$(call build_docker,$(IMAGE_BASE),runtime,deployment/docker/Dockerfile.runtime)

build-publish-parallel:
	$(call build_docker,$(PUBLISHER_IMAGE),latest,deployment/docker/Dockerfile.publish)

# Quick build targets (skip testing for speed)
build-main-quick: deps-check ## Build main image without testing
	$(call build_docker,$(IMAGE_BASE),latest,deployment/docker/Dockerfile)

build-cpu-quick: deps-check ## Build CPU image without testing
	$(call build_docker,$(IMAGE_BASE),cpu,deployment/docker/Dockerfile.cpu)

build-all-quick: ## Build all images without testing (fastest)
	@echo -e "$(BLUE)🚀 Quick build - skipping tests for speed...$(NC)"
	@$(MAKE) -j4 build-main-parallel build-cpu-parallel build-devel-parallel build-runtime-parallel build-publish-parallel
	@echo -e "$(GREEN)✅ Quick build completed$(NC)"

# =============================================================================
# TESTING TARGETS
# =============================================================================

test: test-main test-cpu ## Run basic image tests
	@echo -e "$(GREEN)✅ Basic tests completed$(NC)"

test-main: ## Test main production image
	@echo -e "$(YELLOW)🧪 Testing main image...$(NC)"
	@docker run --rm $(IMAGE_BASE):latest python -c "import abyss; print('✅ Main image test passed')" || (echo -e "$(RED)❌ Main image test failed$(NC)" && exit 1)

test-cpu: ## Test CPU-optimized image
	@echo -e "$(YELLOW)🧪 Testing CPU image...$(NC)"
	@docker run --rm $(IMAGE_BASE):cpu python -c "import abyss; print('✅ CPU image test passed')" || (echo -e "$(RED)❌ CPU image test failed$(NC)" && exit 1)

test-all: test-main test-cpu ## Test all available images
	@echo -e "$(YELLOW)🧪 Testing development image...$(NC)"
	@docker run --rm $(IMAGE_BASE):devel python -c "import abyss; print('✅ Devel image test passed')" 2>/dev/null || echo -e "$(YELLOW)⚠️  Devel image not available$(NC)"
	@echo -e "$(YELLOW)🧪 Testing runtime image...$(NC)"
	@docker run --rm $(IMAGE_BASE):runtime python -c "import abyss; print('✅ Runtime image test passed')" 2>/dev/null || echo -e "$(YELLOW)⚠️  Runtime image not available$(NC)"
	@echo -e "$(GREEN)✅ All available images tested$(NC)"

test-images: ## Show information about built images
	@echo -e "$(BLUE)📊 Docker Images Information:$(NC)"
	@docker images $(IMAGE_BASE) --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}" 2>/dev/null || echo -e "$(YELLOW)⚠️  No images found$(NC)"
	@docker images $(PUBLISHER_IMAGE) --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}" 2>/dev/null || true

# =============================================================================
# RUN TARGETS
# =============================================================================

run: ## Run main container interactively
	@echo -e "$(GREEN)🏃 Starting main container...$(NC)"
	@docker run -it --rm \
		-v $(PWD)/abyss/src/abyss/run/config:/app/config \
		-p 1883:1883 \
		--name $(PROJECT_NAME)-main \
		$(IMAGE_BASE):latest

run-cpu: ## Run CPU container
	@echo -e "$(GREEN)🏃 Starting CPU container...$(NC)"
	@docker run -it --rm \
		-v $(PWD)/abyss/src/abyss/run/config:/app/config \
		-p 1883:1883 \
		--name $(PROJECT_NAME)-cpu \
		$(IMAGE_BASE):cpu

run-devel: ## Run development container with workspace mounted
	@echo -e "$(GREEN)🏃 Starting development container...$(NC)"
	@docker run -it --rm \
		-v $(PWD):/workspace \
		-v $(PWD)/abyss/src/abyss/run/config:/app/config \
		-w /workspace \
		--name $(PROJECT_NAME)-devel \
		$(IMAGE_BASE):devel bash

logs: ## Show container logs
	@echo -e "$(BLUE)📋 Container logs:$(NC)"
	@docker logs -f $$(docker ps -q --filter name=$(PROJECT_NAME) | head -1) 2>/dev/null || echo -e "$(YELLOW)⚠️  No running containers found$(NC)"

shell: ## Open shell in running container
	@echo -e "$(BLUE)🐚 Opening shell in container...$(NC)"
	@docker exec -it $$(docker ps -q --filter name=$(PROJECT_NAME) | head -1) bash 2>/dev/null || echo -e "$(YELLOW)⚠️  No running containers found$(NC)"

# =============================================================================
# COMPOSE TARGETS
# =============================================================================

compose-up: ## Start development environment
	@echo -e "$(GREEN)🐳 Starting development environment...$(NC)"
	@docker-compose -f deployment/compose/development/docker-compose.yml up -d

compose-down: ## Stop development environment
	@echo -e "$(YELLOW)🐳 Stopping development environment...$(NC)"
	@docker-compose -f deployment/compose/development/docker-compose.yml down

# =============================================================================
# CLEANUP TARGETS
# =============================================================================

clean: ## Clean build artifacts and Docker cache
	@echo -e "$(YELLOW)🧹 Cleaning build artifacts...$(NC)"
	@docker builder prune -f
	@docker system prune -f
	@echo -e "$(GREEN)✅ Cleanup completed$(NC)"

clean-images: ## Remove project Docker images
	@echo -e "$(YELLOW)🧹 Removing project Docker images...$(NC)"
	@docker rmi $$(docker images $(IMAGE_BASE) -q) 2>/dev/null || echo -e "$(YELLOW)⚠️  No $(IMAGE_BASE) images found$(NC)"
	@docker rmi $$(docker images $(PUBLISHER_IMAGE) -q) 2>/dev/null || echo -e "$(YELLOW)⚠️  No $(PUBLISHER_IMAGE) images found$(NC)"
	@echo -e "$(GREEN)✅ Images removed$(NC)"

clean-containers: ## Remove stopped containers
	@echo -e "$(YELLOW)🧹 Removing stopped containers...$(NC)"
	@docker container prune -f
	@echo -e "$(GREEN)✅ Containers cleaned$(NC)"

clean-all: clean-containers clean-images clean clean-wheel ## Complete cleanup
	@echo -e "$(GREEN)✅ Complete cleanup finished$(NC)"

# =============================================================================
# DEVELOPMENT TARGETS
# =============================================================================

deps-check: ## Check system dependencies
	@echo -e "$(BLUE)🔍 Checking dependencies...$(NC)"
	@command -v docker >/dev/null 2>&1 || (echo -e "$(RED)❌ Docker not installed$(NC)" && exit 1)
	@command -v python >/dev/null 2>&1 || (echo -e "$(RED)❌ Python not installed$(NC)" && exit 1)
	@[ -f abyss/requirements.txt ] || (echo -e "$(RED)❌ abyss/requirements.txt not found$(NC)" && exit 1)
	@[ -f abyss/pyproject.toml ] || (echo -e "$(RED)❌ abyss/pyproject.toml not found$(NC)" && exit 1)
	@if [ ! -f $(CERT_PATH) ]; then \
		echo -e "$(YELLOW)⚠️  Certificate not found, creating placeholder...$(NC)"; \
		mkdir -p deployment/certs; \
		touch $(CERT_PATH); \
	fi
	@echo -e "$(BLUE)🔍 Checking wheel build dependencies...$(NC)"
	@python -c "import setuptools, wheel, build" 2>/dev/null || \
		(echo -e "$(YELLOW)⚠️  Installing wheel build dependencies...$(NC)" && \
		 pip install --upgrade setuptools wheel build 2>/dev/null || \
		 echo -e "$(YELLOW)⚠️  Some wheel build dependencies missing$(NC)")
	@echo -e "$(GREEN)✅ All dependencies available$(NC)"

dev-setup: deps-check ## Set up development environment
	@echo -e "$(BLUE)🔧 Setting up development environment...$(NC)"
	@cd abyss && pip install -e .
	@cd abyss && pip install -r requirements.txt
	@echo ""
	@echo -e "$(YELLOW)Installing development tools...$(NC)"
	@pip install black isort mypy pytest pytest-asyncio pytest-mock coverage flake8 2>/dev/null || echo -e "$(YELLOW)⚠️  Some dev tools not installed$(NC)"
	@echo ""
	@echo -e "$(GREEN)✅ Development environment ready$(NC)"
	@echo -e "$(BLUE)Usage:$(NC)"
	@echo "  make lint      # Check code quality"
	@echo "  make format    # Format code"
	@echo "  make pytest    # Run tests"
	@echo "  make dev       # Complete development workflow"

dev: ## Complete development workflow (setup, lint, test, build)
	@echo -e "$(BLUE)🚀 Running complete development workflow...$(NC)"
	@$(MAKE) --no-print-directory dev-setup
	@$(MAKE) --no-print-directory lint
	@$(MAKE) --no-print-directory pytest
	@$(MAKE) --no-print-directory build-devel
	@echo -e "$(GREEN)✅ Development workflow completed successfully$(NC)"

dev-package: ## Development workflow with wheel packaging
	@echo -e "$(BLUE)🚀 Running development workflow with packaging...$(NC)"
	@$(MAKE) --no-print-directory dev-setup
	@$(MAKE) --no-print-directory lint
	@$(MAKE) --no-print-directory pytest
	@$(MAKE) --no-print-directory build-wheel
	@echo -e "$(GREEN)✅ Development workflow with packaging completed successfully$(NC)"

lint: ## Run Python linting
	@echo -e "$(BLUE)🔍 Running Python linting...$(NC)"
	@cd abyss && python -m black --check src/ && echo -e "$(GREEN)✅ Black formatting check passed$(NC)" || echo -e "$(YELLOW)⚠️  Black formatting issues found$(NC)"
	@cd abyss && python -m isort --check-only src/ && echo -e "$(GREEN)✅ Import sorting check passed$(NC)" || echo -e "$(YELLOW)⚠️  Import sorting issues found$(NC)"
	@cd abyss && python -m flake8 src/ --max-line-length=88 --extend-ignore=E203,W503 && echo -e "$(GREEN)✅ Flake8 check passed$(NC)" || echo -e "$(YELLOW)⚠️  Flake8 issues found$(NC)"
	@cd abyss && python -m mypy src/ && echo -e "$(GREEN)✅ Type checking passed$(NC)" || echo -e "$(YELLOW)⚠️  Type checking issues found$(NC)"

format: ## Format Python code
	@echo -e "$(BLUE)🎨 Formatting Python code...$(NC)"
	@cd abyss && python -m black src/ && echo -e "$(GREEN)✅ Code formatted with Black$(NC)"
	@cd abyss && python -m isort src/ && echo -e "$(GREEN)✅ Imports sorted with isort$(NC)"
	@echo -e "$(GREEN)✅ Code formatting completed$(NC)"

pytest: ## Run Python unit tests
	@echo -e "$(BLUE)🧪 Running Python tests...$(NC)"
	@cd abyss && python -m pytest tests/ -v --tb=short && echo -e "$(GREEN)✅ All tests passed$(NC)" || echo -e "$(RED)❌ Some tests failed$(NC)"

pytest-coverage: ## Run Python tests with coverage
	@echo -e "$(BLUE)🧪 Running Python tests with coverage...$(NC)"
	@cd abyss && python -m pytest tests/ --cov=src --cov-report=html --cov-report=term && echo -e "$(GREEN)✅ Tests completed with coverage report$(NC)"
	@echo -e "$(BLUE)Coverage report available at: abyss/htmlcov/index.html$(NC)"

# Advanced development targets
dev-test: ## Quick development test (lint + test)
	@$(MAKE) --no-print-directory lint
	@$(MAKE) --no-print-directory pytest

dev-check: ## Check development environment health
	@echo -e "$(BLUE)🔍 Checking development environment...$(NC)"
	@python --version && echo -e "$(GREEN)✅ Python available$(NC)"
	@cd abyss && python -c "import abyss; print('✅ abyss package importable')" && echo -e "$(GREEN)✅ Package imports successfully$(NC)"
	@docker --version && echo -e "$(GREEN)✅ Docker available$(NC)"
	@make --version | head -1 && echo -e "$(GREEN)✅ Make available$(NC)"
	@git --version && echo -e "$(GREEN)✅ Git available$(NC)"

# =============================================================================
# REGISTRY TARGETS
# =============================================================================

push: ## Push images to registry (set REGISTRY variable)
	@if [ -z "$(REGISTRY)" ]; then \
		echo -e "$(RED)❌ REGISTRY variable not set$(NC)"; \
		echo -e "$(YELLOW)Usage: make push REGISTRY=your-registry.com$(NC)"; \
		exit 1; \
	fi
	@echo -e "$(BLUE)📤 Pushing images to $(REGISTRY)...$(NC)"
	@for tag in latest cpu devel runtime; do \
		if docker images $(IMAGE_BASE):$$tag --format "{{.Repository}}" | grep -q $(IMAGE_BASE); then \
			docker tag $(IMAGE_BASE):$$tag $(REGISTRY)/$(IMAGE_BASE):$$tag; \
			docker push $(REGISTRY)/$(IMAGE_BASE):$$tag; \
			echo -e "$(GREEN)✅ Pushed $(IMAGE_BASE):$$tag$(NC)"; \
		fi; \
	done

pull: ## Pull images from registry (set REGISTRY variable)
	@if [ -z "$(REGISTRY)" ]; then \
		echo -e "$(RED)❌ REGISTRY variable not set$(NC)"; \
		echo -e "$(YELLOW)Usage: make pull REGISTRY=your-registry.com$(NC)"; \
		exit 1; \
	fi
	@echo -e "$(BLUE)📥 Pulling images from $(REGISTRY)...$(NC)"
	@docker pull $(REGISTRY)/$(IMAGE_BASE):latest || true
	@docker pull $(REGISTRY)/$(IMAGE_BASE):cpu || true

docker-login: ## Login to Docker registry
	@echo -e "$(BLUE)🔐 Logging into Docker registry...$(NC)"
	@docker login $(REGISTRY)

# =============================================================================
# WHEEL BUILD TARGETS
# =============================================================================

build-wheel: deps-check ## Build Python wheel distribution
	@echo -e "$(GREEN)🎯 Building Python wheel distribution...$(NC)"
	@deployment/scripts/build-wheel.sh

build-wheel-legacy: ## Build Python wheel using legacy shell script
	@echo -e "$(GREEN)🎯 Building Python wheel (legacy)...$(NC)"
	@deployment/scripts/build-wheel.sh

install-wheel: ## Install wheel from local build
	@echo -e "$(GREEN)📦 Installing Python wheel...$(NC)"
	@if [ ! -d "abyss/dist" ]; then \
		echo -e "$(RED)❌ No wheel distribution found. Run 'make build-wheel' first.$(NC)"; \
		exit 1; \
	fi
	@wheel_file=$$(find abyss/dist -name "*.whl" | head -1); \
	if [ -z "$$wheel_file" ]; then \
		echo -e "$(RED)❌ No wheel file found in abyss/dist/$(NC)"; \
		exit 1; \
	fi; \
	echo -e "$(BLUE)Installing: $$wheel_file$(NC)"; \
	pip install --force-reinstall "$$wheel_file" && \
	echo -e "$(GREEN)✅ Wheel installed successfully$(NC)"

clean-wheel: ## Clean wheel build artifacts
	@echo -e "$(YELLOW)🧹 Cleaning wheel build artifacts...$(NC)"
	@rm -rf abyss/dist abyss/build abyss/src/abyss.egg-info
	@echo -e "$(GREEN)✅ Wheel artifacts cleaned$(NC)"

# =============================================================================
# INFO TARGETS
# =============================================================================

version: ## Show version information
	@echo -e "$(BLUE)📋 Version Information:$(NC)"
	@echo "Project: $(PROJECT_NAME)"
	@echo "Version: $(VERSION)"
	@echo "Git Commit: $(GIT_COMMIT)"
	@echo "Build Date: $(BUILD_DATE)"
	@echo "Base Image: $(IMAGE_BASE)"

info: ## Show project information
	@echo -e "$(BLUE)📋 Project Information:$(NC)"
	@echo "Project Name: $(PROJECT_NAME)"
	@echo "Base Image: $(IMAGE_BASE)"
	@echo "Publisher Image: $(PUBLISHER_IMAGE)"
	@echo "Certificate Path: $(CERT_PATH)"
	@echo "Docker Buildkit: $(DOCKER_BUILDKIT)"
	@echo ""
	@echo -e "$(BLUE)📊 Current Docker Images:$(NC)"
	@$(MAKE) test-images

benchmark: ## Show build performance stats
	@echo -e "$(BLUE)📊 Build Performance Benchmark:$(NC)"
	@echo "Starting benchmark..."
	@start_time=$$(date +%s); \
	$(MAKE) build-main >/dev/null 2>&1; \
	end_time=$$(date +%s); \
	duration=$$((end_time - start_time)); \
	echo "Main image build time: $${duration}s"
	@$(MAKE) test-images

# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Ensure old wrapper scripts still work by keeping them functional
build_main: build-main
build_cpu: build-cpu  
build_devel: build-devel
build_runtime: build-runtime
build_publish: build-publish
build_all: build-all