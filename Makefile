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

# Package-specific targets (consolidated from abyss/Makefile)
pkg-clean: ## Clean package build artifacts
	@echo "Cleaning package build artifacts..."
	cd $(ABYSS_DIR) && python3 build_wheel.py --clean

pkg-build: ## Build package wheel
	@echo "Building package wheel..."
	cd $(ABYSS_DIR) && python3 build_wheel.py

pkg-install: ## Build and install package wheel
	@echo "Building and installing package wheel..."
	cd $(ABYSS_DIR) && python3 build_wheel.py --clean --install

pkg-dev-install: ## Install package in editable mode
	@echo "Installing package in development mode..."
	cd $(ABYSS_DIR) && python3 build_wheel.py --clean --editable

pkg-check: ## Validate package without installing
	@echo "Validating package..."
	cd $(ABYSS_DIR) && python3 build_wheel.py --clean --skip-validation
	@if command -v twine >/dev/null 2>&1; then \
		echo "Running twine check..."; \
		cd $(ABYSS_DIR) && twine check dist/*.whl; \
	else \
		echo "twine not available for additional validation"; \
	fi

pkg-lint: ## Run linting on package code
	@echo "Running linting on package..."
	@if command -v flake8 >/dev/null 2>&1; then \
		echo "Running flake8..."; \
		cd $(ABYSS_DIR) && flake8 src/abyss --max-line-length=120 --ignore=E501,W503; \
	fi
	@if command -v pylint >/dev/null 2>&1; then \
		echo "Running pylint..."; \
		cd $(ABYSS_DIR) && pylint src/abyss --disable=C0103,R0903,R0913; \
	fi
	@if command -v mypy >/dev/null 2>&1; then \
		echo "Running mypy..."; \
		cd $(ABYSS_DIR) && mypy src/abyss --ignore-missing-imports; \
	fi

pkg-format: ## Format package code
	@echo "Formatting package code..."
	@if command -v black >/dev/null 2>&1; then \
		echo "Running black..."; \
		cd $(ABYSS_DIR) && black src/abyss tests/; \
	fi
	@if command -v isort >/dev/null 2>&1; then \
		echo "Running isort..."; \
		cd $(ABYSS_DIR) && isort src/abyss tests/; \
	fi

pkg-info: ## Show package information
	@echo "Package Information:"
	@echo "==================="
	@cd $(ABYSS_DIR) && grep -E "^(name|version)" pyproject.toml || echo "Could not read pyproject.toml"
	@echo ""
	@echo "Python version:"
	@python3 --version
	@echo ""
	@echo "Build tools:"
	@python3 -c "import build; print(f'build: {build.__version__}')" 2>/dev/null || echo "build: not installed"
	@python3 -c "import wheel; print(f'wheel: {wheel.__version__}')" 2>/dev/null || echo "wheel: not installed"
	@python3 -c "import setuptools; print(f'setuptools: {setuptools.__version__}')" 2>/dev/null || echo "setuptools: not installed"

# Combined package targets
pkg-clean-build: pkg-clean pkg-build ## Clean and build package
pkg-clean-install: pkg-clean pkg-install ## Clean, build and install package

# CI/CD Build Automation Targets
ci: ## Complete CI pipeline (test, build, validate, security scan)
	@echo "Running complete CI pipeline..."
	@echo "================================"
	@$(MAKE) ci-start-time
	@$(MAKE) validate-config
	@$(MAKE) pkg-lint
	@$(MAKE) test
	@$(MAKE) pkg-build
	@$(MAKE) pkg-check
	@$(MAKE) security-scan
	@$(MAKE) ci-summary

ci-quick: ## Quick CI pipeline (lint, test, build)
	@echo "Running quick CI pipeline..."
	@echo "============================"
	@$(MAKE) pkg-lint
	@$(MAKE) test
	@$(MAKE) pkg-build
	@echo "✅ Quick CI pipeline completed successfully!"

ci-start-time: ## Record CI start time (internal target)
	@echo "CI_START_TIME=$$(date +%s)" > .ci_timing
	@echo "🚀 CI Pipeline started at $$(date)"

ci-summary: ## Show CI pipeline summary (internal target)
	@if [ -f .ci_timing ]; then \
		start_time=$$(grep CI_START_TIME .ci_timing | cut -d= -f2); \
		end_time=$$(date +%s); \
		duration=$$((end_time - start_time)); \
		echo ""; \
		echo "🎉 CI Pipeline Summary"; \
		echo "====================="; \
		echo "✅ All checks passed"; \
		echo "⏱️  Total duration: $${duration}s"; \
		echo "📦 Artifacts ready for deployment"; \
		rm -f .ci_timing; \
	fi

# Build Matrix Support
build-matrix: ## Build across multiple configurations
	@echo "Building across multiple configurations..."
	@echo "========================================="
	@$(MAKE) build-matrix-python
	@$(MAKE) build-matrix-docker

build-matrix-python: ## Build with multiple Python versions
	@echo "Building with multiple Python versions..."
	@for python_ver in python3.8 python3.9 python3.10 python3.11 python3.12; do \
		if command -v $$python_ver >/dev/null 2>&1; then \
			echo "Building with $$python_ver..."; \
			cd $(ABYSS_DIR) && $$python_ver build_wheel.py --clean || echo "⚠️  $$python_ver build failed"; \
		else \
			echo "⚠️  $$python_ver not available"; \
		fi; \
	done

build-matrix-docker: ## Build Docker images for multiple architectures
	@echo "Building Docker images for multiple architectures..."
	@if command -v docker >/dev/null 2>&1 && docker buildx version >/dev/null 2>&1; then \
		echo "Building multi-arch Docker images..."; \
		docker buildx build --platform linux/amd64,linux/arm64 -t uos-depthest-listener:multi-arch . || echo "⚠️  Multi-arch build failed"; \
	else \
		echo "⚠️  Docker buildx not available for multi-arch builds"; \
		echo "🌟 Falling back to RECOMMENDED CPU build"; \
		$(MAKE) docker-cpu; \
	fi

# Dependency Management
deps-check: ## Check for dependency updates
	@echo "Checking for dependency updates..."
	@echo "================================="
	@if command -v pip-audit >/dev/null 2>&1; then \
		echo "Running security audit..."; \
		pip-audit --desc --format=json || echo "⚠️  Security issues found"; \
	else \
		echo "⚠️  pip-audit not installed. Install with: pip install pip-audit"; \
	fi
	@if command -v pip list --outdated >/dev/null 2>&1; then \
		echo "Checking for outdated packages..."; \
		pip list --outdated || echo "⚠️  Could not check outdated packages"; \
	fi

deps-update: ## Update dependencies (interactive)
	@echo "Updating dependencies..."
	@echo "======================="
	@echo "⚠️  This will update dependencies. Continue? [y/N]"; \
	read -r confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		echo "Updating pip..."; \
		pip install --upgrade pip; \
		echo "Updating setuptools and wheel..."; \
		pip install --upgrade setuptools wheel build; \
		echo "Updating development dependencies..."; \
		pip install --upgrade pytest flake8 black isort mypy twine; \
		echo "✅ Dependencies updated. Run 'make test' to validate."; \
	else \
		echo "❌ Dependency update cancelled."; \
	fi

deps-freeze: ## Create requirements snapshot
	@echo "Creating requirements snapshot..."
	pip freeze > requirements.snapshot.txt
	@echo "✅ Requirements snapshot saved to requirements.snapshot.txt"

# Performance Monitoring
build-timing: ## Monitor build performance
	@echo "Monitoring build performance..."
	@echo "=============================="
	@start_time=$$(date +%s); \
	$(MAKE) pkg-clean; \
	clean_time=$$(date +%s); \
	$(MAKE) pkg-build; \
	build_time=$$(date +%s); \
	$(MAKE) docker-cpu; \
	docker_time=$$(date +%s); \
	clean_duration=$$((clean_time - start_time)); \
	build_duration=$$((build_time - clean_time)); \
	docker_duration=$$((docker_time - build_time)); \
	total_duration=$$((docker_time - start_time)); \
	echo ""; \
	echo "⏱️  Build Performance Report (with RECOMMENDED CPU image)"; \
	echo "========================================================="; \
	echo "Clean:      $${clean_duration}s"; \
	echo "Build:      $${build_duration}s"; \
	echo "Docker-CPU: $${docker_duration}s"; \
	echo "Total:      $${total_duration}s"; \
	if [ $$total_duration -gt 300 ]; then \
		echo "⚠️  Build took longer than 5 minutes"; \
	else \
		echo "✅ Build completed within acceptable time"; \
	fi

# Release Automation Targets
release-check: ## Check if ready for release
	@echo "Checking release readiness..."
	@echo "============================"
	@$(MAKE) ci
	@$(MAKE) release-validate-version
	@echo "✅ Release checks completed successfully!"

release-validate-version: ## Validate version consistency
	@echo "Validating version consistency..."
	@pkg_version=$$(cd $(ABYSS_DIR) && grep -E "^version" pyproject.toml | cut -d'"' -f2); \
	if [ -z "$$pkg_version" ]; then \
		echo "❌ Could not extract version from pyproject.toml"; \
		exit 1; \
	fi; \
	echo "📦 Package version: $$pkg_version"; \
	if git tag | grep -q "v$$pkg_version"; then \
		echo "⚠️  Version $$pkg_version already tagged in git"; \
	else \
		echo "✅ Version $$pkg_version is new"; \
	fi

release-patch: ## Create patch release (bump patch version)
	@echo "Creating patch release..."
	@echo "========================"
	@$(MAKE) release-bump-version BUMP_TYPE=patch
	@$(MAKE) release-finalize

release-minor: ## Create minor release (bump minor version)
	@echo "Creating minor release..."
	@echo "========================"
	@$(MAKE) release-bump-version BUMP_TYPE=minor
	@$(MAKE) release-finalize

release-major: ## Create major release (bump major version)
	@echo "Creating major release..."
	@echo "========================"
	@$(MAKE) release-bump-version BUMP_TYPE=major
	@$(MAKE) release-finalize

release-bump-version: ## Bump version (internal target)
	@if [ -z "$(BUMP_TYPE)" ]; then \
		echo "❌ BUMP_TYPE not specified"; \
		exit 1; \
	fi
	@echo "Bumping $(BUMP_TYPE) version..."
	@cd $(ABYSS_DIR) && \
	current_version=$$(grep -E "^version" pyproject.toml | cut -d'"' -f2); \
	echo "Current version: $$current_version"; \
	IFS='.' read -r major minor patch <<< "$$current_version"; \
	case "$(BUMP_TYPE)" in \
		patch) new_version="$$major.$$minor.$$((patch + 1))" ;; \
		minor) new_version="$$major.$$((minor + 1)).0" ;; \
		major) new_version="$$((major + 1)).0.0" ;; \
		*) echo "❌ Invalid BUMP_TYPE: $(BUMP_TYPE)"; exit 1 ;; \
	esac; \
	echo "New version: $$new_version"; \
	sed -i "s/version = \"$$current_version\"/version = \"$$new_version\"/" pyproject.toml; \
	echo "✅ Version bumped to $$new_version"

release-finalize: ## Finalize release (internal target)
	@echo "Finalizing release..."
	@$(MAKE) release-check
	@$(MAKE) release-create-tag
	@$(MAKE) release-build-artifacts
	@echo "🎉 Release completed successfully!"

release-create-tag: ## Create git tag for release
	@echo "Creating git tag..."
	@pkg_version=$$(cd $(ABYSS_DIR) && grep -E "^version" pyproject.toml | cut -d'"' -f2); \
	tag_name="v$$pkg_version"; \
	echo "Creating tag: $$tag_name"; \
	git add $(ABYSS_DIR)/pyproject.toml; \
	git commit -m "chore: bump version to $$pkg_version" || echo "⚠️  No changes to commit"; \
	git tag -a "$$tag_name" -m "Release $$tag_name"; \
	echo "✅ Tag $$tag_name created"

release-build-artifacts: ## Build release artifacts
	@echo "Building release artifacts..."
	@$(MAKE) pkg-clean-build
	@$(MAKE) docker-all
	@pkg_version=$$(cd $(ABYSS_DIR) && grep -E "^version" pyproject.toml | cut -d'"' -f2); \
	release_dir="releases/v$$pkg_version"; \
	mkdir -p "$$release_dir"; \
	cp $(ABYSS_DIR)/dist/*.whl "$$release_dir/"; \
	cp requirements*.txt "$$release_dir/" 2>/dev/null || true; \
	echo "✅ Release artifacts saved to $$release_dir"

release-push: ## Push release to remote (WARNING: This pushes to remote)
	@echo "⚠️  This will push commits and tags to remote. Continue? [y/N]"; \
	read -r confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		echo "Pushing to remote..."; \
		git push origin main; \
		git push origin --tags; \
		echo "✅ Release pushed to remote"; \
	else \
		echo "❌ Push cancelled. Use 'git push origin main && git push origin --tags' manually."; \
	fi

release-dry-run: ## Simulate release process without making changes
	@echo "🧪 Release Dry Run"
	@echo "=================="
	@echo "This would perform the following actions:"
	@echo "1. Run full CI pipeline"
	@echo "2. Validate version consistency"
	@echo "3. Bump version in pyproject.toml"
	@echo "4. Create git tag"
	@echo "5. Build release artifacts"
	@echo "6. Create release directory"
	@echo ""
	@echo "Current state:"
	@$(MAKE) pkg-info
	@echo ""
	@echo "🔒 No actual changes made (dry run mode)"

# Modern Development Workflow Support
pre-commit-install: ## Install pre-commit hooks
	@echo "Installing pre-commit hooks..."
	@if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit install; \
		echo "✅ Pre-commit hooks installed"; \
	else \
		echo "⚠️  pre-commit not installed. Install with: pip install pre-commit"; \
	fi

pre-commit-run: ## Run pre-commit hooks on all files
	@echo "Running pre-commit hooks..."
	@if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit run --all-files; \
	else \
		echo "⚠️  pre-commit not installed. Running manual checks..."; \
		$(MAKE) pkg-format; \
		$(MAKE) pkg-lint; \
	fi

dev-env-setup: ## Setup complete development environment
	@echo "Setting up development environment..."
	@echo "==================================="
	@echo "Installing development dependencies..."
	@pip install --upgrade pip setuptools wheel build
	@pip install pytest flake8 black isort mypy twine pre-commit pip-audit
	@echo "Installing package in editable mode..."
	@$(MAKE) pkg-dev-install
	@echo "Installing pre-commit hooks..."
	@$(MAKE) pre-commit-install
	@echo "✅ Development environment setup complete!"

dev-env-check: ## Check development environment health
	@echo "Checking development environment..."
	@echo "=================================="
	@echo "Python version:"
	@python3 --version
	@echo ""
	@echo "Required tools:"
	@for tool in pip pytest flake8 black isort mypy twine docker git; do \
		if command -v $$tool >/dev/null 2>&1; then \
			echo "✅ $$tool: $$($$tool --version 2>/dev/null | head -1)"; \
		else \
			echo "❌ $$tool: not installed"; \
		fi; \
	done
	@echo ""
	@echo "Optional tools:"
	@for tool in pre-commit pip-audit trivy; do \
		if command -v $$tool >/dev/null 2>&1; then \
			echo "✅ $$tool: available"; \
		else \
			echo "⚠️  $$tool: not installed"; \
		fi; \
	done

dev-reset: ## Reset development environment (clean start)
	@echo "⚠️  This will clean all build artifacts and caches. Continue? [y/N]"; \
	read -r confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		echo "Resetting development environment..."; \
		$(MAKE) clean-all; \
		$(MAKE) docker-clean; \
		$(MAKE) clean-dangling; \
		rm -f .ci_timing requirements.snapshot.txt; \
		echo "✅ Development environment reset complete"; \
	else \
		echo "❌ Reset cancelled"; \
	fi

# Advanced Quality Assurance
qa-full: ## Run comprehensive quality assurance
	@echo "Running comprehensive quality assurance..."
	@echo "========================================="
	@$(MAKE) pkg-format
	@$(MAKE) pkg-lint
	@$(MAKE) validate-config
	@$(MAKE) security-scan
	@$(MAKE) deps-check
	@$(MAKE) test
	@echo "✅ All quality assurance checks passed!"

qa-security: ## Focus on security-specific checks
	@echo "Running security-focused quality assurance..."
	@echo "============================================"
	@$(MAKE) security-scan
	@$(MAKE) deps-check
	@if [ -d ".git" ]; then \
		echo "Checking for secrets in git history..."; \
		if command -v git-secrets >/dev/null 2>&1; then \
			git-secrets --scan; \
		else \
			echo "⚠️  git-secrets not installed for secret scanning"; \
		fi; \
	fi
	@echo "✅ Security checks completed!"

# Build System Health
build-doctor: ## Diagnose build system health
	@echo "🩺 Build System Health Check"
	@echo "============================"
	@echo ""
	@echo "📋 System Information:"
	@echo "OS: $$(uname -s) $$(uname -r)"
	@echo "Architecture: $$(uname -m)"
	@echo "Available memory: $$(free -h 2>/dev/null | awk '/^Mem:/ {print $$2}' || echo 'Unknown')"
	@echo "Available disk: $$(df -h . 2>/dev/null | awk 'NR==2 {print $$4}' || echo 'Unknown')"
	@echo ""
	@$(MAKE) dev-env-check
	@echo ""
	@echo "🔧 Build System Status:"
	@if [ -f "$(BUILD_SCRIPT)" ]; then \
		echo "✅ Build script: $(BUILD_SCRIPT)"; \
	else \
		echo "❌ Build script missing: $(BUILD_SCRIPT)"; \
	fi
	@if [ -f "$(ABYSS_DIR)/pyproject.toml" ]; then \
		echo "✅ Package config: $(ABYSS_DIR)/pyproject.toml"; \
	else \
		echo "❌ Package config missing: $(ABYSS_DIR)/pyproject.toml"; \
	fi
	@echo ""
	@echo "📦 Recent build artifacts:"
	@ls -la $(ABYSS_DIR)/dist/ 2>/dev/null || echo "No build artifacts found"
	@echo ""
	@echo "🐳 Docker images:"
	@$(MAKE) docker-list

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

docker-cpu: check-docker ## 🌟 Build CPU-optimized Docker image (RECOMMENDED)
	@echo "🌟 Building CPU-optimized Docker image (RECOMMENDED BUILD) with caching..."
	@./build-cpu.sh

docker-cpu-fresh: check-docker ## 🌟 Build CPU-optimized Docker image (no cache, RECOMMENDED)
	@echo "🌟 Building CPU-optimized Docker image (RECOMMENDED BUILD) without cache..."
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

quick-start: build docker-cpu ## Build wheel and recommended CPU Docker image
	@echo "🌟 Quick start build completed with RECOMMENDED CPU image!"

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
	@echo "🌟 RECOMMENDED: CPU-optimized builds"
	@echo "  make docker-cpu         # Build CPU image with cache (RECOMMENDED)"
	@echo "  make docker-cpu-fresh   # Build CPU image without cache (RECOMMENDED)"
	@echo ""
	@echo "Cache management:"
	@echo "  make docker-cache-info  # Show cache usage"
	@echo "  make docker-cache-clean # Clean build cache"
	@echo ""
	@echo "Tips:"
	@echo "- 🌟 Use 'make docker-cpu' for most deployments (optimized for CPU)"
	@echo "- Use cached builds for normal development"
	@echo "- Use fresh builds only when dependencies change"
	@echo "- BuildKit cache mounts preserve pip downloads"

# Help target
help: ## Show this help message
	@echo "UOS Drilling Depth Estimation System - Build Targets"
	@echo "=================================================="
	@echo
	@echo "🐍 Python Package Targets:"
	@grep -E '^(build|clean|install|test|validate|pkg-[a-zA-Z_-]*):.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "🐳 Docker Build Targets:"
	@grep -E '^docker-[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "🚀 CI/CD Automation:"
	@grep -E '^(ci|build-matrix|deps-|build-timing):.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "📦 Release Management:"
	@grep -E '^release-[a-zA-Z_-]*:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "🔧 Development Tools:"
	@grep -E '^(dev-|pre-commit|qa-|build-doctor):.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "🛠️ Utility Targets:"
	@grep -E '^(docs-build|security-scan|validate-config|clean-dangling|list-|wheel-info|cache-help):.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "📚 Common Workflows:"
	@echo "  make dev-env-setup  # Setup complete development environment"
	@echo "  make ci             # Run complete CI pipeline"
	@echo "  make pkg-build      # Build Python package"
	@echo "  🌟 make docker-cpu    # Build RECOMMENDED CPU Docker image"
	@echo "  make docker-all     # Build all Docker images"
	@echo "  make qa-full        # Run comprehensive quality checks"
	@echo "  make release-patch  # Create patch release"
	@echo "  make build-doctor   # Diagnose build system health"
	@echo ""
	@echo "🚨 Quick Commands:"
	@echo "  make help           # Show this help"
	@echo "  🌟 make quick-start   # Build package + RECOMMENDED CPU image"
	@echo "  make ci-quick       # Quick CI (lint, test, build)"
	@echo "  make deps-check     # Check dependencies"
	@echo "  make build-timing   # Monitor build performance"