# Repository Layout Guide

This document explains the structure and organization of the UOS Drilling Depth Estimation System repository.

## Overview

This repository follows a **domain-specific organization** optimized for drilling data analysis research and production deployment. The structure balances research flexibility with production requirements.

## Root Directory Structure

```
📁 uos-drilling-wh/
├── 📁 abyss/                    # Main Python package
├── 📁 examples/                 # Examples, research, notebooks
├── 📁 mqtt-multistack/          # Docker deployment stacks
├── 📁 mqtt-compose/             # Alternative Docker compositions
├── 📁 mongo-compose/            # MongoDB deployment
├── 📁 .devnotes/                # Project documentation
├── 📁 .github/                  # GitHub Actions CI/CD
├── 🛠️ Makefile                   # Build automation
├── 🐳 Dockerfile.*              # Multiple Docker build configurations
├── 🔧 build-*.sh                # Build automation scripts
├── 📄 requirements.txt          # Python dependencies
└── 📋 Various config files      # Root-level configurations
```

## Detailed Directory Breakdown

### 📁 `abyss/` - Main Package
**Purpose**: Core Python package for drilling data analysis

```
abyss/
├── src/abyss/                   # Source code
│   ├── mqtt/                    # MQTT components (refactored)
│   │   ├── components/          # Modular MQTT system components
│   │   └── publishers/          # MQTT publishing tools
│   ├── run/                     # Entry points and execution scripts
│   │   ├── config/              # Configuration files
│   │   └── *.py                 # Various runners (MQTT, XLS, JSON)
│   ├── legacy/                  # Archived/deprecated code
│   ├── trained_model/           # ML model artifacts
│   └── *.py                     # Core modules
├── tests/                       # Comprehensive test suite
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   ├── performance/             # Performance tests
│   └── fixtures/                # Test data and utilities
├── docs/                        # Technical documentation
│   ├── MQTT_ARCHITECTURE.md    # System architecture
│   ├── CONFIGURATION_GUIDE.md  # Configuration reference
│   └── *.md                     # Other technical docs
└── pyproject.toml               # Package configuration
```

**Key Design Decisions**:
- **Modular MQTT**: Refactored from monolithic to component-based architecture
- **Entry Points**: All executable scripts in `run/` directory
- **Legacy Code**: Preserved in `legacy/` for reference
- **Test Organization**: Separated by test type for clarity

### 📁 `examples/` - Examples and Research
**Purpose**: Research code, examples, and experimental work

```
examples/
├── notebooks/                   # Research Jupyter notebooks
│   ├── *.ipynb                  # Various analysis notebooks
│   └── data/                    # Example datasets
├── mqtt/                        # MQTT-specific examples
│   ├── *.ipynb                  # MQTT analysis notebooks  
│   ├── *.py                     # Example scripts
│   └── data/                    # MQTT test data
├── machine-learning/            # ML model development
│   └── abyssml/                 # ML package
│       ├── src/                 # Model source code
│       ├── trained_model/       # Model artifacts
│       └── test_data/           # Training/validation data
├── validation-studies/          # Algorithm validation
│   └── validation/              # Validation results
│       ├── Classification/      # Classification results
│       ├── Curves/              # Curve analysis
│       └── parquet/             # Processed data
├── data-analysis/               # Specific analysis projects
│   └── ti_exit_analysis/        # TI exit analysis
└── build/                       # Example build configurations
    ├── Dockerfile.*             # Example Docker builds
    └── setup_py.backup          # Build artifacts
```

**Design Principles**: 
- **Clear categorization** by purpose (notebooks, MQTT, ML, validation, etc.)
- **Research-friendly** structure for data scientists and researchers
- **Self-contained** examples with associated data and configurations

### 📁 `mqtt-multistack/` - Deployment Configurations
**Purpose**: Docker Swarm and production deployment configurations

```
mqtt-multistack/
├── mqtt-broker/                 # Standalone MQTT broker
├── uos-depthest-listener-cpu/   # CPU-optimized listener
├── uos-publisher-json/          # JSON publisher service  
├── uos-publisher-lightweight/   # Lightweight publisher
├── simple-tracking/             # Signal tracking deployment
├── test-deployment/             # Testing infrastructure
│   ├── docker-compose.*.yml     # Various test configurations
│   └── *.sh                     # Deployment scripts
└── *.sh                         # Stack management scripts
```

**Design Philosophy**:
- **Microservice Architecture**: Each component in separate stack
- **Environment Separation**: Different configs for dev/test/prod
- **Orchestration Ready**: Docker Swarm compatible

### 📁 `.devnotes/` - Comprehensive Documentation
**Purpose**: Project documentation optimized for AI assistance and team knowledge

```
.devnotes/
├── abyss/                       # Package-specific docs
├── deployment/                  # Deployment guides
├── research/                    # Research documentation
└── *.md                         # Various project docs
```

**Special Features**:
- **AI-Optimized**: Structured for Claude/AI assistant consumption
- **Hierarchical**: Reduces content duplication by 90%
- **Cross-Referenced**: Links between related documents

## File Naming Conventions

### Docker Files
- `Dockerfile` - Main production image
- `Dockerfile.cpu` - CPU-optimized build
- `Dockerfile.runtime` - Runtime-only image
- `Dockerfile.devel` - Development image
- `Dockerfile.publish*` - Publishing-specific images

### Build Scripts  
- `build-*.sh` - Docker build automation
- `*.sh` - Various utility scripts
- Pattern: `build-<component>.sh`

### Configuration Files
- `mqtt_conf_*.yaml` - MQTT configurations
- `docker-compose.*.yml` - Docker compositions
- Environment-specific naming (e.g., `docker`, `local`, `production`)

## Navigation Tips

### Finding What You Need

| **Looking For** | **Check These Locations** |
|-----------------|---------------------------|
| **Source Code** | `abyss/src/abyss/` |
| **Configuration** | `abyss/src/abyss/run/config/` |
| **Examples** | `examples/notebooks/`, `examples/mqtt/` |
| **Tests** | `abyss/tests/` |
| **Documentation** | `abyss/docs/` or `.devnotes/` |
| **Deployment** | `mqtt-multistack/` |
| **Build Tools** | `Makefile`, `build-*.sh` |

### Common Patterns

1. **Entry Points**: Look in `abyss/src/abyss/run/` for executable scripts
2. **Configuration**: Environment-specific configs use suffixes (`_docker`, `_local`)
3. **Docker**: Production images use base names, specialized builds use suffixes
4. **Tests**: Organized by type (`unit/`, `integration/`, `performance/`)

## Design Principles

### 1. **Separation of Concerns**
- **Production code**: `abyss/src/`
- **Research code**: `_sandbox/`
- **Deployment**: `mqtt-multistack/`
- **Documentation**: `abyss/docs/` and `.devnotes/`

### 2. **Environment Isolation**
- Different configurations for different deployment environments
- Clear separation between development and production code
- Isolated testing infrastructure

### 3. **Research-Friendly**
- `examples/` provides organized experimental space
- Jupyter notebooks preserved with data
- Easy access to example datasets and validation

### 4. **Production-Ready**
- Modular architecture in main package
- Comprehensive testing infrastructure
- Multiple deployment options
- Build automation with caching

## Evolution Notes

### Recent Changes
- **Refactored MQTT**: Moved from monolithic to modular architecture
- **Improved Testing**: Organized tests by type and added comprehensive fixtures
- **Enhanced Documentation**: Added structured technical documentation
- **Docker Optimization**: Implemented BuildKit caching (90% time savings)

### Legacy Considerations
- **Backward Compatibility**: Old code preserved in `legacy/` directories
- **Migration Path**: Gradual migration from old to new architectures
- **Documentation**: Historic decisions documented for context

## Best Practices for Navigation

1. **Start with GETTING_STARTED.md** for orientation
2. **Use the Makefile** - `make help` shows all available operations
3. **Check specific docs** in `abyss/docs/` for technical details
4. **Browse examples** in `examples/` for usage patterns
5. **Reference `.devnotes/`** for comprehensive project information

## Future Considerations

This layout is designed to support:
- **Scaling**: Easy addition of new components
- **Research**: Flexible space for experimentation  
- **Production**: Robust deployment and testing
- **Collaboration**: Clear separation of different work types

The structure balances the needs of researchers, developers, and operations teams while maintaining the flexibility required for ongoing drilling data analysis research.