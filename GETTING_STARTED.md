# Getting Started with UOS Drilling Depth Estimation System

Welcome to the UOS Drilling Depth Estimation System! This guide will help you get oriented and productive quickly.

## Quick Navigation

### 🎯 I want to...

| **Goal** | **Go Here** | **Description** |
|----------|-------------|-----------------|
| **Run the system** | [`abyss/docs/CONFIGURATION_GUIDE.md`](abyss/docs/CONFIGURATION_GUIDE.md) | Configure and run MQTT depth estimation |
| **Understand the architecture** | [`abyss/docs/MQTT_ARCHITECTURE.md`](abyss/docs/MQTT_ARCHITECTURE.md) | System design and components |
| **See examples & notebooks** | [`examples/`](examples/) | Jupyter notebooks, examples, research code |
| **Deploy with Docker** | [`mqtt-multistack/`](mqtt-multistack/) | Docker deployments and orchestration |
| **Run tests** | [`abyss/tests/`](abyss/tests/) | Test suite and validation |
| **Build the system** | [`Makefile`](Makefile) + `make help` | Build targets and automation |
| **Development setup** | [DEVELOPERS.md](DEVELOPERS.md) | Development workflow and tools |

## For Different User Types

### 🔬 **Researchers & Data Scientists**
- **Start here**: [`examples/notebooks/`](examples/notebooks/) - Jupyter notebooks and research examples
- **Data**: [`examples/validation-studies/`](examples/validation-studies/) - Validation datasets and analysis
- **Models**: [`examples/machine-learning/`](examples/machine-learning/) - ML models and training code

### 🚀 **Operations & Deployment**
- **Docker**: [`mqtt-multistack/`](mqtt-multistack/) - Production deployment configurations
- **Build**: `make help` - See all build and deployment targets
- **Config**: [`abyss/src/abyss/run/config/`](abyss/src/abyss/run/config/) - Configuration files

### 👩‍💻 **Developers**
- **Source**: [`abyss/src/abyss/`](abyss/src/abyss/) - Main package source code
- **Tests**: [`abyss/tests/`](abyss/tests/) - Test suite
- **Build**: `make dev-install` - Development installation
- **Architecture**: [`abyss/docs/MQTT_ARCHITECTURE.md`](abyss/docs/MQTT_ARCHITECTURE.md)

## Quick Start Commands

### 1. Install and Build
```bash
# Install in development mode
make dev-install

# Build Python wheel
make build

# Build Docker images (cached)
make docker-all
```

### 2. Run the System
```bash
# Configure first (see Configuration Guide)
# Then run MQTT listener
python -m abyss.run.uos_depth_est_mqtt --config /path/to/config.yaml

# Or with Docker
docker-compose -f mqtt-multistack/uos-depthest-listener-cpu/docker-compose.cpu.yml up
```

### 3. Test and Validate
```bash
# Run tests
make test

# Run specific test category
cd abyss && python -m pytest tests/unit/ -v
```

## Repository Structure Overview

```
📁 Repository Root
├── 📁 abyss/                    # Main package
│   ├── 📁 src/abyss/            # Source code
│   ├── 📁 tests/                # Test suite
│   └── 📁 docs/                 # Technical documentation
├── 📁 examples/                 # Examples, notebooks, research
├── 📁 mqtt-multistack/          # Docker deployment configurations  
├── 📁 .devnotes/                # Comprehensive project documentation
├── 🛠️ Makefile                   # Build automation
└── 📋 Multiple build scripts     # Docker build automation
```

For detailed structure explanation, see [REPOSITORY_LAYOUT.md](REPOSITORY_LAYOUT.md).

## Common Tasks

### Development Workflow
```bash
# 1. Make changes to source code
# 2. Run tests
make test

# 3. Build and test locally
make build
make docker-main

# 4. Deploy for testing
cd mqtt-multistack/test-deployment/
./deploy-test-stack.sh
```

### Working with Examples
```bash
# Start Jupyter for notebooks
cd examples/notebooks/
jupyter lab

# Run MQTT examples
cd examples/mqtt/
python listen-continuous.py
```

### Troubleshooting
- **Build issues**: `make clean && make build`
- **Docker issues**: `make docker-clean && make docker-all`
- **Config issues**: See [`abyss/docs/CONFIGURATION_GUIDE.md`](abyss/docs/CONFIGURATION_GUIDE.md)
- **Architecture questions**: See [`abyss/docs/MQTT_ARCHITECTURE.md`](abyss/docs/MQTT_ARCHITECTURE.md)

## Additional Resources

- **📚 Full Documentation**: [`.devnotes/`](.devnotes/) - Comprehensive project docs
- **🏗️ Architecture**: [`abyss/docs/MQTT_ARCHITECTURE.md`](abyss/docs/MQTT_ARCHITECTURE.md)
- **⚙️ Configuration**: [`abyss/docs/CONFIGURATION_GUIDE.md`](abyss/docs/CONFIGURATION_GUIDE.md)
- **🐳 Docker Guide**: [`docs/docker-caching-guide.md`](docs/docker-caching-guide.md)
- **🧪 Test Strategy**: [`abyss/tests/README_TEST_STRATEGY.md`](abyss/tests/README_TEST_STRATEGY.md)

## Need Help?

1. **Check the documentation** in [`.devnotes/`](.devnotes/) 
2. **Run with verbose logging**: Add `--log-level DEBUG` to commands
3. **Review configuration**: See [`abyss/docs/CONFIGURATION_GUIDE.md`](abyss/docs/CONFIGURATION_GUIDE.md)
4. **Check test examples**: [`abyss/tests/`](abyss/tests/) for usage patterns

---

**Next Steps**: 
- 📖 Read [REPOSITORY_LAYOUT.md](REPOSITORY_LAYOUT.md) for detailed structure
- 🛠️ See [DEVELOPERS.md](DEVELOPERS.md) for development workflows  
- 🚀 Check specific documentation in [`abyss/docs/`](abyss/docs/) for your use case