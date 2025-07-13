# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a deep learning drilling depth estimation system that processes MQTT drilling data to perform real-time depth estimation for industrial drilling operations. The system is built around a Python package called `abyss` that combines machine learning models with traditional signal processing techniques for analyzing drilling telemetry data.

## Architecture

### Core Components

- **abyss Package** (`abyss/src/abyss/`): Main Python package containing:
  - **MQTT Module** (`mqtt/`): Handles MQTT message processing, routing, and analysis
  - **Core Processing** (`uos_depth_est_core.py`): Contains depth estimation algorithms using gradient-based methods and ML inference
  - **Data Processing** (`dataparser.py`): Handles loading and parsing of various data formats (XLS, MAT, TDMS, NPZ)
  - **ML Inference** (`uos_inference.py`): Machine learning model interface for depth estimation
  - **Run Scripts** (`run/`): Entry point scripts for different execution modes
  - **Utilities** (`uos_depth_est_utils.py`): Shared utility functions including `find_in_dict` and `reduce_dict`
  - **Legacy Code** (`legacy/`): Historical components moved during v0.2.4 refactoring (25+ modules)

### MQTT Processing Architecture

The MQTT system maintains **two parallel architectures**:

#### Synchronous Components (`mqtt/components/`)
- **DrillingDataAnalyser**: Main orchestrator for sync processing
- **SimpleMessageCorrelator**: Simplified key-based message correlation (67% smaller, 1.68x faster)
- **MessageProcessor**: Extracts data and performs depth estimation
- **ResultPublisher**: Consolidated publishing with exception-based error handling
- **MessageBuffer**: Stores incoming messages for correlation
- **ConfigurationManager**: Centralized config access with topic building
- **ClientManager**: Manages MQTT client connections

#### Asynchronous Components (`mqtt/async_components/`)  
- **MQTTProcessor**: Modern async message processor
- **CorrelationEngine**: Simplified correlation logic using async/await
- **DataConverter**: Async version of DataFrame conversion
- **MessageHandler**: Async message handling with better error management

### Key Data Flow

1. **MQTT Message Reception**: System listens for Result, Trace, and Heads messages
2. **Message Correlation**: Matches messages based on toolbox/tool ID and timestamps
3. **Data Extraction**: Extracts position, torque, thrust, step data, and head_id
4. **Depth Estimation**: Applies gradient-based and/or ML-based analysis
5. **Result Publishing**: Publishes depth estimates with metadata back to MQTT broker

## Development Commands

### Installation
```bash
# Install the package in development mode (from abyss/ directory)
pip install -e .

# Install from requirements
pip install -r abyss/requirements.txt

# Install wheels dependencies
pip install -r abyss/wheels/requirements.txt
```

### Building
```bash
# Build wheel using Makefile (recommended)
make build          # Build wheel with clean and validation
make install        # Build and install wheel locally
make test           # Run tests

# Build wheel using build script
python build_wheel.py --clean --validate

# Build using shell script
./build_wheel.sh --clean --validate --verbose
```

### Running Tests
```bash
# Run pytest from abyss/ directory
pytest

# Run tests with asyncio support
pytest --asyncio-mode=auto

# Run specific test files
pytest tests/test_processing.py
pytest tests/test_result_publisher.py
pytest tests/test_correlation.py
```

### Linting and Formatting
```bash
# Lint code (if tools available)
make lint

# Format code (if tools available)  
make format

# Manual linting
flake8 abyss/src/abyss --max-line-length=120
pylint abyss/src/abyss --disable=C0103,R0903,R0913
```

### Docker Operations
```bash
# Build main Docker image
docker build -t listener .

# Build CPU-optimized image
./build-cpu.sh

# Build development image
docker build -f Dockerfile.devel -t listener-dev .

# Build all configurations
./build-all.sh

# Run Docker container
docker run -t listener

# Run with custom config
docker run -v $(pwd)/config:/app/config -t listener
```

### Executable Entry Points
The package provides several command-line tools:
```bash
# Main MQTT listener for real-time depth estimation
uos_depthest_listener --config config/mqtt_conf_local.yaml --log-level INFO

# Process XLS files for depth estimation
uos_depthest_xls input_file.xls

# Process JSON data for depth estimation
uos_depthest_json input_data.json

# Publish JSON data to MQTT
uos_publish_json data_directory_path
```

## Configuration

### MQTT Configuration
Primary configuration is handled through YAML files:
- `abyss/src/abyss/run/config/mqtt_conf_local.yaml`: Primary local development config
- `abyss/src/abyss/run/config/mqtt_conf_docker.yaml`: Docker-specific config

### Key Configuration Sections
- **Broker Settings**: MQTT broker connection details
- **Topic Structure**: MQTT topic hierarchy for data ingestion
- **Data Field Mappings**: Paths for extracting drilling data from messages including head_id extraction
- **Estimation Output**: Configuration for publishing results

### Head ID Extraction
The system supports extracting head_id from heads messages using the configuration path:
```yaml
mqtt:
  data_ids:
    head_id: 'AssetManagement.Assets.Heads.0.Identification.SerialNumber'
```

## Important Notes

- The system expects specific drilling configurations (typically 2-stack, multi-step drilling)
- ML models are pre-trained and loaded from `abyss/src/abyss/trained_model/`
- Position values are typically inverted (made positive) during processing
- The system supports both real-time MQTT processing and batch file processing
- **Current Version**: 0.2.5 (enhanced duplicate handling and build automation)
- **Legacy Migration**: v0.2.4 moved 25+ legacy modules to `abyss/legacy/` directory

## Code Quality Guidelines

### When Working with This Codebase:
1. **Prefer async components** over sync equivalents
2. **Use `find_in_dict` and `reduce_dict`** from `uos_depth_est_utils` for data extraction
3. **Follow the head_id extraction pattern** established in `MessageProcessor._extract_head_id_simple()`
4. **Use ConfigurationManager** for all config access (standardized)
5. **Use exception-based error handling** - never return boolean success/failure
6. **Use SimpleMessageCorrelator** instead of legacy MessageCorrelator
7. **Include head_id in all MQTT published messages**

### Improved Patterns (Standardized):
- **Exception-based error handling** with proper chaining (`AbyssError` hierarchy)
- **ConfigurationManager usage** for all configuration access
- **Consolidated method patterns** instead of duplicated code  
- **Key-based message correlation** instead of complex time-bucket grouping
- **Type-safe publishing** with enum-based parameterization

### Avoid These Deprecated Patterns:
- ~~Complex time-bucket correlation logic~~ (use `SimpleMessageCorrelator`)
- ~~Multiple nearly-identical methods~~ (use consolidated patterns with enums)
- ~~Direct config dictionary access~~ (use `ConfigurationManager`)
- ~~Boolean return values for error handling~~ (use exceptions)
- ~~Raw dictionary message handling~~

The async components in `mqtt/async_components/` represent the best practices and should be the foundation for all future MQTT-related development.