# UOS Drilling System: Complete Architecture & Development Reference

## Project Overview

The UOS Drilling Depth Estimation System is a production-grade deep learning platform that processes MQTT drilling data to perform real-time depth estimation for industrial drilling operations. Built around the `abyss` Python package, it combines machine learning models with traditional signal processing for analyzing drilling telemetry data.

**Current State**: v0.2.5 (enhanced duplicate handling and build automation)
**Target Evolution**: v0.3.x (Makefile-first) → v1.0.x (async-only) → v2.0.x (training platform)

## System Architecture

### Core Components Overview

```
abyss/                              # Main Python package
├── src/abyss/
│   ├── mqtt/                       # MQTT processing (dual architecture)
│   │   ├── components/             # Synchronous implementation
│   │   └── async_components/       # Asynchronous implementation (preferred)
│   ├── uos_depth_est_core.py      # Gradient-based depth estimation algorithms
│   ├── dataparser.py              # Multi-format data loading (XLS, MAT, TDMS, NPZ)
│   ├── uos_inference.py           # ML model interface (PatchTSMixer)
│   ├── run/                       # Entry point scripts and configuration
│   ├── trained_model/             # Pre-trained ML models (72 models, 135MB)
│   ├── legacy/                    # Historical components (v0.2.4 migration)
│   └── uos_depth_est_utils.py     # Shared utilities (find_in_dict, reduce_dict)
```

### MQTT Processing Architecture (Dual System)

**Critical Issue**: The system currently maintains two parallel MQTT architectures that need consolidation.

#### Synchronous Components (`mqtt/components/`) - Legacy
- **DrillingDataAnalyser**: Main orchestrator for sync processing
- **SimpleMessageCorrelator**: Key-based message correlation (67% smaller, 1.68x faster)
- **MessageProcessor**: Data extraction and depth estimation
- **ResultPublisher**: Consolidated publishing with exception-based error handling
- **MessageBuffer**: Message storage for correlation
- **ConfigurationManager**: Centralized config access with topic building
- **ClientManager**: MQTT client connection management

#### Asynchronous Components (`mqtt/async_components/`) - Preferred
- **MQTTProcessor**: Modern async message processor
- **CorrelationEngine**: Simplified correlation logic using async/await
- **DataConverter**: Async DataFrame conversion
- **MessageHandler**: Async message handling with improved error management

**Migration Priority**: Async components represent best practices and should be the foundation for all future development.

### Data Flow Architecture

```
MQTT Message Reception
    ↓
Message Correlation (by toolbox/tool ID + timestamps)
    ↓
Data Extraction (position, torque, thrust, step, head_id)
    ↓
Depth Estimation (gradient + ML inference)
    ↓
Result Publishing (with metadata to MQTT broker)
```

### ML Model Architecture

**Current Models**: 72 PatchTSMixer models
- **Cross-Validation**: 24 folds
- **File Types**: 3 drilling configurations  
- **Total Size**: 135MB (causes Docker image bloat to 10-13GB)
- **Target**: ONNX migration for 80-85% size reduction

**Inference Pipeline**:
- Real-time MQTT processing
- Batch file processing (XLS, JSON)
- Three-point depth estimation (entry, transition, exit)

## Development Environment

### Installation Commands
```bash
# Development installation (from abyss/ directory)
pip install -e .

# Requirements installation
pip install -r abyss/requirements.txt
pip install -r abyss/wheels/requirements.txt
```

### Build System (Current + Target)

**Current State**: Mixed build system
- Makefile (partial coverage)
- Python scripts (`build_wheel.py`)
- Shell scripts (`build_wheel.sh`, `build-cpu.sh`, `build-all.sh`)

**Target v0.3.x**: Makefile-first approach
```bash
# Proposed unified commands
make build          # Build wheel with clean and validation
make install        # Build and install wheel locally
make test           # Run comprehensive test suite
make lint           # Code quality checks
make format         # Code formatting
make docker         # Build all Docker configurations
make deploy         # Deploy to target environment
```

### Testing Strategy

**Current Testing**:
```bash
# Pytest execution
pytest                              # All tests
pytest --asyncio-mode=auto         # Async test support
pytest tests/test_processing.py    # Specific modules
pytest tests/test_result_publisher.py
pytest tests/test_correlation.py
```

**Testing Challenges**:
- Parallel test suites for sync/async architectures
- Integration testing across MQTT components
- ML model inference validation

### Docker Architecture

**Current Images**:
- **Main**: `listener` (10-13GB with PyTorch)
- **CPU-optimized**: ONNX-based inference (target: 2GB)
- **Development**: `listener-dev` with debug tools

**Build Commands**:
```bash
docker build -t listener .                    # Main image
./build-cpu.sh                               # CPU-optimized
docker build -f Dockerfile.devel -t listener-dev .  # Development
./build-all.sh                               # All configurations
```

**Container Optimization Targets**:
- 80-85% size reduction via ONNX migration
- Improved startup time
- Reduced memory footprint

## Configuration Management

### MQTT Configuration
**Primary Configs**:
- `abyss/src/abyss/run/config/mqtt_conf_local.yaml`: Local development
- `abyss/src/abyss/run/config/mqtt_conf_docker.yaml`: Docker deployment

**Configuration Sections**:
```yaml
broker:               # MQTT broker connection details
topic_structure:      # MQTT topic hierarchy
data_field_mappings:  # Drilling data extraction paths
estimation_output:    # Result publishing configuration
data_ids:
  head_id: 'AssetManagement.Assets.Heads.0.Identification.SerialNumber'
```

### Configuration Access Patterns

**Preferred**: Use `ConfigurationManager` for all config access
```python
# Good: Centralized configuration access
config_manager = ConfigurationManager(config_path)
broker_config = config_manager.get_broker_config()

# Avoid: Direct dictionary access
config = yaml.load(config_file)
broker_host = config['mqtt']['broker']['host']  # Deprecated
```

## Code Quality Guidelines

### Preferred Patterns (v0.2.5+)

#### 1. **Async-First Development**
```python
# Prefer async components
from abyss.mqtt.async_components import MQTTProcessor, CorrelationEngine

# Avoid sync equivalents unless necessary
# from abyss.mqtt.components import DrillingDataAnalyser  # Legacy
```

#### 2. **Utility Function Usage**
```python
from abyss.uos_depth_est_utils import find_in_dict, reduce_dict

# Extract data using standardized utilities
head_id = find_in_dict(message_data, 'AssetManagement.Assets.Heads.0.Identification.SerialNumber')
drilling_data = reduce_dict(raw_data, required_fields)
```

#### 3. **Exception-Based Error Handling**
```python
# Good: Exception-based flow control
try:
    result = process_drilling_data(data)
    publish_result(result)
except AbyssError as e:
    logger.error(f"Processing failed: {e}")
    handle_error(e)

# Avoid: Boolean return patterns
# success = process_drilling_data(data)  # Deprecated
# if not success: ...
```

#### 4. **Centralized Configuration**
```python
# Good: ConfigurationManager usage
config_manager = ConfigurationManager(config_path)
topics = config_manager.build_topics()

# Avoid: Direct config access
# topics = config['mqtt']['topics']  # Deprecated
```

#### 5. **Modern Message Correlation**
```python
# Good: SimpleMessageCorrelator (67% smaller, 1.68x faster)
correlator = SimpleMessageCorrelator()
correlated_messages = correlator.correlate_by_key(messages)

# Avoid: Legacy MessageCorrelator with time-bucket complexity
```

### Deprecated Patterns (Avoid)

❌ **Complex time-bucket correlation logic** → Use `SimpleMessageCorrelator`
❌ **Multiple nearly-identical methods** → Use consolidated patterns with enums
❌ **Direct config dictionary access** → Use `ConfigurationManager`
❌ **Boolean return values for error handling** → Use exceptions with `AbyssError` hierarchy
❌ **Raw dictionary message handling** → Use structured data extraction utilities

## Deployment Architecture

### Portainer Multi-Stack Pattern

**Current Deployment**:
```
mqtt-multistack/
├── mqtt-broker/                    # Mosquitto MQTT broker
├── uos-depthest-listener-runtime/ # Runtime inference container
├── uos-depthest-listener-cpu/     # CPU-optimized inference
└── uos-publisher-json/            # Data publishing service
```

**Stack Configuration**:
- **Named Volumes**: Persistent configuration and data storage
- **External Networks**: `mqtt-broker_toolbox-network`
- **Resource Limits**: CPU/memory constraints for production

### Entry Points & Execution

**Command-Line Tools**:
```bash
# Real-time MQTT processing
uos_depthest_listener --config config/mqtt_conf_local.yaml --log-level INFO

# Batch processing
uos_depthest_xls input_file.xls        # XLS file processing
uos_depthest_json input_data.json      # JSON data processing
uos_publish_json data_directory_path   # MQTT data publishing
```

## Technical Debt & Migration Priorities

### Priority 1: Build System Consolidation (v0.3.x)
**Current Issues**:
- Mixed build tools (Makefile + Python + Shell scripts)
- Inconsistent command interfaces
- Manual dependency management

**Target Solution**:
- Unified Makefile-based build system
- Standardized command interface
- Automated dependency validation

### Priority 2: Architecture Unification (v1.0.x)
**Current Issues**:
- Dual sync/async MQTT architectures
- Code duplication across parallel systems
- Testing complexity

**Target Solution**:
- Async-only architecture
- Consolidated MQTT processing
- Simplified testing strategy

### Priority 3: Container Optimization (v1.0.x)
**Current Issues**:
- 10-13GB Docker images due to PyTorch
- Slow startup times
- Resource inefficiency

**Target Solution**:
- ONNX-based inference (80-85% size reduction)
- Optimized container layers
- Improved resource utilization

### Priority 4: Training Platform Integration (v2.0.x)
**Future Enhancement**:
- End-user training capabilities
- GUI annotation tools
- PyTorch training + ONNX deployment dual architecture
- MLflow + DVC integration

## Development Workflow Best Practices

### 1. **Branch Strategy**
- Use async components for new features
- Test both sync/async during transition period
- Maintain backward compatibility during migration

### 2. **Testing Requirements**
- All new code must include pytest coverage
- Async components require `--asyncio-mode=auto`
- Integration tests for MQTT message flow

### 3. **Code Review Checklist**
- ✅ Uses ConfigurationManager for config access
- ✅ Implements exception-based error handling
- ✅ Includes head_id in MQTT publications
- ✅ Uses async components where possible
- ✅ Follows established utility patterns

### 4. **Performance Considerations**
- Monitor message correlation performance
- Validate ML inference latency
- Track Docker image size evolution
- Measure MQTT throughput under load

## Future Development Context

This system is evolving toward a comprehensive training platform where end-users can retrain models using their own drilling data. Current architecture decisions (async-first, configuration management, error handling patterns) establish the foundation for this expanded capability.

**Key Evolution Path**:
- **v0.3.x**: Unified build system and deployment
- **v1.0.x**: Async-only architecture with ONNX optimization  
- **v2.0.x**: Integrated training platform with GUI tools

All development should consider this evolution path and maintain patterns that support future training platform integration.