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

### MQTT Processing Architecture

The MQTT system has undergone significant refactoring and currently maintains **two parallel architectures**:

#### Synchronous Components (`mqtt/components/`)
- **DrillingDataAnalyser**: Main orchestrator for sync processing
- **MessageCorrelator**: Complex time-bucket based message correlation 
- **MessageProcessor**: Extracts data and performs depth estimation
- **DataFrameConverter**: Converts MQTT messages to pandas DataFrames
- **ResultPublisher**: Publishes depth estimation results back to MQTT
- **MessageBuffer**: Stores incoming messages for correlation
- **ConfigurationManager**: Handles YAML configuration loading and validation
- **MQTTClientManager**: Manages multiple MQTT client connections

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

### Running Tests
```bash
# Run pytest from abyss/ directory
pytest

# Run tests with asyncio support
pytest --asyncio-mode=auto

# Run specific test files
pytest tests/test_mqtt.py
pytest tests/test_processing.py
pytest tests/test_result_publisher_head_id.py
pytest tests/test_message_processor_heads_id.py
```

### Docker Operations
```bash
# Build main Docker image
docker build -t listener .

# Build development image
docker build -f Dockerfile.devel -t listener-dev .

# Build runtime image  
docker build -f Dockerfile.runtime -t listener-runtime .

# Run Docker container
docker run -t listener

# Run with custom config
docker run -v $(pwd)/config:/app/config -t listener
```

### Executable Entry Points
The package provides several command-line tools:
```bash
# Main MQTT listener for real-time depth estimation
uos_depthest_listener --config mqtt_conf.yaml --log-level INFO

# Process XLS files for depth estimation
uos_depthest_xls input_file.xls

# Process JSON data for depth estimation
uos_depthest_json input_data.json

# Publish JSON data to MQTT
uos_publish_json data.json

# Run async MQTT processor (newer implementation)
uos_depth_est_mqtt --config mqtt_conf.yaml
```

## Configuration

### MQTT Configuration
Primary configuration is handled through YAML files (typically `mqtt_conf.yaml`):
- `abyss/src/abyss/run/mqtt_conf.yaml`: Main configuration
- `abyss/src/abyss/run/config/mqtt_conf_docker.yaml`: Docker-specific config
- `abyss/src/abyss/run/config/mqtt_conf_local.yaml`: Local development config
- Multiple environment-specific configs in `abyss/sandbox/run/`

### Key Configuration Sections
- **Broker Settings**: MQTT broker connection details
- **Topic Structure**: MQTT topic hierarchy for data ingestion
- **Data Field Mappings**: Paths for extracting drilling data from messages including head_id extraction
- **Estimation Output**: Configuration for publishing results

### Head ID Extraction
The system now supports extracting head_id from heads messages using the configuration path:
```yaml
mqtt:
  data_ids:
    head_id: 'AssetManagement.Assets.Heads.0.Identification.SerialNumber'
```

Head_id is extracted using the simple `reduce_dict` utility and included in all published MQTT results.

## Data Processing Workflow

### Message Processing Pipeline
```
MQTT Message → Message Buffer → Correlation → Data Extraction → Depth Estimation → Result Publishing
```

### Head ID Integration
- Head_id is extracted immediately after message correlation using `reduce_dict(heads_data, head_id_path)`
- Included in all published results: successful estimation, insufficient data, and error scenarios
- Uses the `reduce_dict` function from `uos_depth_est_utils.py` for consistent data extraction

## Key Data Formats

- **XLS Files**: Setitec drilling data with columns for Position, Torque, Thrust, Step
- **MQTT Messages**: JSON-formatted messages containing drilling telemetry
- **Output Format**: Three-point estimation (entry, transition, exit positions) with head_id metadata

## Testing Data Locations

- `abyss/test_data/`: Main test datasets organized by source (HAM, MDB, TUHH, UNK)
- `abyss/sandbox/test_data/`: Additional test datasets
- `abyss/src/abyss/dev_data/`: Development reference data
- `abyss/src/abyss/test_data/data_20250326/`: Specific test dataset used for `uos_publish_json` testing

## Docker Compose Environments

- `mqtt-compose/`: MQTT broker setup with Mosquitto
- `mongo-compose/`: MongoDB integration setup  
- `mqtt-multistack/`: Multi-service MQTT environment with broker, listener, and publisher

## Important Notes

- The system expects specific drilling configurations (typically 2-stack, multi-step drilling)
- ML models are pre-trained and loaded from `abyss/src/abyss/trained_model/`
- Position values are typically inverted (made positive) during processing
- The system supports both real-time MQTT processing and batch file processing
- Testing requires specific data formats - see README.md for data structure requirements

## Codebase Analysis and Simplification Opportunities

### Current Architecture Issues

#### 1. **Dual Architecture Complexity** ⚠️
The codebase maintains two parallel MQTT processing architectures:
- **Synchronous components** (`mqtt/components/`): Legacy implementation with complex correlation logic
- **Asynchronous components** (`mqtt/async_components/`): Cleaner, modern implementation

**Recommendation**: Standardize on the async architecture and deprecate sync components.

#### 2. **Overly Complex Message Correlation** 🔴
The sync `MessageCorrelator` (367 lines) implements unnecessary time-bucket grouping:
```python
def _group_messages_by_time_bucket(self, result_messages, trace_messages, heads_messages):
    # Complex nested approach that should be simple key-based matching
```

**Simplification**: The async `CorrelationEngine` (83 lines) demonstrates a much cleaner approach.

#### 3. **Result Publisher Redundancy** 🔴
Three nearly identical methods in `ResultPublisher`:
- `_publish_successful_result()` 
- `_publish_insufficient_data_result()`
- `_publish_error_result()`

**Simplification**: Create a single parameterized publish method to eliminate 90% code duplication.

#### 4. **Configuration Pattern Inconsistency**
Multiple config access patterns throughout codebase:
```python
# Pattern 1: Direct access
self.config['mqtt']['listener']['result']

# Pattern 2: Pre-computed patterns  
self._topic_patterns = {...}

# Pattern 3: Config manager
config_manager.get_mqtt_broker_config()
```

**Recommendation**: Standardize on ConfigurationManager pattern only.

### Test Suite Complexity

#### Issues Identified:
1. **Mock Complexity**: Excessive mocking setup in sync component tests
2. **Duplication**: Parallel test suites for sync and async implementations
3. **Missing Integration**: No end-to-end pipeline tests

#### Simplification Opportunities:
1. Focus testing on async components
2. Add simple integration tests for full MQTT pipeline
3. Reduce mock complexity by using dependency injection

### Data Handling Inconsistencies

#### Message Format Issues:
- `TimestampedData` (legacy sync)
- `Message` (new async)  
- Raw dictionaries (various places)

**Recommendation**: Standardize on the async `Message` model.

#### Error Handling Patterns:
```python
# Inconsistent approaches:
raise ConfigurationError(...)  # Exception-based
return False                   # Boolean return
return None                    # None return
```

**Recommendation**: Use exception-based error handling consistently.

## Next Steps and Priorities

### Immediate Actions (High Priority)
1. **Choose One Architecture**: Migrate remaining functionality to async components
2. **Simplify Message Correlation**: Replace time-bucket approach with simple key matching
3. **Consolidate Result Publishing**: Create single parameterized publish method
4. **Standardize Configuration Access**: Use ConfigurationManager everywhere

### Medium-term Improvements
1. **Unified Message Types**: Complete migration to async `Message` model
2. **Integration Testing**: Add end-to-end MQTT pipeline tests
3. **Dependency Injection**: Implement proper IoC to reduce coupling
4. **Error Handling**: Standardize on exception-based patterns

### Long-term Architectural Goals
1. **Plugin Architecture**: Make depth estimation algorithms pluggable
2. **Configuration Hot-reload**: Support runtime configuration updates
3. **Observability**: Add metrics, tracing, and health monitoring
4. **Streaming Support**: Consider reactive streams for high-throughput scenarios

## Code Quality Guidelines

### When Working with This Codebase:
1. **Prefer async components** over sync equivalents
2. **Use `find_in_dict` and `reduce_dict`** from `uos_depth_est_utils` for data extraction
3. **Follow the head_id extraction pattern** established in `MessageProcessor._extract_head_id_simple()`
4. **Test both success and error scenarios** with proper head_id inclusion
5. **Avoid adding new sync components** - extend async architecture instead
6. **Use ConfigurationManager** for all config access
7. **Include head_id in all MQTT published messages**

### Avoid These Patterns:
- Complex time-bucket correlation logic
- Multiple nearly-identical methods (prefer parameterization)
- Direct config dictionary access
- Raw dictionary message handling
- Excessive mocking in tests

The async components in `mqtt/async_components/` represent the best practices and should be the foundation for all future MQTT-related development.
