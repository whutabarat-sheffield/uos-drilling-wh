
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

The MQTT system has undergone significant refactoring and currently maintains **two parallel architectures**:

#### Synchronous Components (`mqtt/components/`)
- **DrillingDataAnalyser**: Main orchestrator for sync processing (✅ Updated with ConfigurationManager)
- **SimpleMessageCorrelator**: ✅ **NEW** - Simplified key-based message correlation (67% smaller, 1.68x faster)
- **MessageCorrelator**: Legacy complex time-bucket based correlation (deprecated)
- **MessageProcessor**: Extracts data and performs depth estimation
- **DataFrameConverter**: Converts MQTT messages to pandas DataFrames
- **ResultPublisher**: ✅ **ENHANCED** - Consolidated publishing with exception-based error handling
- **MessageBuffer**: Stores incoming messages for correlation
- **ConfigurationManager**: ✅ **ENHANCED** - Centralized config access with topic building
- **MQTTClientManager**: Manages multiple MQTT client connections
- **Exception Hierarchy**: ✅ **NEW** - Comprehensive error handling system (`exceptions.py`)

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
pytest tests/test_processing.py
pytest tests/test_result_publisher_head_id.py
pytest tests/test_message_processor_heads_id.py
pytest tests/mqtt/components/  # Component-specific tests
pytest tests/test_correlation.py
pytest tests/test_uos_publish_json.py
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
# Main MQTT listener for real-time depth estimation (sync components)
uos_depthest_listener --config config/mqtt_conf_local.yaml --log-level INFO

# Process XLS files for depth estimation
uos_depthest_xls input_file.xls

# Process JSON data for depth estimation
uos_depthest_json input_data.json

# Publish JSON data to MQTT
uos_publish_json data_directory_path

# Run async MQTT processor (newer implementation)
uos_depth_est_mqtt --config config/mqtt_conf_local.yaml
```

## Configuration

### MQTT Configuration
Primary configuration is handled through YAML files:
- `abyss/src/abyss/run/config/mqtt_conf_local.yaml`: Primary local development config
- `abyss/src/abyss/run/config/mqtt_conf_docker.yaml`: Docker-specific config
- Multiple environment-specific configs in `abyss/sandbox/run/`

**Note**: The main configuration files are located in the `config/` subdirectory, not at the run root level.

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

Head_id is extracted using the `_extract_head_id_simple()` method with `reduce_dict` utility and included in all published MQTT results.

## Data Processing Workflow

### Message Processing Pipeline
```
MQTT Message → Message Buffer → Correlation → Data Extraction → Depth Estimation → Result Publishing
```

### Head ID Integration
- Head_id is extracted immediately after message correlation using `_extract_head_id_simple()` method
- Handles both string (JSON) and dict data formats automatically
- Included in all published results: successful estimation, insufficient data, and error scenarios
- Uses the `reduce_dict` function from `uos_depth_est_utils.py` for consistent data extraction
- Implementation in `MessageProcessor._extract_head_id_simple()` supports path-based extraction

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

- `mqtt-compose/`: MQTT broker setup with Mosquitto and `uos-depthest-listener` service
  - Provides eclipse-mosquitto MQTT broker on ports 1883 (MQTT) and 9001 (WebSocket)
  - Includes integrated depth estimation listener service
  - Uses external `toolbox-network` for service communication
- `mongo-compose/`: MongoDB integration setup  
- `mqtt-multistack/`: **✅ TESTED** Multi-service MQTT environment with separate Portainer stacks

### MQTT Multi-Stack Architecture (Portainer-Ready)

The `mqtt-multistack/` directory contains production-ready Docker Compose configurations designed for deployment as separate Portainer stacks:

#### 1. **MQTT Broker Stack** (`mqtt-broker/`)
- **Service**: `eclipse-mosquitto:latest` MQTT broker
- **Ports**: 1883 (MQTT), 9001 (WebSocket), 8883 (MQTT over TLS)
- **Persistence**: Named volumes for config, data, and logs
- **Configuration**: Full logging enabled with timestamp support
- **Network**: `toolbox-network` for inter-service communication
- **Features**: Anonymous connections allowed, WebSocket support enabled

#### 2. **Depth Estimation Listener Stack** (`uos-depthest-listener/`)
- **Service**: `uos-depthest-listener:latest` (sync components)
- **Purpose**: Real-time depth estimation from MQTT drilling data
- **Environment**: `LOG_LEVEL=DEBUG`, `PYTHONUNBUFFERED=1`
- **Network**: Connected to `mqtt-broker_toolbox-network` (external)
- **Persistence**: Named volume for configuration storage

#### 3. **Runtime Listener Stack** (`uos-depthest-listener-runtime/`)
- **Service**: `uos-depthest-listener-runtime:latest` (optimized runtime)
- **Purpose**: Production-optimized depth estimation listener
- **Configuration**: Identical to development listener but with runtime image
- **Network**: Connected to `mqtt-broker_toolbox-network` (external)

#### 4. **JSON Publisher Stack** (`uos-publisher-json/`)
- **Service**: `uos-publish-json:latest`
- **Purpose**: Publishes test drilling data from JSON files to MQTT
- **Environment**: `LOG_LEVEL=DEBUG`, `PYTHONUNBUFFERED=1`
- **Network**: Connected to `mqtt-broker_toolbox-network` (external)
- **Use Case**: Testing and data simulation

#### Stack Deployment Notes:
- **Portainer Compatibility**: Each stack designed as separate Portainer deployment
- **Named Volumes**: Uses Portainer-compatible named volumes for persistence
- **External Networks**: Services communicate via shared `toolbox-network`
- **Volume Configuration**: Config files must be copied to named volumes:
  ```bash
  docker run --rm -v mosquitto_config:/config \
    -v /path/to/mqtt-multistack/mqtt-broker/config:/src \
    alpine cp /src/mosquitto.conf /config/
  ```
- **Interactive Mode**: All services run with `stdin_open: true` and `tty: true`
- **Auto-restart**: Services configured with `restart: always` for production

## Important Notes

- The system expects specific drilling configurations (typically 2-stack, multi-step drilling)
- ML models are pre-trained and loaded from `abyss/src/abyss/trained_model/`
- Position values are typically inverted (made positive) during processing
- The system supports both real-time MQTT processing and batch file processing
- Testing requires specific data formats - see README.md for data structure requirements
- **Current Version**: 0.2.4 (major refactoring with component architecture)
- **Current Branch**: 0.2.4-heads (includes head_id integration)
- **Legacy Migration**: v0.2.4 moved 25+ legacy modules to `abyss/legacy/` directory

## Codebase Analysis and Simplification Opportunities

### Current Architecture Issues

#### 1. **Dual Architecture Complexity** ⚠️
The codebase maintains two parallel MQTT processing architectures:
- **Synchronous components** (`mqtt/components/`): Legacy implementation with complex correlation logic
- **Asynchronous components** (`mqtt/async_components/`): Cleaner, modern implementation

**Recommendation**: Standardize on the async architecture and deprecate sync components.

#### 2. **✅ Message Correlation Simplified** - RESOLVED
~~The sync `MessageCorrelator` (367 lines) implements unnecessary time-bucket grouping~~
- **SOLUTION IMPLEMENTED**: `SimpleMessageCorrelator` (204 lines) with key-based matching
- **Performance improved**: 1.68x faster with O(n) complexity vs O(n²)
- **Maintained compatibility**: Legacy correlator still available during transition

#### 3. **✅ Result Publisher Redundancy** - RESOLVED  
~~Three nearly identical methods in `ResultPublisher`~~
- **SOLUTION IMPLEMENTED**: Single `_publish_result_consolidated()` method with `PublishResultType` enum
- **Code reduction**: Eliminated ~150 lines of duplicated code
- **Type safety**: Enum-based parameterization prevents errors

#### 4. **✅ Configuration Pattern Inconsistency** - RESOLVED
~~Multiple config access patterns throughout codebase~~
- **SOLUTION IMPLEMENTED**: StandardizedConfigurationManager usage across all components
- **Enhanced functionality**: Added `build_result_topic()` and other helper methods
- **Backward compatibility**: Legacy patterns still supported during migration

#### 5. **✅ Error Handling Inconsistency** - RESOLVED
~~Inconsistent error handling patterns (exceptions vs boolean returns vs None returns)~~
- **SOLUTION IMPLEMENTED**: Comprehensive exception hierarchy with proper exception chaining
- **Eliminated anti-patterns**: No more boolean success/failure returns
- **Enhanced debugging**: Full error context preserved through exception chains

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

### ✅ Recently Completed Improvements (v0.2.4+)

#### 1. **✅ Simplify Message Correlation** - COMPLETED
- **Replaced complex time-bucket approach** with simple key-based matching in `SimpleMessageCorrelator`
- **67% code reduction**: From 367 lines (MessageCorrelator) to 204 lines (SimpleMessageCorrelator)  
- **1.68x performance improvement** with O(n) vs O(n²) complexity
- **Full backward compatibility** maintained during transition
- **Comprehensive test coverage** with performance comparison tests

#### 2. **✅ Consolidate Result Publishing** - COMPLETED  
- **Eliminated code duplication** with consolidated `_publish_result_consolidated()` method
- **Added type-safe publishing** with `PublishResultType` enum (SUCCESS, INSUFFICIENT_DATA, ERROR)
- **Single parameterized method** replaces three nearly identical methods
- **Maintained full compatibility** while reducing maintenance burden
- **Enhanced error handling** with proper exception propagation

#### 3. **✅ Standardize Configuration Access** - COMPLETED
- **ConfigurationManager usage standardized** across all core components
- **Enhanced ConfigurationManager** with `build_result_topic()` method for centralized topic building
- **Updated components**: `ResultPublisher`, `SimpleMessageCorrelator`, `DrillingDataAnalyser`
- **Backward compatibility maintained** with legacy raw config dictionary support
- **Centralized validation** and error handling for all configuration access

#### 4. **✅ Exception-Based Error Handling** - COMPLETED
- **Comprehensive exception hierarchy** created with base `AbyssError` and specific exceptions:
  - `MQTTPublishError`, `AbyssProcessingError`, `MessageValidationError`, etc.
- **Eliminated boolean return patterns** in favor of proper exception propagation  
- **Exception utility functions** added: `wrap_exception()`, `is_recoverable_error()`, `get_error_severity()`
- **ResultPublisher completely refactored** to use exception-based error handling
- **Exception chaining preserved** to maintain full error context
- **Comprehensive test coverage** for all exception scenarios

### Immediate Actions (High Priority)
1. **Choose One Architecture**: Migrate remaining functionality to async components

### Medium-term Improvements
1. **Unified Message Types**: Complete migration to async `Message` model
2. **Integration Testing**: Add end-to-end MQTT pipeline tests
3. **Dependency Injection**: Implement proper IoC to reduce coupling (⭐ **NEXT PRIORITY**)

### 🎯 Dependency Injection Implementation Plan

#### Current Coupling Issues
The current architecture has several tight coupling patterns that impede testing and flexibility:

```python
# Problem: DrillingDataAnalyser directly instantiates dependencies
class DrillingDataAnalyser:
    def _initialize_components(self):
        self.message_correlator = SimpleMessageCorrelator(config, time_window)  # Hard-coded
        self.message_processor = MessageProcessor(depth_inference, data_converter, config)
        self.depth_inference = DepthInference()  # No interface
        self.client_manager = MQTTClientManager(config, message_handler)
```

#### Proposed Solution: IoC Container + Interface Abstractions

##### Phase 1: Define Abstraction Interfaces
```python
# Core service interfaces
class IMessageCorrelator(Protocol):
    def find_and_process_matches(self, buffers: Dict, processor: Callable) -> bool: ...

class IResultPublisher(Protocol):
    def publish_processing_result(self, result: ProcessingResult, ...) -> None: ...

class IDepthInference(Protocol):
    def estimate_depth(self, data: DataFrame) -> EstimationResult: ...

class IMQTTClientFactory(Protocol):
    def create_result_listener(self) -> mqtt.Client: ...
```

##### Phase 2: Lightweight IoC Container
```python
class ServiceContainer:
    def __init__(self):
        self._services: Dict[type, Any] = {}
        self._factories: Dict[type, Callable] = {}
    
    def register_singleton(self, interface: type, implementation: Any):
        self._services[interface] = implementation
    
    def register_factory(self, interface: type, factory: Callable):
        self._factories[interface] = factory
    
    def resolve(self, interface: type) -> Any:
        # Resolution logic with dependency injection
```

##### Phase 3: Constructor Injection Refactoring
```python
# After: Clean dependency injection
class DrillingDataAnalyser:
    def __init__(self, 
                 correlator: IMessageCorrelator,
                 processor: IMessageProcessor, 
                 publisher: IResultPublisher,
                 client_factory: IMQTTClientFactory,
                 config: ConfigurationManager):
        self.correlator = correlator
        self.processor = processor
        self.publisher = publisher
        # ... no direct instantiation
```

##### Phase 4: Configuration-Based Service Registration  
```python
# Service registration configuration
def configure_services(container: ServiceContainer, config: ConfigurationManager):
    container.register_singleton(IConfigurationManager, config)
    container.register_factory(IMessageCorrelator, 
                              lambda: SimpleMessageCorrelator(config))
    container.register_factory(IResultPublisher, 
                              lambda: ResultPublisher(mqtt_client, config))
    # ...
```

#### Benefits of DI Implementation
1. **🧪 Testability**: Easy mocking and unit test isolation
2. **🔄 Flexibility**: Runtime swapping of implementations  
3. **📦 Loose Coupling**: Components depend on interfaces, not concrete classes
4. **🏗️ Plugin Architecture**: Support for pluggable depth estimation algorithms
5. **⚙️ Configuration-Driven**: Service composition via configuration files

#### Implementation Priority
- **Phase 1** (High): Interface definitions and container basics
- **Phase 2** (Medium): Constructor injection refactoring
- **Phase 3** (Low): Advanced IoC features (decorators, lifecycle management)

### Long-term Architectural Goals
1. **Plugin Architecture**: Make depth estimation algorithms pluggable
2. **Configuration Hot-reload**: Support runtime configuration updates
3. **Observability**: Add metrics, tracing, and health monitoring
4. **Streaming Support**: Consider reactive streams for high-throughput scenarios

## 🚀 Docker Image Optimization: PyTorch to ONNX Migration Plan

### Current Docker Image Size Issues

The current system suffers from significant Docker image bloat due to PyTorch dependencies:

- **Current Sizes**: 
  - `python:3.10.16-slim` base: **~10GB** final image
  - `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime` base: **~13GB** final image
- **Root Cause**: Full PyTorch stack (2.3GB+) for inference-only operations
- **Impact**: Slow deployments, high storage costs, inefficient container registry usage

### PyTorch Usage Analysis

**Current Architecture:**
- **Models**: 72 PatchTSMixer models (24 CV folds × 3 files) = 135MB total
- **Framework**: Custom transformers library + IBM TSFM + PyTorch 2.3.1
- **Usage Pattern**: Pure inference (no training), single-batch processing
- **Compute**: CPU-only inference with occasional `.cpu().detach().numpy()` operations
- **Dependencies**: Heavy ML libraries for lightweight time series inference

**Key Insights:**
- Only **inference-only** usage - no model training or gradient computation
- Simple tensor operations: data loading, forward pass, output extraction
- CPU-bound processing despite CUDA-enabled base images
- Model ensemble where typically only 1-2 models are used in production

### ONNX Migration Benefits

**Estimated Space Savings:**

| Component | Current Size | ONNX Alternative | Savings |
|-----------|-------------|-----------------|---------|
| **PyTorch Core** | ~2.3GB | ONNX Runtime | ~2.1GB |
| **Transformers** | ~350MB | Custom ONNX loader | ~300MB |
| **TSFM Library** | ~12MB | Converted models | ~10MB |
| **CUDA Dependencies** | ~3GB | CPU-only runtime | ~3GB |
| **Total Base Image** | **~10-13GB** | **~1.5-2GB** | **~8-11GB** |

**Performance Benefits:**
- **Faster Inference**: ONNX Runtime optimizations (graph optimization, kernel fusion)
- **Reduced Memory**: Lower runtime memory footprint
- **Better CPU Utilization**: Optimized CPU kernels vs PyTorch CPU fallbacks
- **Startup Time**: Faster model loading and initialization

### ONNX Migration Implementation Plan

#### Phase 1: Model Export & Validation (Weeks 1-2)

**1.1 Export Existing Models**
```python
# Export PatchTSMixer models to ONNX format
import torch
from transformers import PatchTSMixerForPrediction

def export_model_to_onnx(model_path: str, output_path: str):
    model = PatchTSMixerForPrediction.from_pretrained(model_path)
    model.eval()
    
    # Create dummy input matching production shape
    dummy_input = torch.randn(1, 17, 512)  # batch_size=1, channels=17, sequence=512
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,  # Ensure compatibility
        do_constant_folding=True,
        input_names=['time_series_data'],
        output_names=['depth_predictions'],
        dynamic_axes={
            'time_series_data': {0: 'batch_size'},
            'depth_predictions': {0: 'batch_size'}
        }
    )
```

**1.2 Model Validation**
- Export all 24 CV models to ONNX format
- Validate inference accuracy against PyTorch (< 0.1% numerical difference)
- Benchmark inference speed: ONNX vs PyTorch
- Test production input data shapes and edge cases

**1.3 Model Optimization**
```python
# Optimize ONNX models for production
import onnxruntime as ort
from onnxruntime.tools import optimizer

def optimize_onnx_model(input_path: str, output_path: str):
    # Apply graph optimizations
    opt_model = optimizer.optimize_model(
        input_path, 
        model_type='bert',  # transformer-like optimizations
        num_heads=0,  # auto-detect
        hidden_size=0  # auto-detect  
    )
    opt_model.save_model_to_file(output_path)
```

#### Phase 2: Inference Pipeline Refactoring (Weeks 2-3)

**2.1 Create ONNX Inference Engine**
```python
class ONNXDepthInference:
    """ONNX-based replacement for PyTorch DepthInference"""
    
    def __init__(self, model_path: str):
        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']  # CPU-optimized
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def estimate_depth(self, input_data: np.ndarray) -> dict:
        """Run depth estimation inference"""
        # Prepare input (no PyTorch tensors needed)
        input_dict = {self.input_name: input_data.astype(np.float32)}
        
        # Run inference
        outputs = self.session.run([self.output_name], input_dict)
        predictions = outputs[0]
        
        # Extract depth positions (same logic as PyTorch version)
        return self._extract_depth_positions(predictions)
```

**2.2 Update Data Processing Pipeline**
- Replace `torch.tensor()` operations with numpy arrays
- Update `inference_data_operation.py` for numpy-based preprocessing  
- Modify `inference_pipeline.py` to use ONNX session
- Maintain backward compatibility during transition

**2.3 Configuration Management**
```python
# Add ONNX model configuration
onnx_config = {
    "inference_engine": "onnx",  # vs "pytorch"
    "model_path": "/app/trained_model/onnx/",
    "optimization_level": "all",
    "cpu_threads": 4,  # CPU-specific optimization
    "enable_profiling": False
}
```

#### Phase 3: Docker Image Optimization (Week 3)

**3.1 Lightweight Base Image**
```dockerfile
# NEW: ONNX-optimized Dockerfile
FROM python:3.10.16-slim as base

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create optimized Python environment
COPY requirements.onnx.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.onnx.txt

# Copy ONNX models and application code
COPY trained_model/onnx/ /app/trained_model/
COPY abyss/ /app/abyss/

# Expected final size: ~1.5-2GB (vs 10-13GB current)
```

**3.2 Optimized Requirements**
```python
# requirements.onnx.txt - Streamlined dependencies
numpy==1.26.4
pandas==2.2.3
onnxruntime==1.19.2        # ~200MB vs 2.3GB PyTorch
scikit-learn==1.3.2
paho-mqtt==1.6.1
pyyaml==6.0.1
openpyxl==3.1.5

# Removed heavy dependencies:
# torch==2.3.1              # -2.3GB
# transformers               # -350MB  
# accelerate                 # -50MB
# tsfm_public               # -12MB
```

#### Phase 4: Integration & Testing (Week 4)

**4.1 Component Integration**
- Update `MessageProcessor` to use `ONNXDepthInference`
- Modify Docker Compose configurations for new images
- Update build scripts and CI/CD pipelines
- Test MQTT message processing end-to-end

**4.2 Performance Validation**
```python
# Benchmark current vs ONNX performance
def benchmark_inference_performance():
    test_cases = load_production_test_data()
    
    # PyTorch baseline
    pytorch_times = []
    pytorch_model = load_pytorch_model()
    
    # ONNX comparison
    onnx_times = []
    onnx_model = ONNXDepthInference(onnx_model_path)
    
    for test_input in test_cases:
        # Measure both accuracy and speed
        pytorch_result, pytorch_time = time_inference(pytorch_model, test_input)
        onnx_result, onnx_time = time_inference(onnx_model, test_input)
        
        validate_numerical_equivalence(pytorch_result, onnx_result, tolerance=1e-4)
```

**4.3 Migration Strategy**
- **Blue-Green Deployment**: Run both PyTorch and ONNX versions in parallel
- **Gradual Rollout**: Route 10% → 50% → 100% traffic to ONNX version
- **Rollback Plan**: Maintain PyTorch images for emergency rollback
- **Monitoring**: Add inference latency and accuracy metrics

### Implementation Timeline & Milestones

| Phase | Duration | Key Deliverables | Success Criteria |
|-------|----------|------------------|------------------|
| **Phase 1** | 2 weeks | ONNX model exports, validation scripts | <0.1% accuracy difference |
| **Phase 2** | 1 week | ONNX inference engine integration | Functional MQTT pipeline |
| **Phase 3** | 1 week | Optimized Docker images | <2GB final image size |
| **Phase 4** | 1 week | Production deployment, monitoring | 100% traffic on ONNX |

**Total Effort**: ~5 weeks with 1 FTE developer

### Risk Assessment & Mitigation

**Technical Risks:**
- **Model Compatibility**: Some PyTorch operations may not export cleanly to ONNX
  - *Mitigation*: Test export early, identify problematic layers, use ONNX-compatible alternatives
- **Numerical Precision**: Slight differences in floating-point operations
  - *Mitigation*: Comprehensive validation suite, acceptable tolerance thresholds
- **Performance Regression**: ONNX inference could be slower than PyTorch on specific workloads
  - *Mitigation*: Thorough benchmarking, performance testing on production data

**Operational Risks:**
- **Deployment Complexity**: Managing two inference engines during transition
  - *Mitigation*: Feature flags, gradual rollout, comprehensive monitoring
- **Model Update Process**: ONNX export becomes part of model deployment pipeline
  - *Mitigation*: Automated export scripts, CI/CD integration

### Expected Outcomes

**Immediate Benefits:**
- **80-85% smaller Docker images** (2GB vs 10-13GB)
- **50-75% faster container startup times**
- **30-50% reduced deployment times**
- **Significant cost savings** in container registry and cloud storage

**Long-term Benefits:**
- **Inference Performance**: 20-40% faster CPU inference
- **Memory Efficiency**: 40-60% lower runtime memory usage
- **Edge Deployment**: Enables deployment on resource-constrained environments
- **Multi-Platform Support**: Better ARM64 compatibility for edge devices
- **Operational Simplicity**: Single inference runtime, reduced dependency management

This ONNX migration represents a strategic optimization that maintains functional capabilities while dramatically improving deployment efficiency and operational costs.

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. **Head ID Not Appearing in Logs** ⚠️

**Problem**: The system processes messages correctly but head_id is never displayed in logs or results.

**Root Cause**: Missing or incorrect configuration in MQTT config files across different directories.

**Check Configuration Files**:
```bash
# Primary config (should have head_id configuration)
abyss/src/abyss/run/config/mqtt_conf_docker.yaml

# Secondary configs (may be outdated)
abyss/sandbox/run/mqtt_docker_conf.yaml
```

**Solution Steps**:

1. **Verify heads topic subscription** in your config:
   ```yaml
   mqtt:
     listener:
       heads: "AssetManagement/Head"  # Must be present
   ```

2. **Verify head_id data path** in your config:
   ```yaml
   mqtt:
     data_ids:
       head_id: 'AssetManagement.Assets.Heads.0.Identification.SerialNumber'
   ```

3. **Update ALL config files** - the system has multiple config files in different directories:
   - `abyss/src/abyss/run/config/mqtt_conf_docker.yaml` (primary)
   - `abyss/sandbox/run/mqtt_docker_conf.yaml` (secondary)
   - Any custom configs you may be using

4. **Check log level** - head_id extraction logs at INFO level:
   ```bash
   uos_depthest_listener --config config/mqtt_conf_docker.yaml --log-level INFO
   ```

5. **Verify message structure** matches the configured path:
   ```
   Expected: heads_msg.data["Messages"]["Payload"]["AssetManagement"]["Assets"]["Heads"][0]["Identification"]["SerialNumber"]
   ```

**Validation**:
- Look for log entry: `"Extracted head_id: <value>"`
- Look for log entry: `"Heads message is None or empty"` (indicates missing heads subscription)
- Check published MQTT results contain `head_id` field

#### 2. **Duplicate Message Processing Behavior** ℹ️

**Problem**: Listener processes files correctly on first send, but doesn't process identical messages sent again.

**Expected Behavior**: This is **intentional** and **correct** behavior.

**Explanation**: The system implements duplicate message prevention through:
- **Message tracking**: Each message is marked as `processed` after correlation
- **Timestamp-based deduplication**: Messages with identical timestamps are not reprocessed
- **Memory efficiency**: Prevents redundant processing of repeated data

**When Messages Are Reprocessed**:
- Message content changes (different result_id, trace data, etc.)
- Timestamp changes (new measurement time)
- Tool/toolbox ID changes

**To Force Reprocessing** (for testing):
1. Change timestamp in message
2. Modify any data field (result_id, trace values, etc.)
3. Use different tool/toolbox IDs
4. Restart the listener service

#### 3. **Configuration File Version Inconsistencies** ⚠️

**Problem**: Multiple versions of config files exist across directories with different settings.

**File Locations**:
- `abyss/src/abyss/run/config/mqtt_conf_docker.yaml` ✅ **Primary/Updated**
- `abyss/sandbox/run/mqtt_docker_conf.yaml` ❌ **Outdated** 
- `abyss/sandbox/run/mqtt_localhost_conf.yaml`
- `abyss/sandbox/run/mqtt_conf.yaml`

**Solution**: Always use the primary config file or ensure all configs are synchronized:
```bash
# Copy primary config to other locations if needed
cp abyss/src/abyss/run/config/mqtt_conf_docker.yaml abyss/sandbox/run/mqtt_docker_conf.yaml
```

#### 4. **No Messages Being Processed** 🔍

**Diagnostic Steps**:

1. **Check MQTT broker connection**:
   ```bash
   # In logs, look for:
   "Connected to broker successfully"
   "Subscribed to topic: OPCPUBSUB/+/+/ResultManagement"
   ```

2. **Verify topic patterns** match your message topics:
   ```yaml
   # Config topics must match actual message topics
   listener:
     root: "OPCPUBSUB"
     toolboxid: "+"  # or specific ID
     toolid: "+"     # or specific ID
     result: "ResultManagement"
     trace: "ResultManagement/Trace"
     heads: "AssetManagement/Head"
   ```

3. **Check message correlation** - enable DEBUG logging:
   ```bash
   uos_depthest_listener --config config/mqtt_conf_docker.yaml --log-level DEBUG
   ```

4. **Look for correlation logs**:
   ```
   "Found X unprocessed result messages"
   "Found X unprocessed trace messages"
   "Found X unprocessed heads messages"
   ```

#### 5. **Head ID Extraction Fails** 🔍

**Debug Steps**:

1. **Check heads message reception**:
   ```
   # In DEBUG logs, look for:
   "Heads message is <message_content>"
   "Heads message data is a string, attempting to parse JSON"
   "Heads message data is already a dictionary"
   ```

2. **Verify message structure** - check if data path exists:
   ```python
   # Expected structure:
   heads_msg.data["Messages"]["Payload"]["AssetManagement"]["Assets"]["Heads"][0]["Identification"]["SerialNumber"]
   ```

3. **Check for extraction errors**:
   ```
   # In logs, look for:
   "Head ID not found in heads message"
   "Extracted head_id: <value>"  # Success case
   ```

4. **Common path issues**:
   - Missing `Messages.Payload` wrapper
   - Array index `[0]` doesn't exist
   - Different field names (`SerialNumber` vs `serialNumber`)

#### 6. **Performance Issues** 🔍

**Memory Usage**: 
- Clear processed messages periodically
- Monitor buffer sizes in logs
- Consider reducing time window for correlation

**Processing Speed**:
- Use `SimpleMessageCorrelator` (67% faster than legacy `MessageCorrelator`)
- Enable appropriate log levels (INFO/WARN for production)
- Consider async components for high-throughput scenarios

#### 7. **Docker Container Issues** 🐳

**Common Problems**:
- Config file not mounted correctly
- Network connectivity to MQTT broker
- Environment variable configuration

**Solutions**:
```bash
# Verify config mount
docker run -v $(pwd)/config:/app/config -t listener

# Check environment variables
docker run -e LOG_LEVEL=DEBUG -t listener

# Test broker connectivity
docker run --network=host -t listener
```

### Debugging Tools and Commands

#### Enable Debug Logging
```bash
# Local development
uos_depthest_listener --config config/mqtt_conf_local.yaml --log-level DEBUG

# Docker container
docker run -e LOG_LEVEL=DEBUG -t listener
```

#### Check Configuration Loading
```bash
# Verify config file syntax
python -c "import yaml; yaml.safe_load(open('config/mqtt_conf_docker.yaml'))"

# Test config paths
python -c "from abyss.uos_depth_est_utils import reduce_dict; print(reduce_dict({'a': {'b': 'value'}}, 'a.b'))"
```

#### Monitor MQTT Messages
```bash
# Subscribe to all topics (requires mosquitto-clients)
mosquitto_sub -h localhost -t "OPCPUBSUB/+/+/+" -v

# Subscribe to specific result topics
mosquitto_sub -h localhost -t "OPCPUBSUB/+/+/ResultManagement" -v
```

#### Generate Test Data
```bash
# Use the JSON publisher for testing
uos_publish_json /path/to/test/data

# Verify test data structure
python -c "import json; print(json.load(open('test_file.json')))"
```

### Log Analysis Guide

**Key Log Entries to Look For**:

1. **Connection Success**:
   ```
   Connected to broker successfully
   Subscribed to topic: OPCPUBSUB/+/+/ResultManagement
   ```

2. **Message Reception**:
   ```
   Found X unprocessed result messages
   Found X unprocessed trace messages
   Found X unprocessed heads messages
   ```

3. **Head ID Extraction**:
   ```
   Extracted head_id: <value>
   ```

4. **Processing Success**:
   ```
   Processing matched messages
   Published depth estimation result
   ```

5. **Error Indicators**:
   ```
   Head ID not found in heads message
   Missing required result or trace message
   Failed to convert messages to DataFrame
   ```

## Code Quality Guidelines

### When Working with This Codebase:
1. **Prefer async components** over sync equivalents
2. **Use `find_in_dict` and `reduce_dict`** from `uos_depth_est_utils` for data extraction
3. **Follow the head_id extraction pattern** established in `MessageProcessor._extract_head_id_simple()`
4. **Test both success and error scenarios** with proper head_id inclusion
5. **Avoid adding new sync components** - extend async architecture instead
6. **✅ Use ConfigurationManager** for all config access (standardized)
7. **Include head_id in all MQTT published messages**
8. **✅ Use exception-based error handling** - never return boolean success/failure
9. **✅ Use SimpleMessageCorrelator** instead of legacy MessageCorrelator
10. **✅ Follow consolidated publishing patterns** with type-safe enums

### ✅ Improved Patterns (Now Standardized):
- **Exception-based error handling** with proper chaining (`AbyssError` hierarchy)
- **ConfigurationManager usage** for all configuration access
- **Consolidated method patterns** instead of duplicated code  
- **Key-based message correlation** instead of complex time-bucket grouping
- **Type-safe publishing** with enum-based parameterization

### ❌ Avoid These Deprecated Patterns:
- ~~Complex time-bucket correlation logic~~ (use `SimpleMessageCorrelator`)
- ~~Multiple nearly-identical methods~~ (use consolidated patterns with enums)
- ~~Direct config dictionary access~~ (use `ConfigurationManager`)
- ~~Boolean return values for error handling~~ (use exceptions)
- ~~Raw dictionary message handling~~ 
- ~~Excessive mocking in tests~~

The async components in `mqtt/async_components/` represent the best practices and should be the foundation for all future MQTT-related development.