# UOS Drilling Depth Estimation System - AI Coding Agent Instructions

## Project Overview

This is a **Deep Learning Drilling Depth Estimation System** that processes real-time drilling data from MQTT messages to perform depth estimation using machine learning models. The system is designed for industrial drilling operations and provides real-time depth analysis through MQTT message correlation and processing.

### Core Functionality
- **Real-time MQTT Processing**: Listens to drilling data messages (Result, Trace, Heads)
- **Message Correlation**: Matches messages by toolbox/tool ID and timestamps
- **Depth Estimation**: Uses ML models (PatchTSMixer) for three-point depth analysis
- **Result Publishing**: Publishes depth estimates back to MQTT broker
- **Docker Deployment**: Comprehensive containerization with CPU/GPU variants

## Architecture Overview

### Main Components

```
MQTT Broker → DrillingDataAnalyser (Orchestrator)
    ├── ConfigurationManager (YAML config handling)
    ├── MQTTClientManager (Connection management)
    ├── MessageBuffer (Thread-safe message storage)
    ├── SimpleMessageCorrelator (Message matching)
    ├── MessageProcessor (Data extraction & ML inference)
    ├── DataFrameConverter (MQTT to DataFrame conversion)
    ├── DepthInference (ML model - 72 PatchTSMixer models)
    └── ResultPublisher (MQTT result publishing)
```

### Key Data Flow
1. **MQTT Message Reception**: System listens for Result, Trace, and Heads messages
2. **Message Correlation**: Matches messages based on toolbox/tool ID and timestamps (30-second window)
3. **Data Extraction**: Extracts position, torque, thrust, step data, and head_id
4. **Depth Estimation**: Applies ML-based three-point analysis (entry, transition, exit)
5. **Result Publishing**: Publishes depth estimates with metadata back to MQTT broker

## Project Structure

```
/
├── abyss/                          # Main Python package
│   ├── src/abyss/                  # Source code
│   │   ├── mqtt/components/        # Modular MQTT processing components
│   │   │   ├── drilling_analyser.py    # Main orchestrator
│   │   │   ├── config_manager.py       # Configuration management
│   │   │   ├── message_buffer.py       # Thread-safe message storage
│   │   │   ├── simple_correlator.py    # Message correlation
│   │   │   ├── message_processor.py    # Data processing & ML inference
│   │   │   ├── data_converter.py       # MQTT to DataFrame conversion
│   │   │   └── result_publisher.py     # Result publishing
│   │   ├── run/                    # Entry point scripts
│   │   │   ├── uos_depth_est_mqtt.py   # MQTT listener main
│   │   │   ├── uos_depth_est_xls.py    # XLS file processor
│   │   │   ├── uos_depth_est_json.py   # JSON file processor
│   │   │   └── uos_publish_json.py     # JSON data publisher
│   │   ├── uos_depth_est_core.py   # Core depth estimation logic
│   │   └── trained_model/          # ML models (72 PatchTSMixer models, 135MB)
│   ├── tests/                      # Comprehensive test suite (220+ tests)
│   ├── pyproject.toml              # Python packaging configuration
│   └── requirements.txt            # Dependencies
├── _sandbox/                       # Development examples and experiments
├── mqtt-compose/                   # MQTT broker Docker setup
├── mqtt-multistack/                # Multi-service Portainer stacks
├── deployment/                     # Deployment configurations
├── docs/                          # Documentation
├── Dockerfile                     # Main container (CPU-optimized)
├── Dockerfile.cpu                 # CPU-only variant
├── Dockerfile.gpu                 # GPU variant
└── build-publish.sh               # Build automation script
```

## Key Technologies & Dependencies

### Core Technologies
- **Python 3.9+**: Main programming language
- **MQTT**: Message queuing protocol for real-time communication
- **PyTorch**: ML framework for depth estimation models
- **ONNX Runtime**: Model inference optimization
- **Docker**: Containerization and deployment
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

### ML Models
- **PatchTSMixer Models**: 72 models for depth estimation
- **Three-Point Estimation**: Entry, transition, and exit point detection
- **Cross-Validation**: 24 folds across 3 drilling configurations
- **Model Size**: 135MB (target: ONNX migration for 80-85% reduction)

### Build System
- **Docker BuildKit**: Optimized builds (15min → 30s with caching)
- **Multi-stage Builds**: Separate development, runtime, and publish images
- **Automated Testing**: GitHub Actions for wheel validation
- **Makefile**: Comprehensive build targets

## Configuration & Deployment

### Configuration Files
- **`mqtt_conf_local.yaml`**: Local development configuration
- **`mqtt_conf_docker.yaml`**: Docker deployment configuration
- **Key Settings**: Broker connection, topic structure, time windows, duplicate handling

### Docker Deployment Options
1. **CPU-only**: `Dockerfile.cpu` - Optimized for CPU inference
2. **GPU**: `Dockerfile.gpu` - CUDA-enabled for GPU acceleration
3. **Runtime**: `Dockerfile.runtime` - Production-optimized minimal image
4. **Development**: `Dockerfile.devel` - Full development environment

### Entry Points
- **`uos_depthest_listener`**: Main MQTT listener service
- **`uos_depthest_xls`**: Process XLS drilling files
- **`uos_depthest_json`**: Process JSON drilling files
- **`uos_publish_json`**: Publish test data to MQTT

## Development Guidelines

### Code Organization Principles
1. **Modular Architecture**: Components are clearly separated with single responsibilities
2. **Configuration-Driven**: All settings externalized to YAML files
3. **Error Handling**: Comprehensive exception hierarchy with structured logging
4. **Thread Safety**: MessageBuffer and correlators handle concurrent access
5. **Testing**: Extensive test coverage including load tests and integration tests

### Key Classes to Understand
- **`DrillingDataAnalyser`**: Main orchestrator - coordinates all components
- **`MessageBuffer`**: Thread-safe message storage with automatic cleanup
- **`SimpleMessageCorrelator`**: Fast message matching (67% smaller, 1.68x faster than legacy)
- **`MessageProcessor`**: Core processing logic with ML inference
- **`ConfigurationManager`**: Centralized configuration with validation
- **`DepthInference`**: ML model wrapper for depth estimation

### Data Structures
- **`TimestampedData`**: Message wrapper with timestamp and source info
- **`ProcessingResult`**: Structured result from depth estimation
- **Configuration**: YAML-based with typed access methods

### Performance Considerations
- **Current Throughput**: ~100 messages/second
- **Bottlenecks**: Single processing thread, in-memory buffers
- **Future Scaling**: Async processing, RabbitMQ/Kafka integration planned
- **Memory Management**: Automatic cleanup of old messages

### Testing Strategy
1. **Unit Tests**: Component-level testing with mocks
2. **Integration Tests**: End-to-end MQTT message processing
3. **Load Tests**: High-volume concurrent message handling
4. **Performance Tests**: Stress testing with metrics collection

## Common Development Tasks

### Adding New Message Types
1. Update `config_manager.py` to handle new topic patterns
2. Modify `message_processor.py` to extract relevant data
3. Update `data_converter.py` for DataFrame conversion
4. Add tests for new message type handling

### Modifying Depth Estimation Logic
1. Update `uos_depth_est_core.py` for core algorithm changes
2. Modify `message_processor.py` for integration changes
3. Update ML models in `trained_model/` directory
4. Add comprehensive tests for new estimation logic

### Configuration Changes
1. Update YAML schema in `config_manager.py`
2. Add validation for new configuration options
3. Update Docker configurations if needed
4. Document new settings in README

### Performance Optimization
1. Profile using existing load tests in `tests/mqtt/components/`
2. Consider async processing for high-throughput scenarios
3. Optimize ML model inference (ONNX conversion)
4. Monitor memory usage and cleanup intervals

## Error Handling & Debugging

### Structured Logging
- **Log Levels**: DEBUG, INFO, WARNING, ERROR with contextual information
- **Structured Fields**: machine_id, result_id, tool_id, timestamps
- **Performance Logging**: Processing times, throughput metrics
- **Error Tracking**: Rate-limited warnings for repeated failures

### Common Issues & Solutions
1. **Test Discovery Failures**: Clean Python cache, check for duplicate test files
2. **MQTT Connection Issues**: Verify broker configuration, network connectivity
3. **ML Model Loading**: Check model file paths, memory availability
4. **Message Correlation**: Adjust time windows, verify topic structure

### Debugging Tools
- **VS Code Python Extension**: Integrated debugging with pytest
- **Docker Logs**: Container-level debugging
- **MQTT Clients**: mosquitto_pub/sub for message testing
- **Performance Monitoring**: Built-in metrics collection

## Version History & Changelog

### Current Version: 0.2.6
- **Negative Depth Handling**: Detection and handling of invalid depth values
- **Stress Testing Suite**: Comprehensive performance testing framework
- **Docker Build Overhaul**: BuildKit optimization with caching
- **MessageBuffer Enhancements**: Improved monitoring and thread safety
- **Portainer Stack Automation**: Streamlined deployment updates

### Architecture Evolution
- **v0.2.4**: Major refactoring to modular components
- **v0.2.2**: Fixed MQTT message dropping bug
- **v0.2.0**: Transition from two-point to three-point depth estimation
- **v0.1.1**: Switch to wheel-based distribution

## Future Roadmap

### Immediate Priorities
1. **ONNX Migration**: Reduce model size by 80-85%
2. **Async Processing**: Implement async/await for higher throughput
3. **High-Throughput Architecture**: RabbitMQ/Kafka integration
4. **Auto-scaling**: Dynamic worker scaling based on load

### Long-term Vision
1. **Training Platform**: User GUI for model annotation and training
2. **Edge Processing**: Deployment at data source
3. **Multi-region Support**: Geographic distribution
4. **Advanced Analytics**: Real-time anomaly detection

## Important Notes for AI Agents

1. **Always use the modular components** in `abyss/src/abyss/mqtt/components/` - avoid the legacy monolithic classes
2. **Configuration is critical** - all behavior is driven by YAML files; understand the schema
3. **Thread safety matters** - MessageBuffer and correlators handle concurrent access
4. **Testing is comprehensive** - run existing tests before making changes
5. **Docker is the deployment target** - ensure changes work in containerized environments
6. **Performance monitoring** - use built-in metrics for optimization decisions
7. **Error handling** - follow the established exception hierarchy and logging patterns

This system is production-ready and actively used in industrial drilling operations. Maintain high code quality, comprehensive testing, and clear documentation for all changes.
