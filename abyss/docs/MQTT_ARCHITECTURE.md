# MQTT Drilling Data Analysis Architecture

## Overview

The MQTT Drilling Data Analysis system processes real-time drilling data from MQTT messages, performs depth estimation using machine learning models, and publishes results back to MQTT. The system is designed to handle 100+ messages per second with robust parallel processing capabilities.

## Architecture Components

### Core Processing Components

#### 1. DrillingDataAnalyser (`drilling_analyser.py`)
The main orchestrator that coordinates all components and manages the processing flow.

**Responsibilities:**
- Manages MQTT client lifecycle
- Coordinates message buffering and correlation
- Submits work to processing pool
- Handles result publishing
- Monitors system health

**Key Features:**
- Graceful shutdown handling
- Automatic reconnection
- Simple status logging every 30 seconds
- Integration with ProcessPoolExecutor for parallel processing

#### 2. ProcessingPool (`processing_pool.py`)
Manages parallel processing of depth inference tasks using ProcessPoolExecutor.

**Responsibilities:**
- Worker pool management (default: 10 workers)
- Model initialization in each worker process
- Future tracking and result handling
- Performance metrics collection

**Key Design Decisions:**
- Each worker loads its own DepthInference model (~1GB memory)
- Workers are persistent to avoid model reload overhead
- Non-blocking submit with Future-based result handling

#### 3. MessageBuffer (`message_buffer.py`)
Thread-safe buffer for storing incoming MQTT messages by type.

**Responsibilities:**
- Store messages with deduplication
- Provide thread-safe access
- Track buffer health metrics
- Handle buffer overflow (max 1000 messages per type)

**Simplified Metrics:**
- Buffer sizes
- Total messages received
- Duplicate count

#### 4. SimpleMessageCorrelator (`simple_correlator.py`)
Correlates related messages (Results, Traces, Heads) within a time window.

**Responsibilities:**
- Find matching messages across buffers
- Process complete message sets
- Clean up old messages
- Maintain correlation metrics

**Time Window:** 5 seconds (configurable)

### Publishing Components

#### 5. ResultPublisher (`result_publisher.py`)
Publishes processing results to MQTT with configurable validation.

**Responsibilities:**
- Publish depth estimation results
- Delegate formatting to ResultMessageFormatter
- Delegate validation to DepthValidator
- Handle publish errors

#### 6. ResultMessageFormatter (`message_formatter.py`)
Formats processing results into MQTT messages.

**Result Types:**
- SUCCESS: Valid depth estimation
- ERROR: Processing failed
- INSUFFICIENT_DATA: Missing required data
- WARNING: Results with validation warnings

#### 7. DepthValidator (`depth_validator.py`)
Configurable validation for depth estimation results.

**Validation Behaviors:**
- `publish`: Always publish (default)
- `skip`: Don't publish negative values
- `warning`: Publish with warnings

**Features:**
- Sequential negative depth tracking
- Configurable alert thresholds
- Validation statistics

### Supporting Components

#### 8. ConfigurationManager (`config_manager.py`)
Centralized configuration management with validation.

**Features:**
- YAML configuration loading
- Default value handling
- Configuration validation
- Safe nested access with get() method

#### 9. MQTTClientManager (`client_manager.py`)
Manages MQTT client lifecycle and subscriptions.

**Features:**
- Automatic reconnection
- Connection state tracking
- Topic subscription management
- Broker authentication support

#### 10. SimpleThroughputMonitor (`throughput_monitor.py`)
Monitors if the system is keeping up with message arrival rate.

**Status Types:**
- HEALTHY: Processing rate >= 95% of arrival rate
- FALLING_BEHIND: Processing rate < 95% of arrival rate

**Metrics:**
- Arrival rate (messages/second)
- Processing rate (messages/second)
- Spare capacity or shortfall percentage

## Data Flow

```
1. MQTT Messages Arrive
   ├── ResultManagement → MessageBuffer["results"]
   ├── Trace → MessageBuffer["traces"]
   └── Heads → MessageBuffer["heads"]

2. Correlation (every 0.1 seconds)
   └── SimpleMessageCorrelator finds matching sets

3. Parallel Processing
   ├── Submit to ProcessingPool
   ├── Worker performs DepthInference
   └── Return ProcessingResult

4. Result Validation
   └── DepthValidator checks results

5. Publishing
   ├── ResultMessageFormatter creates messages
   └── ResultPublisher sends to MQTT
```

## Configuration

### Processing Configuration
```yaml
mqtt:
  processing:
    workers: 10        # Number of parallel workers
    model_id: 4        # DepthInference model version
```

### Depth Validation Configuration
```yaml
mqtt:
  depth_validation:
    negative_depth_behavior: "publish"  # or "skip", "warning"
    track_sequential_negatives: true
    sequential_threshold: 5
    sequential_window_minutes: 5
```

## Performance Characteristics

### Throughput
- Target: 100 messages/second
- Processing time: ~0.5 seconds per message
- Solution: 10 parallel workers = 20 messages/second capacity

### Memory Usage
- Each worker: ~1GB (model) + overhead
- Total with 10 workers: ~12GB
- Message buffers: <100MB

### Scaling Considerations
- CPU-bound due to Python GIL → ProcessPoolExecutor
- Workers scale linearly up to CPU cores
- Network I/O handled by main thread

## Error Handling

### Exception Hierarchy
```
AbyssError (base)
├── AbyssCommunicationError (MQTT/network issues)
├── AbyssProcessingError (depth estimation failures)
└── MQTTPublishError (publish failures)
```

### Resilience Features
- Automatic MQTT reconnection
- Worker process restart on failure
- Message deduplication
- Graceful shutdown handling

## Monitoring and Observability

### Health Indicators
1. **Throughput Status**: HEALTHY or FALLING_BEHIND
2. **Buffer Sizes**: Monitor for growth
3. **Processing Success Rate**: Track failures
4. **Worker Pool Health**: Active/pending tasks

### Logging
- Structured logging with contextual information
- Simple status logs every 30 seconds
- Warning logs for validation issues
- Error logs with full context

## Future Enhancements

### Phase 2 Considerations
1. **GPU Acceleration**: For faster model inference
2. **Distributed Processing**: Ray/Celery for multi-machine scaling
3. **Advanced Monitoring**: Prometheus/Grafana integration
4. **Dynamic Scaling**: Auto-adjust workers based on load

### Potential Optimizations
1. **Batch Processing**: Process multiple messages per worker call
2. **Model Optimization**: Quantization, pruning, or distillation
3. **Caching**: Cache repeated calculations
4. **Connection Pooling**: Multiple MQTT connections