# MQTT Configuration Guide

This guide explains all configuration options for the MQTT drilling data analysis system.

## Configuration Files

The system uses YAML configuration files located in `src/abyss/run/config/`:
- `mqtt_conf_local.yaml` - Local development configuration
- `mqtt_conf_docker.yaml` - Docker deployment configuration

## Configuration Structure

### MQTT Broker Connection

```yaml
mqtt:
  broker:
    host: "localhost"      # Broker hostname or IP
    port: 1883            # MQTT port (default: 1883)
    username: ""          # Optional authentication
    password: ""          # Optional authentication
```

**Host Options:**
- `"localhost"` - Local development
- `"host.docker.internal"` - Docker containers accessing host
- `"mqtt-broker"` - Docker Compose service name
- IP address (e.g., `"192.168.1.97"`) - Production broker

### Topic Structure

```yaml
mqtt:
  listener:
    root: "OPCPUBSUB"                    # Root topic prefix
    toolboxid: "+"                       # Wildcard or specific ID
    toolid: "+"                          # Wildcard or specific ID
    result: "ResultManagement"           # Result message suffix
    trace: "ResultManagement/Trace"      # Trace message suffix
    heads: "AssetManagement/Heads"       # Heads message suffix
```

**Topic Pattern Examples:**
- Results: `OPCPUBSUB/+/+/ResultManagement`
- Traces: `OPCPUBSUB/+/+/ResultManagement/Trace`
- Heads: `OPCPUBSUB/+/+/AssetManagement/Heads`

**Wildcards:**
- `"+"` - Match any single level (subscribe to all)
- Specific ID - Subscribe only to that tool/toolbox

### Message Handling

```yaml
mqtt:
  listener:
    duplicate_handling: "replace"  # How to handle duplicate messages
```

**Options:**
- `"ignore"` - Keep first message, ignore duplicates
- `"replace"` - Replace with newer message
- `"error"` - Raise error on duplicate

### Data Field Mappings

```yaml
mqtt:
  data_ids:
    # Identification fields
    machine_id: 'ResultManagement.Results.0.ResultMetaData.SerialNumber'
    head_id: 'AssetManagement.Assets.Heads.0.Identification.SerialNumber'
    result_id: 'ResultManagement.Results.0.ResultMetaData.ResultId'
    
    # Trace data arrays
    position: 'ResultManagement.Results.0.ResultContent.Trace.StepTraces.PositionTrace.StepTraceContent[0].Values'
    thrust: 'ResultManagement.Results.0.ResultContent.Trace.StepTraces.IntensityThrustTrace.StepTraceContent[0].Values'
    torque: 'ResultManagement.Results.0.ResultContent.Trace.StepTraces.IntensityTorqueTrace.StepTraceContent[0].Values'
    step: 'ResultManagement.Results.0.ResultContent.Trace.StepTraces.StepNumberTrace.StepTraceContent[0].Values'
    
    # Empty values for calibration
    torque_empty_vals: 'ResultManagement.Results.0.ResultContent.StepResults.0.StepResultValues.IntensityTorqueEmpty'
    thrust_empty_vals: 'ResultManagement.Results.0.ResultContent.StepResults.0.StepResultValues.IntensityThrustEmpty'
```

These paths use JSONPath-like notation to extract data from nested message structures.

### Analyzer Configuration

```yaml
mqtt:
  analyzer:
    correlation_debug: true  # Enable debug mode for correlation diagnostics
```

**Correlation Debug:**
- Shows detailed information about why messages aren't correlating
- Useful for troubleshooting message matching issues
- Produces verbose logging output

### Processing Configuration

```yaml
mqtt:
  processing:
    workers: 10        # Number of parallel worker processes
    model_id: 4        # DepthInference model version
```

**Workers:**
- Each worker loads its own model (~1GB memory)
- Recommended: 10 workers for 100 msg/sec at 0.5s processing
- Adjust based on CPU cores and memory

**Model ID:**
- Selects which pre-trained model to use
- Options: 1, 2, 3, 4 (check model availability)

### Depth Validation

```yaml
mqtt:
  depth_validation:
    negative_depth_behavior: "warning"    # How to handle negative depths
    track_sequential_negatives: true      # Track patterns
    sequential_threshold: 5               # Alert threshold
    sequential_window_minutes: 5          # Time window
```

**Negative Depth Behaviors:**
- `"publish"` - Always publish results (default)
- `"skip"` - Don't publish negative results
- `"warning"` - Publish with warning logs

**Sequential Tracking:**
- Monitors patterns of negative depths
- Alerts after threshold exceeded
- Helps identify systematic issues

### Result Publishing

```yaml
mqtt:
  estimation:
    keypoints: 'ResultManagement.Results.0.ResultContent.DepthEstimation.KeyPoints'
    depth_estimation: 'ResultManagement.Results.0.ResultContent.DepthEstimation.DepthEstimation'
```

These paths serve a dual purpose:
1. **Topic Suffixes**: Used to construct MQTT publish topics
2. **Data Paths**: Define where results would be placed in message payloads

**Published Topics:**
- Keypoints: `OPCPUBSUB/{toolbox_id}/{tool_id}/ResultManagement.Results.0.ResultContent.DepthEstimation.KeyPoints`
- Depth: `OPCPUBSUB/{toolbox_id}/{tool_id}/ResultManagement.Results.0.ResultContent.DepthEstimation.DepthEstimation`

The full JSON path is appended to the base topic to create unique topics for each result type.

## Environment-Specific Settings

### Local Development

```yaml
# mqtt_conf_local.yaml
mqtt:
  broker:
    host: "localhost"
  depth_validation:
    negative_depth_behavior: "warning"  # More verbose for debugging
```

### Docker Deployment

```yaml
# mqtt_conf_docker.yaml  
mqtt:
  broker:
    host: "mqtt-broker"  # Docker service name
  depth_validation:
    negative_depth_behavior: "skip"     # Cleaner production behavior
```

## Configuration Validation

The system validates configuration on startup:
- Required fields must be present
- Data types must match expected formats
- Paths must be syntactically valid

Invalid configuration will prevent startup with descriptive error messages.

## Dynamic Configuration

Some settings can be adjusted at runtime:
- Log levels via environment variables
- Worker count (requires restart)
- Validation behavior (requires restart)

## Configuration Best Practices

1. **Topic Patterns**: Use wildcards during development, specific IDs in production
2. **Worker Count**: Start with CPU cores - 2, adjust based on monitoring
3. **Validation**: Use "warning" for debugging, "skip" or "publish" for production
4. **Duplicate Handling**: "replace" is safest for most scenarios
5. **Sequential Tracking**: Enable to catch systematic issues early

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Check broker host/port
   - Verify network connectivity
   - Check authentication if used

2. **No Messages Received**
   - Verify topic patterns match actual topics
   - Check wildcard vs specific ID usage
   - Enable debug logging

3. **Processing Bottleneck**
   - Increase worker count
   - Check model_id performance
   - Monitor system resources

4. **Invalid Data Extraction**
   - Verify data_ids paths match message structure
   - Use message inspection tools
   - Check for schema changes

5. **Negative Depth Values**
   - Negative depths can occur in drilling operations (e.g., tool retraction, calibration)
   - Configure `negative_depth_behavior`:
     - `"publish"`: Publish all results including negative depths
     - `"warning"`: Publish with warning log
     - `"skip"`: Skip publishing negative depth results
   - System handles negative depths as configured, not as errors

## Example Configurations

### High-Throughput Production

```yaml
mqtt:
  processing:
    workers: 20  # More workers for higher throughput
  depth_validation:
    negative_depth_behavior: "skip"
    sequential_threshold: 10  # Less sensitive
```

### Development/Testing

```yaml
mqtt:
  listener:
    toolboxid: "TEST_TOOLBOX"  # Specific test toolbox
    toolid: "TEST_TOOL"        # Specific test tool
  processing:
    workers: 2  # Minimal resources
  depth_validation:
    negative_depth_behavior: "warning"
    sequential_threshold: 3  # More sensitive
```

### Monitoring Focus

```yaml
mqtt:
  depth_validation:
    track_sequential_negatives: true
    sequential_threshold: 5
    sequential_window_minutes: 10  # Longer window
```