# MQTT Drilling Data Analysis System Documentation

This documentation provides a guide to the MQTT Drilling Data Analysis System, which processes drilling data from MQTT messages to perform depth estimation.

## Overview

The system consists of:
1. A Python module (`abyss.mqtt.example_usage`) that runs the MQTT drilling data analyzer
2. Configuration files (`mqtt_conf_local.yaml` and `mqtt_conf_docker.yaml`) for customizing connection settings and data mappings
3. A component-based architecture with parallel processing capabilities

The system listens for two types of messages:
- Result messages: Contains metadata and operation results
- Trace messages: Contains detailed trace data for drilling operations

When matching result and trace messages are received, the system processes them to estimate drilling depth and publishes the results back to the MQTT broker.

## Installation

### Prerequisites

- Python 3.9+
- The `abyss` package (with the `uos_depth_est_core` module)

### Installation Steps

1. Install required dependencies from the provided tarballs. Note that these tarballs have the same name as publicly available sources but the provided packages contain custom modifications:

```bash
pip install transformers-4.41.0.dev0.tar.gz
pip install tsfm_public-0.2.17.tar.gz
pip install accelerate-1.2.1.tar.gz
```

2. Install the required algorithm along with the trained model:

```bash
pip install abyss-0.1.0.tar.gz
```

3. Navigate to the MQTT module directory:

```bash
cd abyss/src/abyss/mqtt
```

## Configuration

The `mqtt_conf.yaml` file contains all necessary configuration for the MQTT connection and data processing:

### Key Configuration Sections

1. **Broker Settings**: Connection details for the MQTT broker
   ```yaml
   broker:
     host: host.docker.internal
     port: 1883
     username: ""  # Optional
     password: ""  # Optional
   ```

2. **Topic Structure**: Defines the MQTT topics to listen for messages
   ```yaml
   listener:
     root: "OPCPUBSUB"
     toolboxid: "+"  # "+" is a wildcard
     toolid: "+"
     result: "ResultManagement"
     trace: "ResultManagement/Trace"
   ```

3. **Data Field Mappings**: Paths to extract specific data from messages
   ```yaml
   data_ids:
     machine_id: 'ResultManagement.Results.0.ResultMetaData.SerialNumber'
     position: 'ResultManagement.Results.0.ResultContent.Trace.StepTraces.PositionTrace.StepTraceContent[0].Values'
     # additional mappings...
   ```

4. **Estimation Output**: Topics for publishing depth estimation results
   ```yaml
   estimation:
     keypoints: 'ResultManagement.Results.0.ResultContent.DepthEstimation.KeyPoints'
     depth_estimation: 'ResultManagement.Results.0.ResultContent.DepthEstimation.DepthEstimation'
   ```

## Usage

### Basic Usage

Run the script with default configuration:

```bash
python example_usage.py
```

### Command-line Arguments

The script supports the following command-line arguments:

- `--config`: Path to YAML configuration file (default: mqtt_conf_local.yaml)
- `--log-level`: Logging level (choices: DEBUG, INFO, WARNING, ERROR, CRITICAL; default: INFO)

Example:
```bash
python example_usage.py --config=mqtt_conf_docker.yaml --log-level=DEBUG
```

### Logging

The system uses Python's built-in logging module with configurable levels:
- DEBUG: Detailed information, typically for debugging
- INFO: Confirmation that things are working as expected
- WARNING: Something unexpected happened, but the system is still working
- ERROR: Serious problems that prevent some functionality
- CRITICAL: Critical errors that prevent the system from working

## How It Works

1. **Message Listening**: The system creates multiple MQTT clients to listen for result, trace, and heads messages.

2. **Message Buffering**: Messages are buffered in thread-safe buffers with deduplication based on:
   - Same toolbox ID and tool ID
   - Timestamps within the configured time window (default: 5 seconds)
   - Configurable duplicate handling (ignore, replace, error)

3. **Message Correlation**: The SimpleMessageCorrelator finds matching message sets across buffers.

4. **Parallel Processing**: Matched message sets are submitted to a ProcessingPool with multiple workers:
   - Each worker loads its own DepthInference model (~1GB memory)
   - Default: 10 workers for parallel processing
   - Non-blocking Future-based result handling

5. **Depth Validation**: Results are validated with configurable behaviors:
   - Publish, skip, or warn on negative depth values
   - Sequential negative depth tracking

6. **Results Publishing**: Formatted results are published back to the MQTT broker

7. **System Monitoring**: Throughput monitoring tracks if the system is keeping up with message arrival rate

## Example Use Cases

### Industrial Drilling Monitoring

Monitor drilling operations in real-time by analyzing depth data:

```bash
python example_usage.py --log-level=INFO
```

The system will:
1. Connect to the configured MQTT broker
2. Listen for drilling data messages
3. Process and analyze depth information
4. Publish depth estimation results
5. Save matched data pairs for further analysis

### Debugging Data Issues

To troubleshoot data processing problems:

```bash
python example_usage.py --log-level=DEBUG
```

This will provide detailed logs about:
- Message receiving and parsing
- Buffer management
- Matching process
- Data processing steps

### Custom Configuration

For different environments or data structures:

1. Create a custom configuration file (e.g., `production_conf.yaml`)
2. Update broker settings, topic structure, and data mappings
3. Run with the custom configuration:
   ```bash
   python example_usage.py --config=production_conf.yaml
   ```

## Key Components

### DrillingDataAnalyser

The main orchestrator that coordinates all components:
- Manages MQTT client lifecycle
- Coordinates message buffering and correlation
- Submits work to processing pool
- Handles result publishing
- Monitors system health

### ProcessingPool (SimpleProcessingPool)

Manages parallel processing of depth inference tasks:
- Worker pool management (default: 10 workers)
- Model initialization in each worker process
- Future tracking and result handling
- Performance metrics collection

### MessageBuffer

Thread-safe buffer for storing incoming MQTT messages:
- Store messages with deduplication
- Provide thread-safe access
- Track buffer health metrics
- Handle buffer overflow protection

### SimpleMessageCorrelator

Correlates related messages within a time window:
- Find matching messages across buffers
- Process complete message sets
- Clean up old messages
- Maintain correlation metrics

### ResultPublisher

Publishes processing results to MQTT:
- Publish depth estimation results
- Delegate formatting to ResultMessageFormatter
- Delegate validation to DepthValidator
- Handle publish errors

### DepthValidator

Configurable validation for depth estimation results:
- Three validation behaviors: publish, skip, warning
- Sequential negative depth tracking
- Configurable alert thresholds
- Validation statistics

### ConfigurationManager

Centralized configuration management:
- YAML configuration loading
- Default value handling
- Configuration validation
- Safe nested access with get() method

## Customization

To adapt the system to your specific needs:

1. Update the `mqtt_conf.yaml` file with your specific:
   - MQTT broker details
   - Topic structure
   - Data field mappings

2. Modify the depth inference processing in the `process_matching_messages()` method

3. Adjust the time window and cleanup intervals in the `DrillingDataAnalyser` class

## Troubleshooting

Common issues and solutions:

1. **Connection Problems**:
   - Verify broker host and port in config file
   - Check network connectivity
   - Ensure credentials are correct if authentication is required

2. **No Messages Being Processed**:
   - Verify topic structure in configuration matches actual messages
   - Check log for subscription confirmations
   - Use `--log-level=DEBUG` for detailed connection information

3. **Data Extraction Issues**:
   - Ensure data_ids paths in config match actual message structure
   - Save and examine raw messages using DEBUG mode
   - Verify the format of timestamp fields

4. **Depth Estimation Failures**:
   - Ensure enough steps are available in the data (minimum 2)
   - Check the specific error messages in the logs
   - Verify the depth inference model is properly initialized

For persistent issues, examine the logs and saved data files to identify the source of the problem.
