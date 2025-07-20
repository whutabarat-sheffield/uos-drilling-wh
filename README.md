# Deep Learning Drilling Depth Estimation System Documentation

This documentation provides a guide to the Deep Learning Drilling Depth Estimation System, which processes drilling data from MQTT messages to perform depth estimation.

### Changes in 0.2.7

**Organizational & Quality Release** - This version focuses on repository structure, documentation, and comprehensive test coverage improvements.

1. **Repository Reorganization**: Complete restructuring for better maintainability and user experience
   - Renamed `_sandbox` to `examples/` with organized subdirectories for research, MQTT, and notebooks
   - Centralized `config/` directory with deployment templates and environment-specific configurations
   - Added `GETTING_STARTED.md` and `REPOSITORY_LAYOUT.md` for better user onboarding
   - Enhanced Makefile with consolidated build targets and deployment automation

2. **Comprehensive Test Infrastructure**: Major improvements to test coverage and organization
   - Enhanced 4 major test files with critical missing coverage (test_uos_publish_json.py: 87% coverage)
   - Added test_simple_correlator.py with comprehensive correlation testing 
   - Improved test_result_publisher_head_id.py with focused HeadId testing
   - Created unit/integration test separation with shared test utilities
   - Added README_TEST_STRATEGY.md with environment variables and CI/CD guidance

3. **System Robustness & Bug Fixes**: Multiple reliability improvements
   - Fixed MQTT Paho v2 compatibility issues with success reason codes
   - Improved logging messages in MessageProcessor for better debugging
   - Enhanced MQTT configuration for analyzer settings
   - Fixed correlation timing and configuration path mismatches
   - Updated ConfigurationManager with singleton pattern for better resource management

4. **Configuration & Deployment Enhancements**: Better production readiness
   - Added configuration templates for MQTT and Docker deployments
   - Enhanced depth validation configuration options
   - Improved Docker support with lightweight publisher configurations
   - Standalone MQTT publisher module for flexible deployment scenarios

5. **Documentation & Developer Experience**: Significant user experience improvements
   - Comprehensive Getting Started guide with role-based navigation
   - Repository layout documentation explaining structure and organization
   - Enhanced configuration documentation with production examples
   - Developer-focused documentation for contributing and development setup

6. **Code Quality & Maintenance**: Cleaner, more maintainable codebase
   - Removed unused async MQTT components to reduce complexity
   - Consolidated duplicate test files and improved test organization
   - Enhanced build system with better automation and validation
   - Improved error handling and logging throughout the system

**Full Changelog**: https://github.com/whutabarat-sheffield/uos-drilling-wh/compare/v0.2.6-stable...v0.2.7

### Changes in 0.2.6

**Major Feature Release** - This version includes significant architectural improvements and new capabilities.

1. **Negative Depth Handling**: Implemented detection and handling of negative depth estimation values for air drilling scenarios
   - Added EMPTY result type that skips MQTT publication for invalid (negative) depth values
   - Enhanced warning system with individual and sequential (5+ in 5 minutes) negative depth warnings
   - Added negative depth occurrence tracking with windowed statistics

2. **MQTT Publisher Stress Testing Suite**: Comprehensive performance testing framework
   - Three modes: Threading (~80-100 signals/sec), Async (with aiomqtt), and Batch publishing
   - Smart Docker network detection and auto-configuration
   - Real-time performance metrics (throughput, latency, success rate)
   - Detailed stress testing guide and helper scripts

3. **Docker Build System Overhaul**: Optimized build performance with caching
   - BuildKit enabled by default, reducing rebuild times from 15min to 30s
   - Cache-aware builds with optional --no-cache flag
   - New cache management utilities and comprehensive Makefile targets
   - Docker caching guide documentation

4. **MessageBuffer Enhancements**: Improved monitoring and reliability
   - Rate-limited warning system for high drop rates and old messages
   - Performance metrics tracking and thread-safety improvements
   - Comprehensive test suites for load, thread-safety, and warnings

5. **Portainer Stack Automation**: Streamlined deployment updates
   - Automated stack updates with Git-based change detection
   - Health checking and secure API integration
   - Comprehensive documentation and configuration templates

6. **Training Platform Architecture**: Foundation for user training capabilities
   - Designed dual codebase architecture (PyTorch training + ONNX deployment)
   - Planned GUI annotation tools and hybrid labeling workflows
   - Extensive architecture documentation and roadmap

7. **Performance Optimizations**: Reduced logging overhead
   - Changed multiple logging statements from INFO to DEBUG level
   - Conditional progress bars based on logging level
   - Improved overall system performance

8. **Architectural Simplifications**: Cleaner codebase
   - Removed MessageCorrelator class for simpler message handling
   - Streamlined correlation logic
   - Enhanced test coverage

**Full Changelog**: https://github.com/whutabarat-sheffield/uos-drilling-wh/compare/v0.2.5-stable...v0.2.6

### Changes in 0.2.5
1. Enhanced duplicate message handling with configurable strategies: "ignore", "replace", "error"
2. Improved logging format with structured information display and millisecond precision timestamps
3. Added comprehensive build automation with GitHub Actions workflows for wheel creation and validation
4. Enhanced message processing with better duplicate detection and validation across correlator and converter modules
5. Updated configuration management with typed access methods and backward compatibility

### Changes in 0.2.4
1. Added the HeadId element in the published results.
2. Major reorganisation and refactoring of the codebase. Key files are now under `src/abyss/mqtt` and `src/abyss/run`

### Changes in 0.2.3
1. Modified the arrangement of configuration files within a Docker instance. This is to avoid future problems due to mistaken location of config files.

### Changes in 0.2.2
1. Addressed a bug that led to the subscriber always dropping the first MQTT message received. This is done by ensuring that the main function calls the listening thread and the analysis thread independently.

### Changes in 0.2.0
1. Replaced the two-point algorithm with the three-point one, providing entry, transition, and exit location estimates.
![image](https://github.com/user-attachments/assets/5ba444db-c169-4ccb-8041-eb77e0acc99a)

### Changes in 0.1.1
1. Switch to wheels instead of tarballs to further reduce installation size and time
2. Ability to run through an executable in addition to the Python script
3. Addressed primary and secondary bugs due to repurposing of topic names
4. Added more contextual information: the published message now contains the depth estimation results along with the source timestamp, the ResultID, the Tool serial number and the algorithm version
5. Updated the Dockerfile; now it works (tested on Sheffield machine)

## Overview

Currently, the system consists of:
1. A refactored MQTT drilling data analyzer (`uos_depth_est_mqtt.py`) that provides improved message handling and processing
2. Multiple configuration files (`mqtt_conf_local.yaml`, `mqtt_conf_docker.yaml`) for different deployment environments
3. Modular components for message buffering, correlation, processing, and result publishing
4. Comprehensive Docker support with CPU and GPU variants

The system listens for two types of MQTT messages:
- Result messages: Contains metadata and operation results
- Trace messages: Contains detailed trace data (position, torque, thrust) for drilling operations

When matching result and trace messages are received, the system processes them to estimate drilling depth and publishes the results back to the MQTT broker. At the moment the mapping is done by reusing the relevant sections of the MQTT topic.

## Installation

### Prerequisites

- Python 3.9+
- The `abyss` package
- Key dependencies, namely `transformers`, `tsfm_public`, and `accelerate`
- For local testing, an MQTT broker such as `mosquitto` should already be installed
- Docker for virtual machine installation


### Steps for virtual machine installation (using Docker)

1. Clone the entire repository
2. Build the Docker images for publisher and listener
3. Run the Docker images

Details on step #1
```bash
git clone https://github.com/whutabarat-sheffield/uos-drilling-wh.git .
```

Details on step #2 and #3
```bash
#  step #2 build the listener
docker build -t listener .
```

```
# step #3 run the listener
docker run -t listener
```

The listener is callable from the command line as:
```bash 
uos_depthest_listener --config=mqtt_conf_local.yaml --log-level=INFO
```

The docker build will setup a workspace under `/app` directory, with configuration files available in `/app/config/`. 

### Steps for direct installation using sources

1. Clone the repository:
   ```bash
   git clone https://github.com/whutabarat-sheffield/uos-drilling-wh.git
   cd uos-drilling-wh
   ```

2. **Option A: Install directly from source**
   ```bash
   cd abyss/
   pip install .
   ```

3. **Option B: Build wheel and install**
   ```bash
   # Build wheel using the provided build script
   python build_wheel.py --clean --validate
   
   # Install the built wheel
   pip install dist/abyss-*.whl
   ```

4. **Option C: Use Makefile (recommended)**
   ```bash
   # Build and install in one step
   make install
   
   # Or build wheel only
   make build
   
   # Show available targets
   make help
   ```

### Steps for direct installation using wheels
1. Download all of the files in the `release` section apart from the source code files. 
<img width="920" alt="image" src="https://github.airbus.corp/Airbus/uos-drilling/assets/13312/398b8773-cabb-466d-8893-3a5db5f5c1b5">

2. Place them into a single directory, such as `dist`. The exceptions are `listen-continuous.py` and `mqtt_conf.yaml`, which should be placed in a project directory such as `run`. For example:

[<img src="https://github.airbus.corp/Airbus/uos-drilling/assets/13312/1460870b-45b2-4fc7-8bcd-1401e4c8bc35" width="400"/>](image.png)

When running with the new `uos_dephest_listener`, only `mqtt_conf.yaml` needs to be copied into the directory.

7. Install the required dependencies as well as the algorithm from the provided wheels. This is easily done by a single command:
```bash
cd dist
pip install -r requirements.txt
```
This process can take multiple minutes as the installation will download additional dependencies that are not yet available within the Python environment. 

On an Airbus machine:
<img width="661" alt="image" src="https://github.airbus.corp/Airbus/uos-drilling/assets/13312/27a12a95-ddf5-4007-aee6-ae360b2a234d">


8. To test with the python script, go to the `run` directory that contains both `listen-continuous.py` and `mqtt_conf.yaml`, and run it:
```bash
python listen-continuous.py
```
To test with the executable, go to the `run` directory that contains `mqtt_conf.yaml`, and run the executable:
```bash
uos_dephest_listener &
```

## Configuration

The configuration files (`mqtt_conf_local.yaml` for local development, `mqtt_conf_docker.yaml` for Docker deployment) contain all necessary configuration for the MQTT connection and data processing. They are fully commented and include new features like duplicate message handling. More details are provided below under [Usage](#usage).

### Key Configuration Sections

1. **Broker Settings**: Connection details for the MQTT broker; the example below shows the host definition that is needed for testing on Docker images
   ```yaml
   broker:
     host: host.docker.internal
     port: 1883
     username: ""  # Optional
     password: ""  # Optional
   ```

2. **Duplicate Handling**: Configurable strategy for handling duplicate messages
   ```yaml
   listener:
     duplicate_handling: "ignore"  # Options: "ignore", "replace", "error"
   ```

3. **Topic Structure**: Defines the MQTT topics to listen for messages. This was based on the design shared in December 2024.
   ```yaml
   listener:
     root: "OPCPUBSUB"
     toolboxid: "+"  # "+" is a wildcard
     toolid: "+"
     result: "ResultManagement"
     trace: "ResultManagement/Trace"
     heads: "AssetManagement/Head"
   ```

4. **Data Field Mappings**: Paths to extract specific data from messages. This was based on the design shared in December 2024.
   ```yaml
   data_ids:
     machine_id: 'ResultManagement.Results.0.ResultMetaData.SerialNumber'
     head_id: 'AssetManagement.Assets.Heads.0.Identification.SerialNumber'
     result_id: 'ResultManagement.Results.0.ResultMetaData.ResultId'
     position: 'ResultManagement.Results.0.ResultContent.Trace.StepTraces.PositionTrace.StepTraceContent[0].Values'
     thrust: 'ResultManagement.Results.0.ResultContent.Trace.StepTraces.IntensityThrustTrace.StepTraceContent[0].Values'
     torque: 'ResultManagement.Results.0.ResultContent.Trace.StepTraces.IntensityTorqueTrace.StepTraceContent[0].Values'
     # additional mappings...
   ```

9. **Estimation Output**: Topics for publishing depth estimation results. This can be changed to suit.
   ```yaml
   estimation:
     keypoints: 'ResultManagement.Results.0.ResultContent.DepthEstimation.KeyPoints'
     depth_estimation: 'ResultManagement.Results.0.ResultContent.DepthEstimation.DepthEstimation'
   ```

## Usage

### Basic Usage

Run the MQTT drilling data analyzer with default configuration:

```bash
uos_depthest_listener
```

Or with specific configuration:

```bash
uos_depthest_listener --config=mqtt_conf_docker.yaml --log-level=DEBUG
```


### Command-line Arguments

The script supports the following command-line arguments:

- `--config`: Path to YAML configuration file (default: mqtt_conf_local.yaml)
- `--log-level`: Logging level (choices: DEBUG, INFO, WARNING, ERROR, CRITICAL; default: INFO)

Example:
```bash
uos_depthest_listener --config=mqtt_conf_docker.yaml --log-level=DEBUG
```

### Logging

The system uses an enhanced logging format with structured information display:
```
[2024-07-08 14:23:45.123] INFO     MessageProcessor | Processing matched messages [result_id=12345, tool_key=toolbox1/tool1]
```

Logging levels:
- DEBUG: Detailed information, typically for debugging. Reconstructed JSON files are written to disk.
- INFO: Confirmation that things are working as expected. Many tracking messages with structured data are printed.
- WARNING: Something unexpected happened, but the system is still working
- ERROR: Serious problems that prevent some functionality
- CRITICAL: Critical errors that prevent the system from working

## How It Works

1. **Message Listening**: The system creates an MQTT client to listen for result, trace, and heads messages using modular components.

2. **Message Buffering**: Messages are buffered with configurable duplicate handling:
   - "ignore": Skip duplicate messages (default)
   - "replace": Replace existing duplicates with new messages  
   - "error": Raise an error when duplicates are detected

3. **Message Correlation**: Messages are matched based on:
   - Same toolbox ID and tool ID
   - Timestamps within the configured time window (default: 30 seconds)
   - Uses either simple key-based or time-bucket correlation strategies

4. **Processing**: When matching messages are found:
   - Data is extracted according to the configured mappings
   - Head ID information is included for better traceability
   - Matched pairs are saved to file for reference (in DEBUG mode only)
   - The depth inference model processes the data

5. **Results Publishing**: Depth estimation results with head ID are published back to the MQTT broker

6. **Cleanup**: Old messages are periodically removed from buffers with configurable intervals

## Example Use Cases

### Industrial Drilling Monitoring

Monitor drilling operations in real-time by analyzing depth data:

```bash
uos_depthest_listener --log-level=INFO
```

The system will:
1. Connect to the configured MQTT broker
2. Listen for drilling data messages
3. Process and analyse depth information
6. Publish depth estimation results
7. Save matched data pairs for further analysis

### Debugging Data Issues

To troubleshoot data processing problems:

```bash
uos_depthest_listener --log-level=DEBUG
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
   uos_depthest_listener --config=production_conf.yaml
   ```

## Key Components

The system is now built with modular components for better maintainability and testing:

### Core Components

1. **DrillingDataAnalyser**: Main orchestrator class that coordinates all components
2. **ConfigurationManager**: Handles configuration loading and validation with typed access
3. **MessageBuffer**: Manages message buffering with configurable duplicate handling
4. **MessageCorrelator/SimpleMessageCorrelator**: Correlates messages based on timestamps and tool IDs
5. **MessageProcessor**: Processes matched message groups and extracts data
6. **DataConverter**: Converts extracted data for depth estimation
7. **ResultPublisher**: Publishes depth estimation results back to MQTT
8. **ClientManager**: Manages MQTT client connections and subscriptions

### Data Structures

- **TimestampedData**: Stores message data with timestamp and source information
- **ProcessedMessage**: Contains processed data ready for depth estimation

### Key Features

- **Duplicate Detection**: Configurable strategies for handling duplicate messages
- **Enhanced Logging**: Structured logging with millisecond precision timestamps
- **Head ID Support**: Includes drilling head identification in results
- **Error Handling**: Comprehensive error handling with specific exception types

## Customization

To adapt the system to your specific needs:

1. Update the appropriate configuration file (`mqtt_conf_local.yaml` or `mqtt_conf_docker.yaml`) with your specific:
   - MQTT broker details
   - Topic structure and data field mappings
   - Duplicate handling strategy
   - Time windows and cleanup intervals

2. Customize message processing by modifying the `MessageProcessor` component

3. Adjust correlation strategies by selecting between `SimpleMessageCorrelator` and `MessageCorrelator`

4. Configure deployment-specific settings:
   - For local development: Use `mqtt_conf_local.yaml`
   - For Docker deployment: Use `mqtt_conf_docker.yaml`
   - For Portainer deployment: Configure volume mounts appropriately

## Troubleshooting

Common issues and solutions:

1. **Connection Problems**:
   - Verify broker host and port in config file
   - Check network connectivity
   - Ensure credentials are correct if authentication is required
   - For testing in linux localhost, update the broker host to 'localhost'
   - Ensure that a broker such as mosquitto is installed. Testing in Windows without a broker present creates this issue:
 
![image](https://github.airbus.corp/Airbus/uos-drilling/assets/13312/cf83fca7-a50e-485c-84ef-081efa391285)


2. **No Messages Being Processed**:
   - Verify topic structure in configuration matches actual messages
   - Check log for subscription confirmations
   - Use `--log-level=DEBUG` for detailed connection information

3. **Data Extraction Issues**:
   - Ensure data_ids paths in the config file match actual message structure. If the names of the topics have changed, then the appropriate keys needs to be updated 
```yaml
  # Data Field Mappings
  # -----------------
  # These paths define where to find specific data fields in the message payload
  # TODO: Update these paths based on the actual data structure
  data_ids:
    # Machine identification
    machine_id: 'ResultManagement.Results.0.ResultMetaData.SerialNumber'
    
    # Trace data paths
    position: 'ResultManagement.Results.0.ResultContent.Trace.StepTraces.PositionTrace.StepTraceContent[0].Values'
    thrust: 'ResultManagement.Results.0.ResultContent.Trace.StepTraces.IntensityThrustTrace.StepTraceContent[0].Values'
    torque: 'ResultManagement.Results.0.ResultContent.Trace.StepTraces.IntensityTorqueTrace.StepTraceContent[0].Values'
    step: 'ResultManagement.Results.0.ResultContent.Trace.StepTraces.StepNumberTrace.StepTraceContent[0].Values'
    
    # Empty value references
    torque_empty_vals: 'ResultManagement.Results.0.ResultContent.StepResults.0.StepResultValues.IntensityTorqueEmpty'
    thrust_empty_vals: 'ResultManagement.Results.0.ResultContent.StepResults.0.StepResultValues.IntensityThrustEmpty'
    step_vals: 'ResultManagement.Results.0.ResultContent.StepResults.0.StepResultValues.StepNumber'
```
   - Save and examine raw messages using DEBUG mode
   - Verify the format of timestamp fields, which should follow this format: `"%Y-%m-%dT%H:%M:%SZ"`


4. **Depth Estimation Failures**:
   - Ensure that the PSTEP in the xls file looks exactly as the screengrab below: 
![image](https://github.airbus.corp/Airbus/uos-drilling/assets/13312/031ac92f-2373-431b-b752-9bb8bf88be68)
   - As shown in the screengrab, the current model is trained on a specific drilling configuration that has two different types of program steps. 
       + If the data only have one program step type, the algorithm fails. 
       + If the data have three or more program step types, the depth prediction will be meaningless.
   - Check the specific error messages in the logs
   - Verify the depth inference model is properly initialised, with messages such as below:
```bash
INFO:p-3643:t-140230986614592:config.py:<module>:PyTorch version 2.3.1+cpu available.
INFO:p-3643:t-140230986614592:listen-continuous.py:__init__:Loaded configuration from mqtt_conf.yaml
/home/windo/mambaforge/lib/python3.10/site-packages/abyss/trained_model/has_tool_age_predrilling/cv4
```

## What's Changed
* Deploy by @whutabarat-sheffield in https://github.com/whutabarat-sheffield/uos-drilling-wh/pull/2
* Vscode docker by @whutabarat-sheffield in https://github.com/whutabarat-sheffield/uos-drilling-wh/pull/3


**Full Changelog**: https://github.com/whutabarat-sheffield/uos-drilling-wh/compare/v0.2.4-stable...v0.2.5

## New Features in 0.2.5

### Duplicate Message Handling
Configure how the system handles duplicate messages:
```yaml
listener:
  duplicate_handling: "replace"  # Options: "ignore", "replace", "error"
```

### Enhanced Logging
Structured logging format with detailed information:
```
[2024-07-08 14:23:45.123] INFO     SimpleMessageCorrelator | Duplicate messages detected and ignored [result_duplicates=2, trace_duplicates=1, heads_duplicates=0, total_duplicates=3]
```

### Build Automation
- GitHub Actions workflows for automated wheel building
- Comprehensive validation and testing
- Docker multi-platform support

### Improved Configuration Management
- Typed configuration access with `ConfigurationManager`
- Backward compatibility with raw configuration dictionaries
- Better validation and error handling

## Build Automation

The repository includes comprehensive build automation tools for creating wheel distributions:

### Build Scripts

1. **Python Build Script** (`build_wheel.py`):
   ```bash
   # Basic build
   python build_wheel.py
   
   # Build with clean and validation
   python build_wheel.py --clean --validate --verbose
   
   # Custom output directory
   python build_wheel.py --output-dir=wheels/
   ```

2. **Shell Script Wrapper** (`build_wheel.sh`):
   ```bash
   # Simple build
   ./build_wheel.sh
   
   # With options
   ./build_wheel.sh --clean --validate --verbose
   ```

3. **Makefile Targets**:
   ```bash
   make help           # Show available targets
   make build          # Build wheel with clean and validation
   make build-simple   # Quick build without clean/validation
   make install        # Build and install wheel locally
   make clean          # Clean build artifacts
   make wheel-info     # Show information about built wheels
   ```

### Build Features

- **Automatic dependency management**: Installs required build tools
- **Validation**: Tests wheel integrity and installation
- **Cleanup**: Removes build artifacts and temporary files
- **Cross-platform**: Works on Linux, macOS, and Windows
- **Verbose logging**: Detailed build process information
- **Error handling**: Comprehensive error detection and reporting

### GitHub Actions Integration

The build scripts are integrated with GitHub Actions workflows for:
- Automated wheel building on push/PR
- Multi-platform wheel creation (Linux, macOS, Windows)
- Wheel validation and testing
- Release automation
