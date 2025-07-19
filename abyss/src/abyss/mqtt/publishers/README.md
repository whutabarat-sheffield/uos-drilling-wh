# MQTT Publishers Module

A unified MQTT publishing module for drilling data with support for multiple operation modes and realistic drilling patterns.

## Overview

This module consolidates all MQTT publishing functionality into a single, maintainable package with three main modes:

1. **Standard Mode**: Normal publishing with realistic drilling operation patterns
2. **Stress Test Mode**: High-performance mode using threading for load testing
3. **Async Mode**: Ultra-high-performance mode using asyncio (requires aiomqtt)

## Features

- **Realistic Drilling Patterns**: Simulates actual drilling operations (normal, active drilling, bit change, calibration, idle, maintenance)
- **Signal Tracking**: Optional tracking of published signals with unique IDs
- **Test Data Support**: Automatically discovers and loads test data from directory structures
- **Configurable**: YAML-based configuration for broker settings and topic structures
- **High Performance**: Stress test mode can achieve 1000+ signals/second
- **Docker Compatible**: Works seamlessly with existing Docker configurations

## Installation

The module is part of the abyss package. No additional installation required for standard and stress test modes.

For async mode, install the additional dependency:
```bash
pip install aiomqtt
```

## Usage

### Command Line Interface

```bash
# Standard mode with realistic patterns
python -m abyss.mqtt.publishers test_data

# Standard mode without patterns (fixed intervals)
python -m abyss.mqtt.publishers test_data --no-patterns

# Stress test mode (1000 signals/sec for 60 seconds)
python -m abyss.mqtt.publishers test_data --stress-test --rate 1000 --duration 60

# With custom configuration
python -m abyss.mqtt.publishers test_data -c config/mqtt_conf.yaml

# With signal tracking
python -m abyss.mqtt.publishers test_data --track-signals --signal-log tracking.csv

# Infinite publishing (0 repetitions)
python -m abyss.mqtt.publishers test_data -r 0
```

### Docker Usage

The module is fully compatible with existing Docker configurations:

```yaml
# docker-compose.yml
services:
  publisher:
    image: uos-publish-json:latest
    command: python -m abyss.mqtt.publishers /app/test_data -c /app/config/mqtt_conf_docker.yaml
```

### Programmatic Usage

```python
from abyss.mqtt.publishers import StandardPublisher, PublisherConfig

# Create configuration
config = PublisherConfig(
    broker_host="localhost",
    broker_port=1883,
    test_data_path=Path("test_data"),
    repetitions=10,
    track_signals=True
)

# Run publisher
publisher = StandardPublisher(config)
publisher.run()
```

## Configuration

The module uses YAML configuration files with the following structure:

```yaml
mqtt:
  broker:
    host: "localhost"
    port: 1883
    username: ""  # Optional
    password: ""  # Optional
  
  listener:
    root: "OPCPUBSUB"
    result: "ResultManagement"
    trace: "ResultManagement/Trace"
    heads: "AssetManagement/Heads"
```

## Test Data Structure

Test data should be organized in directories containing the following JSON files:
- `ResultManagement.json`
- `Trace.json`
- `Heads.json`

Example structure:
```
test_data/
├── data_20250326/
│   ├── ResultManagement.json
│   ├── Trace.json
│   └── Heads.json
└── data_20250623/
    ├── ResultManagement.json
    ├── Trace.json
    └── Heads.json
```

## Drilling Patterns

The module simulates realistic drilling operations with the following patterns:

- **Normal Operation**: Regular drilling with moderate message rates
- **Active Drilling**: High-frequency messages during active drilling
- **Bit Change**: Reduced activity during tool changes
- **Calibration**: Periodic calibration sequences
- **Idle**: Minimal activity during idle periods
- **Maintenance**: Very low activity during maintenance

Patterns automatically transition based on realistic probabilities and durations.

## Performance

- **Standard Mode**: 10-50 signals/second with realistic patterns
- **Stress Test Mode**: Up to 1000+ signals/second (hardware dependent)
- **Async Mode**: 2000+ signals/second (requires aiomqtt)

## Compatibility

The module includes a compatibility wrapper (`uos_publish_wrapper.py`) that ensures backwards compatibility with existing scripts and Docker configurations. The wrapper automatically redirects to the new module while maintaining the same command-line interface.

## Architecture

```
mqtt/publishers/
├── __init__.py          # Module exports
├── __main__.py          # CLI entry point
├── base.py              # Base publisher class and configuration
├── patterns.py          # Drilling operation patterns
├── standard.py          # Standard publishing mode
├── stress.py            # High-performance stress test mode
├── async_publisher.py   # Ultra-high async mode (optional)
└── README.md            # This file
```

## Migration from Legacy Scripts

To migrate from the old scripts (`uos_publish_json.py`, etc.):

1. Update command from:
   ```bash
   python uos_publish_json.py test_data
   ```
   
2. To:
   ```bash
   python -m abyss.mqtt.publishers test_data
   ```

The compatibility wrapper ensures zero downtime during migration.