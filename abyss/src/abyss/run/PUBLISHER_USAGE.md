# MQTT Publisher Usage Guide

## Overview

The `uos_publish_wrapper.py` provides a unified interface to different MQTT publishers, each optimized for specific use cases.

## Publisher Modes

### 1. Normal Mode (Default)
```bash
python uos_publish_wrapper.py /path/to/data
# or
python uos_publish_json.py /path/to/data
```
- **Behavior**: Updates timestamps to current time
- **Message order**: Random shuffling
- **Delays**: Random delays between messages (configurable)
- **Use case**: Simulating real-time data flow

### 2. Simple Mode (For Exact Matching) ⭐
```bash
python uos_publish_wrapper.py --simple /path/to/data
# or
python uos_publish_wrapper.py --exact-match /path/to/data
# or directly
python uos_publish_json_simple.py /path/to/data
```
- **Behavior**: Preserves original timestamps from JSON files
- **Message order**: Fixed order (Result, Trace, Heads)
- **Delays**: No delays between message types
- **Validation**: Ensures all three files have matching timestamps
- **Use case**: Testing exact timestamp matching system

### 3. Stress Test Mode
```bash
python uos_publish_wrapper.py --stress-test /path/to/data
# or
python uos_publish_json_stress.py /path/to/data
```
- **Behavior**: High-performance multi-threaded publishing
- **Use case**: Load testing and performance benchmarking

### 4. Async Mode
```bash
python uos_publish_wrapper.py --async /path/to/data
# or
python uos_publish_json_async.py /path/to/data
```
- **Behavior**: Ultra-high-performance async publishing
- **Requirements**: `pip install aiomqtt`
- **Use case**: Maximum throughput testing

## Common Arguments

All publishers support these common arguments:
- `--conf`: Configuration file path (default: mqtt_conf_local.yaml)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `--repetitions`: Number of times to publish the dataset

## Environment Variables

- `LOG_LEVEL`: Set logging level (takes priority over --log-level)

## For Exact Matching System

**Always use the simple mode** when testing the exact matching system:
```bash
# Correct for exact matching
python uos_publish_wrapper.py --simple /path/to/data

# Incorrect - will modify timestamps
python uos_publish_wrapper.py /path/to/data
```

The simple publisher ensures:
1. Original timestamps are preserved
2. All three message types have identical timestamps
3. Messages are published atomically (no delays)
4. Deterministic behavior for testing