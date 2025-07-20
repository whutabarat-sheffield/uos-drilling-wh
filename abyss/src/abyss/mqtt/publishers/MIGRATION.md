# Migration Guide: MQTT Publishers

This guide helps migrate from the old publisher scripts to the new unified module.

## Quick Start

### Old Way
```bash
python uos_publish_json.py test_data
python uos_publish_json_stress.py test_data --rate 1000
```

### New Way
```bash
# Using module directly
python -m abyss.mqtt.publishers test_data
python -m abyss.mqtt.publishers test_data --stress-test --rate 1000

# Note: CLI command 'uos_publisher' is not currently implemented
# Use the module form above instead
```

## Docker Usage

### Old Way
```bash
docker run --rm uos-publish-json:latest python uos_publish_json.py /data
```

### New Way
```bash
docker run --rm uos-publish-json:latest python -m abyss.mqtt.publishers /data
```

## Key Improvements

1. **Unified Interface**: All modes (standard, stress, async) in one module
2. **Realistic Patterns**: Simulates actual drilling operations
3. **Better Performance**: Optimized message batching and threading
4. **Signal Tracking**: Built-in support across all modes
5. **Cleaner Code**: No circular dependencies or import issues

## Backwards Compatibility

A compatibility wrapper can be created to redirect old scripts to the new module, ensuring zero downtime during migration. The wrapper would need to be implemented to support legacy command-line interfaces.

## Command Line Changes

### Stress Test Mode
- Old: Run `uos_publish_json_stress.py`
- New: Add `--stress-test` flag

### Async Mode
- Old: Run `uos_publish_json_async.py` (if it existed)
- New: Add `--async-mode` flag (requires aiomqtt installation)
- Note: Async mode is referenced in CLI but async_publisher.py implementation is not present

### Signal Tracking
- Old: Only in main script with `--track-signals`
- New: Available in all modes with `--track-signals`

## Configuration

The YAML configuration format remains unchanged. The new module uses the same configuration structure as the old scripts.

## Test Data

The new module properly iterates through all subdirectories containing the required JSON files (ResultManagement.json, Trace.json, Heads.json).

## For Developers

When building the Docker image, the new module is automatically included as part of the abyss package. No changes to the build process are required.