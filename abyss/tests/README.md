# Abyss Test Suite

This directory contains the test suite for the Abyss MQTT drilling data analysis system.

## Directory Structure

- `unit/`: Unit tests for individual components
- `integration/`: Integration tests for component interactions  
- `performance/`: Performance and load tests
- `e2e/`: End-to-end system tests
- `fixtures/`: Shared test fixtures and utilities
- `conftest.py`: Central pytest configuration

## Running Tests

### Run all tests
```bash
cd abyss
pytest
```

### Run specific test types
```bash
# Unit tests only
pytest tests/unit

# Integration tests
pytest tests/integration -m integration

# Performance tests
pytest tests/performance -m performance

# Exclude slow tests
pytest -m "not slow"
```

### Run with coverage
```bash
pytest --cov=abyss --cov-report=html
# View coverage report
open htmlcov/index.html
```

### Run specific test file
```bash
pytest tests/unit/mqtt/test_config_manager_clean.py
```

### Run with verbose output
```bash
pytest -v
```

## Writing Tests

### Example test file
```python
"""Test module description."""

import pytest
from abyss.mqtt.components.some_module import SomeClass


class TestSomeClass:
    """Test cases for SomeClass."""
    
    def test_basic_functionality(self, sample_mqtt_config):
        """Test basic functionality."""
        # Use fixtures from conftest.py
        instance = SomeClass(sample_mqtt_config)
        assert instance.some_method() == expected_value
    
    @pytest.mark.slow
    def test_slow_operation(self):
        """Test that takes a long time."""
        # This test will be skipped with pytest -m "not slow"
        pass
    
    @pytest.mark.performance
    def test_performance(self, performance_monitor):
        """Test performance characteristics."""
        # performance_monitor fixture tracks metrics
        # Do performance-critical operations
        pass
```

### Using shared fixtures

The `conftest.py` file provides many useful fixtures:

- `sample_mqtt_config`: Standard MQTT configuration
- `temp_config_file`: Temporary config file (auto-cleaned)
- `create_timestamped_data`: Factory for TimestampedData objects
- `mock_mqtt_client`: Mock MQTT client for testing
- `performance_monitor`: Tracks test performance metrics

See `conftest.py` for the complete list.

## Test Markers

Tests can be marked for categorization:

- `@pytest.mark.slow`: Slow tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.performance`: Performance tests
- `@pytest.mark.memory`: Memory usage tests
- `@pytest.mark.threading`: Thread safety tests

## Best Practices

1. **No sys.path manipulation**: Use proper imports
2. **Use shared fixtures**: Don't duplicate fixture code
3. **Mark tests appropriately**: Use markers for test categorization
4. **Keep tests focused**: One test should test one thing
5. **Use descriptive names**: Test names should describe what they test
6. **Clean up resources**: Use fixtures for automatic cleanup