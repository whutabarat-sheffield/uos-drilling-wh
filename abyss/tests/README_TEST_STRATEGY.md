# MQTT Processing Test Strategy

This document explains the testing approach for the MQTT processing components, particularly the distinction between unit tests and integration tests.

## Test Organization

### Unit Tests (`tests/unit/`)
Fast, isolated tests that mock external dependencies:
- **Location**: `tests/unit/mqtt/`
- **Execution Time**: < 1 second per test
- **Dependencies**: All mocked
- **Purpose**: Verify component logic and error handling

Example: `test_processing_pool.py`
- Tests pool initialization, task submission, statistics
- Mocks ProcessPoolExecutor to avoid spawning processes
- Runs in milliseconds

### Integration Tests (`tests/integration/`)
Standalone scripts that test real component interactions:
- **Location**: `tests/integration/`
- **Execution Time**: Several seconds
- **Dependencies**: Real components, actual data files
- **Purpose**: Validate end-to-end functionality

Example: `test_processing_pool_integration.py`
- Tests actual parallel processing
- Uses real JSON test data
- Validates throughput and system behavior

## Running Tests

### Unit Tests (Fast, CI-friendly)
```bash
# Run all unit tests
pytest abyss/tests/unit/

# Run specific component tests
pytest abyss/tests/unit/mqtt/test_processing_pool.py -v

# Skip slow tests
pytest abyss/tests/unit/ -m "not slow"
```

### Integration Tests (Slower, comprehensive)
```bash
# Run with default configuration
python abyss/tests/integration/test_processing_pool_integration.py

# Run with custom configuration
MQTT_CONFIG_PATH=/path/to/config.yaml python abyss/tests/integration/test_processing_pool_integration.py

# Enable verbose output
VERBOSE=1 python abyss/tests/integration/test_processing_pool_integration.py

# Use custom test data
TEST_DATA_PATH=/path/to/test/data python abyss/tests/integration/test_processing_pool_integration.py
```

## Test Data

### Unit Tests
- Use synthetic data or small fixtures
- Data is generated in-memory
- No file I/O required

### Integration Tests  
- Use actual JSON files from `src/abyss/test_data/`
- Transform OPC UA format using shared utilities
- Fall back to synthetic data if files unavailable

## Key Testing Principles

1. **Separation of Concerns**
   - Unit tests verify component behavior
   - Integration tests verify system behavior

2. **Speed vs Coverage Trade-off**
   - Unit tests: Fast feedback during development
   - Integration tests: Confidence before deployment

3. **Deterministic vs Realistic**
   - Unit tests: Deterministic, no timing dependencies
   - Integration tests: Realistic timing, actual concurrency

4. **Test Utilities**
   - Shared test data transformation in `test_data_utils.py`
   - Reduces code duplication
   - Consistent data handling

## CI/CD Integration

### Recommended Pipeline
```yaml
# Fast feedback stage
unit-tests:
  script:
    - pytest abyss/tests/unit/ --cov=abyss/src/abyss/mqtt

# Comprehensive validation stage  
integration-tests:
  script:
    - python abyss/tests/integration/test_processing_pool_integration.py
    - python abyss/tests/integration/test_processing_pool_simple.py
  only:
    - main
    - develop
```

## Adding New Tests

### When to Add Unit Tests
- Testing specific component methods
- Verifying error handling
- Testing edge cases
- Need fast feedback

### When to Add Integration Tests
- Testing component interactions
- Validating performance requirements
- Testing with real data formats
- Simulating production scenarios

## Environment Variables

### Common Variables
- `MQTT_CONFIG_PATH`: Path to MQTT configuration file
- `TEST_DATA_PATH`: Path to test data directory
- `VERBOSE`: Enable detailed output (0/1)

### Test-Specific Variables
Check individual test file headers for additional environment variables.

## Debugging Integration Tests

1. Enable verbose mode: `VERBOSE=1`
2. Check log files in configured log directory
3. Use smaller data sets for faster iteration
4. Run specific test functions individually

## Performance Benchmarks

Integration tests can serve as performance benchmarks:
- `test_processing_pool()`: Baseline processing rate
- `test_throughput_monitoring()`: System load handling
- `test_integrated_system()`: End-to-end throughput

Target metrics are documented in test assertions.