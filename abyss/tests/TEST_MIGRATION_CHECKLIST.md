# Test Migration Checklist

This document tracks the test consolidation effort for Phase 1 implementation, documenting unique test cases from files being removed to ensure no coverage is lost.

## Duplicate Detection Tests ✅

### Files to Consolidate:
1. `test_duplicate_check.py` - Basic script-style test
2. `test_improved_duplicate_check.py` - Enhanced script-style test  
3. `test_duplicate_behavior.py` - Behavior-focused test
4. `test_integration_duplicate_handling.py` - **KEEP** (proper integration test)

### Unique Test Cases to Preserve:

From `test_duplicate_check.py`:
- [x] Same source, same timestamp, same data → duplicate
- [x] Same source, different timestamp (within window) → duplicate
- [x] Same source, different timestamp (outside window) → not duplicate
- [x] Same timestamp, different data → not duplicate

From `test_improved_duplicate_check.py`:
- [x] Dictionary order independence ({"a":1,"b":2} == {"b":2,"a":1})
- [x] Nested dictionary comparison
- [x] List comparison with order
- [x] Numeric precision handling (float comparison)
- [x] Custom time window configuration (2.0 seconds)

From `test_duplicate_behavior.py`:
- [x] 'ignore' mode behavior
- [x] 'replace' mode behavior
- [x] 'error' mode behavior (raises DuplicateMessageError)

### Migration Target:
✅ **COMPLETED**: Consolidated into enhanced `test_message_buffer.py` with proper pytest format.
- Added `TestDuplicateDetection` class with all unique test cases
- Used pytest fixtures and parametrization for different handling modes
- Maintained full test coverage from original files

## Config Manager Tests

### Files to Consolidate:
1. `test_confighandler.py` - **KEEP** (async components, different class)
2. `test_comprehensive_config_manager.py` - Fix imports and MessageCorrelator reference
3. `test_config_manager.py` (in unit/mqtt/) - **DELETE** (duplicate)
4. `test_config_manager_clean.py` - **KEEP** (cleanest implementation)
5. `test_config_manager.py` (in mqtt/components/) - **DELETE** (duplicate)
6. `test_config_manager_integration.py` - **KEEP** (integration test)

### Unique Test Cases to Preserve:

From various config manager tests:
- [x] File not found error handling
- [x] Invalid YAML error handling
- [x] Missing required sections validation
- [x] Type validation for config values
- [x] Default value handling
- [x] Nested config access (get('mqtt.broker.host'))
- [x] Config reload functionality
- [x] Environment variable override
- [x] Config validation against schema

### Migration Target:
✅ **COMPLETED**: 
1. Keep `test_config_manager_clean.py` as primary unit test (has performance tests)
2. Keep `test_config_manager_integration.py` for integration testing
3. Keep `test_confighandler.py` (tests different async ConfigHandler class)
4. Delete duplicate `test_config_manager.py` files
5. Fix `test_comprehensive_config_manager.py` imports

## Head ID Tests

### Files to Consolidate:
1. `test_head_id_extraction.py` - Extraction logic
2. `test_head_id_inclusion.py` - Inclusion in output
3. `test_message_processor_heads_id.py` - Processor integration
4. `test_result_publisher_head_id.py` - Publisher integration

### Unique Test Cases to Preserve:
- [x] Head ID extraction from heads message
- [x] Head ID null when heads message missing
- [x] Head ID included in published results
- [x] Head ID preserved through processing pipeline

### Migration Target:
Add head ID test cases to existing `test_message_processor.py` and `test_result_publisher.py`.

## Script-Style Tests to Convert

### Identification Criteria:
- Files starting with `#!/usr/bin/env python3`
- Files with `if __name__ == "__main__":`
- Files using print() instead of assertions

### Files Identified:
1. `test_duplicate_check.py`
2. `test_improved_duplicate_check.py`
3. `profile_stress_test.py` - **KEEP** as utility, not test

## Test Organization After Migration

### Final Test Structure:
```
tests/
├── unit/
│   ├── mqtt/
│   │   ├── test_config_manager.py (consolidated)
│   │   ├── test_message_buffer.py (+ duplicate tests)
│   │   ├── test_simple_correlator.py
│   │   ├── test_message_processor.py (+ head ID tests)
│   │   └── test_result_publisher.py (+ head ID tests)
│   └── core/
│       ├── test_uos_depth_est_core.py
│       └── test_uos_inference.py
├── integration/
│   ├── test_config_manager_integration.py
│   ├── test_message_buffer_config_manager.py
│   └── test_integration_duplicate_handling.py
├── performance/
│   ├── test_message_buffer_thread_safety.py
│   ├── test_mqtt_system_load.py
│   └── test_throughput_monitoring.py (NEW)
└── fixtures/
    └── ... (unchanged)
```

## Deletion Checklist

Before deleting any test file:
1. [ ] Run the test and capture output
2. [ ] Extract unique test cases
3. [ ] Verify cases are covered in consolidated test
4. [ ] Run consolidated test to ensure coverage
5. [ ] Delete original file
6. [ ] Update any imports/references

## New Tests Added (Phase 1) ✅

1. **test_throughput_monitoring.py** ✅
   - SimpleThroughputMonitor functionality
   - Arrival rate vs processing rate
   - Queue growth detection
   - "Keeping up" status
   - Location: `tests/performance/test_throughput_monitoring.py`

2. **test_bottleneck_profiler.py** ✅
   - Pipeline stage timing
   - Bottleneck identification
   - Sampling accuracy
   - Location: `tests/performance/test_bottleneck_profiler.py`

3. **test_realistic_load_patterns.py** ✅
   - Drilling operation patterns
   - Burst handling
   - Idle period behavior
   - Multi-tool concurrent load
   - Location: `tests/performance/test_realistic_load_patterns.py`

4. **test_diagnostic_correlator.py** ✅
   - Correlation history tracking
   - Orphan message detection
   - Queue growth monitoring
   - Throughput integration
   - Location: `tests/unit/mqtt/test_diagnostic_correlator.py`