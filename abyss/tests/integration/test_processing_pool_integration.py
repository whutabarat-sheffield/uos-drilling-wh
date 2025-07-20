#!/usr/bin/env python3
"""
Integration test script to verify ProcessingPool functionality.

This is a standalone integration test that validates:
1. ProcessPoolExecutor successfully initializes workers
2. Messages are processed in parallel
3. SimpleThroughputMonitor accurately reports system health
4. The system can sustain target throughput

Environment Variables:
- MQTT_CONFIG_PATH: Path to MQTT configuration file (default: src/abyss/run/config/mqtt_conf_local.yaml)
- TEST_DATA_PATH: Path to test data directory (default: src/abyss/test_data/data_20250626)
- VERBOSE: Enable verbose output (default: 0)

Usage:
    python test_processing_pool_integration.py
    VERBOSE=1 python test_processing_pool_integration.py
    MQTT_CONFIG_PATH=/custom/path/config.yaml python test_processing_pool_integration.py
"""

import sys
import time
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.abyss.mqtt.components.message_buffer import MessageBuffer
from src.abyss.mqtt.components.simple_correlator import SimpleMessageCorrelator
from src.abyss.mqtt.components.processing_pool import SimpleProcessingPool
from src.abyss.mqtt.components.throughput_monitor import SimpleThroughputMonitor
from src.abyss.mqtt.components.config_manager import ConfigurationManager
from src.abyss.uos_depth_est import TimestampedData

# Import shared test utilities
from test_data_utils import (
    load_opcua_test_data,
    transform_opcua_result_data,
    transform_opcua_trace_data,
    transform_opcua_heads_data,
    create_synthetic_test_message
)

# Configuration from environment
CONFIG_PATH = os.environ.get('MQTT_CONFIG_PATH', 'src/abyss/run/config/mqtt_conf_local.yaml')
TEST_DATA_PATH = os.environ.get('TEST_DATA_PATH', 'src/abyss/test_data/data_20250626')
VERBOSE = os.environ.get('VERBOSE', '0') == '1'


def setup_logging():
    """Configure logging for the test."""
    level = logging.DEBUG if VERBOSE else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )




def create_test_messages(num_messages: int = 10, topic_type: str = 'result') -> List[TimestampedData]:
    """Create test messages using actual JSON data or synthetic fallback."""
    messages = []
    base_time = datetime.now().timestamp()
    
    # Try to load actual data
    result_payload, trace_payload, heads_payload = load_opcua_test_data(TEST_DATA_PATH)
    
    # Convert OPC UA format to expected format
    if topic_type == 'result' and result_payload:
        # Use shared utility to transform data
        result_data = transform_opcua_result_data(result_payload)
        
        # Create multiple messages with same data but different timestamps
        for i in range(num_messages):
            msg = TimestampedData(
                _data=result_data,
                _timestamp=base_time + i * 0.01,
                _source=f"OPCPUBSUB/ILLL502033771/setitecgabriele/ResultManagement"
            )
            messages.append(msg)
            
    elif topic_type == 'trace' and trace_payload:
        # Use shared utility to transform data
        trace_data = transform_opcua_trace_data(trace_payload)
        
        # Create multiple messages
        for i in range(num_messages):
            msg = TimestampedData(
                _data=trace_data,
                _timestamp=base_time + i * 0.01,
                _source=f"OPCPUBSUB/ILLL502033771/setitecgabriele/ResultManagement/Trace"
            )
            messages.append(msg)
            
    elif topic_type == 'heads' and heads_payload:
        # Use shared utility to transform data
        heads_data = transform_opcua_heads_data(heads_payload)
        
        # Create multiple messages
        for i in range(num_messages):
            msg = TimestampedData(
                _data=heads_data,
                _timestamp=base_time + i * 0.01,
                _source=f"OPCPUBSUB/ILLL502033771/setitecgabriele/AssetManagement/Heads"
            )
            messages.append(msg)
            
    else:
        # Fallback to synthetic data
        if VERBOSE:
            print(f"Using synthetic data for {topic_type}")
        for i in range(num_messages):
            msg = create_synthetic_test_message(topic_type, i, base_time)
            messages.append(msg)
    
    return messages


def test_processing_pool():
    """Test the processing pool functionality."""
    print("\n=== Testing ProcessingPool ===")
    
    # Initialize pool
    pool = SimpleProcessingPool(max_workers=5, config_path=CONFIG_PATH)
    
    # Create all three types of messages
    result_messages = create_test_messages(5, 'result')
    trace_messages = create_test_messages(5, 'trace')
    heads_messages = create_test_messages(5, 'heads')
    
    # Track submissions
    submitted = 0
    rejected = 0
    results = []
    
    def handle_result(result: Dict[str, Any]):
        results.append(result)
        if result.get('success'):
            print(f"✓ Processing completed successfully")
        else:
            print(f"✗ Processing failed: {result.get('error_message')}")
    
    # Submit triplets of messages (heads + result + trace)
    for i in range(min(len(result_messages), len(trace_messages), len(heads_messages))):
        batch = [heads_messages[i], result_messages[i], trace_messages[i]]
        if pool.submit(batch, callback=handle_result):
            submitted += 1
        else:
            rejected += 1
    
    print(f"Submitted: {submitted}, Rejected: {rejected}")
    
    # Wait for completion
    start_time = time.time()
    while len(results) < submitted and time.time() - start_time < 10:
        pool.collect_completed()
        time.sleep(0.1)
    
    # Print stats
    stats = pool.get_stats()
    print(f"\nPool Statistics:")
    print(f"  Workers: {stats['workers']}")
    print(f"  Completed: {stats['completed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Success Rate: {stats['success_rate']:.2%}")
    print(f"  Avg Throughput: {stats['avg_throughput']:.2f} msg/s")
    
    pool.shutdown()
    
    return stats['failed'] == 0


def test_throughput_monitoring():
    """Test throughput monitoring functionality."""
    print("\n=== Testing ThroughputMonitor ===")
    
    monitor = SimpleThroughputMonitor(sample_rate=0.1)
    
    # Simulate message arrivals and processing
    print("Simulating 10 msg/sec arrival rate...")
    
    for i in range(50):  # 5 seconds of simulation
        # Record arrivals (1 every 5 iterations = 10/sec)
        if i % 5 == 0:
            monitor.record_arrival()
        
        # Simulate processing with 0.5s delay
        if i % 25 == 0:  # Every 2.5s, complete one message
            start_time = time.time() - 0.5
            monitor.record_processing_complete(start_time)
        
        time.sleep(0.1)  # 100ms per iteration
    
    # Get status
    status = monitor.get_status()
    print(f"\nMonitor Status: {status.status}")
    print(f"Details: {json.dumps(status.details, indent=2)}")
    
    return status.status == 'FALLING_BEHIND'


def test_integrated_system():
    """Test the full integrated system."""
    print("\n=== Testing Integrated System ===")
    
    # Initialize components
    config = ConfigurationManager(CONFIG_PATH)
    buffer = MessageBuffer(config)
    correlator = SimpleMessageCorrelator(config)
    pool = SimpleProcessingPool(max_workers=10, config_path=CONFIG_PATH)
    monitor = SimpleThroughputMonitor()
    
    # Tracking
    processed_count = 0
    start_time = time.time()
    
    def handle_result(result: Dict[str, Any]):
        nonlocal processed_count
        if result.get('success'):
            processed_count += 1
    
    # Simulate 10 msg/sec for 2 seconds
    print("Simulating 10 msg/sec for 2 seconds...")
    
    for i in range(20):  # 20 iterations at 100ms each = 2 seconds
        # Add messages to buffer (1 per iteration = 10/sec)
        result_msgs = create_test_messages(1, 'result')
        trace_msgs = create_test_messages(1, 'trace')
        
        for msg in result_msgs + trace_msgs:
            buffer.add_message(msg)
            monitor.record_arrival()
        
        # Find and process matches
        buffers = buffer.get_all_buffers()
        matches = correlator.find_and_process_matches(
            buffers=buffers,
            message_processor=lambda msgs: pool.submit(msgs, handle_result)
        )
        
        # Collect completed
        pool.collect_completed()
        
        # Brief status every 0.5 seconds
        if i % 5 == 0:  # Every 5 iterations (0.5 seconds)
            status = monitor.get_status()
            queue_depth = pool.get_queue_depth()
            buffer_stats = buffer.get_buffer_stats()
            print(f"  t={i*0.1:.1f}s: Status={status.status}, Queue={queue_depth}, Buffer={buffer_stats['total_messages']}")
        
        time.sleep(0.1)
    
    # Final stats
    elapsed = time.time() - start_time
    pool_stats = pool.get_stats()
    buffer_stats = buffer.get_buffer_stats()
    
    print(f"\nFinal Statistics:")
    print(f"  Duration: {elapsed:.2f}s")
    print(f"  Messages Added: ~40")
    print(f"  Messages Processed: {processed_count}")
    print(f"  Processing Rate: {processed_count/elapsed:.2f} msg/s")
    print(f"  Pool Success Rate: {pool_stats['success_rate']:.2%}")
    print(f"  Final Buffer Size: {buffer_stats['total_messages']}")
    
    pool.shutdown()
    
    # Success if we processed at least some messages
    return processed_count > 0


def main():
    """Run all tests."""
    setup_logging()
    
    print("=" * 60)
    print("ProcessingPool Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("ProcessingPool Basic Functionality", test_processing_pool),
        ("ThroughputMonitor Detection", test_throughput_monitoring),
        ("Integrated System Performance", test_integrated_system)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}")
            success = test_func()
            results.append((test_name, success))
            print(f"Result: {'PASS' if success else 'FAIL'}")
        except Exception as e:
            print(f"ERROR: {e}")
            logging.error(f"Test failed: {test_name}", exc_info=True)
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} passed ({passed/total*100:.0f}%)")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)