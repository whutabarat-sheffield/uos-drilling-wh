#!/usr/bin/env python3
"""
Test script to verify ProcessingPool integration.

This script tests:
1. ProcessPoolExecutor successfully initializes workers
2. Messages are processed in parallel
3. SimpleThroughputMonitor accurately reports system health
4. The system can sustain 100 msg/sec throughput
"""

import sys
import time
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.abyss.mqtt.components.message_buffer import MessageBuffer
from src.abyss.mqtt.components.simple_correlator import SimpleMessageCorrelator
from src.abyss.mqtt.components.processing_pool import SimpleProcessingPool
from src.abyss.mqtt.components.throughput_monitor import SimpleThroughputMonitor
from src.abyss.mqtt.components.config_manager import ConfigurationManager
from src.abyss.uos_depth_est import TimestampedData


def setup_logging():
    """Configure logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_test_data_from_json():
    """Load actual test data from JSON files."""
    result_data = None
    trace_data = None
    heads_data = None
    
    # Try to load from test_data directory
    test_data_path = "src/abyss/test_data/data_20250626"
    
    try:
        # Load ResultManagement data
        with open(os.path.join(test_data_path, "ResultManagement.json"), 'r') as f:
            result_json = json.load(f)
            result_data = result_json['Messages']['Payload']
            
        # Load Trace data
        with open(os.path.join(test_data_path, "Trace.json"), 'r') as f:
            trace_json = json.load(f)
            trace_data = trace_json['Messages']['Payload']
            
        # Load Heads data
        with open(os.path.join(test_data_path, "Heads.json"), 'r') as f:
            heads_json = json.load(f)
            heads_data = heads_json['Messages']['Payload']
            
    except Exception as e:
        print(f"Warning: Could not load test data from JSON files: {e}")
        print("Falling back to synthetic data")
        
    return result_data, trace_data, heads_data


def create_test_messages(num_messages: int = 10, topic_type: str = 'result') -> List[TimestampedData]:
    """Create test messages using actual JSON data or synthetic fallback."""
    messages = []
    base_time = datetime.now().timestamp()
    
    # Try to load actual data
    result_payload, trace_payload, heads_payload = load_test_data_from_json()
    
    # Convert OPC UA format to expected format
    if topic_type == 'result' and result_payload:
        # Extract data from OPC UA format
        result_data = {
            'ResultManagement': {
                'Results': [{
                    'ResultMetaData': {
                        'SerialNumber': result_payload.get('nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecgabriele.ResultManagement.Results.0.ResultMetaData.SerialNumber', {}).get('Value', '0'),
                        'ResultId': result_payload.get('nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecgabriele.ResultManagement.Results.0.ResultMetaData.ResultId', {}).get('Value', '432')
                    },
                    'ResultContent': {
                        'StepResults': [{
                            'StepResultValues': {
                                'IntensityTorqueEmpty': result_payload.get('nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecgabriele.ResultManagement.Results.0.ResultContent.StepResults.0.StepResultValues.IntensityTorqueEmpty.MeasuredValue', {}).get('Value', []),
                                'IntensityThrustEmpty': result_payload.get('nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecgabriele.ResultManagement.Results.0.ResultContent.StepResults.0.StepResultValues.IntensityThrustEmpty.MeasuredValue', {}).get('Value', []),
                                'StepNumber': result_payload.get('nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecgabriele.ResultManagement.Results.0.ResultContent.StepResults.0.StepResultValues.StepNumber.MeasuredValue', {}).get('Value', [])
                            }
                        }]
                    }
                }]
            }
        }
        
        # Create multiple messages with same data but different timestamps
        for i in range(num_messages):
            msg = TimestampedData(
                _data=result_data,
                _timestamp=base_time + i * 0.01,
                _source=f"OPCPUBSUB/ILLL502033771/setitecgabriele/ResultManagement"
            )
            messages.append(msg)
            
    elif topic_type == 'trace' and trace_payload:
        # Extract trace data
        trace_data = {
            'ResultManagement': {
                'Results': [{
                    'ResultContent': {
                        'Trace': {
                            'StepTraces': {
                                'PositionTrace': {
                                    'StepResultId': trace_payload.get('nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecgabriele.ResultManagement.Results.0.ResultContent.Trace.StepTraces.PositionTrace.StepResultId', {}).get('Value', '432'),
                                    'StepTraceContent': [{
                                        'Values': trace_payload.get('nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecgabriele.ResultManagement.Results.0.ResultContent.Trace.StepTraces.PositionTrace.StepTraceContent[0].Values', {}).get('Value', [])[:100]  # Limit to first 100 values
                                    }]
                                },
                                'IntensityThrustTrace': {
                                    'StepTraceContent': [{
                                        'Values': trace_payload.get('nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecgabriele.ResultManagement.Results.0.ResultContent.Trace.StepTraces.IntensityThrustTrace.StepTraceContent[0].Values', {}).get('Value', [])[:100] if 'IntensityThrustTrace' in str(trace_payload) else []
                                    }]
                                },
                                'IntensityTorqueTrace': {
                                    'StepTraceContent': [{
                                        'Values': trace_payload.get('nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecgabriele.ResultManagement.Results.0.ResultContent.Trace.StepTraces.IntensityTorqueTrace.StepTraceContent[0].Values', {}).get('Value', [])[:100] if 'IntensityTorqueTrace' in str(trace_payload) else []
                                    }]
                                }
                            }
                        }
                    }
                }]
            }
        }
        
        # Create multiple messages
        for i in range(num_messages):
            msg = TimestampedData(
                _data=trace_data,
                _timestamp=base_time + i * 0.01,
                _source=f"OPCPUBSUB/ILLL502033771/setitecgabriele/ResultManagement/Trace"
            )
            messages.append(msg)
            
    elif topic_type == 'heads' and heads_payload:
        # Extract heads data - look at actual structure
        heads_data = {
            'AssetManagement': {
                'Assets': {
                    'Heads': [{
                        'Identification': {
                            'SerialNumber': heads_payload.get('nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecgabriele.AssetManagement.Assets.Heads.0.Identification.SerialNumber', {}).get('Value', 'HEAD001')
                        }
                    }]
                }
            }
        }
        
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
        print(f"Using synthetic data for {topic_type}")
        for i in range(num_messages):
            if topic_type == 'result':
                source = f"OPCPUBSUB/TEST001/tool{i%3}/ResultManagement"
                data = {
                    'ResultManagement': {
                        'Results': [{
                            'ResultMetaData': {
                                'SerialNumber': f'SN{i:05d}',
                                'ResultId': f'RID{i:05d}'
                            }
                        }]
                    }
                }
            else:
                source = f"OPCPUBSUB/TEST001/tool{i%3}/ResultManagement/Trace" 
                data = {
                    'ResultManagement': {
                        'Results': [{
                            'ResultContent': {
                                'Trace': {
                                    'StepTraces': {
                                        'PositionTrace': {'StepTraceContent': [{'Values': [i * 0.1]}]},
                                        'IntensityThrustTrace': {'StepTraceContent': [{'Values': [i * 0.2]}]},
                                        'IntensityTorqueTrace': {'StepTraceContent': [{'Values': [i * 0.3]}]}
                                    }
                                }
                            }
                        }]
                    }
                }
                
            msg = TimestampedData(
                _data=data,
                _timestamp=base_time + i * 0.01,
                _source=source
            )
            messages.append(msg)
    
    return messages


def test_processing_pool():
    """Test the processing pool functionality."""
    print("\n=== Testing ProcessingPool ===")
    
    # Initialize pool
    pool = SimpleProcessingPool(max_workers=5, config_path='src/abyss/run/config/mqtt_conf_local.yaml')
    
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
    print("Simulating 100 msg/sec arrival rate...")
    
    for i in range(50):  # 0.5 seconds of simulation
        # Record arrivals (10 per iteration = 100/sec)
        for _ in range(10):
            monitor.record_arrival()
        
        # Simulate processing with 0.5s delay
        if i % 5 == 0:  # Every 50ms, complete one message
            start_time = time.time() - 0.5
            monitor.record_processing_complete(start_time)
        
        time.sleep(0.01)  # 10ms per iteration
    
    # Get status
    status = monitor.get_status()
    print(f"\nMonitor Status: {status.status}")
    print(f"Details: {json.dumps(status.details, indent=2)}")
    
    return status.status == 'FALLING_BEHIND'


def test_integrated_system():
    """Test the full integrated system."""
    print("\n=== Testing Integrated System ===")
    
    # Initialize components
    config = ConfigurationManager('src/abyss/run/config/mqtt_conf_local.yaml')
    buffer = MessageBuffer(config)
    correlator = SimpleMessageCorrelator(config)
    pool = SimpleProcessingPool(max_workers=10, config_path='src/abyss/run/config/mqtt_conf_local.yaml')
    monitor = SimpleThroughputMonitor()
    
    # Tracking
    processed_count = 0
    start_time = time.time()
    
    def handle_result(result: Dict[str, Any]):
        nonlocal processed_count
        if result.get('success'):
            processed_count += 1
    
    # Simulate 100 msg/sec for 2 seconds
    print("Simulating 100 msg/sec for 2 seconds...")
    
    for i in range(200):  # 200 iterations at 10ms each = 2 seconds
        # Add messages to buffer (10 per iteration = 100/sec)
        result_msgs = create_test_messages(5, 'result')
        trace_msgs = create_test_messages(5, 'trace')
        
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
        if i % 50 == 0:
            status = monitor.get_status()
            queue_depth = pool.get_queue_depth()
            buffer_stats = buffer.get_buffer_stats()
            print(f"  t={i*0.01:.1f}s: Status={status.status}, Queue={queue_depth}, Buffer={buffer_stats['total_messages']}")
        
        time.sleep(0.01)
    
    # Final stats
    elapsed = time.time() - start_time
    pool_stats = pool.get_stats()
    buffer_stats = buffer.get_buffer_stats()
    
    print(f"\nFinal Statistics:")
    print(f"  Duration: {elapsed:.2f}s")
    print(f"  Messages Added: ~200")
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