#!/usr/bin/env python3
"""
Simple test to verify ProcessingPool is working correctly.
"""

import sys
import os
import json
import time
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.abyss.mqtt.components.processing_pool import SimpleProcessingPool
from src.abyss.uos_depth_est import TimestampedData


def setup_logging():
    """Configure logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_actual_messages():
    """Load actual messages from test data files."""
    test_data_path = "src/abyss/test_data/data_20250626"
    timestamp = datetime.now().timestamp()
    
    messages = []
    
    # Load and create messages in the exact format expected
    try:
        # Load ResultManagement
        with open(os.path.join(test_data_path, "ResultManagement.json"), 'r') as f:
            result_json = json.load(f)
            # Use the full JSON structure as-is
            result_msg = TimestampedData(
                _data=result_json,
                _timestamp=timestamp,
                _source="OPCPUBSUB/ILLL502033771/setitecgabriele/ResultManagement"
            )
            messages.append(result_msg)
            
        # Load Trace
        with open(os.path.join(test_data_path, "Trace.json"), 'r') as f:
            trace_json = json.load(f)
            trace_msg = TimestampedData(
                _data=trace_json,
                _timestamp=timestamp + 0.001,
                _source="OPCPUBSUB/ILLL502033771/setitecgabriele/ResultManagement/Trace"
            )
            messages.append(trace_msg)
            
        # Load Heads
        with open(os.path.join(test_data_path, "Heads.json"), 'r') as f:
            heads_json = json.load(f)
            heads_msg = TimestampedData(
                _data=heads_json,
                _timestamp=timestamp + 0.002,
                _source="OPCPUBSUB/ILLL502033771/setitecgabriele/AssetManagement/Heads"
            )
            messages.append(heads_msg)
            
        return messages
        
    except Exception as e:
        logging.error(f"Failed to load test data: {e}")
        return []


def main():
    """Run simple test of ProcessingPool."""
    setup_logging()
    
    print("\n=== Simple ProcessingPool Test ===\n")
    
    # Initialize pool
    pool = SimpleProcessingPool(max_workers=2, config_path='src/abyss/run/config/mqtt_conf_local.yaml')
    
    # Load actual messages
    messages = load_actual_messages()
    if not messages:
        print("Failed to load test messages!")
        return False
        
    print(f"Loaded {len(messages)} test messages")
    
    # Track results
    results = []
    
    def handle_result(result):
        results.append(result)
        if result.get('success'):
            print(f"✓ Processing successful! Got keypoints: {result.get('keypoints') is not None}")
        else:
            print(f"✗ Processing failed: {result.get('error_message', 'Unknown error')}")
    
    # Submit the batch
    print("\nSubmitting message batch...")
    submitted = pool.submit(messages, callback=handle_result)
    
    if submitted:
        print("Messages submitted successfully")
    else:
        print("Failed to submit messages (pool full)")
        
    # Wait for completion
    print("\nWaiting for processing...")
    start_time = time.time()
    timeout = 10  # 10 second timeout
    
    while len(results) < 1 and time.time() - start_time < timeout:
        pool.collect_completed()
        time.sleep(0.1)
    
    # Check results
    if results:
        result = results[0]
        print(f"\nProcessing completed in {time.time() - start_time:.2f} seconds")
        print(f"Success: {result.get('success')}")
        if result.get('keypoints'):
            print(f"Keypoints shape: {len(result['keypoints'])} points")
        if result.get('depth_estimation'):
            print(f"Depth estimation: {result['depth_estimation']}")
    else:
        print(f"\nTimeout: No results after {timeout} seconds")
        
    # Show pool stats
    stats = pool.get_stats()
    print(f"\nPool Statistics:")
    print(f"  Submitted: {stats['submitted']}")
    print(f"  Completed: {stats['completed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Queue depth: {stats['queue_depth']}")
    
    # Cleanup
    pool.shutdown()
    
    return len(results) > 0 and results[0].get('success', False)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)