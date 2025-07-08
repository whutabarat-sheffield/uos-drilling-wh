#!/usr/bin/env python3
"""
Test script to verify MessageBuffer._check_for_duplicate functionality
"""

import sys
import os
import json
from datetime import datetime

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'abyss', 'src'))

from abyss.uos_depth_est import TimestampedData
from abyss.mqtt.components.message_buffer import MessageBuffer

def test_duplicate_detection():
    """Test the duplicate detection functionality"""
    
    # Create test configuration
    config = {
        'mqtt': {
            'listener': {
                'root': 'OPCPUBSUB',
                'result': 'ResultManagement',
                'trace': 'ResultManagement/Trace',
                'duplicate_handling': 'ignore'
            }
        }
    }
    
    # Create MessageBuffer instance
    buffer = MessageBuffer(config)
    
    # Test data
    timestamp = datetime.now().timestamp()
    test_data = {"test": "data", "value": 123}
    source = "OPCPUBSUB/toolbox1/tool1/ResultManagement"
    
    # Create test messages
    message1 = TimestampedData(timestamp, test_data, source)
    message2 = TimestampedData(timestamp + 0.5, test_data, source)  # Same data, slightly different timestamp
    message3 = TimestampedData(timestamp + 2.0, test_data, source)  # Same data, but outside 1-second window
    message4 = TimestampedData(timestamp, {"different": "data"}, source)  # Same timestamp, different data
    
    existing_messages = [message1]
    
    print("Testing duplicate detection...")
    
    # Test 1: Same source, same timestamp, same data (should be duplicate)
    print("\nTest 1: Exact duplicate")
    is_dup, index = buffer._check_for_duplicate(message1, existing_messages)
    print(f"Result: is_duplicate={is_dup}, index={index}")
    print(f"Expected: is_duplicate=True, index=0")
    
    # Test 2: Same source, close timestamp (0.5s), same data (should be duplicate)
    print("\nTest 2: Close timestamp duplicate")
    is_dup, index = buffer._check_for_duplicate(message2, existing_messages)
    print(f"Result: is_duplicate={is_dup}, index={index}")
    print(f"Expected: is_duplicate=True, index=0")
    
    # Test 3: Same source, far timestamp (2s), same data (should NOT be duplicate)
    print("\nTest 3: Far timestamp, same data")
    is_dup, index = buffer._check_for_duplicate(message3, existing_messages)
    print(f"Result: is_duplicate={is_dup}, index={index}")
    print(f"Expected: is_duplicate=False, index=None")
    
    # Test 4: Same source, same timestamp, different data (should NOT be duplicate)
    print("\nTest 4: Same timestamp, different data")
    is_dup, index = buffer._check_for_duplicate(message4, existing_messages)
    print(f"Result: is_duplicate={is_dup}, index={index}")
    print(f"Expected: is_duplicate=False, index=None")
    
    # Test 5: Test data comparison edge cases
    print("\nTest 5: Data comparison edge cases")
    
    # Test with complex nested data
    complex_data1 = {"nested": {"array": [1, 2, 3], "string": "test"}}
    complex_data2 = {"nested": {"array": [1, 2, 3], "string": "test"}}  # Same content
    complex_data3 = {"nested": {"array": [1, 2, 4], "string": "test"}}  # Different content
    
    msg_complex1 = TimestampedData(timestamp, complex_data1, source)
    msg_complex2 = TimestampedData(timestamp, complex_data2, source)
    msg_complex3 = TimestampedData(timestamp, complex_data3, source)
    
    existing_complex = [msg_complex1]
    
    is_dup, index = buffer._check_for_duplicate(msg_complex2, existing_complex)
    print(f"Complex data same: is_duplicate={is_dup}, index={index}")
    
    is_dup, index = buffer._check_for_duplicate(msg_complex3, existing_complex)
    print(f"Complex data different: is_duplicate={is_dup}, index={index}")
    
    # Test 6: Test string comparison method directly
    print("\nTest 6: Direct string comparison")
    result1 = buffer._compare_message_data(test_data, test_data)
    result2 = buffer._compare_message_data(complex_data1, complex_data2)
    result3 = buffer._compare_message_data(complex_data1, complex_data3)
    
    print(f"Same simple data: {result1}")
    print(f"Same complex data: {result2}")
    print(f"Different complex data: {result3}")

if __name__ == "__main__":
    test_duplicate_detection()
