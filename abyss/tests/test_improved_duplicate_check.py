#!/usr/bin/env python3
"""
Comprehensive test for improved MessageBuffer._check_for_duplicate functionality
"""

import sys
import os
from datetime import datetime

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'abyss', 'src'))

from abyss.uos_depth_est import TimestampedData
from abyss.mqtt.components.message_buffer import MessageBuffer

def test_improved_duplicate_detection():
    """Test the improved duplicate detection functionality"""
    
    # Create test configuration with custom time window
    config = {
        'mqtt': {
            'listener': {
                'root': 'OPCPUBSUB',
                'result': 'ResultManagement',
                'trace': 'ResultManagement/Trace',
                'duplicate_handling': 'ignore',
                'duplicate_time_window': 2.0  # Custom 2-second window
            }
        }
    }
    
    # Create MessageBuffer instance
    buffer = MessageBuffer(config)
    
    print("Testing improved duplicate detection...")
    
    # Test 1: Dictionary order independence
    print("\nTest 1: Dictionary order independence")
    timestamp = datetime.now().timestamp()
    source = "OPCPUBSUB/toolbox1/tool1/ResultManagement"
    
    data1 = {"b": 2, "a": 1, "c": 3}  # Different order
    data2 = {"a": 1, "b": 2, "c": 3}  # Different order
    
    msg1 = TimestampedData(timestamp, data1, source)
    msg2 = TimestampedData(timestamp, data2, source)
    
    result = buffer._compare_message_data(data1, data2)
    print(f"Dictionary order test: {result} (expected: True)")
    
    # Test 2: Floating point precision
    print("\nTest 2: Floating point precision")
    data3 = {"value": 0.1 + 0.2}  # 0.30000000000000004
    data4 = {"value": 0.3}        # 0.3
    
    result = buffer._compare_message_data(data3, data4)
    print(f"Float precision test: {result} (expected: True)")
    
    # Test 3: Mixed numeric types
    print("\nTest 3: Mixed numeric types")
    data5 = {"count": 42}     # int
    data6 = {"count": 42.0}   # float
    
    result = buffer._compare_message_data(data5, data6)
    print(f"Mixed numeric types: {result} (expected: True)")
    
    # Test 4: Nested structures
    print("\nTest 4: Nested structures")
    data7 = {
        "outer": {
            "inner": [1, 2, {"nested": "value"}],
            "number": 3.14159
        }
    }
    data8 = {
        "outer": {
            "number": 3.14159,
            "inner": [1, 2, {"nested": "value"}]
        }
    }
    
    result = buffer._compare_message_data(data7, data8)
    print(f"Nested structures: {result} (expected: True)")
    
    # Test 5: Custom time window
    print("\nTest 5: Custom time window (2 seconds)")
    base_time = datetime.now().timestamp()
    
    msg_base = TimestampedData(base_time, {"test": "data"}, source)
    msg_1_5s = TimestampedData(base_time + 1.5, {"test": "data"}, source)  # Within 2s window
    msg_2_5s = TimestampedData(base_time + 2.5, {"test": "data"}, source)  # Outside 2s window
    
    existing = [msg_base]
    
    is_dup_1_5, _ = buffer._check_for_duplicate(msg_1_5s, existing)
    is_dup_2_5, _ = buffer._check_for_duplicate(msg_2_5s, existing)
    
    print(f"1.5s difference: is_duplicate={is_dup_1_5} (expected: True)")
    print(f"2.5s difference: is_duplicate={is_dup_2_5} (expected: False)")
    
    # Test 6: None handling
    print("\nTest 6: None handling")
    result_none_none = buffer._compare_message_data(None, None)
    result_none_data = buffer._compare_message_data(None, {"test": "data"})
    result_data_none = buffer._compare_message_data({"test": "data"}, None)
    
    print(f"None vs None: {result_none_none} (expected: True)")
    print(f"None vs data: {result_none_data} (expected: False)")
    print(f"Data vs None: {result_data_none} (expected: False)")
    
    # Test 7: List comparison
    print("\nTest 7: List comparison")
    list1 = [1, 2, {"key": "value"}, [3, 4]]
    list2 = [1, 2, {"key": "value"}, [3, 4]]
    list3 = [1, 2, {"key": "different"}, [3, 4]]
    
    result_same_lists = buffer._compare_message_data(list1, list2)
    result_diff_lists = buffer._compare_message_data(list1, list3)
    
    print(f"Same lists: {result_same_lists} (expected: True)")
    print(f"Different lists: {result_diff_lists} (expected: False)")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    test_improved_duplicate_detection()
