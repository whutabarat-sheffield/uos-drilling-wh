#!/usr/bin/env python3
"""
Test MessageBuffer duplicate handling behavior through public interface
"""

import sys
import os
import json
from datetime import datetime

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'abyss', 'src'))

from abyss.uos_depth_est import TimestampedData
from abyss.mqtt.components.message_buffer import MessageBuffer

# Define the exception locally for testing since it might not be in the installed version
class DuplicateMessageError(Exception):
    pass

def test_duplicate_behavior_ignore():
    """Test duplicate handling with 'ignore' strategy"""
    print("=== Testing Duplicate Behavior: IGNORE Strategy ===")
    
    config = {
        'mqtt': {
            'listener': {
                'root': 'OPCPUBSUB',
                'result': 'ResultManagement',
                'trace': 'ResultManagement/Trace',
                'duplicate_handling': 'ignore',
                'duplicate_time_window': 1.0
            }
        }
    }
    
    buffer = MessageBuffer(config)
    timestamp = datetime.now().timestamp()
    test_data = {"test": "data", "value": 123}
    source = "OPCPUBSUB/toolbox1/tool1/ResultManagement"
    
    # Create test messages
    message1 = TimestampedData(timestamp, test_data, source)
    message2 = TimestampedData(timestamp + 0.5, test_data, source)  # Duplicate within time window
    message3 = TimestampedData(timestamp + 2.0, test_data, source)  # Outside time window
    message4 = TimestampedData(timestamp, {"different": "data"}, source)  # Different data
    
    # Test 1: Add original message
    print("\nTest 1: Adding original message")
    result1 = buffer.add_message(message1)
    stats1 = buffer.get_buffer_stats()
    print(f"Result: {result1} (expected: True)")
    print(f"Total messages: {stats1['total_messages']} (expected: 1)")
    
    # Test 2: Add duplicate message (should be ignored)
    print("\nTest 2: Adding duplicate message within time window")
    result2 = buffer.add_message(message2)
    stats2 = buffer.get_buffer_stats()
    print(f"Result: {result2} (expected: False - ignored)")
    print(f"Total messages: {stats2['total_messages']} (expected: 1 - no increase)")
    
    # Test 3: Add message outside time window (should be added)
    print("\nTest 3: Adding similar message outside time window")
    result3 = buffer.add_message(message3)
    stats3 = buffer.get_buffer_stats()
    print(f"Result: {result3} (expected: True)")
    print(f"Total messages: {stats3['total_messages']} (expected: 2)")
    
    # Test 4: Add message with different data (should be added)
    print("\nTest 4: Adding message with different data")
    result4 = buffer.add_message(message4)
    stats4 = buffer.get_buffer_stats()
    print(f"Result: {result4} (expected: True)")
    print(f"Total messages: {stats4['total_messages']} (expected: 3)")
    
    return result1, result2, result3, result4

def test_duplicate_behavior_replace():
    """Test duplicate handling with 'replace' strategy"""
    print("\n\n=== Testing Duplicate Behavior: REPLACE Strategy ===")
    
    config = {
        'mqtt': {
            'listener': {
                'root': 'OPCPUBSUB',
                'result': 'ResultManagement',
                'trace': 'ResultManagement/Trace',
                'duplicate_handling': 'replace',
                'duplicate_time_window': 1.0
            }
        }
    }
    
    buffer = MessageBuffer(config)
    timestamp = datetime.now().timestamp()
    original_data = {"version": 1, "value": "original"}
    updated_data = {"version": 2, "value": "updated"}
    source = "OPCPUBSUB/toolbox1/tool1/ResultManagement"
    
    # Create test messages
    message1 = TimestampedData(timestamp, original_data, source)
    message2 = TimestampedData(timestamp + 0.5, updated_data, source)  # Different data, same source/time
    
    # Test 1: Add original message
    print("\nTest 1: Adding original message")
    result1 = buffer.add_message(message1)
    stats1 = buffer.get_buffer_stats()
    print(f"Result: {result1} (expected: True)")
    print(f"Total messages: {stats1['total_messages']} (expected: 1)")
    
    # Get the buffer contents to verify original data
    topic_pattern = 'OPCPUBSUB/+/+/ResultManagement'
    messages_before = buffer.get_messages_by_topic(topic_pattern)
    if messages_before:
        print(f"Original data: {messages_before[0].data}")
    
    # Test 2: Add updated message (should replace if same data, add if different)
    print("\nTest 2: Adding message with different data at similar time")
    result2 = buffer.add_message(message2)
    stats2 = buffer.get_buffer_stats()
    messages_after = buffer.get_messages_by_topic(topic_pattern)
    
    print(f"Result: {result2} (expected: True - different data)")
    print(f"Total messages: {stats2['total_messages']} (expected: 2)")
    if len(messages_after) >= 2:
        print(f"Message 1 data: {messages_after[0].data}")
        print(f"Message 2 data: {messages_after[1].data}")
    
    return result1, result2

def test_duplicate_behavior_error():
    """Test duplicate handling with 'error' strategy"""
    print("\n\n=== Testing Duplicate Behavior: ERROR Strategy ===")
    
    config = {
        'mqtt': {
            'listener': {
                'root': 'OPCPUBSUB',
                'result': 'ResultManagement',
                'trace': 'ResultManagement/Trace',
                'duplicate_handling': 'error',
                'duplicate_time_window': 1.0
            }
        }
    }
    
    buffer = MessageBuffer(config)
    timestamp = datetime.now().timestamp()
    test_data = {"test": "data", "value": 123}
    source = "OPCPUBSUB/toolbox1/tool1/ResultManagement"
    
    # Create test messages
    message1 = TimestampedData(timestamp, test_data, source)
    message2 = TimestampedData(timestamp + 0.5, test_data, source)  # Exact duplicate
    
    # Test 1: Add original message
    print("\nTest 1: Adding original message")
    result1 = buffer.add_message(message1)
    stats1 = buffer.get_buffer_stats()
    print(f"Result: {result1} (expected: True)")
    print(f"Total messages: {stats1['total_messages']} (expected: 1)")
    
    # Test 2: Add duplicate message (should raise error)
    print("\nTest 2: Adding duplicate message (should raise error)")
    try:
        result2 = buffer.add_message(message2)
        print(f"ERROR: Expected exception but got result: {result2}")
        return False
    except Exception as e:
        # Check if it's any kind of duplicate-related exception
        if "duplicate" in str(e).lower() or "DuplicateMessageError" in str(type(e).__name__):
            print(f"SUCCESS: Got expected duplicate exception: {type(e).__name__}: {e}")
            stats2 = buffer.get_buffer_stats()
            print(f"Total messages after error: {stats2['total_messages']} (expected: 1)")
            return True
        else:
            print(f"ERROR: Got unexpected exception type: {type(e).__name__}: {e}")
            return False

def test_complex_data_comparison():
    """Test duplicate detection with complex nested data structures"""
    print("\n\n=== Testing Complex Data Comparison ===")
    
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
    
    buffer = MessageBuffer(config)
    timestamp = datetime.now().timestamp()
    source = "OPCPUBSUB/toolbox1/tool1/ResultManagement"
    
    # Complex nested data with different key orders
    data1 = {
        "outer": {
            "b": 2,
            "a": 1,
            "nested": {
                "list": [1, 2, {"key": "value"}],
                "number": 3.14159
            }
        },
        "simple": "test"
    }
    
    data2 = {
        "simple": "test",
        "outer": {
            "a": 1,
            "nested": {
                "number": 3.14159,
                "list": [1, 2, {"key": "value"}]
            },
            "b": 2
        }
    }
    
    data3 = {
        "simple": "test",
        "outer": {
            "a": 1,
            "nested": {
                "number": 3.14159,
                "list": [1, 2, {"key": "different"}]  # Changed value
            },
            "b": 2
        }
    }
    
    # Test messages
    message1 = TimestampedData(timestamp, data1, source)
    message2 = TimestampedData(timestamp + 0.5, data2, source)  # Same data, different order
    message3 = TimestampedData(timestamp + 0.7, data3, source)  # Different data
    
    # Test 1: Add original complex message
    print("\nTest 1: Adding original complex message")
    result1 = buffer.add_message(message1)
    stats1 = buffer.get_buffer_stats()
    print(f"Result: {result1} (expected: True)")
    print(f"Total messages: {stats1['total_messages']} (expected: 1)")
    
    # Test 2: Add same data with different key order (should be duplicate)
    print("\nTest 2: Adding same data with different key order")
    result2 = buffer.add_message(message2)
    stats2 = buffer.get_buffer_stats()
    print(f"Result: {result2} (expected: False - duplicate)")
    print(f"Total messages: {stats2['total_messages']} (expected: 1)")
    
    # Test 3: Add actually different data (should be added)
    print("\nTest 3: Adding actually different data")
    result3 = buffer.add_message(message3)
    stats3 = buffer.get_buffer_stats()
    print(f"Result: {result3} (expected: True)")
    print(f"Total messages: {stats3['total_messages']} (expected: 2)")
    
    return result1, result2, result3

def main():
    """Run all duplicate behavior tests"""
    print("Testing MessageBuffer Duplicate Handling via Public Interface")
    print("=" * 70)
    
    try:
        # Test ignore strategy
        ignore_results = test_duplicate_behavior_ignore()
        
        # Test replace strategy
        replace_results = test_duplicate_behavior_replace()
        
        # Test error strategy
        error_result = test_duplicate_behavior_error()
        
        # Test complex data comparison
        complex_results = test_complex_data_comparison()
        
        # Summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        print(f"Ignore strategy tests: {ignore_results}")
        print(f"Replace strategy tests: {replace_results}")
        print(f"Error strategy test: {error_result}")
        print(f"Complex data tests: {complex_results}")
        
        # Verify expected results
        ignore_expected = (True, False, True, True)
        replace_expected = (True, True)
        complex_expected = (True, False, True)
        
        print(f"\nIgnore strategy: {'PASS' if ignore_results == ignore_expected else 'FAIL'}")
        print(f"Replace strategy: {'PASS' if replace_results == replace_expected else 'FAIL'}")
        print(f"Error strategy: {'PASS' if error_result else 'FAIL'}")
        print(f"Complex data: {'PASS' if complex_results == complex_expected else 'FAIL'}")
        
    except Exception as e:
        print(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
