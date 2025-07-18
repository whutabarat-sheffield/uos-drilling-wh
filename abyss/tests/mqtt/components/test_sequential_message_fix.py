"""
Test to validate the fix for the sequential message dropping bug.

This test specifically recreates the conditions that caused message 422
to be dropped in the sequence 421, 422, 423, 424.
"""

import pytest
import time
import threading
from datetime import datetime
from unittest.mock import Mock, patch

from abyss.mqtt.components.message_buffer import MessageBuffer
from abyss.mqtt.components.simple_correlator import SimpleMessageCorrelator
from abyss.uos_depth_est import TimestampedData


class TestSequentialMessageFix:
    """Test that the exact matching implementation fixes the sequential message bug."""
    
    @pytest.fixture
    def config(self):
        """Standard test configuration with heads required."""
        return {
            'mqtt': {
                'listener': {
                    'root': 'mqtt',
                    'result': 'ResultManagement',
                    'trace': 'Trace',
                    'heads': 'AssetManagement'
                }
            }
        }
    
    def create_message(self, msg_type: str, tool_key: str, timestamp_str: str, 
                      result_id: int = None):
        """Helper to create test messages."""
        toolbox_id, tool_id = tool_key.split('/')
        topic_map = {
            'result': 'ResultManagement',
            'trace': 'Trace',
            'heads': 'AssetManagement'
        }
        
        source = f"mqtt/{toolbox_id}/{tool_id}/{topic_map[msg_type]}"
        
        data = {
            'SourceTimestamp': timestamp_str,
            'Value': {'test': 'data'}
        }
        
        if result_id is not None:
            data['ResultId'] = result_id
        
        return TimestampedData(
            _timestamp=datetime.now().timestamp(),
            _source=source,
            _data=data
        )
    
    def test_sequential_messages_not_dropped_exact_match(self, config):
        """
        Test that sequential messages (421, 422, 423, 424) are NOT dropped
        with exact matching implementation.
        """
        # Create buffer with small size to trigger cleanup
        buffer = MessageBuffer(config, max_buffer_size=10, cleanup_interval=60)
        correlator = SimpleMessageCorrelator(config)
        
        # Track processed results
        processed_results = []
        
        def message_processor(messages):
            for msg in messages:
                if 'ResultId' in msg.data:
                    processed_results.append(msg.data['ResultId'])
        
        # Simulate the exact scenario from the bug report
        tool_key = "E00401/009F45BACA"
        
        # Add messages for result IDs 421, 422, 423, 424
        # Each with its own unique timestamp (simulating real data)
        for i, result_id in enumerate([421, 422, 423, 424]):
            # Each message gets a unique timestamp
            timestamp = f"2024-01-18T10:30:{i:02d}Z"
            
            # Add all three message types with same timestamp
            result_msg = self.create_message('result', tool_key, timestamp, result_id)
            trace_msg = self.create_message('trace', tool_key, timestamp)
            heads_msg = self.create_message('heads', tool_key, timestamp)
            
            # Add messages
            assert buffer.add_message(result_msg)
            assert buffer.add_message(trace_msg)
            assert buffer.add_message(heads_msg)
            
            # Simulate some processing delay
            time.sleep(0.001)
        
        # Process all matches
        buffers = buffer.get_all_buffers()
        correlator.find_and_process_matches(buffers, message_processor)
        
        # Verify ALL messages were processed
        assert sorted(processed_results) == [421, 422, 423, 424], \
            f"Expected [421, 422, 423, 424], got {sorted(processed_results)}"
        
        # Verify no messages were dropped
        stats = buffer.get_buffer_stats()
        assert stats['messages_dropped'] == 0, \
            f"Messages were dropped: {stats['messages_dropped']}"
        assert stats['exact_matches_completed'] == 4, \
            f"Expected 4 matches, got {stats['exact_matches_completed']}"
    
    def test_high_volume_no_drops(self, config):
        """Test that no messages are dropped even under high volume."""
        buffer = MessageBuffer(config, max_buffer_size=50, cleanup_interval=60)
        correlator = SimpleMessageCorrelator(config)
        
        processed_results = []
        
        def message_processor(messages):
            for msg in messages:
                if 'ResultId' in msg.data:
                    processed_results.append(msg.data['ResultId'])
        
        # Simulate high volume - 100 message sets
        tool_key = "E00401/009F45BACA"
        expected_results = []
        
        for result_id in range(1000, 1100):
            timestamp = f"2024-01-18T10:{result_id%60:02d}:{result_id//60:02d}Z"
            expected_results.append(result_id)
            
            # Add all three message types
            buffer.add_message(self.create_message('result', tool_key, timestamp, result_id))
            buffer.add_message(self.create_message('trace', tool_key, timestamp))
            buffer.add_message(self.create_message('heads', tool_key, timestamp))
        
        # Process all
        buffers = buffer.get_all_buffers()
        correlator.find_and_process_matches(buffers, message_processor)
        
        # Verify all processed
        assert sorted(processed_results) == expected_results
        
        stats = buffer.get_buffer_stats()
        assert stats['messages_dropped'] == 0
        assert stats['exact_matches_completed'] == 100
    
    def test_concurrent_processing_no_drops(self, config):
        """Test that concurrent processing doesn't cause drops."""
        buffer = MessageBuffer(config, max_buffer_size=20, cleanup_interval=1)
        correlator = SimpleMessageCorrelator(config)
        
        processed_results = []
        processed_lock = threading.Lock()
        
        def message_processor(messages):
            with processed_lock:
                for msg in messages:
                    if 'ResultId' in msg.data:
                        processed_results.append(msg.data['ResultId'])
        
        # Function to continuously process
        stop_processing = False
        def continuous_processor():
            while not stop_processing:
                buffers = buffer.get_all_buffers()
                correlator.find_and_process_matches(buffers, message_processor)
                time.sleep(0.01)
        
        # Start processing thread
        processor_thread = threading.Thread(target=continuous_processor)
        processor_thread.start()
        
        # Add messages while processing is happening
        tool_key = "E00401/009F45BACA"
        expected_results = []
        
        for result_id in range(500, 550):
            timestamp = f"2024-01-18T11:{result_id%60:02d}:{result_id//60:02d}Z"
            expected_results.append(result_id)
            
            buffer.add_message(self.create_message('result', tool_key, timestamp, result_id))
            buffer.add_message(self.create_message('trace', tool_key, timestamp))
            buffer.add_message(self.create_message('heads', tool_key, timestamp))
            
            time.sleep(0.002)  # Small delay between additions
        
        # Let processing catch up
        time.sleep(0.5)
        
        # Stop processing
        stop_processing = True
        processor_thread.join()
        
        # Final process to catch any remaining
        buffers = buffer.get_all_buffers()
        correlator.find_and_process_matches(buffers, message_processor)
        
        # Verify all processed (may have duplicates due to concurrent processing)
        with processed_lock:
            # Remove duplicates and sort
            unique_results = sorted(list(set(processed_results)))
            assert unique_results == expected_results
        
        stats = buffer.get_buffer_stats()
        assert stats['messages_dropped'] == 0
    
    def test_exact_timestamp_requirement(self, config):
        """Test that only messages with exactly matching timestamps are correlated."""
        buffer = MessageBuffer(config)
        correlator = SimpleMessageCorrelator(config)
        
        processed_groups = []
        
        def message_processor(messages):
            group = {
                'timestamps': set(),
                'result_ids': []
            }
            for msg in messages:
                group['timestamps'].add(msg.data.get('SourceTimestamp'))
                if 'ResultId' in msg.data:
                    group['result_ids'].append(msg.data['ResultId'])
            processed_groups.append(group)
        
        tool_key = "E00401/009F45BACA"
        
        # Add messages with slightly different timestamps
        buffer.add_message(self.create_message('result', tool_key, "2024-01-18T10:30:45.000Z", 100))
        buffer.add_message(self.create_message('trace', tool_key, "2024-01-18T10:30:45.000Z"))
        buffer.add_message(self.create_message('heads', tool_key, "2024-01-18T10:30:45.000Z"))
        
        # Different timestamp - should NOT match
        buffer.add_message(self.create_message('result', tool_key, "2024-01-18T10:30:45.001Z", 101))
        buffer.add_message(self.create_message('trace', tool_key, "2024-01-18T10:30:45.001Z"))
        buffer.add_message(self.create_message('heads', tool_key, "2024-01-18T10:30:45.002Z"))  # Different!
        
        # Process
        buffers = buffer.get_all_buffers()
        correlator.find_and_process_matches(buffers, message_processor)
        
        # Should only have one matched group
        assert len(processed_groups) == 1
        assert processed_groups[0]['result_ids'] == [100]
        assert len(processed_groups[0]['timestamps']) == 1  # All same timestamp
        
        stats = buffer.get_buffer_stats()
        assert stats['exact_matches_completed'] == 1
        assert stats['total_active_keys'] == 2  # Two incomplete sets remain