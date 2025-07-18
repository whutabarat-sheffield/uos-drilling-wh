"""
Test for Sequential Message Dropping Bug

This test reproduces the issue where message 422 gets dropped in a sequence
of 421, 422, 423, 424 due to buffer cleanup removing unprocessed messages.
"""

import pytest
import time
import threading
from unittest.mock import Mock, MagicMock
from datetime import datetime

from abyss.uos_depth_est import TimestampedData
from abyss.mqtt.components.message_buffer import MessageBuffer
from abyss.mqtt.components.simple_correlator import SimpleMessageCorrelator
from abyss.mqtt.components.config_manager import ConfigurationManager


class TestSequentialMessageBug:
    """Test case for the sequential message dropping bug."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            'mqtt': {
                'broker': {'host': 'localhost', 'port': 1883},
                'listener': {
                    'root': 'mqtt',
                    'result': 'ResultManagement',
                    'trace': 'Trace',
                    'duplicate_handling': 'ignore',
                    'duplicate_time_window': 1.0
                }
            },
            'analyser': {
                'time_window': 30,
                'cleanup_interval': 2,  # Short interval for testing
                'max_buffer_size': 5  # Small buffer to trigger cleanup
            }
        }
    
    @pytest.fixture
    def message_buffer(self, config):
        """Create message buffer with small size to trigger cleanup."""
        config_manager = ConfigurationManager.from_dict(config)
        return MessageBuffer(
            config=config_manager,
            cleanup_interval=2,
            max_buffer_size=5,  # Small buffer to force cleanup
            max_age_seconds=300
        )
    
    @pytest.fixture
    def correlator(self, config):
        """Create message correlator."""
        config_manager = ConfigurationManager.from_dict(config)
        return SimpleMessageCorrelator(config=config_manager, time_window=30)
    
    def create_message(self, msg_type: str, toolbox_id: str, tool_id: str, 
                      result_id: int, timestamp: float) -> TimestampedData:
        """Create a test message."""
        source = f"mqtt/{toolbox_id}/{tool_id}/{msg_type}"
        data = {
            'resultID': result_id,
            'toolboxID': toolbox_id,
            'toolID': tool_id,
            'timestamp': timestamp
        }
        return TimestampedData(
            timestamp=timestamp,
            source=source,
            data=data
        )
    
    def test_sequential_message_dropping_bug(self, message_buffer, correlator):
        """
        Test that demonstrates the bug where message 422 gets dropped
        when messages arrive in sequence: 421, 422, 423, 424.
        """
        toolbox_id = "E00401"
        tool_id = "009F45BACA"
        base_time = datetime.now().timestamp()
        
        # Track which messages were processed
        processed_results = []
        
        def mock_processor(messages):
            """Mock message processor that tracks processed messages."""
            for msg in messages:
                if 'ResultManagement' in msg.source:
                    processed_results.append(msg.data['resultID'])
        
        # Simulate messages arriving in sequence
        messages = []
        for i, result_id in enumerate([421, 422, 423, 424]):
            timestamp = base_time + i * 0.5  # 0.5 second intervals
            
            # Create result and trace messages
            result_msg = self.create_message('ResultManagement', toolbox_id, tool_id, result_id, timestamp)
            trace_msg = self.create_message('Trace', toolbox_id, tool_id, result_id, timestamp + 0.1)
            
            messages.append((result_msg, trace_msg))
        
        # Add messages to buffer
        for result_msg, trace_msg in messages:
            message_buffer.add_message(result_msg)
            message_buffer.add_message(trace_msg)
            
            # Get buffer stats after each addition
            stats = message_buffer.get_buffer_stats()
            print(f"After adding resultID {result_msg.data['resultID']}: {stats}")
        
        # Force cleanup by adding more messages to exceed buffer size
        # This simulates the condition where cleanup removes unprocessed messages
        for i in range(5):
            dummy_time = base_time + 100 + i
            dummy_result = self.create_message('ResultManagement', 'dummy', 'dummy', 1000 + i, dummy_time)
            dummy_trace = self.create_message('Trace', 'dummy', 'dummy', 1000 + i, dummy_time)
            message_buffer.add_message(dummy_result)
            message_buffer.add_message(dummy_trace)
        
        # Now process messages
        buffers = message_buffer.get_all_buffers()
        correlator.find_and_process_matches(buffers, mock_processor)
        
        # Check which messages were processed
        print(f"Processed results: {sorted(processed_results)}")
        
        # The bug would cause 422 to be missing
        # With the fix, all messages should be processed
        assert 421 in processed_results, "Message 421 should be processed"
        assert 422 in processed_results, "Message 422 should NOT be dropped (this was the bug)"
        assert 423 in processed_results, "Message 423 should be processed"
        assert 424 in processed_results, "Message 424 should be processed"
    
    def test_buffer_preserves_unprocessed_messages(self, message_buffer):
        """
        Test that the buffer cleanup preserves unprocessed messages
        and removes processed ones first.
        """
        base_time = datetime.now().timestamp()
        
        # Create a mix of processed and unprocessed messages
        for i in range(8):
            msg = self.create_message(
                'ResultManagement', 
                'toolbox', 
                'tool', 
                100 + i, 
                base_time + i
            )
            message_buffer.add_message(msg)
            
            # Mark even-numbered messages as processed
            if i % 2 == 0:
                msg.processed = True
        
        # Get initial state
        initial_stats = message_buffer.get_buffer_stats()
        print(f"Initial buffer state: {initial_stats}")
        
        # Force cleanup
        message_buffer.cleanup_old_messages()
        
        # Check that unprocessed messages are preserved
        buffers = message_buffer.get_all_buffers()
        result_buffer = buffers.get('mqtt/+/+/ResultManagement', [])
        
        unprocessed_ids = []
        processed_ids = []
        for msg in result_buffer:
            if getattr(msg, 'processed', False):
                processed_ids.append(msg.data['resultID'])
            else:
                unprocessed_ids.append(msg.data['resultID'])
        
        print(f"After cleanup - Unprocessed: {unprocessed_ids}, Processed: {processed_ids}")
        
        # With our fix, unprocessed messages (odd numbers) should be preserved
        # and processed messages (even numbers) should be removed first
        assert len(unprocessed_ids) > 0, "Unprocessed messages should be preserved"
        
        # Check that if any messages were removed, processed ones were removed first
        all_original_ids = list(range(100, 108))
        remaining_ids = unprocessed_ids + processed_ids
        removed_ids = [id for id in all_original_ids if id not in remaining_ids]
        
        if removed_ids:
            # All removed IDs should be even (processed)
            for removed_id in removed_ids:
                assert removed_id % 2 == 0, f"Removed message {removed_id} should be processed (even)"
    
    def test_concurrent_processing_and_cleanup(self, message_buffer, correlator):
        """
        Test that concurrent processing and cleanup don't cause message loss.
        """
        toolbox_id = "E00401"
        tool_id = "009F45BACA"
        base_time = datetime.now().timestamp()
        
        processed_results = []
        cleanup_count = [0]
        
        def mock_processor(messages):
            """Mock processor with delay to simulate processing time."""
            time.sleep(0.05)  # Simulate processing delay
            for msg in messages:
                if 'ResultManagement' in msg.source:
                    processed_results.append(msg.data['resultID'])
        
        def cleanup_thread():
            """Thread that performs cleanup while processing is happening."""
            while cleanup_count[0] < 3:
                time.sleep(0.1)
                message_buffer.cleanup_old_messages()
                cleanup_count[0] += 1
        
        # Start cleanup thread
        cleanup = threading.Thread(target=cleanup_thread)
        cleanup.daemon = True
        cleanup.start()
        
        # Add messages and process them while cleanup is running
        for i in range(10):
            result_id = 500 + i
            timestamp = base_time + i * 0.1
            
            result_msg = self.create_message('ResultManagement', toolbox_id, tool_id, result_id, timestamp)
            trace_msg = self.create_message('Trace', toolbox_id, tool_id, result_id, timestamp)
            
            message_buffer.add_message(result_msg)
            message_buffer.add_message(trace_msg)
            
            # Process messages
            buffers = message_buffer.get_all_buffers()
            correlator.find_and_process_matches(buffers, mock_processor)
        
        # Wait for cleanup thread to finish
        cleanup.join(timeout=1)
        
        # Process any remaining messages
        buffers = message_buffer.get_all_buffers()
        correlator.find_and_process_matches(buffers, mock_processor)
        
        print(f"Processed results with concurrent cleanup: {sorted(processed_results)}")
        
        # All messages should be processed despite concurrent cleanup
        expected_results = list(range(500, 510))
        assert sorted(processed_results) == expected_results, \
            f"All messages should be processed. Missing: {set(expected_results) - set(processed_results)}"