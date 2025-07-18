"""
Test for Sequential Message Dropping Bug - Exact Match Implementation

This test verifies that the exact matching implementation correctly handles
sequential messages and doesn't drop message 422 in a sequence of 421, 422, 423, 424.
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


class TestSequentialMessageBugFixed:
    """Test that the sequential message dropping bug is fixed with exact matching."""
    
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
                    'heads': 'AssetManagement',
                    'duplicate_handling': 'ignore',
                    'duplicate_time_window': 1.0
                }
            },
            'analyser': {
                'time_window': 30,  # Ignored in exact match mode
                'cleanup_interval': 2,
                'max_buffer_size': 10  # Increased for exact matching
            }
        }
    
    @pytest.fixture
    def message_buffer(self, config):
        """Create message buffer with exact matching."""
        return MessageBuffer(
            config=config,
            cleanup_interval=60,  # Longer cleanup for exact matching
            max_buffer_size=100,  # Larger buffer for exact matching
            max_age_seconds=300
        )
    
    @pytest.fixture
    def correlator(self, config):
        """Create message correlator (exact match mode)."""
        return SimpleMessageCorrelator(config=config, time_window=30)
    
    def create_message_set(self, msg_type: str, toolbox_id: str, tool_id: str, 
                          result_id: int, source_timestamp: str) -> TimestampedData:
        """Create a test message with exact SourceTimestamp."""
        topic_map = {
            'ResultManagement': 'result',
            'Trace': 'trace', 
            'AssetManagement': 'heads'
        }
        
        source = f"mqtt/{toolbox_id}/{tool_id}/{msg_type}"
        
        # All messages must have SourceTimestamp for exact matching
        data = {
            'SourceTimestamp': source_timestamp,
            'toolboxID': toolbox_id,
            'toolID': tool_id
        }
        
        # Add type-specific data
        if msg_type == 'ResultManagement':
            data['ResultId'] = result_id
        elif msg_type == 'AssetManagement':
            data['HeadsId'] = f'HEADS{result_id}'
        
        return TimestampedData(
            _timestamp=time.time(),  # Current time for receive timestamp
            _source=source,
            _data=data
        )
    
    def test_sequential_messages_not_dropped_exact_match(self, message_buffer, correlator):
        """
        Test that sequential messages (421, 422, 423, 424) are NOT dropped
        with exact matching implementation.
        """
        toolbox_id = "E00401"
        tool_id = "009F45BACA"
        
        # Track which messages were processed
        processed_results = []
        
        def mock_processor(messages):
            """Mock message processor that tracks processed messages."""
            for msg in messages:
                if 'ResultManagement' in msg.source and 'ResultId' in msg.data:
                    processed_results.append(msg.data['ResultId'])
        
        # Create messages with exact timestamps for each result ID
        for i, result_id in enumerate([421, 422, 423, 424]):
            # Each message set has its own unique SourceTimestamp
            source_timestamp = f'2024-01-18T10:30:{i:02d}Z'
            
            # Create all three required messages with same SourceTimestamp
            result_msg = self.create_message_set('ResultManagement', toolbox_id, tool_id, 
                                               result_id, source_timestamp)
            trace_msg = self.create_message_set('Trace', toolbox_id, tool_id, 
                                              result_id, source_timestamp)
            heads_msg = self.create_message_set('AssetManagement', toolbox_id, tool_id, 
                                              result_id, source_timestamp)
            
            # Add messages to buffer
            message_buffer.add_message(result_msg)
            message_buffer.add_message(trace_msg)
            message_buffer.add_message(heads_msg)
            
            # Log buffer stats after each set
            stats = message_buffer.get_buffer_stats()
            print(f"After adding resultID {result_id}: {stats}")
        
        # With exact matching, all message sets should be matched immediately
        final_stats = message_buffer.get_buffer_stats()
        assert final_stats['exact_matches_completed'] == 4
        assert final_stats['messages_dropped'] == 0
        
        # Process matched messages
        buffers = message_buffer.get_all_buffers()
        correlator.find_and_process_matches(buffers, mock_processor)
        
        # Verify ALL messages were processed (bug is fixed!)
        print(f"Processed results: {sorted(processed_results)}")
        assert sorted(processed_results) == [421, 422, 423, 424]
        assert 422 in processed_results, "Message 422 should NOT be dropped with exact matching!"
    
    def test_buffer_handles_many_sequential_messages(self, message_buffer):
        """
        Test that buffer can handle many sequential messages without dropping any.
        """
        toolbox_id = "E00401"
        tool_id = "009F45BACA"
        
        # Add 100 message sets
        for i in range(100):
            result_id = 1000 + i
            source_timestamp = f'2024-01-18T10:{i//60:02d}:{i%60:02d}Z'
            
            # Create complete message set
            result_msg = self.create_message_set('ResultManagement', toolbox_id, tool_id,
                                               result_id, source_timestamp)
            trace_msg = self.create_message_set('Trace', toolbox_id, tool_id,
                                              result_id, source_timestamp)
            heads_msg = self.create_message_set('AssetManagement', toolbox_id, tool_id,
                                              result_id, source_timestamp)
            
            message_buffer.add_message(result_msg)
            message_buffer.add_message(trace_msg)
            message_buffer.add_message(heads_msg)
        
        # All should be matched
        stats = message_buffer.get_buffer_stats()
        assert stats['exact_matches_completed'] == 100
        assert stats['messages_dropped'] == 0
        assert stats['total_active_keys'] == 0  # All matched, no pending
    
    def test_incomplete_sets_remain_pending(self, message_buffer):
        """
        Test that incomplete message sets remain pending and don't interfere
        with complete sets.
        """
        toolbox_id = "E00401"
        tool_id = "009F45BACA"
        
        # Add some complete sets
        for i in range(5):
            source_timestamp = f'2024-01-18T10:30:{i:02d}Z'
            result_id = 500 + i
            
            result_msg = self.create_message_set('ResultManagement', toolbox_id, tool_id,
                                               result_id, source_timestamp)
            trace_msg = self.create_message_set('Trace', toolbox_id, tool_id,
                                              result_id, source_timestamp)
            heads_msg = self.create_message_set('AssetManagement', toolbox_id, tool_id,
                                              result_id, source_timestamp)
            
            message_buffer.add_message(result_msg)
            message_buffer.add_message(trace_msg)
            message_buffer.add_message(heads_msg)
        
        # Add some incomplete sets (missing heads)
        for i in range(5, 10):
            source_timestamp = f'2024-01-18T10:30:{i:02d}Z'
            result_id = 500 + i
            
            result_msg = self.create_message_set('ResultManagement', toolbox_id, tool_id,
                                               result_id, source_timestamp)
            trace_msg = self.create_message_set('Trace', toolbox_id, tool_id,
                                              result_id, source_timestamp)
            # No heads message!
            
            message_buffer.add_message(result_msg)
            message_buffer.add_message(trace_msg)
        
        stats = message_buffer.get_buffer_stats()
        assert stats['exact_matches_completed'] == 5  # Only complete sets
        assert stats['total_active_keys'] == 5  # Incomplete sets remain pending
    
    def test_exact_matching_prevents_race_conditions(self, message_buffer, correlator):
        """
        Test that exact matching prevents race conditions that could cause
        message dropping in fuzzy matching.
        """
        processed_results = []
        processing_lock = threading.Lock()
        
        def mock_processor(messages):
            """Thread-safe message processor."""
            with processing_lock:
                for msg in messages:
                    if 'ResultManagement' in msg.source and 'ResultId' in msg.data:
                        processed_results.append(msg.data['ResultId'])
        
        def add_message_set(result_id, delay=0):
            """Add a complete message set with optional delay."""
            if delay:
                time.sleep(delay)
            
            source_timestamp = f'2024-01-18T10:30:{result_id%60:02d}Z'
            toolbox_id = "E00401"
            tool_id = "009F45BACA"
            
            result_msg = self.create_message_set('ResultManagement', toolbox_id, tool_id,
                                               result_id, source_timestamp)
            trace_msg = self.create_message_set('Trace', toolbox_id, tool_id,
                                              result_id, source_timestamp)
            heads_msg = self.create_message_set('AssetManagement', toolbox_id, tool_id,
                                              result_id, source_timestamp)
            
            message_buffer.add_message(result_msg)
            message_buffer.add_message(trace_msg)
            message_buffer.add_message(heads_msg)
        
        # Start multiple threads adding messages concurrently
        threads = []
        for i in range(10):
            t = threading.Thread(target=add_message_set, args=(600 + i, i * 0.01))
            t.start()
            threads.append(t)
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Process all matches
        buffers = message_buffer.get_all_buffers()
        correlator.find_and_process_matches(buffers, mock_processor)
        
        # All messages should be processed
        assert sorted(processed_results) == list(range(600, 610))
        
        # Check buffer stats
        stats = message_buffer.get_buffer_stats()
        assert stats['exact_matches_completed'] == 10
        assert stats['messages_dropped'] == 0