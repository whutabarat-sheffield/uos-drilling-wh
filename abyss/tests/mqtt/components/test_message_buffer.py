"""
Test suite for MessageBuffer class - Exact Matching Implementation.

Tests the MessageBuffer that uses exact (tool_key, source_timestamp) matching
instead of fuzzy time windows.
"""

import pytest
import time
from datetime import datetime
from collections import defaultdict

# Import the classes we need to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from abyss.mqtt.components.message_buffer import MessageBuffer, MessageSet
from abyss.uos_depth_est import TimestampedData


class TestMessageBuffer:
    """Test cases for MessageBuffer class with exact matching."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
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
    
    @pytest.fixture
    def message_buffer(self, sample_config):
        """Create MessageBuffer instance for testing."""
        return MessageBuffer(
            config=sample_config,
            cleanup_interval=60,
            max_buffer_size=100,
            max_age_seconds=300
        )
    
    @pytest.fixture
    def sample_result_message(self):
        """Create sample result message with SourceTimestamp."""
        return TimestampedData(
            _timestamp=time.time(),
            _data={
                'test': 'result_data',
                'SourceTimestamp': '2024-01-18T10:30:45Z',
                'ResultId': 100
            },
            _source='mqtt/toolbox1/tool1/ResultManagement'
        )
    
    @pytest.fixture
    def sample_trace_message(self):
        """Create sample trace message with SourceTimestamp."""
        return TimestampedData(
            _timestamp=time.time(),
            _data={
                'test': 'trace_data',
                'SourceTimestamp': '2024-01-18T10:30:45Z'
            },
            _source='mqtt/toolbox1/tool1/Trace'
        )
    
    def test_initialization(self, sample_config):
        """Test MessageBuffer initialization."""
        buffer = MessageBuffer(sample_config)
        
        assert buffer.config == sample_config
        assert isinstance(buffer.buffers, defaultdict)
        assert buffer.cleanup_interval == 60
        assert buffer.max_buffer_size == 10000
        assert buffer.max_age_seconds == 300
    
    @pytest.fixture
    def sample_heads_message(self):
        """Create sample heads message with SourceTimestamp."""
        return TimestampedData(
            _timestamp=time.time(),
            _data={
                'test': 'heads_data',
                'SourceTimestamp': '2024-01-18T10:30:45Z',
                'HeadsId': 'HEADS123'
            },
            _source='mqtt/toolbox1/tool1/AssetManagement'
        )

    def test_add_result_message(self, message_buffer, sample_result_message):
        """Test adding a result message to buffer."""
        success = message_buffer.add_message(sample_result_message)
        
        assert success is True
        
        # Check the message was added to the exact match buffer
        stats = message_buffer.get_buffer_stats()
        assert stats['total_messages'] == 1
        assert stats['total_active_keys'] == 1  # One pending match
    
    def test_exact_matching_complete_set(self, message_buffer, sample_result_message, 
                                         sample_trace_message, sample_heads_message):
        """Test exact matching with complete message set."""
        # Add all three messages with same timestamp
        assert message_buffer.add_message(sample_result_message) is True
        assert message_buffer.add_message(sample_trace_message) is True
        assert message_buffer.add_message(sample_heads_message) is True
        
        # Check that exact match was completed
        stats = message_buffer.get_buffer_stats()
        assert stats['exact_matches_completed'] == 1
        assert stats['total_active_keys'] == 0  # No pending matches
        assert stats['completed_matches_pending'] == 1
    
    def test_no_match_different_timestamps(self, message_buffer):
        """Test that messages with different timestamps don't match."""
        # Create messages with different SourceTimestamps
        result_msg = TimestampedData(
            _timestamp=time.time(),
            _data={'SourceTimestamp': '2024-01-18T10:30:45Z'},
            _source='mqtt/toolbox1/tool1/ResultManagement'
        )
        trace_msg = TimestampedData(
            _timestamp=time.time(),
            _data={'SourceTimestamp': '2024-01-18T10:30:46Z'},  # Different!
            _source='mqtt/toolbox1/tool1/Trace'
        )
        heads_msg = TimestampedData(
            _timestamp=time.time(),
            _data={'SourceTimestamp': '2024-01-18T10:30:47Z'},  # Different!
            _source='mqtt/toolbox1/tool1/AssetManagement'
        )
        
        message_buffer.add_message(result_msg)
        message_buffer.add_message(trace_msg)
        message_buffer.add_message(heads_msg)
        
        # No matches should be found
        stats = message_buffer.get_buffer_stats()
        assert stats['exact_matches_completed'] == 0
        assert stats['total_active_keys'] == 3  # Three separate pending matches
    
    def test_missing_source_timestamp(self, message_buffer):
        """Test handling of messages without SourceTimestamp."""
        # Message without SourceTimestamp
        invalid_msg = TimestampedData(
            _timestamp=time.time(),
            _data={'test': 'data'},  # No SourceTimestamp!
            _source='mqtt/toolbox1/tool1/ResultManagement'
        )
        
        success = message_buffer.add_message(invalid_msg)
        assert success is False
        
        stats = message_buffer.get_buffer_stats()
        assert stats['timestamp_extraction_failures'] > 0
    
    def test_sequential_messages_preservation(self, message_buffer):
        """Test that sequential messages are not dropped (bug fix verification)."""
        tool_key = 'toolbox1/tool1'
        
        # Add messages for result IDs 421, 422, 423, 424
        for i, result_id in enumerate([421, 422, 423, 424]):
            # Each set has its own timestamp
            timestamp = f'2024-01-18T10:30:{i:02d}Z'
            
            result_msg = TimestampedData(
                _timestamp=time.time(),
                _data={
                    'SourceTimestamp': timestamp,
                    'ResultId': result_id
                },
                _source=f'mqtt/{tool_key}/ResultManagement'
            )
            trace_msg = TimestampedData(
                _timestamp=time.time(),
                _data={'SourceTimestamp': timestamp},
                _source=f'mqtt/{tool_key}/Trace'
            )
            heads_msg = TimestampedData(
                _timestamp=time.time(),
                _data={
                    'SourceTimestamp': timestamp,
                    'HeadsId': f'HEADS{result_id}'
                },
                _source=f'mqtt/{tool_key}/AssetManagement'
            )
            
            assert message_buffer.add_message(result_msg)
            assert message_buffer.add_message(trace_msg)
            assert message_buffer.add_message(heads_msg)
        
        # All 4 should have matched
        stats = message_buffer.get_buffer_stats()
        assert stats['exact_matches_completed'] == 4
        assert stats['messages_dropped'] == 0
    
    def test_expiry_cleanup(self, message_buffer):
        """Test cleanup of expired unmatched messages."""
        # Create buffer with short expiry
        buffer = MessageBuffer(
            config=message_buffer.config,
            cleanup_interval=1,
            max_age_seconds=2
        )
        
        # Add incomplete match (only result, no trace/heads)
        result_msg = TimestampedData(
            _timestamp=time.time(),
            _data={
                'SourceTimestamp': '2024-01-18T10:30:45Z',
                'ResultId': 100
            },
            _source='mqtt/toolbox1/tool1/ResultManagement'
        )
        
        buffer.add_message(result_msg)
        
        stats_before = buffer.get_buffer_stats()
        assert stats_before['total_active_keys'] == 1
        
        # Wait for expiry
        time.sleep(3)
        
        # Force cleanup check
        buffer.cleanup_expired_messages()
        
        stats_after = buffer.get_buffer_stats()
        assert stats_after['total_active_keys'] == 0
        assert stats_after['messages_expired'] > 0
    
    def test_get_buffer_stats(self, message_buffer, sample_result_message, 
                             sample_trace_message, sample_heads_message):
        """Test buffer statistics for exact matching."""
        # Add complete message set
        message_buffer.add_message(sample_result_message)
        message_buffer.add_message(sample_trace_message)
        message_buffer.add_message(sample_heads_message)
        
        stats = message_buffer.get_buffer_stats()
        
        # Check exact match statistics
        assert 'exact_matches_completed' in stats
        assert 'total_active_keys' in stats
        assert 'completed_matches_pending' in stats
        assert 'total_messages' in stats
        assert 'timestamp_extraction_failures' in stats
        
        assert stats['exact_matches_completed'] == 1
        assert stats['total_active_keys'] == 0
        assert stats['completed_matches_pending'] == 1
    
    def test_get_metrics(self, message_buffer):
        """Test buffer metrics calculation."""
        # Add some successful and failed operations
        msg_with_timestamp = TimestampedData(
            _timestamp=time.time(),
            _data={'SourceTimestamp': '2024-01-18T10:30:45Z'},
            _source='mqtt/toolbox1/tool1/ResultManagement'
        )
        msg_without_timestamp = TimestampedData(
            _timestamp=time.time(),
            _data={'no': 'timestamp'},
            _source='mqtt/toolbox1/tool1/ResultManagement'
        )
        
        message_buffer.add_message(msg_with_timestamp)
        message_buffer.add_message(msg_without_timestamp)
        
        metrics = message_buffer.get_metrics()
        
        assert 'exact_match_rate' in metrics
        assert 'extraction_failure_rate' in metrics
        assert 'messages_per_second' in metrics
        assert 'average_match_age' in metrics
    
    def test_nested_timestamp_extraction(self, message_buffer):
        """Test extraction of SourceTimestamp from nested structures."""
        # Test nested timestamp structure
        nested_msg = TimestampedData(
            _timestamp=time.time(),
            _data={
                'MessagePayload': {
                    'Value': {'test': 'data'},
                    'SourceTimestamp': '2024-01-18T10:30:45Z'
                }
            },
            _source='mqtt/toolbox1/tool1/ResultManagement'
        )
        
        success = message_buffer.add_message(nested_msg)
        assert success is True
        
        # Verify timestamp was extracted correctly
        stats = message_buffer.get_buffer_stats()
        assert stats['total_messages'] == 1
        assert stats['timestamp_extraction_failures'] == 0
    
    def test_multiple_tools_same_timestamp(self, message_buffer):
        """Test that different tools with same timestamp don't interfere."""
        timestamp = '2024-01-18T10:30:45Z'
        
        # Tool 1 messages
        for msg_type in ['ResultManagement', 'Trace', 'AssetManagement']:
            msg = TimestampedData(
                _timestamp=time.time(),
                _data={'SourceTimestamp': timestamp},
                _source=f'mqtt/toolbox1/tool1/{msg_type}'
            )
            message_buffer.add_message(msg)
        
        # Tool 2 messages (same timestamp, different tool)
        for msg_type in ['ResultManagement', 'Trace', 'AssetManagement']:
            msg = TimestampedData(
                _timestamp=time.time(),
                _data={'SourceTimestamp': timestamp},
                _source=f'mqtt/toolbox1/tool2/{msg_type}'
            )
            message_buffer.add_message(msg)
        
        stats = message_buffer.get_buffer_stats()
        assert stats['exact_matches_completed'] == 2  # Both tools matched
        assert stats['total_active_keys'] == 0


if __name__ == '__main__':
    pytest.main([__file__])