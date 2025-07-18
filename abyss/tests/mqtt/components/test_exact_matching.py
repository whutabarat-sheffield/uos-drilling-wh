"""
Test suite for exact matching implementation.

Tests the new MessageBuffer and SimpleMessageCorrelator that use
exact (tool_key, source_timestamp) matching instead of fuzzy time windows.
"""

import pytest
import time
import json
from datetime import datetime
from unittest.mock import Mock, patch

from abyss.mqtt.components.message_buffer_exact import MessageBuffer, MessageSet
from abyss.mqtt.components.simple_correlator_exact import SimpleMessageCorrelator
from abyss.uos_depth_est import TimestampedData


class TestExactMessageBuffer:
    """Test the exact-match MessageBuffer implementation."""
    
    @pytest.fixture
    def config(self):
        """Standard test configuration."""
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
    def buffer(self, config):
        """Create a MessageBuffer instance."""
        return MessageBuffer(config, cleanup_interval=60, max_age_seconds=300)
    
    def create_message(self, msg_type: str, tool_key: str, timestamp_str: str, 
                      result_id: int = None, nested: bool = False):
        """Helper to create test messages with various structures."""
        toolbox_id, tool_id = tool_key.split('/')
        topic_map = {
            'result': 'ResultManagement',
            'trace': 'Trace',
            'heads': 'AssetManagement'
        }
        
        source = f"mqtt/{toolbox_id}/{tool_id}/{topic_map[msg_type]}"
        
        if nested:
            # Test nested structure (like real data)
            data = {
                'MessagePayload': {
                    'Value': {'test': 'data'},
                    'SourceTimestamp': timestamp_str
                }
            }
        else:
            # Simple structure
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
    
    def test_exact_matching_same_timestamp(self, buffer):
        """Test that messages with exactly the same timestamp match."""
        tool_key = "toolbox1/tool1"
        timestamp = "2024-01-18T10:30:45Z"
        
        # Add result, trace, and heads with same timestamp
        result_msg = self.create_message('result', tool_key, timestamp, result_id=123)
        trace_msg = self.create_message('trace', tool_key, timestamp)
        heads_msg = self.create_message('heads', tool_key, timestamp)
        
        assert buffer.add_message(result_msg)
        assert buffer.add_message(trace_msg)
        assert buffer.add_message(heads_msg)
        
        # Check that match was found
        stats = buffer.get_buffer_stats()
        assert stats['exact_matches_completed'] == 1
        assert stats['total_active_keys'] == 0  # No pending matches
        assert stats['completed_matches_pending'] == 1  # Ready for processing
    
    def test_no_match_different_timestamps(self, buffer):
        """Test that messages with different timestamps don't match."""
        tool_key = "toolbox1/tool1"
        
        # Add result, trace, and heads with different timestamps
        result_msg = self.create_message('result', tool_key, "2024-01-18T10:30:45Z", result_id=123)
        trace_msg = self.create_message('trace', tool_key, "2024-01-18T10:30:46Z")  # 1 second later
        heads_msg = self.create_message('heads', tool_key, "2024-01-18T10:30:47Z")  # 2 seconds later
        
        assert buffer.add_message(result_msg)
        assert buffer.add_message(trace_msg)
        assert buffer.add_message(heads_msg)
        
        # Check that no match was found
        stats = buffer.get_buffer_stats()
        assert stats['exact_matches_completed'] == 0
        assert stats['total_active_keys'] == 3  # Three separate pending matches
    
    def test_multiple_tool_keys_same_timestamp(self, buffer):
        """Test that different tools with same timestamp don't interfere."""
        timestamp = "2024-01-18T10:30:45Z"
        
        # Tool 1 messages
        buffer.add_message(self.create_message('result', 'box1/tool1', timestamp, 123))
        buffer.add_message(self.create_message('trace', 'box1/tool1', timestamp))
        buffer.add_message(self.create_message('heads', 'box1/tool1', timestamp))
        
        # Tool 2 messages (same timestamp, different tool)
        buffer.add_message(self.create_message('result', 'box1/tool2', timestamp, 124))
        buffer.add_message(self.create_message('trace', 'box1/tool2', timestamp))
        buffer.add_message(self.create_message('heads', 'box1/tool2', timestamp))
        
        stats = buffer.get_buffer_stats()
        assert stats['exact_matches_completed'] == 2  # Both tools matched
        assert stats['total_active_keys'] == 0
    
    def test_nested_timestamp_extraction(self, buffer):
        """Test extraction of SourceTimestamp from nested structures."""
        tool_key = "toolbox1/tool1"
        timestamp = "2024-01-18T10:30:45Z"
        
        # Create messages with nested timestamp
        result_msg = self.create_message('result', tool_key, timestamp, result_id=123, nested=True)
        trace_msg = self.create_message('trace', tool_key, timestamp, nested=True)
        heads_msg = self.create_message('heads', tool_key, timestamp, nested=True)
        
        assert buffer.add_message(result_msg)
        assert buffer.add_message(trace_msg)
        assert buffer.add_message(heads_msg)
        
        stats = buffer.get_buffer_stats()
        assert stats['exact_matches_completed'] == 1
    
    def test_sequential_message_preservation(self, buffer):
        """Test that sequential messages are not dropped (the original bug)."""
        tool_key = "E00401/009F45BACA"
        base_time = "2024-01-18T10:30:00Z"
        
        # Add messages for result IDs 421, 422, 423, 424
        for i, result_id in enumerate([421, 422, 423, 424]):
            # Each has its own timestamp
            timestamp = f"2024-01-18T10:30:{i:02d}Z"
            
            result_msg = self.create_message('result', tool_key, timestamp, result_id)
            trace_msg = self.create_message('trace', tool_key, timestamp)
            heads_msg = self.create_message('heads', tool_key, timestamp)
            
            assert buffer.add_message(result_msg)
            assert buffer.add_message(trace_msg)
            assert buffer.add_message(heads_msg)
        
        # All 4 should have matched (now that heads is included)
        stats = buffer.get_buffer_stats()
        assert stats['exact_matches_completed'] == 4
        assert stats['total_active_keys'] == 0
        assert stats['messages_dropped'] == 0
    
    def test_expiry_of_unmatched_messages(self, buffer):
        """Test that old unmatched messages are expired and logged."""
        # Create buffer with short expiry
        buffer = MessageBuffer(buffer.config, cleanup_interval=1, max_age_seconds=2)
        
        # Add only result (incomplete match)
        buffer.add_message(self.create_message('result', 'box1/tool1', "2024-01-18T10:30:45Z", 123))
        
        stats_before = buffer.get_buffer_stats()
        assert stats_before['total_active_keys'] == 1
        
        # Wait for expiry
        time.sleep(3)
        
        stats_after = buffer.get_buffer_stats()
        assert stats_after['total_active_keys'] == 0
        assert stats_after['messages_expired'] > 0
    
    def test_heads_message_requirement(self, buffer):
        """Test that heads message is always required."""
        tool_key = "toolbox1/tool1"
        timestamp = "2024-01-18T10:30:45Z"
        
        # Add result and trace (but no heads)
        buffer.add_message(self.create_message('result', tool_key, timestamp, 123))
        buffer.add_message(self.create_message('trace', tool_key, timestamp))
        
        # Should not match without heads (heads is always required)
        stats = buffer.get_buffer_stats()
        assert stats['exact_matches_completed'] == 0
        assert stats['total_active_keys'] == 1
        
        # Now add heads message
        buffer.add_message(self.create_message('heads', tool_key, timestamp))
        stats = buffer.get_buffer_stats()
        assert stats['exact_matches_completed'] == 1
        assert stats['total_active_keys'] == 0
    
    def test_api_compatibility_methods(self, buffer):
        """Test that API compatibility methods work correctly."""
        tool_key = "toolbox1/tool1"
        timestamp = "2024-01-18T10:30:45Z"
        
        # Add all three message types
        result_msg = self.create_message('result', tool_key, timestamp, 123)
        trace_msg = self.create_message('trace', tool_key, timestamp)
        heads_msg = self.create_message('heads', tool_key, timestamp)
        buffer.add_message(result_msg)
        buffer.add_message(trace_msg)
        buffer.add_message(heads_msg)
        
        # Test get_all_buffers
        buffers = buffer.get_all_buffers()
        assert len(buffers) > 0
        
        # Test get_messages_by_topic
        result_pattern = "mqtt/+/+/ResultManagement"
        result_messages = buffer.get_messages_by_topic(result_pattern)
        # After match, messages should be in completed queue, not active
        
        # Test clear_buffer
        buffer.clear_buffer()
        stats = buffer.get_buffer_stats()
        assert stats['total_messages'] == 0
        
        # Test get_metrics
        metrics = buffer.get_metrics()
        assert 'exact_match_rate' in metrics
        assert 'extraction_failure_rate' in metrics


class TestSimpleMessageCorrelator:
    """Test the exact-match SimpleMessageCorrelator implementation."""
    
    @pytest.fixture
    def config(self):
        """Standard test configuration."""
        return {
            'mqtt': {
                'listener': {
                    'root': 'mqtt',
                    'result': 'ResultManagement',
                    'trace': 'Trace'
                }
            }
        }
    
    @pytest.fixture
    def correlator(self, config):
        """Create a SimpleMessageCorrelator instance."""
        return SimpleMessageCorrelator(config)
    
    def test_find_and_process_exact_matches(self, correlator):
        """Test that correlator correctly processes exact matches."""
        # Create test messages with matching timestamps
        timestamp = "2024-01-18T10:30:45Z"
        
        result_msg = TimestampedData(
            _timestamp=time.time(),
            _source="mqtt/box1/tool1/ResultManagement",
            _data={'SourceTimestamp': timestamp, 'ResultId': 100}
        )
        
        trace_msg = TimestampedData(
            _timestamp=time.time(),
            _source="mqtt/box1/tool1/Trace",
            _data={'SourceTimestamp': timestamp, 'TraceData': [1, 2, 3]}
        )
        
        heads_msg = TimestampedData(
            _timestamp=time.time(),
            _source="mqtt/box1/tool1/AssetManagement",
            _data={'SourceTimestamp': timestamp, 'AssetData': {'info': 'test'}}
        )
        
        # Create buffers dict as MessageBuffer would provide
        buffers = {
            'mqtt/+/+/ResultManagement': [result_msg],
            'mqtt/+/+/Trace': [trace_msg],
            'mqtt/+/+/AssetManagement': [heads_msg]
        }
        
        # Track processed messages
        processed = []
        def processor(messages):
            processed.extend(messages)
        
        # Process matches
        found = correlator.find_and_process_matches(buffers, processor)
        
        assert found is True
        assert len(processed) == 3
        assert result_msg in processed
        assert trace_msg in processed
        assert heads_msg in processed
        
        # Check metrics
        stats = correlator.get_correlation_stats(buffers)
        assert stats['matches_processed'] == 1
        assert stats['exact_matches_found'] == 1
    
    def test_no_match_different_timestamps(self, correlator):
        """Test that messages with different timestamps don't match."""
        result_msg = TimestampedData(
            _timestamp=time.time(),
            _source="mqtt/box1/tool1/ResultManagement",
            _data={'SourceTimestamp': "2024-01-18T10:30:45Z", 'ResultId': 100}
        )
        
        trace_msg = TimestampedData(
            _timestamp=time.time(),
            _source="mqtt/box1/tool1/Trace",
            _data={'SourceTimestamp': "2024-01-18T10:30:46Z", 'TraceData': [1, 2, 3]}
        )
        
        buffers = {
            'mqtt/+/+/ResultManagement': [result_msg],
            'mqtt/+/+/Trace': [trace_msg]
        }
        
        processed = []
        def processor(messages):
            processed.extend(messages)
        
        found = correlator.find_and_process_matches(buffers, processor)
        
        assert found is False
        assert len(processed) == 0
        
        stats = correlator.get_correlation_stats(buffers)
        assert stats['complete_groups'] == 0
        assert stats['incomplete_groups'] == 2  # Two separate timestamps
    
    def test_correlation_stats(self, correlator):
        """Test correlation statistics reporting."""
        # Create mixed messages
        messages = []
        
        # Complete triple (result, trace, heads)
        messages.append(TimestampedData(
            _timestamp=time.time(),
            _source="mqtt/box1/tool1/ResultManagement",
            _data={'SourceTimestamp': "2024-01-18T10:30:45Z"}
        ))
        messages.append(TimestampedData(
            _timestamp=time.time(),
            _source="mqtt/box1/tool1/Trace",
            _data={'SourceTimestamp': "2024-01-18T10:30:45Z"}
        ))
        messages.append(TimestampedData(
            _timestamp=time.time(),
            _source="mqtt/box1/tool1/AssetManagement",
            _data={'SourceTimestamp': "2024-01-18T10:30:45Z"}
        ))
        
        # Incomplete (only result)
        messages.append(TimestampedData(
            _timestamp=time.time(),
            _source="mqtt/box1/tool2/ResultManagement",
            _data={'SourceTimestamp': "2024-01-18T10:30:46Z"}
        ))
        
        buffers = {
            'mqtt/+/+/ResultManagement': [m for m in messages if 'Result' in m.source],
            'mqtt/+/+/Trace': [m for m in messages if 'Trace' in m.source],
            'mqtt/+/+/AssetManagement': [m for m in messages if 'Asset' in m.source]
        }
        
        stats = correlator.get_correlation_stats(buffers)
        
        assert stats['total_messages'] == 4  # 3 from complete group + 1 incomplete
        assert stats['exact_groups'] == 2  # Two different timestamps
        assert stats['complete_groups'] == 1  # Only one has all three types
        assert stats['incomplete_groups'] == 1
        assert stats['correlation_approach'] == 'exact_match_only'


class TestExactMatchIntegration:
    """Integration tests for the complete exact-match system."""
    
    def test_end_to_end_exact_matching(self):
        """Test complete flow from buffer to correlator."""
        config = {
            'mqtt': {
                'listener': {
                    'root': 'mqtt',
                    'result': 'ResultManagement',
                    'trace': 'Trace'
                }
            }
        }
        
        buffer = MessageBuffer(config)
        correlator = SimpleMessageCorrelator(config)
        
        # Track what gets processed
        processed_results = []
        
        def processor(messages):
            for msg in messages:
                if 'ResultId' in msg.data:
                    processed_results.append(msg.data['ResultId'])
        
        # Add matching messages
        timestamp = "2024-01-18T10:30:45Z"
        
        result_msg = TimestampedData(
            _timestamp=time.time(),
            _source="mqtt/box1/tool1/ResultManagement",
            _data={'SourceTimestamp': timestamp, 'ResultId': 100}
        )
        trace_msg = TimestampedData(
            _timestamp=time.time(),
            _source="mqtt/box1/tool1/Trace",
            _data={'SourceTimestamp': timestamp}
        )
        heads_msg = TimestampedData(
            _timestamp=time.time(),
            _source="mqtt/box1/tool1/AssetManagement",
            _data={'SourceTimestamp': timestamp}
        )
        
        buffer.add_message(result_msg)
        buffer.add_message(trace_msg)
        buffer.add_message(heads_msg)
        
        # Get buffers and process
        buffers = buffer.get_all_buffers()
        correlator.find_and_process_matches(buffers, processor)
        
        # Verify processing
        assert 100 in processed_results
        
        # Check buffer stats
        buffer_stats = buffer.get_buffer_stats()
        assert buffer_stats['exact_matches_completed'] == 1
        
        # Check correlator stats
        corr_stats = correlator.get_correlation_stats(buffers)
        assert corr_stats['matches_processed'] == 1
    
    def test_sequential_messages_not_dropped(self):
        """Integration test for the sequential message bug fix."""
        config = {
            'mqtt': {
                'listener': {
                    'root': 'mqtt',
                    'result': 'ResultManagement',
                    'trace': 'Trace'
                }
            }
        }
        
        buffer = MessageBuffer(config, max_buffer_size=5)  # Small buffer
        correlator = SimpleMessageCorrelator(config)
        
        processed_results = []
        
        def processor(messages):
            for msg in messages:
                if 'ResultId' in msg.data:
                    processed_results.append(msg.data['ResultId'])
        
        # Add sequential messages
        tool_key = "E00401/009F45BACA"
        
        for i, result_id in enumerate([421, 422, 423, 424]):
            timestamp = f"2024-01-18T10:30:{i:02d}Z"
            
            result_msg = TimestampedData(
                _timestamp=time.time(),
                _source=f"mqtt/{tool_key}/ResultManagement",
                _data={'SourceTimestamp': timestamp, 'ResultId': result_id}
            )
            trace_msg = TimestampedData(
                _timestamp=time.time(),
                _source=f"mqtt/{tool_key}/Trace",
                _data={'SourceTimestamp': timestamp}
            )
            heads_msg = TimestampedData(
                _timestamp=time.time(),
                _source=f"mqtt/{tool_key}/AssetManagement",
                _data={'SourceTimestamp': timestamp}
            )
            
            buffer.add_message(result_msg)
            buffer.add_message(trace_msg)
            buffer.add_message(heads_msg)
        
        # Process all matches
        buffers = buffer.get_all_buffers()
        correlator.find_and_process_matches(buffers, processor)
        
        # All messages should be processed
        assert sorted(processed_results) == [421, 422, 423, 424]
        
        # No messages should be dropped
        metrics = buffer.get_metrics()
        assert metrics['messages_dropped'] == 0
        assert metrics['exact_matches_completed'] == 4