"""
Test suite for MessageBuffer class.
"""

import pytest
import time
from datetime import datetime
from collections import defaultdict

# Import the classes we need to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from abyss.mqtt.components.message_buffer import MessageBuffer
from abyss.uos_depth_est import TimestampedData


class TestMessageBuffer:
    """Test cases for MessageBuffer class."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'mqtt': {
                'listener': {
                    'root': 'test/root',
                    'result': 'Result',
                    'trace': 'Trace',
                    'heads': 'Heads'
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
        """Create sample result message."""
        return TimestampedData(
            _timestamp=time.time(),
            _data={'test': 'result_data'},
            _source='test/root/toolbox1/tool1/Result'
        )
    
    @pytest.fixture
    def sample_trace_message(self):
        """Create sample trace message."""
        return TimestampedData(
            _timestamp=time.time(),
            _data={'test': 'trace_data'},
            _source='test/root/toolbox1/tool1/Trace'
        )
    
    def test_initialization(self, sample_config):
        """Test MessageBuffer initialization."""
        buffer = MessageBuffer(sample_config)
        
        assert buffer.config == sample_config
        assert isinstance(buffer.buffers, defaultdict)
        assert buffer.cleanup_interval == 60
        assert buffer.max_buffer_size == 10000
        assert buffer.max_age_seconds == 300
    
    def test_add_result_message(self, message_buffer, sample_result_message):
        """Test adding a result message to buffer."""
        success = message_buffer.add_message(sample_result_message)
        
        assert success is True
        assert len(message_buffer.buffers) == 1
        
        # Check the message was added to correct buffer
        result_topic = 'test/root/+/+/Result'
        assert result_topic in message_buffer.buffers
        assert len(message_buffer.buffers[result_topic]) == 1
        assert message_buffer.buffers[result_topic][0] == sample_result_message
    
    def test_add_trace_message(self, message_buffer, sample_trace_message):
        """Test adding a trace message to buffer."""
        success = message_buffer.add_message(sample_trace_message)
        
        assert success is True
        
        # Check the message was added to correct buffer
        trace_topic = 'test/root/+/+/Trace'
        assert trace_topic in message_buffer.buffers
        assert len(message_buffer.buffers[trace_topic]) == 1
    
    def test_add_invalid_message(self, message_buffer):
        """Test adding invalid messages."""
        # Test None message
        success = message_buffer.add_message(None)
        assert success is False
        
        # Test message with no source
        invalid_msg = TimestampedData(
            _timestamp=time.time(),
            _data={'test': 'data'},
            _source=''
        )
        success = message_buffer.add_message(invalid_msg)
        assert success is False
    
    def test_add_non_essential_message(self, message_buffer):
        """Test adding non-essential message."""
        non_essential_msg = TimestampedData(
            _timestamp=time.time(),
            _data={'test': 'data'},
            _source='test/root/toolbox1/tool1/SomeOtherTopic'
        )
        
        success = message_buffer.add_message(non_essential_msg)
        assert success is False
        assert len(message_buffer.buffers) == 0
    
    def test_cleanup_old_messages(self, message_buffer):
        """Test cleanup of old messages."""
        # Add old message
        old_timestamp = time.time() - 400  # Older than max_age_seconds (300)
        old_message = TimestampedData(
            _timestamp=old_timestamp,
            _data={'test': 'old_data'},
            _source='test/root/toolbox1/tool1/Result'
        )
        
        # Add recent message
        recent_message = TimestampedData(
            _timestamp=time.time(),
            _data={'test': 'recent_data'},
            _source='test/root/toolbox1/tool1/Result'
        )
        
        message_buffer.add_message(old_message)
        message_buffer.add_message(recent_message)
        
        result_topic = 'test/root/+/+/Result'
        assert len(message_buffer.buffers[result_topic]) == 2
        
        # Trigger cleanup
        message_buffer.cleanup_old_messages()
        
        # Old message should be removed, recent message should remain
        assert len(message_buffer.buffers[result_topic]) == 1
        assert message_buffer.buffers[result_topic][0] == recent_message
    
    def test_buffer_size_limit(self, message_buffer):
        """Test buffer size limit enforcement."""
        message_buffer.max_buffer_size = 5  # Set low limit for testing
        
        # Add more messages than the limit
        for i in range(10):
            msg = TimestampedData(
                _timestamp=time.time() + i,
                _data={'test': f'data_{i}'},
                _source='test/root/toolbox1/tool1/Result'
            )
            message_buffer.add_message(msg)
        
        # Should trigger cleanup and keep only newest messages
        result_topic = 'test/root/+/+/Result'
        assert len(message_buffer.buffers[result_topic]) <= message_buffer.max_buffer_size
    
    def test_get_buffer_stats(self, message_buffer, sample_result_message, sample_trace_message):
        """Test buffer statistics."""
        message_buffer.add_message(sample_result_message)
        message_buffer.add_message(sample_trace_message)
        
        stats = message_buffer.get_buffer_stats()
        
        assert 'total_buffers' in stats
        assert 'total_messages' in stats
        assert 'buffer_sizes' in stats
        assert stats['total_buffers'] == 2
        assert stats['total_messages'] == 2
    
    def test_get_messages_by_topic(self, message_buffer, sample_result_message):
        """Test retrieving messages by topic."""
        message_buffer.add_message(sample_result_message)
        
        result_topic = 'test/root/+/+/Result'
        messages = message_buffer.get_messages_by_topic(result_topic)
        
        assert len(messages) == 1
        assert messages[0] == sample_result_message
    
    def test_clear_buffer(self, message_buffer, sample_result_message, sample_trace_message):
        """Test clearing buffers."""
        message_buffer.add_message(sample_result_message)
        message_buffer.add_message(sample_trace_message)
        
        assert message_buffer.get_buffer_stats()['total_messages'] == 2
        
        # Clear specific buffer
        result_topic = 'test/root/+/+/Result'
        message_buffer.clear_buffer(result_topic)
        
        assert len(message_buffer.buffers[result_topic]) == 0
        assert message_buffer.get_buffer_stats()['total_messages'] == 1
        
        # Clear all buffers
        message_buffer.clear_buffer()
        assert message_buffer.get_buffer_stats()['total_messages'] == 0


if __name__ == '__main__':
    pytest.main([__file__])