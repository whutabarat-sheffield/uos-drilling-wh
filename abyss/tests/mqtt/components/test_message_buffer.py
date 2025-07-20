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

from abyss.mqtt.components.message_buffer import MessageBuffer, DuplicateMessageError
from abyss.uos_depth_est import TimestampedData


class TestMessageBuffer:
    """Test cases for MessageBuffer class."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'mqtt': {
                'broker': {
                    'host': 'localhost',
                    'port': 1883,
                    'username': 'test_user',
                    'password': 'test_pass',
                    'keepalive': 60
                },
                'listener': {
                    'root': 'test/root',
                    'result': 'Result',
                    'trace': 'Trace',
                    'heads': 'Heads'
                }
            }
        }
    
    @pytest.fixture
    def message_buffer(self, sample_config, tmp_path):
        """Create MessageBuffer instance for testing."""
        import yaml
        from abyss.mqtt.components.config_manager import ConfigurationManager
        
        # Create temporary config file from sample_config
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        # Create ConfigurationManager and MessageBuffer
        config_manager = ConfigurationManager(str(config_file))
        return MessageBuffer(
            config=config_manager,
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
    
    def test_initialization(self, sample_config, tmp_path):
        """Test MessageBuffer initialization."""
        import yaml
        from abyss.mqtt.components.config_manager import ConfigurationManager
        
        # Create temporary config file from sample_config
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        # Create ConfigurationManager and MessageBuffer
        config_manager = ConfigurationManager(str(config_file))
        buffer = MessageBuffer(config_manager)
        
        assert buffer.config_manager is not None
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
        message_buffer.cleanup_target_size = int(5 * 0.8)  # Recalculate cleanup target size
        
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
        buffer_size = len(message_buffer.buffers[result_topic])
        
        # Due to hysteresis cleanup behavior, buffer should be kept at or below max_buffer_size
        # The final size should be around cleanup_target_size after cleanup
        assert buffer_size <= message_buffer.max_buffer_size
    
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


class TestDuplicateDetection:
    """Test cases for duplicate message detection functionality."""
    
    @pytest.fixture
    def config_with_duplicate_handling(self, request):
        """Configuration with parameterized duplicate handling."""
        return {
            'mqtt': {
                'broker': {
                    'host': 'localhost',
                    'port': 1883
                },
                'listener': {
                    'root': 'OPCPUBSUB',
                    'result': 'ResultManagement',
                    'trace': 'ResultManagement/Trace',
                    'duplicate_handling': request.param,
                    'duplicate_time_window': 1.0
                }
            }
        }
    
    def test_basic_duplicate_detection(self, tmp_path):
        """Test basic duplicate detection scenarios."""
        import yaml
        from abyss.mqtt.components.config_manager import ConfigurationManager
        
        config = {
            'mqtt': {
                'broker': {
                    'host': 'localhost',
                    'port': 1883
                },
                'listener': {
                    'root': 'OPCPUBSUB',
                    'result': 'ResultManagement',
                    'trace': 'ResultManagement/Trace',
                    'duplicate_handling': 'ignore',
                    'duplicate_time_window': 1.0
                }
            }
        }
        
        # Create temporary config file
        config_file = tmp_path / "dup_test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Create ConfigurationManager and MessageBuffer
        config_manager = ConfigurationManager(str(config_file))
        buffer = MessageBuffer(config_manager)
        timestamp = time.time()
        test_data = {"test": "data", "value": 123}
        source = "OPCPUBSUB/toolbox1/tool1/ResultManagement"
        
        # Test 1: Same source, same timestamp, same data → duplicate
        msg1 = TimestampedData(timestamp, test_data, source)
        msg2 = TimestampedData(timestamp, test_data, source)
        
        assert buffer.add_message(msg1) is True
        assert buffer.add_message(msg2) is False  # Duplicate ignored
        
        # Test 2: Same source, different timestamp within window → duplicate
        msg3 = TimestampedData(timestamp + 0.5, test_data, source)
        assert buffer.add_message(msg3) is False
        
        # Test 3: Same source, different timestamp outside window → not duplicate
        msg4 = TimestampedData(timestamp + 2.0, test_data, source)
        assert buffer.add_message(msg4) is True
        
        # Test 4: Same timestamp, different data → not duplicate
        msg5 = TimestampedData(timestamp, {"different": "data"}, source)
        assert buffer.add_message(msg5) is True
    
    @pytest.mark.parametrize('config_with_duplicate_handling', ['ignore'], indirect=True)
    def test_duplicate_handling_ignore(self, config_with_duplicate_handling, tmp_path):
        """Test duplicate handling with 'ignore' strategy."""
        import yaml
        from abyss.mqtt.components.config_manager import ConfigurationManager
        
        # Create temporary config file
        config_file = tmp_path / "ignore_test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_with_duplicate_handling, f)
        
        # Create ConfigurationManager and MessageBuffer
        config_manager = ConfigurationManager(str(config_file))
        buffer = MessageBuffer(config_manager)
        timestamp = time.time()
        test_data = {"test": "data", "value": 123}
        source = "OPCPUBSUB/toolbox1/tool1/ResultManagement"
        
        msg1 = TimestampedData(timestamp, test_data, source)
        msg2 = TimestampedData(timestamp + 0.5, test_data, source)
        
        # First message should be added
        assert buffer.add_message(msg1) is True
        assert buffer.get_buffer_stats()['total_messages'] == 1
        
        # Duplicate should be ignored
        assert buffer.add_message(msg2) is False
        assert buffer.get_buffer_stats()['total_messages'] == 1
    
    @pytest.mark.parametrize('config_with_duplicate_handling', ['replace'], indirect=True)
    def test_duplicate_handling_replace(self, config_with_duplicate_handling, tmp_path):
        """Test duplicate handling with 'replace' strategy."""
        import yaml
        from abyss.mqtt.components.config_manager import ConfigurationManager
        
        # Create temporary config file
        config_file = tmp_path / "replace_test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_with_duplicate_handling, f)
        
        # Create ConfigurationManager and MessageBuffer
        config_manager = ConfigurationManager(str(config_file))
        buffer = MessageBuffer(config_manager)
        timestamp = time.time()
        original_data = {"version": 1, "value": "original"}
        updated_data = {"version": 1, "value": "original"}  # Same data for true duplicate
        source = "OPCPUBSUB/toolbox1/tool1/ResultManagement"
        
        msg1 = TimestampedData(timestamp, original_data, source)
        msg2 = TimestampedData(timestamp + 0.5, updated_data, source)
        
        # First message should be added
        assert buffer.add_message(msg1) is True
        assert buffer.get_buffer_stats()['total_messages'] == 1
        
        # Duplicate should replace the original
        assert buffer.add_message(msg2) is True
        assert buffer.get_buffer_stats()['total_messages'] == 1
        
        # Verify the message was replaced (checking timestamp)
        topic_pattern = 'OPCPUBSUB/+/+/ResultManagement'
        messages = buffer.get_messages_by_topic(topic_pattern)
        assert len(messages) == 1
        assert messages[0].timestamp == timestamp + 0.5
    
    @pytest.mark.parametrize('config_with_duplicate_handling', ['error'], indirect=True)
    def test_duplicate_handling_error(self, config_with_duplicate_handling, tmp_path):
        """Test duplicate handling with 'error' strategy."""
        import yaml
        from abyss.mqtt.components.config_manager import ConfigurationManager
        
        # Create temporary config file
        config_file = tmp_path / "error_test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_with_duplicate_handling, f)
        
        # Create ConfigurationManager and MessageBuffer
        config_manager = ConfigurationManager(str(config_file))
        buffer = MessageBuffer(config_manager)
        timestamp = time.time()
        test_data = {"test": "data", "value": 123}
        source = "OPCPUBSUB/toolbox1/tool1/ResultManagement"
        
        msg1 = TimestampedData(timestamp, test_data, source)
        msg2 = TimestampedData(timestamp + 0.5, test_data, source)
        
        # First message should be added
        assert buffer.add_message(msg1) is True
        
        # Duplicate should raise error
        with pytest.raises(DuplicateMessageError) as exc_info:
            buffer.add_message(msg2)
        
        assert "Duplicate message detected" in str(exc_info.value)
        assert buffer.get_buffer_stats()['total_messages'] == 1
    
    def test_complex_data_comparison(self, tmp_path):
        """Test duplicate detection with complex nested data structures."""
        import yaml
        from abyss.mqtt.components.config_manager import ConfigurationManager
        
        config = {
            'mqtt': {
                'broker': {
                    'host': 'localhost',
                    'port': 1883
                },
                'listener': {
                    'root': 'OPCPUBSUB',
                    'result': 'ResultManagement',
                    'trace': 'ResultManagement/Trace',
                    'duplicate_handling': 'ignore'
                }
            }
        }
        
        # Create temporary config file
        config_file = tmp_path / "complex_test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Create ConfigurationManager and MessageBuffer
        config_manager = ConfigurationManager(str(config_file))
        buffer = MessageBuffer(config_manager)
        timestamp = time.time()
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
        
        msg1 = TimestampedData(timestamp, data1, source)
        msg2 = TimestampedData(timestamp + 0.5, data2, source)  # Same data, different order
        msg3 = TimestampedData(timestamp + 0.7, data3, source)  # Different data
        
        # First message should be added
        assert buffer.add_message(msg1) is True
        assert buffer.get_buffer_stats()['total_messages'] == 1
        
        # Same data with different key order should be duplicate
        assert buffer.add_message(msg2) is False
        assert buffer.get_buffer_stats()['total_messages'] == 1
        
        # Actually different data should be added
        assert buffer.add_message(msg3) is True
        assert buffer.get_buffer_stats()['total_messages'] == 2
    
    def test_custom_time_window(self, tmp_path):
        """Test duplicate detection with custom time window."""
        import yaml
        from abyss.mqtt.components.config_manager import ConfigurationManager
        
        config = {
            'mqtt': {
                'broker': {
                    'host': 'localhost',
                    'port': 1883
                },
                'listener': {
                    'root': 'OPCPUBSUB',
                    'result': 'ResultManagement',
                    'trace': 'ResultManagement/Trace',
                    'duplicate_handling': 'ignore',
                    'duplicate_time_window': 2.0  # 2 second window
                }
            }
        }
        
        # Create temporary config file
        config_file = tmp_path / "time_window_test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Create ConfigurationManager and MessageBuffer
        config_manager = ConfigurationManager(str(config_file))
        buffer = MessageBuffer(config_manager)
        timestamp = time.time()
        test_data = {"test": "data"}
        source = "OPCPUBSUB/toolbox1/tool1/ResultManagement"
        
        msg1 = TimestampedData(timestamp, test_data, source)
        msg2 = TimestampedData(timestamp + 1.5, test_data, source)  # Within 2s window
        msg3 = TimestampedData(timestamp + 2.5, test_data, source)  # Outside 2s window
        
        assert buffer.add_message(msg1) is True
        assert buffer.add_message(msg2) is False  # Duplicate
        assert buffer.add_message(msg3) is True   # Not duplicate
    
    def test_numeric_precision_in_duplicates(self, tmp_path):
        """Test duplicate detection handles floating point precision."""
        import yaml
        from abyss.mqtt.components.config_manager import ConfigurationManager
        
        config = {
            'mqtt': {
                'broker': {
                    'host': 'localhost',
                    'port': 1883
                },
                'listener': {
                    'root': 'OPCPUBSUB',
                    'result': 'ResultManagement',
                    'trace': 'ResultManagement/Trace',
                    'duplicate_handling': 'ignore'
                }
            }
        }
        
        # Create temporary config file
        config_file = tmp_path / "precision_test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Create ConfigurationManager and MessageBuffer
        config_manager = ConfigurationManager(str(config_file))
        buffer = MessageBuffer(config_manager)
        timestamp = time.time()
        source = "OPCPUBSUB/toolbox1/tool1/ResultManagement"
        
        # Test floating point comparison
        data1 = {"value": 3.14159265359}
        data2 = {"value": 3.14159265359}  # Exact same
        data3 = {"value": 2.71828}        # Clearly different value (e)
        
        msg1 = TimestampedData(timestamp, data1, source)
        msg2 = TimestampedData(timestamp + 0.1, data2, source)
        msg3 = TimestampedData(timestamp + 0.2, data3, source)
        
        assert buffer.add_message(msg1) is True
        assert buffer.add_message(msg2) is False  # Duplicate
        assert buffer.add_message(msg3) is True   # Different value


if __name__ == '__main__':
    pytest.main([__file__])