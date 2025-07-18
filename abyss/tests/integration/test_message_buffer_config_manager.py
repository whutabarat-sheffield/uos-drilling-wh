"""
Test the improved MessageBuffer with ConfigurationManager integration.

This test verifies that:
1. MessageBuffer can accept both ConfigurationManager and raw config
2. Configuration access is consistent between both methods
3. Backward compatibility is maintained
"""

import pytest
from abyss.mqtt.components.config_manager import ConfigurationManager
from abyss.mqtt.components.message_buffer import MessageBuffer


class TestMessageBufferConfigManager:
    """Test MessageBuffer integration with ConfigurationManager."""
    
    def test_message_buffer_with_config_manager(self, temp_config_file):
        """Test MessageBuffer with ConfigurationManager instance."""
        # Create ConfigurationManager
        config_manager = ConfigurationManager(temp_config_file)
        
        # Create MessageBuffer with ConfigurationManager
        message_buffer = MessageBuffer(config=config_manager)
        
        # Test that configuration is properly accessed
        assert message_buffer.duplicate_handling == 'ignore'
        assert message_buffer.config_manager is not None
        
        # Test topic patterns
        expected_patterns = {
            'result': 'test/drilling/+/+/Result',
            'trace': 'test/drilling/+/+/Trace',
            'heads': 'test/drilling/+/+/Heads'
        }
        
        for key, expected in expected_patterns.items():
            assert message_buffer._topic_patterns[key] == expected, \
                f"Topic pattern mismatch for {key}: expected {expected}, got {message_buffer._topic_patterns[key]}"
    
    def test_message_buffer_with_raw_config(self, minimal_config):
        """Test MessageBuffer with raw config dictionary (backward compatibility)."""
        # Create MessageBuffer with raw config
        message_buffer = MessageBuffer(config=minimal_config)
        
        # Test that configuration is properly accessed
        assert message_buffer.duplicate_handling == 'ignore'  # default value
        assert message_buffer.config_manager is None  # Should be None for raw config
        
        # Test topic patterns
        expected_patterns = {
            'result': 'test/+/+/R',
            'trace': 'test/+/+/T',
            'heads': 'test/+/+/H'
        }
        
        for key, expected in expected_patterns.items():
            assert message_buffer._topic_patterns[key] == expected, \
                f"Topic pattern mismatch for {key}: expected {expected}, got {message_buffer._topic_patterns[key]}"
    
    def test_custom_duplicate_handling_config(self, tmp_path):
        """Test MessageBuffer with custom duplicate handling configuration."""
        import yaml
        
        # Create config with custom duplicate handling
        config_data = {
            'mqtt': {
                'broker': {
                    'host': 'localhost',
                    'port': 1883
                },
                'listener': {
                    'duplicate_handling': 'error',
                    'duplicate_time_window': 2.5,
                    'root': 'TEST_ROOT',
                    'result': 'TestResult',
                    'trace': 'TestTrace',
                    'heads': 'TestHeads'
                }
            }
        }
        
        config_file = tmp_path / "custom_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Create ConfigurationManager
        config_manager = ConfigurationManager(str(config_file))
        
        # Create MessageBuffer with ConfigurationManager
        message_buffer = MessageBuffer(config=config_manager)
        
        # Test that custom configuration is properly accessed
        assert message_buffer.duplicate_handling == 'error'
        # Note: duplicate_time_window is internal, not exposed as attribute
    
    def test_both_approaches_equivalent(self, temp_config_file):
        """Test that both approaches produce equivalent results."""
        # Create both versions
        config_manager = ConfigurationManager(temp_config_file)
        raw_config = config_manager.get_raw_config()
        
        buffer_with_manager = MessageBuffer(config=config_manager)
        buffer_with_raw = MessageBuffer(config=raw_config)
        
        # Test equivalence
        assert buffer_with_manager.duplicate_handling == buffer_with_raw.duplicate_handling
        assert buffer_with_manager._topic_patterns == buffer_with_raw._topic_patterns
    
    def test_missing_heads_pattern(self):
        """Test MessageBuffer with config missing heads pattern."""
        # Create raw config without heads
        config = {
            'mqtt': {
                'listener': {
                    'duplicate_handling': 'replace',
                    'duplicate_time_window': 1.5,
                    'root': 'RAW_ROOT',
                    'result': 'RawResult',
                    'trace': 'RawTrace'
                    # Note: no 'heads' configured
                }
            }
        }
        
        # Create MessageBuffer with raw config
        message_buffer = MessageBuffer(config=config)
        
        # Test that configuration is properly accessed
        assert message_buffer.duplicate_handling == 'replace'
        
        # Test topic patterns
        expected_patterns = {
            'result': 'RAW_ROOT/+/+/RawResult',
            'trace': 'RAW_ROOT/+/+/RawTrace'
        }
        
        for key, expected in expected_patterns.items():
            assert message_buffer._topic_patterns[key] == expected
        
        # Test that heads pattern is created with empty value when not configured
        # (MessageBuffer creates it anyway with empty string)
        assert 'heads' in message_buffer._topic_patterns
        assert message_buffer._topic_patterns['heads'] == 'RAW_ROOT/+/+/'
    
    @pytest.mark.performance
    def test_message_buffer_performance(self, config_manager, performance_monitor):
        """Test MessageBuffer performance with large number of messages."""
        from abyss.uos_depth_est import TimestampedData
        import time
        
        message_buffer = MessageBuffer(config=config_manager)
        
        # Add many messages
        for i in range(1000):
            data = TimestampedData(
                _timestamp=time.time(),
                _data={'id': i, 'value': i * 2},
                _source=f'test/drilling/tb{i % 10}/tool{i % 5}/Result'
            )
            message_buffer.add_message(data)
        
        # Check buffer size - messages are stored by full topic path
        # Count total messages across all buffers
        total_messages = sum(len(buffer) for buffer in message_buffer.buffers.values())
        assert total_messages == 1000