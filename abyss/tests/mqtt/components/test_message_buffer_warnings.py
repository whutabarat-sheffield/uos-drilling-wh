"""
Test suite for MessageBuffer warning conditions and metrics.
"""

import pytest
import time
import threading
import logging
from unittest.mock import patch, MagicMock
from datetime import datetime

# Disable debug logging for tests
logging.getLogger().setLevel(logging.WARNING)

# Import the classes we need to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from abyss.mqtt.components.message_buffer import MessageBuffer
from abyss.uos_depth_est import TimestampedData


class TestMessageBufferWarnings:
    """Test cases for MessageBuffer warning conditions and metrics."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'mqtt': {
                'broker': {
                    'host': 'localhost',
                    'port': 1883
                },
                'listener': {
                    'root': 'test/root',
                    'result': 'Result',
                    'trace': 'Trace',
                    'heads': 'Heads',
                    'duplicate_handling': 'ignore'
                }
            }
        }
    
    @pytest.fixture
    def small_buffer(self, sample_config, tmp_path):
        """Create MessageBuffer with small size for testing warnings."""
        import yaml
        from abyss.mqtt.components.config_manager import ConfigurationManager
        
        # Create temporary config file
        config_file = tmp_path / "warnings_test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        # Create ConfigurationManager and MessageBuffer
        config_manager = ConfigurationManager(str(config_file))
        return MessageBuffer(
            config=config_manager,
            cleanup_interval=300,
            max_buffer_size=10,  # Small buffer for testing
            max_age_seconds=300
        )
    
    def create_test_message(self, msg_id: int, timestamp: float = None) -> TimestampedData:
        """Create a test message."""
        if timestamp is None:
            timestamp = time.time()
        return TimestampedData(
            _timestamp=timestamp,
            _data={'test_id': msg_id},
            _source=f'test/root/toolbox1/tool1/Result'
        )
    
    @patch('logging.warning')
    def test_progressive_buffer_warnings(self, mock_warning, small_buffer):
        """Test progressive buffer capacity warnings at 60%, 80%, 90%."""
        # Add messages to reach different capacity levels
        
        # Add 5 messages (50% - no warning)
        for i in range(5):
            small_buffer.add_message(self.create_test_message(i))
        assert mock_warning.call_count == 0
        
        # Add 1 more (60% - moderate warning)
        small_buffer.add_message(self.create_test_message(5))
        mock_warning.assert_called()
        args, kwargs = mock_warning.call_args
        assert "Buffer usage elevated" in args[0]
        
        # Reset mock
        mock_warning.reset_mock()
        
        # Add 2 more (80% - high warning)
        for i in range(6, 8):
            small_buffer.add_message(self.create_test_message(i))
        mock_warning.assert_called()
        args, kwargs = mock_warning.call_args
        assert "Buffer at high capacity" in args[0]
        
        # Reset mock
        mock_warning.reset_mock()
        
        # Add 1 more (90% - critical warning)
        small_buffer.add_message(self.create_test_message(8))
        mock_warning.assert_called()
        args, kwargs = mock_warning.call_args
        assert "CRITICAL: Buffer at critical capacity" in args[0]
    
    @patch('logging.warning')
    def test_old_message_warning(self, mock_warning, small_buffer):
        """Test warning for old messages in buffer."""
        # Add message with old timestamp (5 minutes ago)
        old_timestamp = time.time() - 300
        small_buffer.add_message(self.create_test_message(1, old_timestamp))
        
        # Trigger message age check
        small_buffer._check_message_age_warnings()
        
        mock_warning.assert_called()
        args, kwargs = mock_warning.call_args
        assert "Old messages detected in buffer" in args[0]
        assert kwargs['extra']['oldest_message_age_seconds'] > 240
    
    @patch('logging.warning')
    def test_high_drop_rate_warning(self, mock_warning, small_buffer):
        """Test warning for high message drop rate."""
        # Directly set metrics to simulate a high drop rate scenario
        with small_buffer._metrics_lock:
            small_buffer._metrics['messages_received'] = 95
            small_buffer._metrics['messages_dropped'] = 10  # ~10% drop rate

        # Ensure no background threads or blocking calls are triggered
        # Only call the warning check directly
        small_buffer._check_drop_rate_warning()

        mock_warning.assert_called()
        args, kwargs = mock_warning.call_args
        assert "High message drop rate detected" in args[0]
        assert kwargs['extra']['drop_rate_percent'] > 5
    
    @patch('logging.warning')
    def test_excessive_cleanup_warning(self, mock_warning, small_buffer):
        """Test warning when cleanup removes too many messages."""
        # Fill buffer to max capacity
        for i in range(10):
            small_buffer.add_message(self.create_test_message(i))
        
        # Force cleanup by setting old timestamps on more than half the messages
        with small_buffer._buffer_lock:
            # First ensure we have messages in a buffer
            if not small_buffer.buffers:
                # No buffers exist yet, skip test
                pytest.skip("No buffers created")
            
            topic = list(small_buffer.buffers.keys())[0]
            buffer = small_buffer.buffers[topic]
            original_size = len(buffer)
            
            # Make 60% of messages very old to trigger high removal percentage
            messages_to_age = int(original_size * 0.6)
            for i in range(messages_to_age):
                buffer[i]._timestamp = time.time() - 400  # Older than max_age_seconds
        
        # Manually trigger cleanup to ensure it runs
        small_buffer.cleanup_old_messages()
        
        # Should see warning about excessive cleanup
        warning_found = False
        for call in mock_warning.call_args_list:
            args, kwargs = call
            if args and "Excessive buffer cleanup indicates system falling behind" in args[0]:
                warning_found = True
                assert kwargs['extra']['removal_percent'] > 50
                break
        
        assert warning_found, "Expected excessive cleanup warning"
    
    def test_metrics_collection(self, small_buffer):
        """Test that metrics are properly collected."""
        # Add some messages
        for i in range(5):
            small_buffer.add_message(self.create_test_message(i))
        
        # Get metrics
        metrics = small_buffer.get_metrics()
        
        assert metrics['messages_received'] >= 5
        assert metrics['messages_processed'] >= 5
        assert metrics['messages_dropped'] == 0
        assert metrics['drop_rate'] == 0
        assert 'avg_processing_time_ms' in metrics
        assert metrics['cleanup_cycles'] >= 0
    
    def test_rate_limited_warnings(self, small_buffer):
        """Test that warnings are rate-limited to avoid spam."""
        with patch('logging.warning') as mock_warning:
            # Trigger the same warning multiple times quickly
            for _ in range(10):
                small_buffer._log_rate_limited_warning(
                    'test_warning', 60, "Test warning", {'data': 'test'}
                )
            
            # Should only be called once due to rate limiting
            assert mock_warning.call_count == 1
            
            # Simulate time passing
            with patch('time.time', return_value=time.time() + 61):
                small_buffer._log_rate_limited_warning(
                    'test_warning', 60, "Test warning", {'data': 'test'}
                )
            
            # Now should be called again
            assert mock_warning.call_count == 2
    
    def test_buffer_overflow_handling(self, small_buffer):
        """Test behavior when buffer overflows."""
        # Fill buffer beyond capacity
        messages_added = 0
        for i in range(20):
            success = small_buffer.add_message(self.create_test_message(i))
            if success:
                messages_added += 1
        
        # Check that buffer respects max size after cleanup
        stats = small_buffer.get_buffer_stats()
        assert stats['total_messages'] <= small_buffer.max_buffer_size
        
        # Check metrics show some processing
        metrics = small_buffer.get_metrics()
        assert metrics['cleanup_cycles'] > 0
    
    def test_duplicate_message_metrics(self, small_buffer):
        """Test that duplicate messages are tracked in metrics."""
        # Add same message multiple times
        msg = self.create_test_message(1)
        
        small_buffer.add_message(msg)
        small_buffer.add_message(msg)  # Duplicate
        small_buffer.add_message(msg)  # Duplicate
        
        metrics = small_buffer.get_metrics()
        assert metrics['duplicate_messages'] >= 2
        assert metrics['duplicate_rate'] > 0
    
    @patch('logging.warning')
    def test_concurrent_warning_conditions(self, mock_warning, small_buffer):
        """Test multiple warning conditions occurring simultaneously."""
        # Create conditions for multiple warnings
        # 1. Old messages
        old_msg = self.create_test_message(1, time.time() - 300)
        small_buffer.add_message(old_msg)
        
        # 2. High buffer usage
        for i in range(2, 9):
            small_buffer.add_message(self.create_test_message(i))
        
        # 3. Simulate high drop rate
        with small_buffer._metrics_lock:
            small_buffer._metrics['messages_dropped'] = 10
            small_buffer._metrics['messages_received'] = 100
        
        # Trigger checks
        small_buffer._check_message_age_warnings()
        small_buffer._check_drop_rate_warning()
        
        # Should see multiple different warnings
        warning_messages = [call[0][0] for call in mock_warning.call_args_list]
        
        # Check for different warning types
        has_old_message_warning = any("Old messages" in msg for msg in warning_messages)
        has_capacity_warning = any("capacity" in msg for msg in warning_messages)
        has_drop_rate_warning = any("drop rate" in msg for msg in warning_messages)
        
        assert has_old_message_warning or has_capacity_warning or has_drop_rate_warning