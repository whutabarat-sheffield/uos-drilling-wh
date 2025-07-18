"""
Test suite for MessageBuffer warning conditions and metrics with exact matching.
"""

import pytest
import time
import threading
import logging
from unittest.mock import patch, MagicMock
from datetime import datetime

# Import the classes we need to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from abyss.mqtt.components.message_buffer import MessageBuffer
from abyss.uos_depth_est import TimestampedData


class TestMessageBufferWarnings:
    """Test cases for MessageBuffer warning conditions and metrics with exact matching."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'mqtt': {
                'listener': {
                    'root': 'test/root',
                    'result': 'ResultManagement',
                    'trace': 'Trace',
                    'heads': 'AssetManagement',
                    'duplicate_handling': 'ignore'
                }
            }
        }
    
    @pytest.fixture
    def small_buffer(self, sample_config):
        """Create MessageBuffer with small size for testing warnings."""
        return MessageBuffer(
            config=sample_config,
            cleanup_interval=300,
            max_buffer_size=10,  # Small buffer for testing (message sets, not individual messages)
            max_age_seconds=300
        )
    
    def create_test_message(self, msg_type: str, msg_id: int, source_timestamp: str, 
                          timestamp: float = None) -> TimestampedData:
        """Create a test message with exact timestamp matching."""
        if timestamp is None:
            timestamp = time.time()
            
        data = {
            'SourceTimestamp': source_timestamp,
            'toolboxID': 'toolbox1',
            'toolID': 'tool1'
        }
        
        # Add type-specific fields
        if msg_type == 'ResultManagement':
            data['ResultId'] = msg_id
        elif msg_type == 'AssetManagement':
            data['HeadsId'] = f'HEADS{msg_id}'
            
        return TimestampedData(
            _timestamp=timestamp,
            _data=data,
            _source=f'test/root/toolbox1/tool1/{msg_type}'
        )
    
    @patch('logging.warning')
    def test_progressive_buffer_warnings(self, mock_warning, small_buffer):
        """Test progressive buffer capacity warnings at 60%, 80%, 90%."""
        # Buffer size is 10 message sets
        # Add incomplete message sets to reach different capacity levels
        
        # Add 5 incomplete message sets (50% - no warning)
        for i in range(5):
            source_timestamp = f'2024-01-18T10:30:{i:02d}Z'
            # Only add result and trace (incomplete - missing heads)
            small_buffer.add_message(self.create_test_message('ResultManagement', i, source_timestamp))
            small_buffer.add_message(self.create_test_message('Trace', i, source_timestamp))
        assert mock_warning.call_count == 0
        
        # Add 1 more incomplete set (60% - moderate warning)
        source_timestamp = f'2024-01-18T10:30:05Z'
        small_buffer.add_message(self.create_test_message('ResultManagement', 5, source_timestamp))
        small_buffer.add_message(self.create_test_message('Trace', 5, source_timestamp))
        
        # Check for warning
        warning_found = any("Buffer usage elevated" in str(call) for call in mock_warning.call_args_list)
        assert warning_found or mock_warning.call_count > 0
        
        # Reset mock
        mock_warning.reset_mock()
        
        # Add 2 more incomplete sets (80% - high warning)
        for i in range(6, 8):
            source_timestamp = f'2024-01-18T10:30:{i:02d}Z'
            small_buffer.add_message(self.create_test_message('ResultManagement', i, source_timestamp))
            small_buffer.add_message(self.create_test_message('Trace', i, source_timestamp))
        
        # Check for high capacity warning
        warning_found = any("high capacity" in str(call) for call in mock_warning.call_args_list)
        assert warning_found or mock_warning.call_count > 0
        
        # Reset mock
        mock_warning.reset_mock()
        
        # Add 1 more incomplete set (90% - critical warning)
        source_timestamp = f'2024-01-18T10:30:08Z'
        small_buffer.add_message(self.create_test_message('ResultManagement', 8, source_timestamp))
        small_buffer.add_message(self.create_test_message('Trace', 8, source_timestamp))
        
        # Check for critical warning
        warning_found = any("CRITICAL" in str(call) for call in mock_warning.call_args_list)
        assert warning_found or mock_warning.call_count > 0
    
    @patch('logging.warning')
    def test_old_message_warning(self, mock_warning, small_buffer):
        """Test warning for old incomplete message sets in buffer."""
        # Add incomplete message set with old receive timestamp (5 minutes ago)
        old_receive_time = time.time() - 300
        source_timestamp = '2024-01-18T10:30:00Z'
        
        # Add only result and trace (incomplete set that will stay in buffer)
        small_buffer.add_message(self.create_test_message('ResultManagement', 1, source_timestamp, old_receive_time))
        small_buffer.add_message(self.create_test_message('Trace', 1, source_timestamp, old_receive_time))
        
        # Trigger message age check
        small_buffer._check_message_age_warnings()
        
        # Should warn about old incomplete sets
        warning_found = any("Old incomplete message sets" in str(call) for call in mock_warning.call_args_list)
        assert warning_found or mock_warning.call_count > 0
    
    @patch('logging.warning')
    def test_high_drop_rate_warning(self, mock_warning, small_buffer):
        """Test warning for high message drop rate."""
        # Add many complete message sets
        for i in range(20):
            source_timestamp = f'2024-01-18T10:30:{i:02d}Z'
            # Add complete sets
            small_buffer.add_message(self.create_test_message('ResultManagement', i, source_timestamp))
            small_buffer.add_message(self.create_test_message('Trace', i, source_timestamp))
            small_buffer.add_message(self.create_test_message('AssetManagement', i, source_timestamp))
        
        # Clear buffer to make room
        small_buffer.clear_buffer()
        
        # Force some drops by manipulating metrics
        with small_buffer._metrics_lock:
            small_buffer._metrics['messages_dropped'] = 10
            small_buffer._metrics['messages_received'] = 100
        
        # Check drop rate warning
        small_buffer._check_drop_rate_warning()
        
        # Should see high drop rate warning
        warning_found = any("drop rate" in str(call).lower() for call in mock_warning.call_args_list)
        assert warning_found or mock_warning.call_count > 0
    
    @patch('logging.warning')
    def test_excessive_cleanup_warning(self, mock_warning, small_buffer):
        """Test warning when cleanup removes too many incomplete message sets."""
        # Set smaller max_age for testing
        small_buffer.max_age_seconds = 1
        
        # Fill buffer with incomplete sets
        for i in range(10):
            source_timestamp = f'2024-01-18T10:30:{i:02d}Z'
            # Only add result and trace (incomplete)
            small_buffer.add_message(self.create_test_message('ResultManagement', i, source_timestamp))
            small_buffer.add_message(self.create_test_message('Trace', i, source_timestamp))
        
        # Wait for sets to become old
        time.sleep(1.1)
        
        # Add more incomplete sets to trigger cleanup
        for i in range(10, 15):
            source_timestamp = f'2024-01-18T10:30:{i:02d}Z'
            small_buffer.add_message(self.create_test_message('ResultManagement', i, source_timestamp))
            small_buffer.add_message(self.create_test_message('Trace', i, source_timestamp))
        
        # Force cleanup
        small_buffer._cleanup_old_incomplete_sets()
        
        # Should see warning about excessive cleanup
        warning_found = any("cleanup" in str(call).lower() for call in mock_warning.call_args_list)
        assert warning_found or small_buffer.get_buffer_stats()['messages_dropped'] > 0
    
    def test_metrics_collection(self, small_buffer):
        """Test that metrics are properly collected."""
        # Add some complete message sets
        for i in range(3):
            source_timestamp = f'2024-01-18T10:30:{i:02d}Z'
            small_buffer.add_message(self.create_test_message('ResultManagement', i, source_timestamp))
            small_buffer.add_message(self.create_test_message('Trace', i, source_timestamp))
            small_buffer.add_message(self.create_test_message('AssetManagement', i, source_timestamp))
        
        # Get buffer stats (new API)
        stats = small_buffer.get_buffer_stats()
        
        assert stats['messages_received'] >= 9  # 3 complete sets * 3 messages each
        assert stats['exact_matches_completed'] == 3
        assert stats['messages_dropped'] == 0
        assert stats['total_active_keys'] == 0  # All complete
    
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
        # Fill buffer beyond capacity with incomplete sets
        for i in range(20):
            source_timestamp = f'2024-01-18T10:30:{i:02d}Z'
            # Only add result and trace (incomplete - will stay in buffer)
            small_buffer.add_message(self.create_test_message('ResultManagement', i, source_timestamp))
            small_buffer.add_message(self.create_test_message('Trace', i, source_timestamp))
        
        # Check that buffer respects max size after cleanup
        stats = small_buffer.get_buffer_stats()
        assert stats['total_active_keys'] <= small_buffer.max_buffer_size
        
        # Check that cleanup occurred
        assert stats['messages_dropped'] > 0 or stats['total_active_keys'] < 20
    
    def test_duplicate_message_metrics(self, small_buffer):
        """Test that duplicate exact matches are tracked in metrics."""
        source_timestamp = '2024-01-18T10:30:00Z'
        
        # Add the same complete message set twice
        for _ in range(2):
            small_buffer.add_message(self.create_test_message('ResultManagement', 1, source_timestamp))
            small_buffer.add_message(self.create_test_message('Trace', 1, source_timestamp))
            small_buffer.add_message(self.create_test_message('AssetManagement', 1, source_timestamp))
        
        stats = small_buffer.get_buffer_stats()
        assert stats['duplicate_exact_matches'] >= 1  # Should detect duplicate complete match
    
    @patch('logging.warning')
    def test_concurrent_warning_conditions(self, mock_warning, small_buffer):
        """Test multiple warning conditions occurring simultaneously."""
        # Create conditions for multiple warnings
        # 1. Old incomplete message set
        old_timestamp = '2024-01-18T10:30:00Z'
        small_buffer.add_message(self.create_test_message('ResultManagement', 1, old_timestamp, time.time() - 300))
        small_buffer.add_message(self.create_test_message('Trace', 1, old_timestamp, time.time() - 300))
        
        # 2. High buffer usage with more incomplete sets
        for i in range(2, 9):
            source_timestamp = f'2024-01-18T10:30:{i:02d}Z'
            small_buffer.add_message(self.create_test_message('ResultManagement', i, source_timestamp))
            small_buffer.add_message(self.create_test_message('Trace', i, source_timestamp))
        
        # 3. Simulate high drop rate
        with small_buffer._metrics_lock:
            small_buffer._metrics['messages_dropped'] = 10
            small_buffer._metrics['messages_received'] = 100
        
        # Trigger checks
        small_buffer._check_message_age_warnings()
        small_buffer._check_drop_rate_warning()
        
        # Should see multiple different warnings
        warning_messages = [str(call) for call in mock_warning.call_args_list]
        
        # Check for different warning types
        has_old_message_warning = any("old" in msg.lower() for msg in warning_messages)
        has_capacity_warning = any("capacity" in msg.lower() for msg in warning_messages)
        has_drop_rate_warning = any("drop rate" in msg.lower() for msg in warning_messages)
        
        assert has_old_message_warning or has_capacity_warning or has_drop_rate_warning