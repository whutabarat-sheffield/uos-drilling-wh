import pytest
import tempfile
import yaml
import os
from unittest.mock import Mock, patch
from abyss.mqtt.components.simple_correlator import SimpleMessageCorrelator
from abyss.mqtt.components.config_manager import ConfigurationManager
from abyss.uos_depth_est import TimestampedData
import time
import logging


class TestSimpleCorrelator:
    """Test the simplified message correlator"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration"""
        config_data = {
            'mqtt': {
                'broker': {
                    'host': 'localhost',
                    'port': 1883,
                    'username': '',
                    'password': ''
                },
                'listener': {
                    'root': 'OPCPUBSUB',
                    'result': 'ResultManagement',
                    'trace': 'ResultManagement/Trace',
                    'heads': 'AssetManagement/Head'
                },
                'estimation': {
                    'keypoints': 'Estimation/Keypoints',
                    'depth_estimation': 'Estimation/DepthEstimation'
                }
            }
        }
        
        # Create temporary config file and ConfigurationManager
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            config_manager = ConfigurationManager(config_file)
            yield config_manager
        finally:
            os.unlink(config_file)
    
    @pytest.fixture
    def simple_correlator(self, mock_config):
        """Create simple correlator"""
        return SimpleMessageCorrelator(mock_config, time_window=5.0)
    
    def test_simple_correlator_basic_matching(self, simple_correlator):
        """Test basic message matching with simple correlator"""
        timestamp = time.time()
        
        # Create test messages
        result_msg = TimestampedData(
            _timestamp=timestamp,
            _data='{"test": "result"}',
            _source='OPCPUBSUB/toolbox1/tool1/ResultManagement'
        )
        
        trace_msg = TimestampedData(
            _timestamp=timestamp + 1.0,  # 1 second later
            _data='{"test": "trace"}',
            _source='OPCPUBSUB/toolbox1/tool1/ResultManagement/Trace'
        )
        
        heads_msg = TimestampedData(
            _timestamp=timestamp + 0.5,  # 0.5 seconds later
            _data='{"test": "heads"}',
            _source='OPCPUBSUB/toolbox1/tool1/AssetManagement/Head'
        )
        
        # Create buffers
        buffers = {
            'OPCPUBSUB/+/+/ResultManagement': [result_msg],
            'OPCPUBSUB/+/+/ResultManagement/Trace': [trace_msg],
            'OPCPUBSUB/+/+/AssetManagement/Head': [heads_msg]
        }
        
        # Mock message processor
        processed_matches = []
        def mock_processor(messages):
            processed_matches.append(messages)
        
        # Test correlation
        found_matches = simple_correlator.find_and_process_matches(buffers, mock_processor)
        
        # Verify results
        assert found_matches is True
        assert len(processed_matches) == 1
        assert len(processed_matches[0]) == 3  # result + trace + heads
        
        # Verify messages are marked as processed
        assert getattr(result_msg, 'processed', False) is True
        assert getattr(trace_msg, 'processed', False) is True
        assert getattr(heads_msg, 'processed', False) is True
    
    def test_simple_correlator_no_heads_message(self, simple_correlator):
        """Test correlation without heads message"""
        timestamp = time.time()
        
        # Create test messages (no heads)
        result_msg = TimestampedData(
            _timestamp=timestamp,
            _data='{"test": "result"}',
            _source='OPCPUBSUB/toolbox2/tool2/ResultManagement'
        )
        
        trace_msg = TimestampedData(
            _timestamp=timestamp + 2.0,  # 2 seconds later
            _data='{"test": "trace"}',
            _source='OPCPUBSUB/toolbox2/tool2/ResultManagement/Trace'
        )
        
        # Create buffers (no heads buffer)
        buffers = {
            'OPCPUBSUB/+/+/ResultManagement': [result_msg],
            'OPCPUBSUB/+/+/ResultManagement/Trace': [trace_msg]
        }
        
        # Mock message processor
        processed_matches = []
        def mock_processor(messages):
            processed_matches.append(messages)
        
        # Test correlation
        found_matches = simple_correlator.find_and_process_matches(buffers, mock_processor)
        
        # Verify results
        assert found_matches is True
        assert len(processed_matches) == 1
        assert len(processed_matches[0]) == 2  # result + trace only
    
    def test_simple_correlator_time_window_rejection(self, simple_correlator):
        """Test that messages outside time window are rejected"""
        timestamp = time.time()
        
        # Create test messages with large time difference
        result_msg = TimestampedData(
            _timestamp=timestamp,
            _data='{"test": "result"}',
            _source='OPCPUBSUB/toolbox3/tool3/ResultManagement'
        )
        
        trace_msg = TimestampedData(
            _timestamp=timestamp + 10.0,  # 10 seconds later (outside 5s window)
            _data='{"test": "trace"}',
            _source='OPCPUBSUB/toolbox3/tool3/ResultManagement/Trace'
        )
        
        # Create buffers
        buffers = {
            'OPCPUBSUB/+/+/ResultManagement': [result_msg],
            'OPCPUBSUB/+/+/ResultManagement/Trace': [trace_msg]
        }
        
        # Mock message processor
        processed_matches = []
        def mock_processor(messages):
            processed_matches.append(messages)
        
        # Test correlation
        found_matches = simple_correlator.find_and_process_matches(buffers, mock_processor)
        
        # Verify no matches found
        assert found_matches is False
        assert len(processed_matches) == 0
    
    def test_simple_correlator_different_tool_keys(self, simple_correlator):
        """Test that messages with different tool keys don't match"""
        timestamp = time.time()
        
        # Create test messages with different tool keys
        result_msg = TimestampedData(
            _timestamp=timestamp,
            _data='{"test": "result"}',
            _source='OPCPUBSUB/toolbox1/tool1/ResultManagement'
        )
        
        trace_msg = TimestampedData(
            _timestamp=timestamp + 1.0,
            _data='{"test": "trace"}',
            _source='OPCPUBSUB/toolbox2/tool2/ResultManagement/Trace'  # Different tool
        )
        
        # Create buffers
        buffers = {
            'OPCPUBSUB/+/+/ResultManagement': [result_msg],
            'OPCPUBSUB/+/+/ResultManagement/Trace': [trace_msg]
        }
        
        # Mock message processor
        processed_matches = []
        def mock_processor(messages):
            processed_matches.append(messages)
        
        # Test correlation
        found_matches = simple_correlator.find_and_process_matches(buffers, mock_processor)
        
        # Verify no matches found
        assert found_matches is False
        assert len(processed_matches) == 0
    
    def test_group_by_tool_key(self, simple_correlator):
        """Test tool key grouping functionality"""
        messages = [
            TimestampedData(_timestamp=1.0, _data='{}', _source='OPCPUBSUB/tb1/t1/ResultManagement'),
            TimestampedData(_timestamp=2.0, _data='{}', _source='OPCPUBSUB/tb1/t2/ResultManagement'),
            TimestampedData(_timestamp=3.0, _data='{}', _source='OPCPUBSUB/tb2/t1/ResultManagement'),
            TimestampedData(_timestamp=4.0, _data='{}', _source='OPCPUBSUB/tb1/t1/ResultManagement'),
        ]
        
        grouped = simple_correlator._group_by_tool_key(messages)
        
        assert len(grouped) == 3  # tb1/t1, tb1/t2, tb2/t1
        assert len(grouped['tb1/t1']) == 2
        assert len(grouped['tb1/t2']) == 1
        assert len(grouped['tb2/t1']) == 1
    
    def test_correlation_stats(self, simple_correlator):
        """Test correlation statistics"""
        # Create some test messages
        messages = [
            TimestampedData(_timestamp=1.0, _data='{}', _source='test'),
            TimestampedData(_timestamp=2.0, _data='{}', _source='test')
        ]
        
        buffers = {
            'OPCPUBSUB/+/+/ResultManagement': messages,
            'OPCPUBSUB/+/+/ResultManagement/Trace': []
        }
        
        stats = simple_correlator.get_correlation_stats(buffers)
        
        assert stats['total_messages'] == 2
        assert stats['time_window'] == 5.0
        assert stats['correlation_approach'] == 'simple_key_based'
        assert 'unprocessed_messages' in stats
    
    def test_error_handling_in_find_and_process_matches(self, simple_correlator):
        """Test graceful error handling when message processor raises exception"""
        timestamp = time.time()
        
        # Create multiple valid messages that would normally correlate
        result_msgs = [
            TimestampedData(
                _timestamp=timestamp,
                _data='{"test": "result1"}',
                _source='OPCPUBSUB/toolbox1/tool1/ResultManagement'
            ),
            TimestampedData(
                _timestamp=timestamp + 2,
                _data='{"test": "result2"}',
                _source='OPCPUBSUB/toolbox2/tool2/ResultManagement'
            )
        ]
        
        trace_msgs = [
            TimestampedData(
                _timestamp=timestamp + 0.5,
                _data='{"test": "trace1"}',
                _source='OPCPUBSUB/toolbox1/tool1/ResultManagement/Trace'
            ),
            TimestampedData(
                _timestamp=timestamp + 2.5,
                _data='{"test": "trace2"}',
                _source='OPCPUBSUB/toolbox2/tool2/ResultManagement/Trace'
            )
        ]
        
        buffers = {
            'OPCPUBSUB/+/+/ResultManagement': result_msgs,
            'OPCPUBSUB/+/+/ResultManagement/Trace': trace_msgs
        }
        
        # Mock message processor that raises exception on first call
        call_count = 0
        def failing_processor(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Processing failed")
            # Second call succeeds
        
        # Test correlation with failing processor
        with patch('logging.error') as mock_log_error:
            found_matches = simple_correlator.find_and_process_matches(buffers, failing_processor)
            
            # Verify error handling - should still find matches despite one failure
            assert found_matches is True
            assert call_count == 2  # Both matches attempted
            mock_log_error.assert_called_once()
            call_args = mock_log_error.call_args
            assert "Error processing message match" in call_args[0][0]
    
    def test_extract_tool_key_with_invalid_source(self, simple_correlator):
        """Test handling of malformed source topics"""
        test_cases = [
            ("", None),  # Empty string
            ("OPCPUBSUB", None),  # Single segment
            ("OPCPUBSUB/toolbox", None),  # Missing tool segment
        ]
        
        for source, expected in test_cases:
            result = simple_correlator._extract_tool_key(source)
            assert result == expected
        
        # Test None handling separately
        with patch('logging.warning') as mock_warning:
            result = simple_correlator._extract_tool_key(None)
            assert result is None
            mock_warning.assert_called()
    
    def test_multiple_messages_same_tool_key_unbalanced(self, simple_correlator):
        """Test correlation with multiple unbalanced messages from same tool"""
        base_time = time.time()
        
        # Create 3 result messages
        result_msgs = [
            TimestampedData(
                _timestamp=base_time + i,
                _data=f'{{"seq": {i}}}',
                _source='OPCPUBSUB/toolbox1/tool1/ResultManagement'
            ) for i in range(3)
        ]
        
        # Create only 2 trace messages (unbalanced)
        trace_msgs = [
            TimestampedData(
                _timestamp=base_time + i + 0.5,
                _data=f'{{"seq": {i}}}',
                _source='OPCPUBSUB/toolbox1/tool1/ResultManagement/Trace'
            ) for i in range(2)
        ]
        
        # Create 1 heads message
        heads_msg = TimestampedData(
            _timestamp=base_time + 0.2,
            _data='{"type": "heads"}',
            _source='OPCPUBSUB/toolbox1/tool1/AssetManagement/Head'
        )
        
        buffers = {
            'OPCPUBSUB/+/+/ResultManagement': result_msgs,
            'OPCPUBSUB/+/+/ResultManagement/Trace': trace_msgs,
            'OPCPUBSUB/+/+/AssetManagement/Head': [heads_msg]
        }
        
        processed_matches = []
        def mock_processor(messages):
            processed_matches.append(messages)
        
        # Test correlation
        found_matches = simple_correlator.find_and_process_matches(buffers, mock_processor)
        
        # Verify results
        assert found_matches is True
        assert len(processed_matches) == 2  # Only 2 complete pairs
        
        # Verify each result message matched only once
        assert result_msgs[0].processed is True
        assert result_msgs[1].processed is True
        assert result_msgs[2].processed is False  # No matching trace
        
        # Verify heads message reused for first match only
        assert heads_msg.processed is True
    
    def test_cleanup_orphaned_heads_messages(self, simple_correlator):
        """Test removal of old heads messages that can't be correlated"""
        current_time = time.time()
        
        # Create old heads messages (> 60s)
        old_heads = [
            TimestampedData(
                _timestamp=current_time - 70,  # 70 seconds old
                _data='{"old": true}',
                _source='OPCPUBSUB/toolbox1/tool1/AssetManagement/Head'
            ),
            TimestampedData(
                _timestamp=current_time - 65,  # 65 seconds old
                _data='{"old": true}',
                _source='OPCPUBSUB/toolbox2/tool2/AssetManagement/Head'
            )
        ]
        
        # Create recent heads message
        recent_heads = TimestampedData(
            _timestamp=current_time - 30,  # 30 seconds old
            _data='{"old": false}',
            _source='OPCPUBSUB/toolbox3/tool3/AssetManagement/Head'
        )
        
        # Need at least one result/trace message to trigger correlation logic
        result_msg = TimestampedData(
            _timestamp=current_time,
            _data='{"test": "result"}',
            _source='OPCPUBSUB/toolbox4/tool4/ResultManagement'
        )
        
        trace_msg = TimestampedData(
            _timestamp=current_time + 0.5,
            _data='{"test": "trace"}',
            _source='OPCPUBSUB/toolbox4/tool4/ResultManagement/Trace'
        )
        
        buffers = {
            'OPCPUBSUB/+/+/ResultManagement': [result_msg],
            'OPCPUBSUB/+/+/ResultManagement/Trace': [trace_msg],
            'OPCPUBSUB/+/+/AssetManagement/Head': old_heads + [recent_heads]
        }
        
        with patch('logging.info') as mock_info:
            # Run correlation (will trigger cleanup)
            simple_correlator.find_and_process_matches(buffers, lambda x: None)
            
            # Verify old messages marked as processed
            assert old_heads[0].processed is True
            assert old_heads[1].processed is True
            assert recent_heads.processed is False
            
            # Verify cleanup was logged
            cleanup_logged = any(
                "Cleaned up 2 orphaned heads messages" in str(call)
                for call in mock_info.call_args_list
            )
            assert cleanup_logged
    
    def test_correlation_health_high_unprocessed_warning(self, simple_correlator):
        """Test warning when unprocessed messages exceed threshold"""
        timestamp = time.time()
        
        # Create 150 unprocessed messages (> 100 threshold)
        result_msgs = [
            TimestampedData(
                _timestamp=timestamp + i * 10,  # Outside time window
                _data=f'{{"seq": {i}}}',
                _source=f'OPCPUBSUB/toolbox{i}/tool{i}/ResultManagement'
            ) for i in range(75)
        ]
        
        trace_msgs = [
            TimestampedData(
                _timestamp=timestamp + i * 10 + 100,  # Way outside time window
                _data=f'{{"seq": {i}}}',
                _source=f'OPCPUBSUB/toolbox{i}/tool{i}/ResultManagement/Trace'
            ) for i in range(75)
        ]
        
        buffers = {
            'OPCPUBSUB/+/+/ResultManagement': result_msgs,
            'OPCPUBSUB/+/+/ResultManagement/Trace': trace_msgs
        }
        
        with patch('logging.warning') as mock_warning:
            # First call should log warning
            simple_correlator.find_and_process_matches(buffers, lambda x: None)
            
            warning_logged = any(
                "High number of unprocessed messages" in str(call)
                for call in mock_warning.call_args_list
            )
            assert warning_logged
            
            # Reset mock
            mock_warning.reset_mock()
            
            # Second call within 60s should NOT log (rate limiting)
            simple_correlator.find_and_process_matches(buffers, lambda x: None)
            assert mock_warning.call_count == 0
    
    def test_tool_correlation_failure_tracking(self, simple_correlator):
        """Test warning after repeated correlation failures for a tool"""
        timestamp = time.time()
        
        # Create messages that won't correlate (outside time window)
        result_msg = TimestampedData(
            _timestamp=timestamp,
            _data='{"test": "result"}',
            _source='OPCPUBSUB/toolbox1/tool1/ResultManagement'
        )
        
        trace_msg = TimestampedData(
            _timestamp=timestamp + 40,  # 40s later (outside 5s window)
            _data='{"test": "trace"}',
            _source='OPCPUBSUB/toolbox1/tool1/ResultManagement/Trace'
        )
        
        buffers = {
            'OPCPUBSUB/+/+/ResultManagement': [result_msg],
            'OPCPUBSUB/+/+/ResultManagement/Trace': [trace_msg]
        }
        
        with patch('logging.warning') as mock_warning:
            # Run correlation 6 times
            for i in range(6):
                simple_correlator.find_and_process_matches(buffers, lambda x: None)
            
            # Should warn after 5 failures
            warning_logged = any(
                "Repeated correlation failures for tool" in str(call) and
                "toolbox1/tool1" in str(call)
                for call in mock_warning.call_args_list
            )
            assert warning_logged
            
            # Verify failure count in warning
            assert any(
                "'consecutive_failures': 6" in str(call) or
                "consecutive_failures': 5" in str(call)
                for call in mock_warning.call_args_list
            )


@pytest.mark.integration
class TestSimpleCorrelatorIntegration:
    """Integration tests with real ConfigurationManager"""
    
    def test_integration_with_real_config_manager(self):
        """Test correlator with actual ConfigurationManager"""
        # Create real config file
        config_data = {
            'mqtt': {
                'broker': {
                    'host': 'test-broker',
                    'port': 1883
                },
                'listener': {
                    'root': 'TEST/ROOT',
                    'result': 'Results',
                    'trace': 'Results/Traces',
                    'heads': 'Assets/Heads',
                    'time_window': 10.0  # Move time_window to listener section
                },
                'analyzer': {
                    'correlation_debug': True  # Use correct config path
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            # Create real ConfigurationManager
            config_manager = ConfigurationManager(config_file)
            
            # Create correlator with real config - don't pass time_window to use config value
            correlator = SimpleMessageCorrelator(config_manager, time_window=30.0)  # Use default to trigger config lookup
            
            # Verify configuration was properly loaded
            assert correlator.time_window == 10.0  # Should be from config
            assert correlator.debug_mode is True
            
            # Test basic correlation with real config topics
            timestamp = time.time()
            result_msg = TimestampedData(
                _timestamp=timestamp,
                _data='{"test": "result"}',
                _source='TEST/ROOT/box1/tool1/Results'
            )
            
            trace_msg = TimestampedData(
                _timestamp=timestamp + 1.0,
                _data='{"test": "trace"}',
                _source='TEST/ROOT/box1/tool1/Results/Traces'
            )
            
            buffers = {
                'TEST/ROOT/+/+/Results': [result_msg],
                'TEST/ROOT/+/+/Results/Traces': [trace_msg],
                'TEST/ROOT/+/+/Assets/Heads': []
            }
            
            processed = []
            def processor(msgs):
                processed.append(msgs)
            
            # Run correlation
            found = correlator.find_and_process_matches(buffers, processor)
            
            assert found is True
            assert len(processed) == 1
            assert len(processed[0]) == 2
            
        finally:
            os.unlink(config_file)


class TestSimpleCorrelatorChaos:
    """Chaos testing - multiple failure modes combined"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for chaos testing"""
        config_data = {
            'mqtt': {
                'broker': {
                    'host': 'localhost',
                    'port': 1883
                },
                'listener': {
                    'root': 'OPCPUBSUB',
                    'result': 'ResultManagement',
                    'trace': 'ResultManagement/Trace',
                    'heads': 'AssetManagement/Head'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            config_manager = ConfigurationManager(config_file)
            yield config_manager
        finally:
            os.unlink(config_file)
    
    @pytest.fixture
    def chaos_correlator(self, mock_config):
        """Create correlator for chaos testing"""
        return SimpleMessageCorrelator(mock_config, time_window=5.0)
    
    def test_chaos_multiple_failure_modes(self, chaos_correlator):
        """Test behavior under multiple simultaneous failures"""
        current_time = time.time()
        
        # Mix of valid and invalid messages
        messages = {
            'OPCPUBSUB/+/+/ResultManagement': [
                # Valid message
                TimestampedData(
                    _timestamp=current_time,
                    _data='{"valid": true}',
                    _source='OPCPUBSUB/box1/tool1/ResultManagement'
                ),
                # Message with None source (will cause exception in tool key extraction)
                TimestampedData(
                    _timestamp=current_time,
                    _data='{"valid": false}',
                    _source=None
                ),
                # Old message
                TimestampedData(
                    _timestamp=current_time - 100,
                    _data='{"old": true}',
                    _source='OPCPUBSUB/box2/tool2/ResultManagement'
                ),
            ],
            'OPCPUBSUB/+/+/ResultManagement/Trace': [
                # Matching trace for valid message
                TimestampedData(
                    _timestamp=current_time + 0.5,
                    _data='{"valid": true}',
                    _source='OPCPUBSUB/box1/tool1/ResultManagement/Trace'
                ),
                # Trace with malformed source
                TimestampedData(
                    _timestamp=current_time,
                    _data='{"malformed": true}',
                    _source='INVALID/SOURCE'
                ),
            ],
            'OPCPUBSUB/+/+/AssetManagement/Head': [
                # Very old heads message
                TimestampedData(
                    _timestamp=current_time - 120,
                    _data='{"ancient": true}',
                    _source='OPCPUBSUB/box3/tool3/AssetManagement/Head'
                ),
            ]
        }
        
        # Processor that fails on first call, succeeds on second
        call_count = 0
        processed = []
        
        def chaotic_processor(msgs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First call fails")
            processed.append(msgs)
        
        # Run correlation with all the chaos
        with patch('logging.error') as mock_error, \
             patch('logging.warning') as mock_warning, \
             patch('logging.info') as mock_info:
            
            # First attempt - processor will fail
            result1 = chaos_correlator.find_and_process_matches(messages, chaotic_processor)
            assert result1 is False  # Failed due to processor exception
            
            # Second attempt - processor will succeed
            result2 = chaos_correlator.find_and_process_matches(messages, chaotic_processor)
            
            # Despite all the chaos, valid messages should still correlate
            assert len(processed) == 1
            assert len(processed[0]) == 2  # Valid result + trace pair
            
            # Verify various issues were logged
            assert mock_error.call_count >= 1  # Processor exception
            assert mock_warning.call_count >= 1  # None source warning
            
            # Verify old heads cleanup happened
            old_heads = messages['OPCPUBSUB/+/+/AssetManagement/Head'][0]
            assert old_heads.processed is True


class TestCorrelatorComparison:
    """Compare simple and complex correlators"""
    
    @pytest.fixture
    def mock_config(self):
        return {
            'mqtt': {
                'listener': {
                    'root': 'OPCPUBSUB',
                    'result': 'ResultManagement',
                    'trace': 'ResultManagement/Trace',
                    'heads': 'AssetManagement/Head'
                }
            }
        }
    
