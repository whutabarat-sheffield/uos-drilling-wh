"""
Test suite for DrillingDataAnalyser orchestrator class.
"""

import pytest
import tempfile
import os
import yaml
import time
import threading
from unittest.mock import Mock, patch, MagicMock

# Get the expected version dynamically
try:
    from importlib.metadata import version
    EXPECTED_VERSION = version('abyss')
except Exception:
    EXPECTED_VERSION = "0.2.7"  # Fallback version same as in the code

# Import the classes we need to test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from abyss.mqtt.components.drilling_analyser import DrillingDataAnalyser
from abyss.mqtt.components.config_manager import ConfigurationError
from abyss.uos_depth_est import TimestampedData


class TestDrillingDataAnalyser:
    """Test cases for DrillingDataAnalyser orchestrator class."""
    
    @pytest.fixture
    def sample_config_dict(self):
        """Sample configuration dictionary."""
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
                    'root': 'drilling/data',
                    'result': 'Result',
                    'trace': 'Trace',
                    'heads': 'Heads',
                    'time_window': 30.0,
                    'cleanup_interval': 60,
                    'max_buffer_size': 10000
                },
                'estimation': {
                    'keypoints': 'Keypoints',
                    'depth_estimation': 'DepthEstimation'
                }
            }
        }
    
    @pytest.fixture
    def config_file(self, sample_config_dict):
        """Create temporary configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config_dict, f)
            config_path = f.name
        
        yield config_path
        
        # Cleanup
        if os.path.exists(config_path):
            os.unlink(config_path)
    
    @pytest.fixture
    def mock_depth_inference(self):
        """Mock depth inference for testing."""
        with patch('abyss.mqtt.components.drilling_analyser.DepthInference') as mock_di:
            mock_instance = Mock()
            mock_instance.infer3_common.return_value = [0.0, 5.0, 10.0]
            mock_di.return_value = mock_instance
            yield mock_instance
    
    def test_initialization_success(self, config_file, mock_depth_inference):
        """Test successful DrillingDataAnalyser initialization."""
        analyser = DrillingDataAnalyser(config_file)
        
        assert analyser.config_path == config_file
        assert analyser.ALGO_VERSION == EXPECTED_VERSION
        assert analyser.processing_active is False
        assert analyser.processing_thread is None
        
        # Check that components are initialized
        assert hasattr(analyser, 'config_manager')
        assert hasattr(analyser, 'message_buffer')
        assert hasattr(analyser, 'message_correlator')
        assert hasattr(analyser, 'data_converter')
        assert hasattr(analyser, 'message_processor')
        assert hasattr(analyser, 'client_manager')
        assert hasattr(analyser, 'depth_inference')
    
    def test_initialization_invalid_config(self):
        """Test initialization with invalid config file."""
        with pytest.raises((ConfigurationError, FileNotFoundError)):
            DrillingDataAnalyser('/nonexistent/config.yaml')
    
    def test_get_status(self, config_file, mock_depth_inference):
        """Test status retrieval."""
        analyser = DrillingDataAnalyser(config_file)
        
        status = analyser.get_status()
        
        assert 'processing_active' in status
        assert 'processing_thread_alive' in status
        assert 'algo_version' in status
        assert 'config_path' in status
        assert 'buffer_stats' in status
        assert 'config_summary' in status
        
        assert status['processing_active'] is False
        assert status['algo_version'] == EXPECTED_VERSION
        assert status['config_path'] == config_file
    
    @patch('paho.mqtt.client.Client')
    def test_setup_result_publisher(self, mock_mqtt_client, config_file, mock_depth_inference):
        """Test result publisher setup."""
        # Mock the MQTT client
        mock_client_instance = Mock()
        mock_mqtt_client.return_value = mock_client_instance
        
        analyser = DrillingDataAnalyser(config_file)
        
        # Manually create a result client to test setup
        analyser.client_manager.clients['result'] = mock_client_instance
        analyser._setup_result_publisher()
        
        assert analyser.result_publisher is not None
    
    def test_process_matched_messages_success(self, config_file, mock_depth_inference):
        """Test successful processing of matched messages."""
        analyser = DrillingDataAnalyser(config_file)
        
        # Mock the message processor result
        mock_result = Mock()
        mock_result.success = True
        mock_result.keypoints = [0.0, 5.0, 10.0]
        mock_result.depth_estimation = [5.0, 5.0]
        mock_result.machine_id = "TEST_MACHINE"
        mock_result.result_id = "TEST_RESULT"
        mock_result.error_message = None
        
        analyser.message_processor.process_matching_messages = Mock(return_value=mock_result)
        
        # Mock result publisher
        analyser.result_publisher = Mock()
        analyser.result_publisher.publish_processing_result = Mock(return_value=True)
        
        # Create test messages
        test_messages = [
            TimestampedData(
                _timestamp=time.time(),
                _data={'test': 'data'},
                _source='drilling/data/toolbox1/tool1/Result'
            )
        ]
        
        # Test processing
        analyser._process_matched_messages(test_messages)
        
        # Verify calls
        analyser.message_processor.process_matching_messages.assert_called_once_with(test_messages)
        analyser.result_publisher.publish_processing_result.assert_called_once()
    
    def test_process_matched_messages_failure(self, config_file, mock_depth_inference):
        """Test processing of matched messages when processing fails."""
        analyser = DrillingDataAnalyser(config_file)
        
        # Mock failed processing result
        mock_result = Mock()
        mock_result.success = False
        mock_result.error_message = "Processing failed"
        
        analyser.message_processor.process_matching_messages = Mock(return_value=mock_result)
        analyser.result_publisher = Mock()
        
        # Create test messages
        test_messages = [
            TimestampedData(
                _timestamp=time.time(),
                _data={'test': 'data'},
                _source='drilling/data/toolbox1/tool1/Result'
            )
        ]
        
        # Test processing
        analyser._process_matched_messages(test_messages)
        
        # Verify no publishing occurred
        analyser.result_publisher.publish_processing_result.assert_not_called()
    
    def test_continuous_processing_loop(self, config_file, mock_depth_inference):
        """Test continuous processing loop behavior."""
        analyser = DrillingDataAnalyser(config_file)
        
        # Mock the correlator to return False (no matches)
        analyser.message_correlator.find_and_process_matches = Mock(return_value=False)
        analyser.message_buffer.get_all_buffers = Mock(return_value={})
        
        # Start processing for a short time
        analyser.processing_active = True
        
        # Run processing in a thread for a brief moment
        def run_processing():
            start_time = time.time()
            while analyser.processing_active and (time.time() - start_time) < 0.5:
                analyser.continuous_processing()
                break  # Exit after one iteration for testing
        
        processing_thread = threading.Thread(target=run_processing)
        processing_thread.start()
        
        # Let it run briefly then stop
        time.sleep(0.1)
        analyser.processing_active = False
        processing_thread.join(timeout=1.0)
        
        # Verify the correlator was called
        analyser.message_correlator.find_and_process_matches.assert_called()
    
    @patch('paho.mqtt.client.Client')
    def test_shutdown(self, mock_mqtt_client, config_file, mock_depth_inference):
        """Test graceful shutdown."""
        # Mock MQTT clients
        mock_result_client = Mock()
        mock_trace_client = Mock()
        mock_heads_client = Mock()
        
        def mock_client_factory(client_id):
            if 'result' in client_id:
                return mock_result_client
            elif 'trace' in client_id:
                return mock_trace_client
            elif 'heads' in client_id:
                return mock_heads_client
            return Mock()
        
        mock_mqtt_client.side_effect = mock_client_factory
        
        analyser = DrillingDataAnalyser(config_file)
        
        # Simulate some clients being created
        analyser.client_manager.clients = {
            'result': mock_result_client,
            'trace': mock_trace_client,
            'heads': mock_heads_client
        }
        
        # Start processing
        analyser.processing_active = True
        analyser.processing_thread = Mock()
        analyser.processing_thread.is_alive.return_value = True
        
        # Test shutdown
        analyser.shutdown()
        
        # Verify shutdown behavior
        assert analyser.processing_active is False
        analyser.processing_thread.join.assert_called_with(timeout=2.0)
        
        # Verify all clients were stopped and disconnected
        mock_result_client.loop_stop.assert_called_once()
        mock_result_client.disconnect.assert_called_once()
        mock_trace_client.loop_stop.assert_called_once()
        mock_trace_client.disconnect.assert_called_once()
        mock_heads_client.loop_stop.assert_called_once()
        mock_heads_client.disconnect.assert_called_once()
    
    def test_context_manager(self, config_file, mock_depth_inference):
        """Test context manager functionality."""
        with patch.object(DrillingDataAnalyser, 'shutdown') as mock_shutdown:
            with DrillingDataAnalyser(config_file) as analyser:
                assert isinstance(analyser, DrillingDataAnalyser)
            
            # Verify shutdown was called on exit
            mock_shutdown.assert_called_once()
    
    def test_system_status_logging(self, config_file, mock_depth_inference):
        """Test system status logging functionality."""
        analyser = DrillingDataAnalyser(config_file)
        
        # Mock the dependencies
        analyser.message_buffer.get_buffer_stats = Mock(return_value={'total_messages': 0})
        analyser.message_correlator.get_correlation_stats = Mock(return_value={'total_unprocessed': 0})
        
        # Test status logging (should not raise exceptions)
        analyser._log_system_status()
        
        # Verify the mocked methods were called
        analyser.message_buffer.get_buffer_stats.assert_called_once()
        analyser.message_correlator.get_correlation_stats.assert_called_once()
    
    def test_invalid_message_source_format(self, config_file, mock_depth_inference):
        """Test handling of messages with invalid source format."""
        analyser = DrillingDataAnalyser(config_file)
        
        # Mock successful processing
        mock_result = Mock()
        mock_result.success = True
        mock_result.keypoints = [0.0, 5.0, 10.0]
        mock_result.depth_estimation = [5.0, 5.0]
        
        analyser.message_processor.process_matching_messages = Mock(return_value=mock_result)
        analyser.result_publisher = Mock()
        
        # Create message with invalid source format
        invalid_message = TimestampedData(
            _timestamp=time.time(),
            _data={'test': 'data'},
            _source='invalid_format'  # Missing required path parts
        )
        
        # Test processing - should handle gracefully
        analyser._process_matched_messages([invalid_message])
        
        # Verify no publishing occurred due to invalid source
        analyser.result_publisher.publish_processing_result.assert_not_called()
    
    def test_workflow_statistics_tracking(self, config_file, mock_depth_inference):
        """Test workflow statistics tracking functionality."""
        analyser = DrillingDataAnalyser(config_file)
        
        # Verify initial workflow stats
        assert analyser._workflow_stats['messages_arrived'] == 0
        assert analyser._workflow_stats['messages_processed'] == 0
        assert analyser._workflow_stats['processing_times'] == []
        assert analyser._workflow_stats['last_processed'] == 0
        
        # Test message arrival tracking by simulating MessageBuffer reference
        analyser._workflow_stats['messages_arrived'] += 1  # Simulate arrival
        
        # Mock successful processing result
        mock_result = Mock()
        mock_result.success = True
        mock_result.keypoints = [0.0, 5.0, 10.0]
        mock_result.depth_estimation = [5.0, 5.0]
        mock_result.machine_id = "TEST_MACHINE"
        mock_result.result_id = "TEST_RESULT"
        
        analyser.message_processor.process_matching_messages = Mock(return_value=mock_result)
        analyser.result_publisher = Mock()
        analyser.result_publisher.publish_processing_result = Mock(return_value=True)
        
        # Create test message
        test_message = TimestampedData(
            _timestamp=time.time(),
            _data={'test': 'data'},
            _source='drilling/data/toolbox1/tool1/Result'
        )
        
        # Process message
        analyser._process_matched_messages([test_message])
        
        # Verify workflow stats were updated
        assert analyser._workflow_stats['messages_processed'] == 1
        assert analyser._workflow_stats['last_processed'] > 0
        assert len(analyser._workflow_stats['processing_times']) == 1
        assert analyser._workflow_stats['processing_times'][0] >= 0  # Should be in milliseconds (can be 0 for very fast processing)
        
        # Test get_status includes workflow stats
        status = analyser.get_status()
        assert 'workflow' in status
        assert status['workflow']['health'] == 'HEALTHY'
        assert status['workflow']['last_minute_arrivals'] == 1
        assert status['workflow']['last_minute_processed'] == 1
        assert status['workflow']['avg_processing_ms'] >= 0


if __name__ == '__main__':
    pytest.main([__file__])