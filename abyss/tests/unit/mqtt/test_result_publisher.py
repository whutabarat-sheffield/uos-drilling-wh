import pytest
from unittest.mock import Mock, MagicMock, call
import json
import logging
from datetime import datetime

from abyss.mqtt.components.result_publisher import ResultPublisher, PublishResultType
from abyss.mqtt.components.message_processor import ProcessingResult


class TestResultPublisher:
    """Test the consolidated result publisher"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        return {
            'mqtt': {
                'listener': {
                    'root': 'OPCPUBSUB'
                },
                'estimation': {
                    'keypoints': 'Estimation/Keypoints',
                    'depth_estimation': 'Estimation/DepthEstimation'
                }
            }
        }
    
    @pytest.fixture
    def mock_mqtt_client(self):
        """Mock MQTT client"""
        client = Mock()
        # Mock successful publish response
        publish_result = Mock()
        publish_result.rc = 0  # MQTT_ERR_SUCCESS
        client.publish.return_value = publish_result
        return client
    
    @pytest.fixture
    def result_publisher(self, mock_mqtt_client, mock_config):
        """Create result publisher with mocked dependencies"""
        return ResultPublisher(mock_mqtt_client, mock_config)
    
    @pytest.fixture
    def sample_processing_result(self):
        """Sample processing result for testing"""
        return ProcessingResult(
            success=True,
            keypoints=[1.0, 2.0, 3.0, 4.0, 5.0],
            depth_estimation=[1.0, 1.0, 1.0, 1.0],
            machine_id="TEST_MACHINE",
            result_id="TEST_RESULT",
            head_id="TEST_HEAD_ID",
            error_message=None
        )
    
    def test_publish_successful_result(self, result_publisher, sample_processing_result, mock_mqtt_client):
        """Test publishing successful results"""
        result_publisher._publish_successful_result(
            sample_processing_result, "toolbox1", "tool1", "2023-01-01T12:00:00Z", "0.2.6"
        )
        
        # Should publish to both keypoints and depth estimation topics
        assert mock_mqtt_client.publish.call_count == 2
        
        # Verify topics
        keyp_call = mock_mqtt_client.publish.call_args_list[0]
        dest_call = mock_mqtt_client.publish.call_args_list[1]
        
        assert keyp_call[0][0] == "OPCPUBSUB/toolbox1/tool1/Estimation/Keypoints"
        assert dest_call[0][0] == "OPCPUBSUB/toolbox1/tool1/Estimation/DepthEstimation"
        
        # Verify payload content
        keyp_payload = json.loads(keyp_call[0][1])
        dest_payload = json.loads(dest_call[0][1])
        
        assert keyp_payload['Value'] == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert keyp_payload['SourceTimestamp'] == "2023-01-01T12:00:00Z"
        assert keyp_payload['AlgoVersion'] == "0.2.6"
        assert keyp_payload['HeadId'] == "TEST_HEAD_ID"
        
        assert dest_payload['Value'] == [1.0, 1.0, 1.0, 1.0]
        assert dest_payload['HeadId'] == "TEST_HEAD_ID"
    
    def test_publish_no_steps_result(self, result_publisher, sample_processing_result, mock_mqtt_client):
        """Test publishing when there are no steps (steps < 3)"""
        # Modify result to have no keypoints (steps < 3)
        no_steps_result = ProcessingResult(
            success=True,
            keypoints=[],  # Empty means steps < 3
            depth_estimation=[],
            machine_id="TEST_MACHINE",
            result_id="TEST_RESULT",
            head_id="TEST_HEAD_ID",
            error_message=None
        )
        
        result_publisher._publish_no_steps_result(
            no_steps_result, "toolbox2", "tool2", "2023-01-01T13:00:00Z"
        )
        
        # Should only publish to keypoints topic with special message
        assert mock_mqtt_client.publish.call_count == 1
        
        keyp_call = mock_mqtt_client.publish.call_args_list[0]
        assert keyp_call[0][0] == "OPCPUBSUB/toolbox2/tool2/Estimation/Keypoints"
        
        keyp_payload = json.loads(keyp_call[0][1])
        assert keyp_payload['Value'] == 'Not enough steps to estimate keypoints'
        assert keyp_payload['SourceTimestamp'] == "2023-01-01T13:00:00Z"
        assert keyp_payload['HeadId'] == "TEST_HEAD_ID"
        assert 'AlgoVersion' not in keyp_payload
    
    def test_publish_error_result(self, result_publisher, sample_processing_result, mock_mqtt_client):
        """Test publishing error results"""
        # Create an error result
        error_result = ProcessingResult(
            success=False,
            keypoints=[],
            depth_estimation=[],
            machine_id="TEST_MACHINE", 
            result_id="TEST_RESULT",
            head_id="TEST_HEAD_ID",
            error_message="Test error message"
        )
        
        result_publisher._publish_error_result(
            error_result, "toolbox3", "tool3", "2023-01-01T14:00:00Z", 
            error_message="Custom error message"
        )
        
        # Should publish to both topics with error messages
        assert mock_mqtt_client.publish.call_count == 2
        
        keyp_payload = json.loads(mock_mqtt_client.publish.call_args_list[0][0][1])
        dest_payload = json.loads(mock_mqtt_client.publish.call_args_list[1][0][1])
        
        assert keyp_payload['Value'] == 'Error in keypoint estimation: Custom error message'
        assert keyp_payload['Error'] is True
        assert dest_payload['Value'] == 'Error in depth estimation: Custom error message'
        assert dest_payload['Error'] is True
    
    def test_publish_processing_result_main_method(self, result_publisher, sample_processing_result, mock_mqtt_client):
        """Test the main publish_processing_result method"""
        timestamp = 1672574400.0  # 2023-01-01 12:00:00 UTC
        
        result_publisher.publish_processing_result(
            sample_processing_result, "toolbox1", "tool1", timestamp, "0.2.6"
        )
        
        # Should publish successfully
        assert mock_mqtt_client.publish.call_count == 2
        
        # Verify timestamp formatting
        keyp_call = mock_mqtt_client.publish.call_args_list[0]
        keyp_payload = json.loads(keyp_call[0][1])
        assert keyp_payload['SourceTimestamp'] == "2023-01-01T12:00:00Z"
    
    def test_topic_construction_with_config_manager(self, mock_mqtt_client):
        """Test topic construction when using ConfigurationManager"""
        from abyss.mqtt.components.config_manager import ConfigurationManager
        import tempfile
        import yaml
        
        # Create config with custom paths
        config_data = {
            'mqtt': {
                'listener': {'root': 'CUSTOM/ROOT'},
                'estimation': {
                    'keypoints': 'Custom/KP',
                    'depth_estimation': 'Custom/DE'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config_manager = ConfigurationManager(config_path)
            publisher = ResultPublisher(mock_mqtt_client, config_manager)
            
            # Create dummy result
            result = ProcessingResult(
                success=True,
                keypoints=[1, 2, 3, 4, 5],
                depth_estimation=[1, 1, 1, 1],
                machine_id="M1",
                result_id="R1",
                head_id="H1",
                error_message=None
            )
            
            publisher.publish_processing_result(result, "tb1", "t1", 1234567890.0, "1.0")
            
            # Verify custom topics were used
            keyp_call = mock_mqtt_client.publish.call_args_list[0]
            dest_call = mock_mqtt_client.publish.call_args_list[1]
            
            assert keyp_call[0][0] == "CUSTOM/ROOT/tb1/t1/Custom/KP"
            assert dest_call[0][0] == "CUSTOM/ROOT/tb1/t1/Custom/DE"
        finally:
            import os
            os.unlink(config_path)
    
    def test_consolidated_method_success_type(self, result_publisher, sample_processing_result, mock_mqtt_client):
        """Test the consolidated publish method with SUCCESS type"""
        result_publisher._publish_result_consolidated(
            PublishResultType.SUCCESS,
            sample_processing_result,
            "tb1", "t1", "2023-01-01T15:00:00Z",
            algo_version="0.2.6"
        )
        
        # Should behave same as _publish_successful_result
        assert mock_mqtt_client.publish.call_count == 2
        keyp_payload = json.loads(mock_mqtt_client.publish.call_args_list[0][0][1])
        assert keyp_payload['AlgoVersion'] == "0.2.6"
    
    def test_consolidated_method_no_steps_type(self, result_publisher, sample_processing_result, mock_mqtt_client):
        """Test consolidated method with NO_STEPS type"""
        result_publisher._publish_result_consolidated(
            PublishResultType.NO_STEPS,
            sample_processing_result,
            "tb2", "t2", "2023-01-01T16:00:00Z"
        )
        
        # Should only publish keypoints with special message
        assert mock_mqtt_client.publish.call_count == 1
        keyp_payload = json.loads(mock_mqtt_client.publish.call_args_list[0][0][1])
        assert keyp_payload['Value'] == 'Not enough steps to estimate keypoints'
        assert 'AlgoVersion' not in keyp_payload
    
    def test_consolidated_method_error_type(self, result_publisher, sample_processing_result, mock_mqtt_client):
        """Test consolidated method with ERROR type"""
        result_publisher._publish_result_consolidated(
            PublishResultType.ERROR,
            sample_processing_result,
            "tb3", "t3", "2023-01-01T17:00:00Z",
            error_message="Test error"
        )
        
        # Method returns None, verify by checking MQTT calls
        assert mock_mqtt_client.publish.call_count == 2
        
        # Verify keypoints data
        keyp_payload = json.loads(mock_mqtt_client.publish.call_args_list[0][0][1])
        assert keyp_payload['Value'] == 'Error in keypoint estimation: Test error'
        assert keyp_payload['Error'] is True
    
    def test_mqtt_publish_failure(self, result_publisher, sample_processing_result, mock_mqtt_client):
        """Test handling of MQTT publish failures"""
        from abyss.mqtt.components.exceptions import MQTTPublishError
        
        # Mock failed publish
        publish_result = Mock()
        publish_result.rc = 1  # MQTT_ERR_INVAL
        mock_mqtt_client.publish.return_value = publish_result
        
        # Should raise MQTTPublishError due to publish failure
        with pytest.raises(MQTTPublishError, match="MQTT publish failed for topic"):
            result_publisher._publish_successful_result(
                sample_processing_result, "tb", "t", "2023-01-01T18:00:00Z", "0.2.6"
            )
    
    def test_unknown_result_type_error(self, result_publisher, sample_processing_result):
        """Test error handling for unknown result type"""
        with pytest.raises(ValueError, match="Unknown result type"):
            result_publisher._publish_result_consolidated(
                "INVALID_TYPE",  # Invalid type
                sample_processing_result,
                "tb", "t", "2023-01-01T19:00:00Z"
            )
    
    def test_publish_processing_result_success_flow(self, result_publisher, sample_processing_result, mock_mqtt_client):
        """Test the main publish_processing_result method for success flow"""
        timestamp = 1672574400.0  # 2023-01-01 12:00:00 UTC
        
        # Call method - should not raise any exception
        result_publisher.publish_processing_result(
            sample_processing_result, "tb", "t", timestamp, "0.2.6"
        )
        
        # Verify 2 publishes (keypoints + depth estimation)
        assert mock_mqtt_client.publish.call_count == 2
    
    def test_publish_processing_result_no_steps_flow(self, result_publisher, mock_mqtt_client):
        """Test main method with no steps result"""
        no_steps_result = ProcessingResult(
            success=True,
            keypoints=[],  # Empty = no steps
            depth_estimation=[],
            machine_id="M1",
            result_id="R1", 
            head_id="H1",
            error_message=None
        )
        
        result_publisher.publish_processing_result(
            no_steps_result, "tb", "t", 1672574400.0, "0.2.6"
        )
        
        # Should only publish 1 message (keypoints only)
        assert mock_mqtt_client.publish.call_count == 1
    
    def test_publish_processing_result_error_flow(self, result_publisher, mock_mqtt_client):
        """Test main method with error result"""
        error_result = ProcessingResult(
            success=False,
            keypoints=[],
            depth_estimation=[],
            machine_id="M1",
            result_id="R1",
            head_id="H1",
            error_message="Processing failed"
        )
        
        result_publisher.publish_processing_result(
            error_result, "tb", "t", 1672574400.0, "0.2.6"
        )
        
        # Should publish 2 error messages
        assert mock_mqtt_client.publish.call_count == 2
        
        # Both should contain error flag
        keyp_payload = json.loads(mock_mqtt_client.publish.call_args_list[0][0][1])
        dest_payload = json.loads(mock_mqtt_client.publish.call_args_list[1][0][1])
        assert keyp_payload.get('Error') is True
        assert dest_payload.get('Error') is True
    
    def test_missing_config_values(self):
        """Test error handling when config is missing required values"""
        incomplete_config = {
            'mqtt': {
                'listener': {
                    'root': 'ROOT'
                },
                'estimation': {
                    # Missing keypoints and depth_estimation
                }
            }
        }
        
        mock_client = Mock()
        with pytest.raises(KeyError):
            ResultPublisher(mock_client, incomplete_config)
    
    def test_timestamp_formatting(self, result_publisher, sample_processing_result, mock_mqtt_client):
        """Test various timestamp formats are handled correctly"""
        # Unix timestamp
        result_publisher.publish_processing_result(
            sample_processing_result, "tb", "t", 1672574400.0, "0.2.6"
        )
        
        keyp_payload = json.loads(mock_mqtt_client.publish.call_args_list[0][0][1])
        assert keyp_payload['SourceTimestamp'] == "2023-01-01T12:00:00Z"
        
        # Clear previous calls
        mock_mqtt_client.reset_mock()
        
        # Different timestamp
        result_publisher.publish_processing_result(
            sample_processing_result, "tb", "t", 1609459200.0, "0.2.6"  # 2021-01-01
        )
        
        keyp_payload = json.loads(mock_mqtt_client.publish.call_args_list[0][0][1])
        assert keyp_payload['SourceTimestamp'] == "2021-01-01T00:00:00Z"
    
    def test_get_stats(self, result_publisher):
        """Test getting publisher statistics"""
        stats = result_publisher.get_stats()
        
        assert isinstance(stats, dict)
        assert 'messages_published' in stats
        assert 'publish_errors' in stats
        assert 'negative_depth_stats' in stats
        assert stats['negative_depth_stats']['total_occurrences'] == 0
        assert stats['negative_depth_stats']['recent_count'] == 0
    
    def test_thread_safety_concurrent_publishes(self, result_publisher, sample_processing_result, mock_mqtt_client):
        """Test thread safety with concurrent publish operations"""
        import threading
        import time
        
        publish_count = 10
        threads = []
        
        def publish_task(index):
            result_publisher.publish_processing_result(
                sample_processing_result,
                f"tb{index}",
                f"t{index}", 
                1672574400.0 + index,
                "0.2.6"
            )
        
        # Start multiple threads
        for i in range(publish_count):
            thread = threading.Thread(target=publish_task, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Each publish creates 2 MQTT calls
        assert mock_mqtt_client.publish.call_count == publish_count * 2
        
        # Verify stats are consistent
        stats = result_publisher.get_stats()
        assert stats['messages_published'] == publish_count * 2


class TestResultPublisherEdgeCases:
    """Test edge cases and error scenarios"""
    
    @pytest.fixture
    def mock_mqtt_client(self):
        client = Mock()
        publish_result = Mock()
        publish_result.rc = 0
        client.publish.return_value = publish_result
        return client
    
    @pytest.fixture
    def mock_config(self):
        return {
            'mqtt': {
                'listener': {'root': 'ROOT'},
                'estimation': {
                    'keypoints': 'KP',
                    'depth_estimation': 'DE'
                }
            }
        }
    
    def test_empty_keypoints_but_success_true(self, mock_mqtt_client, mock_config):
        """Test handling when success=True but keypoints are empty"""
        publisher = ResultPublisher(mock_mqtt_client, mock_config)
        
        result = ProcessingResult(
            success=True,
            keypoints=[],  # Empty despite success=True
            depth_estimation=[],
            machine_id="M1",
            result_id="R1",
            head_id="H1",
            error_message=None
        )
        
        publisher.publish_processing_result(result, "tb", "t", 1234567890.0, "1.0")
        
        # Should treat as NO_STEPS case
        assert mock_mqtt_client.publish.call_count == 1
        payload = json.loads(mock_mqtt_client.publish.call_args[0][1])
        assert payload['Value'] == 'Not enough steps to estimate keypoints'
    
    def test_none_values_in_arrays(self, mock_mqtt_client, mock_config):
        """Test handling of None values in keypoints/depth arrays"""
        publisher = ResultPublisher(mock_mqtt_client, mock_config)
        
        result = ProcessingResult(
            success=True,
            keypoints=[1.0, None, 3.0, None, 5.0],  # Contains None
            depth_estimation=[1.0, None, None, 1.0],
            machine_id="M1", 
            result_id="R1",
            head_id="H1",
            error_message=None
        )
        
        # Should handle gracefully - JSON serialization will convert None to null
        publisher.publish_processing_result(result, "tb", "t", 1234567890.0, "1.0")
        
        assert mock_mqtt_client.publish.call_count == 2
        keyp_payload = json.loads(mock_mqtt_client.publish.call_args_list[0][0][1])
        # JSON null values preserved
        assert keyp_payload['Value'] == [1.0, None, 3.0, None, 5.0]
    
    def test_very_long_arrays(self, mock_mqtt_client, mock_config):
        """Test publishing very long keypoint/depth arrays"""
        publisher = ResultPublisher(mock_mqtt_client, mock_config)
        
        # Create arrays with 1000 elements
        long_keypoints = list(range(1000))
        long_depths = list(range(999))
        
        result = ProcessingResult(
            success=True,
            keypoints=long_keypoints,
            depth_estimation=long_depths,
            machine_id="M1",
            result_id="R1",
            head_id="H1",
            error_message=None
        )
        
        publisher.publish_processing_result(result, "tb", "t", 1234567890.0, "1.0")
        
        # Should publish successfully
        assert mock_mqtt_client.publish.call_count == 2
        
        # Verify full arrays were published
        keyp_payload = json.loads(mock_mqtt_client.publish.call_args_list[0][0][1])
        assert len(keyp_payload['Value']) == 1000
    
    def test_special_characters_in_identifiers(self, mock_mqtt_client, mock_config):
        """Test handling of special characters in toolbox/tool IDs"""
        publisher = ResultPublisher(mock_mqtt_client, mock_config)
        
        result = ProcessingResult(
            success=True,
            keypoints=[1, 2, 3, 4, 5],
            depth_estimation=[1, 1, 1, 1],
            machine_id="M-1/2",  # Special chars
            result_id="R@1",
            head_id="H#1",
            error_message=None
        )
        
        # Should handle special chars in topic construction
        publisher.publish_processing_result(
            result, "tb-1/2", "t@3", 1234567890.0, "1.0"
        )
        
        # Verify topics contain the special chars
        keyp_call = mock_mqtt_client.publish.call_args_list[0]
        assert "tb-1/2" in keyp_call[0][0]
        assert "t@3" in keyp_call[0][0]


class TestNegativeDepthHandling:
    """Test handling of negative depth estimation values"""
    
    @pytest.fixture
    def mock_mqtt_client(self):
        client = Mock()
        publish_result = Mock()
        publish_result.rc = 0
        client.publish.return_value = publish_result
        return client
    
    @pytest.fixture
    def mock_config(self):
        return {
            'mqtt': {
                'listener': {'root': 'ROOT'},
                'estimation': {
                    'keypoints': 'KP',
                    'depth_estimation': 'DE'
                }
            }
        }
    
    @pytest.fixture
    def result_publisher(self, mock_mqtt_client, mock_config):
        return ResultPublisher(mock_mqtt_client, mock_config)
    
    def test_negative_depth_detection_single_value(self, result_publisher, mock_mqtt_client, caplog):
        """Test detection of single negative depth value"""
        # Create result with negative depth
        result_with_negative_depth = ProcessingResult(
            success=True,
            keypoints=[1.0, 2.0, 1.5],  # Third keypoint is lower, causing negative depth
            depth_estimation=[1.0, -0.5],  # Second depth is negative
            machine_id="M1",
            result_id="R1",
            head_id="H1",
            error_message=None
        )
        
        # Set log level to capture warnings
        with caplog.at_level(logging.WARNING):
            result_publisher.publish_processing_result(
                result_with_negative_depth, "tb", "t", 1672574400.0, "0.2.6"
            )
        
        # Current implementation skips publish when negative depths detected
        assert mock_mqtt_client.publish.call_count == 0
        
        # Check for warning log
        warning_logs = [r for r in caplog.records if r.levelname == 'WARNING']
        assert len(warning_logs) >= 1
        assert "Negative depth estimation detected" in warning_logs[0].message
        assert warning_logs[0].negative_values == [-0.5]
        assert warning_logs[0].all_depths == [1.0, -0.5]
    
    def test_negative_depth_detection_multiple_values(self, result_publisher, mock_mqtt_client, caplog):
        """Test detection of multiple negative depth values"""
        # Create result with multiple negative depths
        result_with_negatives = ProcessingResult(
            success=True,
            keypoints=[1.0, 0.5, 0.0, 2.0],
            depth_estimation=[-2.0, -1.0, 2.0],  # Two negative values
            machine_id="M1",
            result_id="R1", 
            head_id="H1",
            error_message=None
        )
        
        with caplog.at_level(logging.WARNING):
            result_publisher.publish_processing_result(
                result_with_negatives, "tb", "t", 1672574400.0, "0.2.6"
            )
        
        # Should skip publish
        assert mock_mqtt_client.publish.call_count == 0
        
        # Check warning contains all negative values
        warning_logs = [r for r in caplog.records if r.levelname == 'WARNING']
        assert len(warning_logs) >= 1
        assert warning_logs[0].negative_values == [-2.0, -1.0]
    
    def test_all_positive_depths_no_warning(self, result_publisher, mock_mqtt_client, caplog):
        """Test that no warning is logged for all positive depths"""
        result_all_positive = ProcessingResult(
            success=True,
            keypoints=[1.0, 2.0, 3.0, 4.0],
            depth_estimation=[1.0, 1.0, 1.0],  # All positive
            machine_id="M1",
            result_id="R1",
            head_id="H1",
            error_message=None
        )
        
        with caplog.at_level(logging.WARNING):
            result_publisher.publish_processing_result(
                result_all_positive, "tb", "t", 1672574400.0, "0.2.6"
            )
        
        # No warnings should be logged
        warning_logs = [r for r in caplog.records 
                       if r.levelname == 'WARNING' and 'negative' in r.message.lower()]
        assert len(warning_logs) == 0
    
    def test_sequential_negative_depth_warning(self, result_publisher, mock_mqtt_client, caplog):
        """Test warning when 5 negative depths occur in close sequence"""
        # Create multiple results with negative depths
        negative_result = ProcessingResult(
            success=True,
            keypoints=[2.0, 1.0],  # Decreasing = negative depth
            depth_estimation=[-1.0],
            machine_id="M1",
            result_id="R1",
            head_id="H1",
            error_message=None
        )
        
        with caplog.at_level(logging.WARNING):
            # Publish 5 negative results
            for i in range(5):
                result_publisher.publish_processing_result(
                    negative_result, "tb", f"t{i}", 1672574400.0 + i, "0.2.6"
                )
        
        # Should have additional warning about sequential negatives
        sequential_warnings = [w for w in caplog.records 
                             if "Multiple negative depth estimations detected in close sequence" in w.message]
        assert len(sequential_warnings) == 1
        assert sequential_warnings[0].count == 5
        assert sequential_warnings[0].time_window_seconds == 60
    
    def test_negative_depth_stats_tracking(self, result_publisher, mock_mqtt_client):
        """Test that negative depth occurrences are tracked in stats"""
        # Create result with negative depth
        negative_result = ProcessingResult(
            success=True,
            keypoints=[3.0, 2.0],
            depth_estimation=[-1.0],
            machine_id="M1",
            result_id="R1",
            head_id="H1",
            error_message=None
        )
        
        # Publish several negative results
        for i in range(3):
            result_publisher.publish_processing_result(
                negative_result, "tb", f"t{i}", 1672574400.0 + i, "0.2.4"
            )
        
        stats = result_publisher.get_stats()
        assert stats['negative_depth_stats']['total_occurrences'] == 3
        assert stats['negative_depth_stats']['recent_count'] == 3
    
    def test_old_negative_depths_expire_from_window(self, result_publisher, mock_mqtt_client):
        """Test that old negative depth occurrences expire from the tracking window"""
        import time
        
        # Create result with negative depth
        negative_result = ProcessingResult(
            success=True,
            keypoints=[5.0, 3.0],
            depth_estimation=[-1.0],
            machine_id="M1",
            result_id="R1", 
            head_id="H1",
            error_message=None
        )
        
        # Mock time to control window expiration
        original_time = time.time
        current_mock_time = [1672574400.0]
        
        def mock_time():
            return current_mock_time[0]
        
        time.time = mock_time
        
        try:
            # Publish a negative result
            result_publisher.publish_processing_result(
                negative_result, "tb", "t1", current_mock_time[0], "0.2.6"
            )
            
            # Advance time by 65 seconds (past 60 second window)
            current_mock_time[0] += 65
            
            # Trigger window cleanup by publishing another negative result
            result_publisher.publish_processing_result(
                negative_result, "tb", "t2", current_mock_time[0], "0.2.6"
            )
            
            stats = result_publisher.get_stats()
            # Should only count the recent one, not the old one
            assert stats['negative_depth_stats']['recent_count'] == 1
        finally:
            time.time = original_time