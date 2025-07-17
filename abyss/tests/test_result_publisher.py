import pytest
from unittest.mock import Mock, MagicMock, call
import json
import logging
from datetime import datetime

try:
    from abyss.mqtt.components.result_publisher import ResultPublisher, PublishResultType
    from abyss.mqtt.components.message_processor import ProcessingResult
except ImportError:
    # Fallback for development environments
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src_path))
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
            keypoints=[1.0, 2.0, 3.0],
            depth_estimation=[10.0, 20.0, 30.0],
            machine_id="TEST_MACHINE",
            result_id="TEST_RESULT",
            head_id="TEST_HEAD",
            error_message=None
        )
    
    def test_publish_successful_result(self, result_publisher, sample_processing_result, mock_mqtt_client):
        """Test publishing successful result"""
        # Test data
        toolbox_id = "toolbox1"
        tool_id = "tool1"
        dt_string = "2023-01-01T12:00:00Z"
        algo_version = "0.2.6"
        
        # Call method - should not raise any exception
        result_publisher._publish_successful_result(
            sample_processing_result, toolbox_id, tool_id, dt_string, algo_version
        )
        
        # Verify MQTT publish calls
        assert mock_mqtt_client.publish.call_count == 2
        
        # Check keypoints topic and data
        keyp_call = mock_mqtt_client.publish.call_args_list[0]
        keyp_topic = keyp_call[0][0]
        keyp_payload = json.loads(keyp_call[0][1])
        
        assert keyp_topic == "OPCPUBSUB/toolbox1/tool1/Estimation/Keypoints"
        assert keyp_payload['Value'] == [1.0, 2.0, 3.0]
        assert keyp_payload['AlgoVersion'] == "0.2.6"
        assert keyp_payload['MachineId'] == "TEST_MACHINE"
        assert keyp_payload['ResultId'] == "TEST_RESULT"
        assert keyp_payload['HeadId'] == "TEST_HEAD"
        assert keyp_payload['SourceTimestamp'] == dt_string
        
        # Check depth estimation topic and data
        dest_call = mock_mqtt_client.publish.call_args_list[1]
        dest_topic = dest_call[0][0]
        dest_payload = json.loads(dest_call[0][1])
        
        assert dest_topic == "OPCPUBSUB/toolbox1/tool1/Estimation/DepthEstimation"
        assert dest_payload['Value'] == [10.0, 20.0, 30.0]
        assert dest_payload['AlgoVersion'] == "0.2.6"
        assert dest_payload['MachineId'] == "TEST_MACHINE"
        assert dest_payload['ResultId'] == "TEST_RESULT"
        assert dest_payload['HeadId'] == "TEST_HEAD"
        assert dest_payload['SourceTimestamp'] == dt_string
    
    def test_publish_insufficient_data_result(self, result_publisher, sample_processing_result, mock_mqtt_client):
        """Test publishing insufficient data result"""
        # Test data
        toolbox_id = "toolbox2"
        tool_id = "tool2"
        dt_string = "2023-01-01T13:00:00Z"
        
        # Call method - should not raise any exception
        result_publisher._publish_insufficient_data_result(
            sample_processing_result, toolbox_id, tool_id, dt_string
        )
        
        # Verify MQTT publish calls
        assert mock_mqtt_client.publish.call_count == 2
        
        # Check keypoints topic and data
        keyp_call = mock_mqtt_client.publish.call_args_list[0]
        keyp_payload = json.loads(keyp_call[0][1])
        
        assert keyp_payload['Value'] == 'Not enough steps to estimate keypoints'
        assert 'AlgoVersion' not in keyp_payload
        assert keyp_payload['MachineId'] == "TEST_MACHINE"
        
        # Check depth estimation topic and data
        dest_call = mock_mqtt_client.publish.call_args_list[1]
        dest_payload = json.loads(dest_call[0][1])
        
        assert dest_payload['Value'] == 'Not enough steps to estimate depth'
        assert 'AlgoVersion' not in dest_payload
        assert dest_payload['MachineId'] == "TEST_MACHINE"
    
    def test_publish_error_result(self, result_publisher, sample_processing_result, mock_mqtt_client):
        """Test publishing error result"""
        # Test data
        toolbox_id = "toolbox3"
        tool_id = "tool3"
        dt_string = "2023-01-01T14:00:00Z"
        error_message = "Test error occurred"
        
        # Call method - should not raise any exception
        result_publisher._publish_error_result(
            sample_processing_result, toolbox_id, tool_id, dt_string, error_message
        )
        
        # Verify MQTT publish calls
        assert mock_mqtt_client.publish.call_count == 2
        
        # Check keypoints topic and data
        keyp_call = mock_mqtt_client.publish.call_args_list[0]
        keyp_payload = json.loads(keyp_call[0][1])
        
        assert keyp_payload['Value'] == 'Error in keypoint estimation: Test error occurred'
        assert keyp_payload['Error'] is True
        assert keyp_payload['MachineId'] == "TEST_MACHINE"
        
        # Check depth estimation topic and data
        dest_call = mock_mqtt_client.publish.call_args_list[1]
        dest_payload = json.loads(dest_call[0][1])
        
        assert dest_payload['Value'] == 'Error in depth estimation: Test error occurred'
        assert dest_payload['Error'] is True
        assert dest_payload['MachineId'] == "TEST_MACHINE"
    
    def test_consolidated_method_success_type(self, result_publisher, sample_processing_result, mock_mqtt_client):
        """Test consolidated method with SUCCESS type"""
        result_publisher._publish_result_consolidated(
            PublishResultType.SUCCESS,
            sample_processing_result,
            "tb1", "t1", "2023-01-01T15:00:00Z",
            algo_version="0.2.4"
        )
        
        # Method returns None, verify by checking MQTT calls
        assert mock_mqtt_client.publish.call_count == 2
        
        # Verify keypoints data
        keyp_payload = json.loads(mock_mqtt_client.publish.call_args_list[0][0][1])
        assert keyp_payload['Value'] == [1.0, 2.0, 3.0]
        assert keyp_payload['AlgoVersion'] == "0.2.6"
    
    def test_consolidated_method_insufficient_data_type(self, result_publisher, sample_processing_result, mock_mqtt_client):
        """Test consolidated method with INSUFFICIENT_DATA type"""
        result_publisher._publish_result_consolidated(
            PublishResultType.INSUFFICIENT_DATA,
            sample_processing_result,
            "tb2", "t2", "2023-01-01T16:00:00Z"
        )
        
        # Method returns None, verify by checking MQTT calls
        assert mock_mqtt_client.publish.call_count == 2
        
        # Verify keypoints data
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
        assert mock_mqtt_client.publish.call_count == 2
        
        # Verify timestamp conversion
        keyp_payload = json.loads(mock_mqtt_client.publish.call_args_list[0][0][1])
        assert keyp_payload['SourceTimestamp'] == "2023-01-01T12:00:00Z"
    
    def test_publish_processing_result_error_flow(self, result_publisher, mock_mqtt_client):
        """Test the main publish_processing_result method for error flow"""
        # Create failed processing result
        failed_result = ProcessingResult(
            success=False,
            keypoints=None,
            depth_estimation=None,
            machine_id="TEST_MACHINE",
            result_id="TEST_RESULT",
            head_id="TEST_HEAD",
            error_message="Processing failed"
        )
        
        timestamp = 1672574400.0
        
        # Call method - should not raise any exception

        
        result_publisher.publish_processing_result(
            failed_result, "tb", "t", timestamp, "0.2.6"
        )
        assert mock_mqtt_client.publish.call_count == 2
        
        # Verify error message in payload
        keyp_payload = json.loads(mock_mqtt_client.publish.call_args_list[0][0][1])
        assert "Error in keypoint estimation: Processing failed" in keyp_payload['Value']
        assert keyp_payload['Error'] is True
    
    def test_publish_processing_result_insufficient_data_flow(self, result_publisher, mock_mqtt_client):
        """Test the main publish_processing_result method for insufficient data flow"""
        # Create result with missing data
        insufficient_result = ProcessingResult(
            success=True,
            keypoints=None,  # Missing keypoints
            depth_estimation=None,  # Missing depth estimation
            machine_id="TEST_MACHINE",
            result_id="TEST_RESULT",
            head_id="TEST_HEAD",
            error_message=None
        )
        
        timestamp = 1672574400.0
        
        # Call method - should not raise any exception

        
        result_publisher.publish_processing_result(
            insufficient_result, "tb", "t", timestamp, "0.2.6"
        )
        assert mock_mqtt_client.publish.call_count == 2
        
        # Verify insufficient data message
        keyp_payload = json.loads(mock_mqtt_client.publish.call_args_list[0][0][1])
        assert keyp_payload['Value'] == 'Not enough steps to estimate keypoints'
        assert 'AlgoVersion' not in keyp_payload
    
    def test_build_topic(self, result_publisher):
        """Test topic building functionality"""
        topic = result_publisher._build_topic("toolbox1", "tool1", "keypoints")
        assert topic == "OPCPUBSUB/toolbox1/tool1/Estimation/Keypoints"
        
        topic = result_publisher._build_topic("toolbox2", "tool2", "depth_estimation")
        assert topic == "OPCPUBSUB/toolbox2/tool2/Estimation/DepthEstimation"
    
    def test_publish_custom_result(self, result_publisher, mock_mqtt_client):
        """Test custom result publishing"""
        # Call method - should not raise any exception

        result_publisher.publish_custom_result(
            "tb", "t", "keypoints", [1, 2, 3],
            timestamp=1672574400.0,
            additional_fields={"CustomField": "CustomValue"}
        )
        assert mock_mqtt_client.publish.call_count == 1
        
        # Verify payload
        payload = json.loads(mock_mqtt_client.publish.call_args[0][1])
        assert payload['Value'] == [1, 2, 3]
        assert payload['SourceTimestamp'] == "2023-01-01T12:00:00Z"
        assert payload['CustomField'] == "CustomValue"
    
    def test_get_publisher_stats(self, result_publisher, mock_mqtt_client):
        """Test publisher statistics"""
        # Mock is_connected method
        mock_mqtt_client.is_connected.return_value = True
        
        stats = result_publisher.get_publisher_stats()
        
        assert stats['mqtt_client_connected'] is True
        assert stats['config_loaded'] is True
        assert stats['publisher_ready'] is True
        assert 'negative_depth_stats' in stats
        assert stats['negative_depth_stats']['total_occurrences'] == 0
        assert stats['negative_depth_stats']['recent_count'] == 0


class TestResultPublisherComparison:
    """Compare consolidated vs individual methods to ensure identical behavior"""
    
    @pytest.fixture
    def mock_config(self):
        return {
            'mqtt': {
                'listener': {'root': 'OPCPUBSUB'},
                'estimation': {
                    'keypoints': 'Estimation/Keypoints',
                    'depth_estimation': 'Estimation/DepthEstimation'
                }
            }
        }
    
    @pytest.fixture
    def sample_processing_result(self):
        return ProcessingResult(
            success=True,
            keypoints=[1.0, 2.0, 3.0],
            depth_estimation=[10.0, 20.0, 30.0],
            machine_id="TEST_MACHINE",
            result_id="TEST_RESULT",
            head_id="TEST_HEAD",
            error_message=None
        )
    
    def test_success_methods_identical_output(self, mock_config, sample_processing_result):
        """Test that consolidated and individual success methods produce identical output"""
        # Setup two identical publishers
        client1 = Mock()
        client2 = Mock()
        
        # Mock successful publish
        publish_result = Mock()
        publish_result.rc = 0
        client1.publish.return_value = publish_result
        client2.publish.return_value = publish_result
        
        publisher1 = ResultPublisher(client1, mock_config)
        publisher2 = ResultPublisher(client2, mock_config)
        
        # Test parameters
        toolbox_id = "tb1"
        tool_id = "t1"
        dt_string = "2023-01-01T12:00:00Z"
        algo_version = "0.2.6"
        
        # Call both methods
        publisher1._publish_successful_result(
            sample_processing_result, toolbox_id, tool_id, dt_string, algo_version
        )
        publisher2._publish_result_consolidated(
            PublishResultType.SUCCESS, sample_processing_result,
            toolbox_id, tool_id, dt_string, algo_version=algo_version
        )
        
        # Both methods return None, verify by checking MQTT calls
        
        # Both should make same number of calls
        assert client1.publish.call_count == client2.publish.call_count == 2
        
        # Extract and compare payloads
        payload1_keyp = json.loads(client1.publish.call_args_list[0][0][1])
        payload2_keyp = json.loads(client2.publish.call_args_list[0][0][1])
        
        payload1_dest = json.loads(client1.publish.call_args_list[1][0][1])
        payload2_dest = json.loads(client2.publish.call_args_list[1][0][1])
        
        # Payloads should be identical
        assert payload1_keyp == payload2_keyp
        assert payload1_dest == payload2_dest
        
        # Verify key content
        assert payload1_keyp['Value'] == [1.0, 2.0, 3.0]
        assert payload1_keyp['AlgoVersion'] == "0.2.4"
        assert payload1_dest['Value'] == [10.0, 20.0, 30.0]
    
    def test_insufficient_data_methods_identical_output(self, mock_config, sample_processing_result):
        """Test that consolidated and individual insufficient data methods produce identical output"""
        # Setup identical publishers
        client1 = Mock()
        client2 = Mock()
        
        publish_result = Mock()
        publish_result.rc = 0
        client1.publish.return_value = publish_result
        client2.publish.return_value = publish_result
        
        publisher1 = ResultPublisher(client1, mock_config)
        publisher2 = ResultPublisher(client2, mock_config)
        
        # Test parameters
        toolbox_id = "tb2"
        tool_id = "t2"
        dt_string = "2023-01-01T13:00:00Z"
        
        # Call both methods
        publisher1._publish_insufficient_data_result(
            sample_processing_result, toolbox_id, tool_id, dt_string
        )
        publisher2._publish_result_consolidated(
            PublishResultType.INSUFFICIENT_DATA, sample_processing_result,
            toolbox_id, tool_id, dt_string
        )
        
        # Both methods return None, verify by checking MQTT calls
        assert client1.publish.call_count == client2.publish.call_count == 2
        
        # Compare payloads
        payload1_keyp = json.loads(client1.publish.call_args_list[0][0][1])
        payload2_keyp = json.loads(client2.publish.call_args_list[0][0][1])
        
        assert payload1_keyp == payload2_keyp
        assert payload1_keyp['Value'] == 'Not enough steps to estimate keypoints'
        assert 'AlgoVersion' not in payload1_keyp
    
    def test_error_methods_identical_output(self, mock_config, sample_processing_result):
        """Test that consolidated and individual error methods produce identical output"""
        # Setup identical publishers
        client1 = Mock()
        client2 = Mock()
        
        publish_result = Mock()
        publish_result.rc = 0
        client1.publish.return_value = publish_result
        client2.publish.return_value = publish_result
        
        publisher1 = ResultPublisher(client1, mock_config)
        publisher2 = ResultPublisher(client2, mock_config)
        
        # Test parameters
        toolbox_id = "tb3"
        tool_id = "t3"
        dt_string = "2023-01-01T14:00:00Z"
        error_message = "Test error occurred"
        
        # Call both methods
        publisher1._publish_error_result(
            sample_processing_result, toolbox_id, tool_id, dt_string, error_message
        )
        publisher2._publish_result_consolidated(
            PublishResultType.ERROR, sample_processing_result,
            toolbox_id, tool_id, dt_string, error_message=error_message
        )
        
        # Both methods return None, verify by checking MQTT calls
        assert client1.publish.call_count == client2.publish.call_count == 2
        
        # Compare payloads
        payload1_keyp = json.loads(client1.publish.call_args_list[0][0][1])
        payload2_keyp = json.loads(client2.publish.call_args_list[0][0][1])
        
        assert payload1_keyp == payload2_keyp
        assert payload1_keyp['Value'] == 'Error in keypoint estimation: Test error occurred'
        assert payload1_keyp['Error'] is True


class TestNegativeDepthHandling:
    """Test handling of negative depth estimation values"""
    
    @pytest.fixture
    def mock_config(self):
        return {
            'mqtt': {
                'listener': {'root': 'OPCPUBSUB'},
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
        publish_result = Mock()
        publish_result.rc = 0
        client.publish.return_value = publish_result
        return client
    
    @pytest.fixture
    def result_publisher(self, mock_mqtt_client, mock_config):
        """Create result publisher with mocked dependencies"""
        return ResultPublisher(mock_mqtt_client, mock_config)
    
    def test_negative_depth_detection_single_value(self, result_publisher, mock_mqtt_client, caplog):
        """Test detection of single negative depth value"""
        # Create result with negative depth
        result_with_negative_depth = ProcessingResult(
            success=True,
            keypoints=[1.0, 2.0, 1.5],  # Third keypoint is lower, causing negative depth
            depth_estimation=[1.0, -0.5],  # Second depth is negative
            machine_id="TEST_MACHINE",
            result_id="TEST_RESULT",
            head_id="TEST_HEAD",
            error_message=None
        )
        
        # Clear any previous logs
        caplog.clear()
        
        # Publish result
        with caplog.at_level(logging.WARNING):
            result_publisher.publish_processing_result(
                result_with_negative_depth, "tb", "t", 1672574400.0, "0.2.6"
            )
        
        # Verify no MQTT publish occurred
        assert mock_mqtt_client.publish.call_count == 0
        
        # Verify warning was logged
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
            keypoints=[5.0, 3.0, 2.0, 4.0],
            depth_estimation=[-2.0, -1.0, 2.0],  # Two negative values
            machine_id="TEST_MACHINE",
            result_id="TEST_RESULT",
            head_id="TEST_HEAD",
            error_message=None
        )
        
        caplog.clear()
        
        # Publish result
        with caplog.at_level(logging.WARNING):
            result_publisher.publish_processing_result(
                result_with_negatives, "tb", "t", 1672574400.0, "0.2.6"
            )
        
        # Verify no MQTT publish occurred
        assert mock_mqtt_client.publish.call_count == 0
        
        # Verify warning contains all negative values
        warning_logs = [r for r in caplog.records if r.levelname == 'WARNING']
        assert len(warning_logs) >= 1
        assert warning_logs[0].negative_values == [-2.0, -1.0]
    
    def test_positive_depths_published_normally(self, result_publisher, mock_mqtt_client):
        """Test that positive depths are published normally"""
        # Create result with all positive depths
        positive_result = ProcessingResult(
            success=True,
            keypoints=[1.0, 2.0, 3.0],
            depth_estimation=[1.0, 1.0],  # All positive
            machine_id="TEST_MACHINE",
            result_id="TEST_RESULT",
            head_id="TEST_HEAD",
            error_message=None
        )
        
        # Publish result
        result_publisher.publish_processing_result(
            positive_result, "tb", "t", 1672574400.0, "0.2.6"
        )
        
        # Verify MQTT publish occurred normally
        assert mock_mqtt_client.publish.call_count == 2
    
    def test_sequential_negative_depth_warning(self, result_publisher, mock_mqtt_client, caplog):
        """Test warning when 5 negative depths occur in close sequence"""
        # Create multiple results with negative depths
        negative_result = ProcessingResult(
            success=True,
            keypoints=[2.0, 1.0],
            depth_estimation=[-1.0],
            machine_id="TEST_MACHINE",
            result_id="TEST_RESULT",
            head_id="TEST_HEAD",
            error_message=None
        )
        
        caplog.clear()
        
        # Publish 5 negative results
        with caplog.at_level(logging.WARNING):
            for i in range(5):
                result_publisher.publish_processing_result(
                    negative_result, "tb", f"t{i}", 1672574400.0 + i, "0.2.6"
                )
        
        # Verify no MQTT publishes occurred
        assert mock_mqtt_client.publish.call_count == 0
        
        # Check for sequential warning
        warning_logs = [r for r in caplog.records if r.levelname == 'WARNING']
        sequential_warnings = [w for w in warning_logs 
                             if "Multiple negative depth estimations detected in close sequence" in w.message]
        assert len(sequential_warnings) >= 1
        assert sequential_warnings[0].count == 5
        assert sequential_warnings[0].threshold == 5
    
    def test_negative_depth_stats_tracking(self, result_publisher, mock_mqtt_client):
        """Test that negative depth occurrences are tracked in stats"""
        # Create result with negative depth
        negative_result = ProcessingResult(
            success=True,
            keypoints=[2.0, 1.0],
            depth_estimation=[-1.0],
            machine_id="TEST_MACHINE",
            result_id="TEST_RESULT",
            head_id="TEST_HEAD",
            error_message=None
        )
        
        # Publish several negative results
        for i in range(3):
            result_publisher.publish_processing_result(
                negative_result, "tb", f"t{i}", 1672574400.0 + i, "0.2.4"
            )
        
        # Check stats
        stats = result_publisher.get_publisher_stats()
        assert stats['negative_depth_stats']['total_occurrences'] == 3
        assert stats['negative_depth_stats']['recent_count'] == 3
    
    def test_old_negative_depths_expire_from_window(self, result_publisher, mock_mqtt_client):
        """Test that old negative depth occurrences expire from the tracking window"""
        import time
        
        # Create result with negative depth
        negative_result = ProcessingResult(
            success=True,
            keypoints=[2.0, 1.0],
            depth_estimation=[-1.0],
            machine_id="TEST_MACHINE",
            result_id="TEST_RESULT",
            head_id="TEST_HEAD",
            error_message=None
        )
        
        # Mock time to control window expiration
        original_time = time.time
        current_mock_time = [1672574400.0]
        
        def mock_time():
            return current_mock_time[0]
        
        # Temporarily replace time.time
        time.time = mock_time
        
        try:
            # Publish a negative result
            result_publisher.publish_processing_result(
                negative_result, "tb", "t1", current_mock_time[0], "0.2.6"
            )
            
            # Advance time beyond the window (5 minutes = 300 seconds)
            current_mock_time[0] += 400  # 400 seconds later
            
            # Trigger window cleanup by publishing another negative result
            result_publisher.publish_processing_result(
                negative_result, "tb", "t2", current_mock_time[0], "0.2.6"
            )
            
            # Check stats - should only show 1 recent occurrence
            stats = result_publisher.get_publisher_stats()
            assert stats['negative_depth_stats']['recent_count'] == 1
            
        finally:
            # Restore original time function
            time.time = original_time