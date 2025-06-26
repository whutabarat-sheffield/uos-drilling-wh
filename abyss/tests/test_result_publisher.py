import pytest
from unittest.mock import Mock, MagicMock, call
import json
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
        algo_version = "0.2.4"
        
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
        assert keyp_payload['AlgoVersion'] == "0.2.4"
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
        assert dest_payload['AlgoVersion'] == "0.2.4"
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
        result = result_publisher._publish_result_consolidated(
            PublishResultType.SUCCESS,
            sample_processing_result,
            "tb1", "t1", "2023-01-01T15:00:00Z",
            algo_version="0.2.4"
        )
        
        assert result is True
        assert mock_mqtt_client.publish.call_count == 2
        
        # Verify keypoints data
        keyp_payload = json.loads(mock_mqtt_client.publish.call_args_list[0][0][1])
        assert keyp_payload['Value'] == [1.0, 2.0, 3.0]
        assert keyp_payload['AlgoVersion'] == "0.2.4"
    
    def test_consolidated_method_insufficient_data_type(self, result_publisher, sample_processing_result, mock_mqtt_client):
        """Test consolidated method with INSUFFICIENT_DATA type"""
        result = result_publisher._publish_result_consolidated(
            PublishResultType.INSUFFICIENT_DATA,
            sample_processing_result,
            "tb2", "t2", "2023-01-01T16:00:00Z"
        )
        
        assert result is True
        assert mock_mqtt_client.publish.call_count == 2
        
        # Verify keypoints data
        keyp_payload = json.loads(mock_mqtt_client.publish.call_args_list[0][0][1])
        assert keyp_payload['Value'] == 'Not enough steps to estimate keypoints'
        assert 'AlgoVersion' not in keyp_payload
    
    def test_consolidated_method_error_type(self, result_publisher, sample_processing_result, mock_mqtt_client):
        """Test consolidated method with ERROR type"""
        result = result_publisher._publish_result_consolidated(
            PublishResultType.ERROR,
            sample_processing_result,
            "tb3", "t3", "2023-01-01T17:00:00Z",
            error_message="Test error"
        )
        
        assert result is True
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
        with pytest.raises(MQTTPublishError, match="Failed to publish success result"):
            result_publisher._publish_successful_result(
                sample_processing_result, "tb", "t", "2023-01-01T18:00:00Z", "0.2.4"
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
            sample_processing_result, "tb", "t", timestamp, "0.2.4"
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
            failed_result, "tb", "t", timestamp, "0.2.4"
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
            insufficient_result, "tb", "t", timestamp, "0.2.4"
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
        algo_version = "0.2.4"
        
        # Call both methods
        result1 = publisher1._publish_successful_result(
            sample_processing_result, toolbox_id, tool_id, dt_string, algo_version
        )
        result2 = publisher2._publish_result_consolidated(
            PublishResultType.SUCCESS, sample_processing_result,
            toolbox_id, tool_id, dt_string, algo_version=algo_version
        )
        
        # Both should succeed
        assert result1 == result2 == True
        
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
        result1 = publisher1._publish_insufficient_data_result(
            sample_processing_result, toolbox_id, tool_id, dt_string
        )
        result2 = publisher2._publish_result_consolidated(
            PublishResultType.INSUFFICIENT_DATA, sample_processing_result,
            toolbox_id, tool_id, dt_string
        )
        
        # Both should succeed
        assert result1 == result2 == True
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
        result1 = publisher1._publish_error_result(
            sample_processing_result, toolbox_id, tool_id, dt_string, error_message
        )
        result2 = publisher2._publish_result_consolidated(
            PublishResultType.ERROR, sample_processing_result,
            toolbox_id, tool_id, dt_string, error_message=error_message
        )
        
        # Both should succeed
        assert result1 == result2 == True
        assert client1.publish.call_count == client2.publish.call_count == 2
        
        # Compare payloads
        payload1_keyp = json.loads(client1.publish.call_args_list[0][0][1])
        payload2_keyp = json.loads(client2.publish.call_args_list[0][0][1])
        
        assert payload1_keyp == payload2_keyp
        assert payload1_keyp['Value'] == 'Error in keypoint estimation: Test error occurred'
        assert payload1_keyp['Error'] is True