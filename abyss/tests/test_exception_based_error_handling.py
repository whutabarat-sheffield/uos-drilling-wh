import pytest
import json
from unittest.mock import Mock
from abyss.mqtt.components.result_publisher import ResultPublisher
from abyss.mqtt.components.message_processor import ProcessingResult
from abyss.mqtt.components.exceptions import (
    MQTTPublishError,
    AbyssProcessingError
)


class TestExceptionBasedErrorHandling:
    """Test exception-based error handling in MQTT components"""
    
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
    def mock_mqtt_client_success(self):
        client = Mock()
        publish_result = Mock()
        publish_result.rc = 0  # MQTT_ERR_SUCCESS
        client.publish.return_value = publish_result
        return client
    
    @pytest.fixture
    def mock_mqtt_client_failure(self):
        client = Mock()
        publish_result = Mock()
        publish_result.rc = 1  # MQTT_ERR_INVAL
        client.publish.return_value = publish_result
        return client
    
    @pytest.fixture
    def mock_mqtt_client_exception(self):
        client = Mock()
        client.publish.side_effect = ConnectionError("Network error")
        return client
    
    @pytest.fixture
    def sample_processing_result(self):
        return ProcessingResult(
            success=True,
            keypoints=[1.0, 2.0, 3.0],
            depth_estimation=[10.0, 20.0, 30.0],
            machine_id="TEST_MACHINE",
            result_id="TEST_RESULT",
            head_id="TEST_HEAD_ID",
            error_message=None
        )
    
    def test_successful_publish_no_exception(self, mock_mqtt_client_success, mock_config, sample_processing_result):
        """Test that successful publishing doesn't raise exceptions"""
        result_publisher = ResultPublisher(mock_mqtt_client_success, mock_config)
        
        # Should not raise any exception
        result_publisher.publish_processing_result(
            sample_processing_result, "tb1", "t1", 1672574400.0, "0.2.4"
        )
        
        # Verify MQTT publish was called
        assert mock_mqtt_client_success.publish.call_count == 2
    
    def test_mqtt_publish_error_raises_exception(self, mock_mqtt_client_failure, mock_config, sample_processing_result):
        """Test that MQTT publish failures raise AbyssProcessingError with chained MQTTPublishError"""
        result_publisher = ResultPublisher(mock_mqtt_client_failure, mock_config)
        
        with pytest.raises(AbyssProcessingError, match="Failed to publish processing result"):
            result_publisher.publish_processing_result(
                sample_processing_result, "tb1", "t1", 1672574400.0, "0.2.4"
            )
    
    def test_mqtt_connection_error_raises_exception(self, mock_mqtt_client_exception, mock_config, sample_processing_result):
        """Test that MQTT connection errors are wrapped in AbyssProcessingError"""
        result_publisher = ResultPublisher(mock_mqtt_client_exception, mock_config)
        
        with pytest.raises(AbyssProcessingError, match="Failed to publish processing result"):
            result_publisher.publish_processing_result(
                sample_processing_result, "tb1", "t1", 1672574400.0, "0.2.4"
            )
    
    def test_invalid_timestamp_raises_exception(self, mock_mqtt_client_success, mock_config, sample_processing_result):
        """Test that invalid timestamps raise AbyssProcessingError"""
        result_publisher = ResultPublisher(mock_mqtt_client_success, mock_config)
        
        with pytest.raises(AbyssProcessingError, match="Failed to publish processing result"):
            result_publisher.publish_processing_result(
                sample_processing_result, "tb1", "t1", "invalid_timestamp", "0.2.4"
            )
    
    def test_custom_result_publish_error_raises_exception(self, mock_mqtt_client_failure, mock_config):
        """Test that custom result publish failures raise MQTTPublishError"""
        result_publisher = ResultPublisher(mock_mqtt_client_failure, mock_config)
        
        with pytest.raises(MQTTPublishError, match="Failed to publish custom result"):
            result_publisher.publish_custom_result("tb1", "t1", "keypoints", [1, 2, 3])
    
    def test_custom_result_success_no_exception(self, mock_mqtt_client_success, mock_config):
        """Test that successful custom result publishing doesn't raise exceptions"""
        result_publisher = ResultPublisher(mock_mqtt_client_success, mock_config)
        
        # Should not raise any exception
        result_publisher.publish_custom_result(
            "tb1", "t1", "keypoints", [1, 2, 3], 
            timestamp=1672574400.0,
            additional_fields={"HeadId": "TEST_HEAD"}
        )
        
        # Verify MQTT publish was called once
        assert mock_mqtt_client_success.publish.call_count == 1
        
        # Verify payload content
        payload = json.loads(mock_mqtt_client_success.publish.call_args[0][1])
        assert payload['Value'] == [1, 2, 3]
        assert payload['HeadId'] == "TEST_HEAD"
    
    def test_error_result_publishing_success(self, mock_mqtt_client_success, mock_config):
        """Test that error result publishing works without exceptions"""
        result_publisher = ResultPublisher(mock_mqtt_client_success, mock_config)
        
        failed_result = ProcessingResult(
            success=False,
            keypoints=None,
            depth_estimation=None,
            machine_id="TEST_MACHINE",
            result_id="TEST_RESULT",
            head_id="TEST_HEAD_ID",
            error_message="Processing failed"
        )
        
        # Should not raise any exception
        result_publisher.publish_processing_result(
            failed_result, "tb1", "t1", 1672574400.0, "0.2.4"
        )
        
        # Verify MQTT publish was called twice (keypoints + depth)
        assert mock_mqtt_client_success.publish.call_count == 2
        
        # Verify error messages in payloads
        keyp_payload = json.loads(mock_mqtt_client_success.publish.call_args_list[0][0][1])
        dest_payload = json.loads(mock_mqtt_client_success.publish.call_args_list[1][0][1])
        
        assert "Error in keypoint estimation" in keyp_payload['Value']
        assert "Error in depth estimation" in dest_payload['Value']
        assert keyp_payload['Error'] is True
        assert dest_payload['Error'] is True
    
    def test_insufficient_data_result_publishing_success(self, mock_mqtt_client_success, mock_config):
        """Test that insufficient data result publishing works without exceptions"""
        result_publisher = ResultPublisher(mock_mqtt_client_success, mock_config)
        
        insufficient_result = ProcessingResult(
            success=True,
            keypoints=None,  # Missing data
            depth_estimation=None,  # Missing data
            machine_id="TEST_MACHINE",
            result_id="TEST_RESULT",
            head_id="TEST_HEAD_ID",
            error_message=None
        )
        
        # Should not raise any exception
        result_publisher.publish_processing_result(
            insufficient_result, "tb1", "t1", 1672574400.0, "0.2.4"
        )
        
        # Verify MQTT publish was called twice
        assert mock_mqtt_client_success.publish.call_count == 2
        
        # Verify insufficient data messages in payloads
        keyp_payload = json.loads(mock_mqtt_client_success.publish.call_args_list[0][0][1])
        dest_payload = json.loads(mock_mqtt_client_success.publish.call_args_list[1][0][1])
        
        assert keyp_payload['Value'] == 'Not enough steps to estimate keypoints'
        assert dest_payload['Value'] == 'Not enough steps to estimate depth'
        assert 'AlgoVersion' not in keyp_payload
        assert 'AlgoVersion' not in dest_payload
    
    def test_partial_publish_failure_raises_exception(self, mock_config, sample_processing_result):
        """Test that partial publish failures (first succeeds, second fails) raise exceptions"""
        # Create a mock client that succeeds on first call, fails on second
        client = Mock()
        success_result = Mock()
        success_result.rc = 0
        failure_result = Mock()
        failure_result.rc = 1
        
        client.publish.side_effect = [success_result, failure_result]
        
        result_publisher = ResultPublisher(client, mock_config)
        
        with pytest.raises(AbyssProcessingError, match="Failed to publish processing result"):
            result_publisher.publish_processing_result(
                sample_processing_result, "tb1", "t1", 1672574400.0, "0.2.4"
            )
        
        # Verify both publish calls were attempted
        assert client.publish.call_count == 2
    
    def test_exception_chaining_preserves_original_error(self, mock_mqtt_client_exception, mock_config, sample_processing_result):
        """Test that exception chaining preserves the original error information"""
        result_publisher = ResultPublisher(mock_mqtt_client_exception, mock_config)
        
        with pytest.raises(AbyssProcessingError) as exc_info:
            result_publisher.publish_processing_result(
                sample_processing_result, "tb1", "t1", 1672574400.0, "0.2.4"
            )
        
        # Verify exception chaining preserved original error through the chain
        assert exc_info.value.__cause__ is not None
        # The cause should be an MQTTPublishError which has ConnectionError as its cause
        mqtt_error = exc_info.value.__cause__
        assert isinstance(mqtt_error, MQTTPublishError)
        assert mqtt_error.__cause__ is not None
        assert isinstance(mqtt_error.__cause__, ConnectionError)
        assert "Network error" in str(mqtt_error.__cause__)
        
        # Verify wrapped error message includes context
        assert "Failed to publish processing result" in str(exc_info.value)


class TestExceptionUtilities:
    """Test the exception utility functions"""
    
    def test_wrap_exception_preserves_original(self):
        from abyss.mqtt.components.exceptions import wrap_exception, MQTTPublishError
        
        original_error = ValueError("Original error message")
        wrapped = wrap_exception(original_error, MQTTPublishError, "Custom message")
        
        assert isinstance(wrapped, MQTTPublishError)
        assert "Custom message" in str(wrapped)
        assert "ValueError" in str(wrapped)
        assert "Original error message" in str(wrapped)
        assert wrapped.__cause__ is original_error
    
    def test_is_recoverable_error(self):
        from abyss.mqtt.components.exceptions import (
            is_recoverable_error, 
            MQTTConnectionError, 
            MessageValidationError,
            AbyssConfigurationError,
            BufferOverflowError
        )
        
        # Communication errors should be recoverable
        assert is_recoverable_error(MQTTConnectionError("Connection failed"))
        
        # Some processing errors should be recoverable
        assert is_recoverable_error(MessageValidationError("Invalid message"))
        
        # Configuration errors should not be recoverable
        assert not is_recoverable_error(AbyssConfigurationError("Config missing"))
        
        # System errors should not be recoverable
        assert not is_recoverable_error(BufferOverflowError("Buffer full"))
        
        # Unknown errors should not be recoverable by default
        assert not is_recoverable_error(RuntimeError("Unknown error"))
    
    def test_get_error_severity(self):
        from abyss.mqtt.components.exceptions import (
            get_error_severity,
            BufferOverflowError,
            AbyssConfigurationError,
            MQTTConnectionError,
            MessageValidationError
        )
        
        # System errors should be critical
        assert get_error_severity(BufferOverflowError("Buffer full")) == 'critical'
        
        # Configuration errors should be error level
        assert get_error_severity(AbyssConfigurationError("Config missing")) == 'error'
        
        # Communication errors should be warning level
        assert get_error_severity(MQTTConnectionError("Connection failed")) == 'warning'
        
        # Processing errors should be warning level
        assert get_error_severity(MessageValidationError("Invalid message")) == 'warning'
        
        # Unknown errors should default to error level
        assert get_error_severity(RuntimeError("Unknown error")) == 'error'