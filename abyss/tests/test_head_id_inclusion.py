import pytest
import json
from unittest.mock import Mock
from abyss.mqtt.components.result_publisher import ResultPublisher, PublishResultType
from abyss.mqtt.components.message_processor import ProcessingResult


class TestHeadIdInclusion:
    """Verify that head_id is included in every published message"""
    
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
        client = Mock()
        publish_result = Mock()
        publish_result.rc = 0
        client.publish.return_value = publish_result
        return client
    
    @pytest.fixture
    def result_publisher(self, mock_mqtt_client, mock_config):
        return ResultPublisher(mock_mqtt_client, mock_config)
    
    @pytest.fixture
    def sample_processing_result(self):
        return ProcessingResult(
            success=True,
            keypoints=[1.0, 2.0, 3.0],
            depth_estimation=[10.0, 20.0, 30.0],
            machine_id="TEST_MACHINE",
            result_id="TEST_RESULT",
            head_id="TEST_HEAD_ID_123",
            error_message=None
        )
    
    def verify_head_id_in_payload(self, mock_client, expected_head_id="TEST_HEAD_ID_123"):
        """Helper to verify head_id is in all published payloads"""
        assert mock_client.publish.call_count >= 1
        
        for call in mock_client.publish.call_args_list:
            payload_json = call[0][1]
            payload = json.loads(payload_json)
            
            assert 'HeadId' in payload, f"HeadId missing from payload: {payload}"
            assert payload['HeadId'] == expected_head_id, f"HeadId mismatch. Expected: {expected_head_id}, Got: {payload['HeadId']}"
    
    def test_head_id_in_successful_result(self, result_publisher, sample_processing_result, mock_mqtt_client):
        """Test head_id is included in successful result messages"""
        result_publisher._publish_successful_result(
            sample_processing_result, "tb1", "t1", "2023-01-01T12:00:00Z", "0.2.6"
        )
        
        self.verify_head_id_in_payload(mock_mqtt_client)
    
    def test_head_id_in_insufficient_data_result(self, result_publisher, sample_processing_result, mock_mqtt_client):
        """Test head_id is included in insufficient data result messages"""
        result_publisher._publish_insufficient_data_result(
            sample_processing_result, "tb2", "t2", "2023-01-01T13:00:00Z"
        )
        
        self.verify_head_id_in_payload(mock_mqtt_client)
    
    def test_head_id_in_error_result(self, result_publisher, sample_processing_result, mock_mqtt_client):
        """Test head_id is included in error result messages"""
        result_publisher._publish_error_result(
            sample_processing_result, "tb3", "t3", "2023-01-01T14:00:00Z", "Test error"
        )
        
        self.verify_head_id_in_payload(mock_mqtt_client)
    
    def test_head_id_in_consolidated_success(self, result_publisher, sample_processing_result, mock_mqtt_client):
        """Test head_id is included in consolidated method - SUCCESS type"""
        result_publisher._publish_result_consolidated(
            PublishResultType.SUCCESS, sample_processing_result,
            "tb4", "t4", "2023-01-01T15:00:00Z", algo_version="0.2.6"
        )
        
        self.verify_head_id_in_payload(mock_mqtt_client)
    
    def test_head_id_in_consolidated_insufficient_data(self, result_publisher, sample_processing_result, mock_mqtt_client):
        """Test head_id is included in consolidated method - INSUFFICIENT_DATA type"""
        result_publisher._publish_result_consolidated(
            PublishResultType.INSUFFICIENT_DATA, sample_processing_result,
            "tb5", "t5", "2023-01-01T16:00:00Z"
        )
        
        self.verify_head_id_in_payload(mock_mqtt_client)
    
    def test_head_id_in_consolidated_error(self, result_publisher, sample_processing_result, mock_mqtt_client):
        """Test head_id is included in consolidated method - ERROR type"""
        result_publisher._publish_result_consolidated(
            PublishResultType.ERROR, sample_processing_result,
            "tb6", "t6", "2023-01-01T17:00:00Z", error_message="Test error"
        )
        
        self.verify_head_id_in_payload(mock_mqtt_client)
    
    def test_head_id_in_main_publish_processing_result_success(self, result_publisher, sample_processing_result, mock_mqtt_client):
        """Test head_id is included in main publish_processing_result method - success flow"""
        timestamp = 1672574400.0
        
        result_publisher.publish_processing_result(
            sample_processing_result, "tb7", "t7", timestamp, "0.2.6"
        )
        
        self.verify_head_id_in_payload(mock_mqtt_client)
    
    def test_head_id_in_main_publish_processing_result_error(self, result_publisher, mock_mqtt_client):
        """Test head_id is included in main publish_processing_result method - error flow"""
        failed_result = ProcessingResult(
            success=False,
            keypoints=None,
            depth_estimation=None,
            machine_id="TEST_MACHINE",
            result_id="TEST_RESULT",
            head_id="ERROR_HEAD_ID_456",
            error_message="Processing failed"
        )
        
        timestamp = 1672574400.0
        
        result_publisher.publish_processing_result(
            failed_result, "tb8", "t8", timestamp, "0.2.6"
        )
        
        self.verify_head_id_in_payload(mock_mqtt_client, "ERROR_HEAD_ID_456")
    
    def test_head_id_in_main_publish_processing_result_insufficient(self, result_publisher, mock_mqtt_client):
        """Test head_id is included in main publish_processing_result method - insufficient data flow"""
        insufficient_result = ProcessingResult(
            success=True,
            keypoints=None,
            depth_estimation=None,
            machine_id="TEST_MACHINE",
            result_id="TEST_RESULT",
            head_id="INSUFFICIENT_HEAD_ID_789",
            error_message=None
        )
        
        timestamp = 1672574400.0
        
        result_publisher.publish_processing_result(
            insufficient_result, "tb9", "t9", timestamp, "0.2.6"
        )
        
        self.verify_head_id_in_payload(mock_mqtt_client, "INSUFFICIENT_HEAD_ID_789")
    
    def test_custom_result_does_not_include_head_id_by_default(self, result_publisher, mock_mqtt_client):
        """Test that publish_custom_result does NOT include head_id by default"""
        result_publisher.publish_custom_result(
            "tb10", "t10", "keypoints", [1, 2, 3], timestamp=1672574400.0
        )
        
        assert mock_mqtt_client.publish.call_count == 1
        payload = json.loads(mock_mqtt_client.publish.call_args[0][1])
        
        # Should NOT have HeadId by default
        assert 'HeadId' not in payload
        assert payload['Value'] == [1, 2, 3]
        assert payload['SourceTimestamp'] == "2023-01-01T12:00:00Z"
    
    def test_custom_result_can_include_head_id_via_additional_fields(self, result_publisher, mock_mqtt_client):
        """Test that publish_custom_result can include head_id via additional_fields"""
        result_publisher.publish_custom_result(
            "tb11", "t11", "keypoints", [1, 2, 3], 
            timestamp=1672574400.0,
            additional_fields={"HeadId": "CUSTOM_HEAD_ID_999"}
        )
        
        assert mock_mqtt_client.publish.call_count == 1
        payload = json.loads(mock_mqtt_client.publish.call_args[0][1])
        
        # Should have HeadId from additional_fields
        assert 'HeadId' in payload
        assert payload['HeadId'] == "CUSTOM_HEAD_ID_999"
        assert payload['Value'] == [1, 2, 3]
    
    def test_all_processing_result_methods_include_head_id(self, result_publisher, sample_processing_result, mock_mqtt_client):
        """Comprehensive test: verify head_id is in all messages from processing result methods"""
        # Test all three individual methods
        result_publisher._publish_successful_result(
            sample_processing_result, "tb_s", "t_s", "2023-01-01T12:00:00Z", "0.2.6"
        )
        
        result_publisher._publish_insufficient_data_result(
            sample_processing_result, "tb_i", "t_i", "2023-01-01T13:00:00Z"
        )
        
        result_publisher._publish_error_result(
            sample_processing_result, "tb_e", "t_e", "2023-01-01T14:00:00Z", "error"
        )
        
        # Should have 6 calls total (2 per method: keypoints + depth_estimation)
        assert mock_mqtt_client.publish.call_count == 6
        
        # Verify ALL payloads contain head_id
        for call in mock_mqtt_client.publish.call_args_list:
            payload = json.loads(call[0][1])
            assert 'HeadId' in payload, f"HeadId missing from payload: {payload}"
            assert payload['HeadId'] == "TEST_HEAD_ID_123"