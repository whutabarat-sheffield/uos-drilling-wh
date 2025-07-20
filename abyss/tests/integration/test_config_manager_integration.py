import pytest
import json
from unittest.mock import Mock, MagicMock
from abyss.mqtt.components.result_publisher import ResultPublisher
from abyss.mqtt.components.config_manager import ConfigurationManager
from abyss.mqtt.components.message_processor import ProcessingResult


class TestConfigurationManagerIntegration:
    """Test integration of components with ConfigurationManager"""
    
    @pytest.fixture
    def mock_mqtt_client(self):
        client = Mock()
        publish_result = Mock()
        publish_result.rc = 0
        client.publish.return_value = publish_result
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
    
    def test_result_publisher_with_config_manager(self, mock_mqtt_client, sample_processing_result, tmp_path):
        """Test ResultPublisher using ConfigurationManager"""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
mqtt:
  broker:
    host: "localhost"
    port: 1883
  listener:
    root: "OPCPUBSUB"
    result: "ResultManagement"
    trace: "ResultManagement/Trace"
    heads: "AssetManagement/Head"
    time_window: 5.0
    cleanup_interval: 60
    max_buffer_size: 1000
  estimation:
    keypoints: "Estimation/Keypoints"
    depth_estimation: "Estimation/DepthEstimation"
"""
        config_file.write_text(config_content)
        
        # Create ConfigurationManager
        config_manager = ConfigurationManager(str(config_file))
        
        # Create ResultPublisher with ConfigurationManager
        result_publisher = ResultPublisher(mock_mqtt_client, config_manager)
        
        # Test publishing
        result_publisher.publish_processing_result(
            sample_processing_result, "toolbox1", "tool1", 1672574400.0, "0.2.6"
        )
        assert mock_mqtt_client.publish.call_count == 2
        
        # Verify topics are built correctly using ConfigurationManager
        keyp_call = mock_mqtt_client.publish.call_args_list[0]
        dest_call = mock_mqtt_client.publish.call_args_list[1]
        
        assert keyp_call[0][0] == "OPCPUBSUB/toolbox1/tool1/Estimation/Keypoints"
        assert dest_call[0][0] == "OPCPUBSUB/toolbox1/tool1/Estimation/DepthEstimation"
        
        # Verify payload content
        keyp_payload = json.loads(keyp_call[0][1])
        dest_payload = json.loads(dest_call[0][1])
        
        assert keyp_payload['Value'] == [1.0, 2.0, 3.0]
        assert keyp_payload['HeadId'] == "TEST_HEAD_ID"
        assert keyp_payload['AlgoVersion'] == "0.2.6"
        
        assert dest_payload['Value'] == [10.0, 20.0, 30.0]
        assert dest_payload['HeadId'] == "TEST_HEAD_ID"
        assert dest_payload['AlgoVersion'] == "0.2.6"
    
    def test_result_publisher_backward_compatibility(self, mock_mqtt_client, sample_processing_result, tmp_path):
        """Test ResultPublisher works with ConfigurationManager"""
        # Create temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
mqtt:
  broker:
    host: "localhost"
    port: 1883
  listener:
    root: "OPCPUBSUB"
    result: "ResultManagement"
    trace: "ResultManagement/Trace"
  estimation:
    keypoints: "Estimation/Keypoints"
    depth_estimation: "Estimation/DepthEstimation"
"""
        config_file.write_text(config_content)
        
        # Create ResultPublisher with ConfigurationManager
        config_manager = ConfigurationManager(str(config_file))
        result_publisher = ResultPublisher(mock_mqtt_client, config_manager)
        
        # Test publishing
        result_publisher.publish_processing_result(
            sample_processing_result, "toolbox2", "tool2", 1672574400.0, "0.2.6"
        )
        assert mock_mqtt_client.publish.call_count == 2
        
        # Verify topics are built correctly using legacy method
        keyp_call = mock_mqtt_client.publish.call_args_list[0]
        dest_call = mock_mqtt_client.publish.call_args_list[1]
        
        assert keyp_call[0][0] == "OPCPUBSUB/toolbox2/tool2/Estimation/Keypoints"
        assert dest_call[0][0] == "OPCPUBSUB/toolbox2/tool2/Estimation/DepthEstimation"
    
    def test_config_manager_build_result_topic(self, tmp_path):
        """Test ConfigurationManager's build_result_topic method"""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
mqtt:
  broker:
    host: "localhost"
    port: 1883
  listener:
    root: "OPCPUBSUB"
    result: "ResultManagement"
    trace: "ResultManagement/Trace"
    heads: "AssetManagement/Head"
  estimation:
    keypoints: "Estimation/Keypoints"
    depth_estimation: "Estimation/DepthEstimation"
    custom_type: "Custom/Endpoint"
"""
        config_file.write_text(config_content)
        
        config_manager = ConfigurationManager(str(config_file))
        
        # Test building various topics
        keyp_topic = config_manager.build_result_topic("tb1", "t1", "keypoints")
        dest_topic = config_manager.build_result_topic("tb2", "t2", "depth_estimation")
        custom_topic = config_manager.build_result_topic("tb3", "t3", "custom_type")
        
        assert keyp_topic == "OPCPUBSUB/tb1/t1/Estimation/Keypoints"
        assert dest_topic == "OPCPUBSUB/tb2/t2/Estimation/DepthEstimation"
        assert custom_topic == "OPCPUBSUB/tb3/t3/Custom/Endpoint"
    
    def test_config_manager_missing_estimation_endpoint(self, tmp_path):
        """Test ConfigurationManager error handling for missing estimation endpoint"""
        config_file = tmp_path / "test_config.yaml"
        config_content = """
mqtt:
  broker:
    host: "localhost"
    port: 1883
  listener:
    root: "OPCPUBSUB"
    result: "ResultManagement"
    trace: "ResultManagement/Trace"
    heads: "AssetManagement/Head"
  estimation:
    keypoints: "Estimation/Keypoints"
    # depth_estimation missing
"""
        config_file.write_text(config_content)
        
        config_manager = ConfigurationManager(str(config_file))
        
        # Should work for keypoints
        keyp_topic = config_manager.build_result_topic("tb1", "t1", "keypoints")
        assert keyp_topic == "OPCPUBSUB/tb1/t1/Estimation/Keypoints"
        
        # Should raise error for missing endpoint
        with pytest.raises(Exception, match="estimation endpoint"):
            config_manager.build_result_topic("tb1", "t1", "depth_estimation")
    
    def test_config_manager_missing_root(self, tmp_path):
        """Test ConfigurationManager validation for missing root"""
        config_file = tmp_path / "test_config.yaml"
        config_content = """
mqtt:
  broker:
    host: "localhost"
    port: 1883
  listener:
    # root missing
    result: "ResultManagement"
    trace: "ResultManagement/Trace"
    heads: "AssetManagement/Head"
  estimation:
    keypoints: "Estimation/Keypoints"
"""
        config_file.write_text(config_content)
        
        # Should raise error during ConfigurationManager initialization
        with pytest.raises(Exception, match="Missing required listener field: root"):
            ConfigurationManager(str(config_file))
    
    def test_simple_correlator_with_config_manager(self, tmp_path):
        """Test SimpleMessageCorrelator using ConfigurationManager"""
        from abyss.mqtt.components.simple_correlator import SimpleMessageCorrelator
        
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
mqtt:
  broker:
    host: "localhost"
    port: 1883
  listener:
    root: "OPCPUBSUB"
    result: "ResultManagement"
    trace: "ResultManagement/Trace"
    heads: "AssetManagement/Head"
    time_window: 10.0
    cleanup_interval: 60
    max_buffer_size: 1000
  estimation:
    keypoints: "Estimation/Keypoints"
    depth_estimation: "Estimation/DepthEstimation"
"""
        config_file.write_text(config_content)
        
        # Create ConfigurationManager
        config_manager = ConfigurationManager(str(config_file))
        
        # Create SimpleMessageCorrelator with ConfigurationManager
        correlator = SimpleMessageCorrelator(config_manager)
        
        # Verify it uses ConfigurationManager settings
        assert correlator.config_manager is config_manager
        assert correlator.time_window == 10.0  # From config
        
        # Verify topic patterns are set correctly
        expected_patterns = {
            'result': 'OPCPUBSUB/+/+/ResultManagement',
            'trace': 'OPCPUBSUB/+/+/ResultManagement/Trace',
            'heads': 'OPCPUBSUB/+/+/AssetManagement/Head'
        }
        assert correlator._topic_patterns == expected_patterns
    
    def test_simple_correlator_backward_compatibility(self, tmp_path):
        """Test SimpleMessageCorrelator works with ConfigurationManager"""
        from abyss.mqtt.components.simple_correlator import SimpleMessageCorrelator
        
        # Create temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
mqtt:
  broker:
    host: "localhost"
    port: 1883
  listener:
    root: "OPCPUBSUB"
    result: "ResultManagement"
    trace: "ResultManagement/Trace"
    heads: "AssetManagement/Head"
    time_window: 15.0
  estimation:
    keypoints: "Estimation/Keypoints"
    depth_estimation: "Estimation/DepthEstimation"
"""
        config_file.write_text(config_content)
        
        # Create SimpleMessageCorrelator with ConfigurationManager
        config_manager = ConfigurationManager(str(config_file))
        correlator = SimpleMessageCorrelator(config_manager, time_window=15.0)
        
        # Verify it uses ConfigurationManager
        assert correlator.config_manager is not None
        assert correlator.time_window == 15.0
        
        # Verify topic patterns are set correctly using legacy method
        expected_patterns = {
            'result': 'OPCPUBSUB/+/+/ResultManagement',
            'trace': 'OPCPUBSUB/+/+/ResultManagement/Trace',
            'heads': 'OPCPUBSUB/+/+/AssetManagement/Head'
        }
        assert correlator._topic_patterns == expected_patterns