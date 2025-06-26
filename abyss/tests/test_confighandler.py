import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from abyss.mqtt.async_components.confighandler import ConfigHandler, MQTTConfig, ListenerConfig, BrokerConfig, DataIdsConfig, EstimationConfig, PublisherConfig

class TestConfigHandler:
    
    @pytest.fixture
    def mock_config_handler(self):
        # Create mock objects for all required components
        broker_config = BrokerConfig(host="localhost", port=1883)
        listener_config = ListenerConfig(root="drilling", toolboxid="tb1", toolid="tool1", result="result", trace="trace")
        data_ids_config = MagicMock(spec=DataIdsConfig)
        estimation_config = MagicMock(spec=EstimationConfig)
        publisher_config = MagicMock(spec=PublisherConfig)
        
        mqtt_config = MQTTConfig(
            broker=broker_config,
            listener=listener_config,
            data_ids=data_ids_config,
            estimation=estimation_config,
            publisher=publisher_config
        )
        
        return ConfigHandler(mqtt=mqtt_config)
    
    def test_is_trace_topic_with_valid_topic(self, mock_config_handler):
        # Given a valid trace topic
        topic = "drilling/tb1/tool1/trace"
        
        # When checking if it's a trace topic
        result = mock_config_handler.is_trace_topic(topic)
        
        # Then it should return True
        assert result is True
    
    def test_is_trace_topic_with_result_topic(self, mock_config_handler):
        # Given a result topic (not a trace topic)
        topic = "drilling/tb1/tool1/result"
        
        # When checking if it's a trace topic
        result = mock_config_handler.is_trace_topic(topic)
        
        # Then it should return False
        assert result is False
    
    def test_is_trace_topic_with_too_short_topic(self, mock_config_handler):
        # Given a topic with too few parts
        topic = "drilling/tb1/trace"
        
        # When checking if it's a trace topic
        result = mock_config_handler.is_trace_topic(topic)
        
        # Then it should return False
        assert result is False
    
    def test_is_trace_topic_with_invalid_suffix(self, mock_config_handler):
        # Given a topic with invalid suffix
        topic = "drilling/tb1/tool1/something_else"
        
        # When checking if it's a trace topic
        result = mock_config_handler.is_trace_topic(topic)
        
        # Then it should return False
        assert result is False
    
    def test_is_trace_topic_with_multi_part_trace(self, mock_config_handler):
        # Given a config with multi-part trace path
        mock_config_handler.mqtt.listener.trace = "trace/data"
        
        # When checking valid multi-part topic
        valid_topic = "drilling/tb1/tool1/trace/data"
        result = mock_config_handler.is_trace_topic(valid_topic)
        
        # Then it should return True
        assert result is True
        
        # When checking invalid multi-part topic
        invalid_topic = "drilling/tb1/tool1/trace"
        result = mock_config_handler.is_trace_topic(invalid_topic)
        
        # Then it should return False
        assert result is False