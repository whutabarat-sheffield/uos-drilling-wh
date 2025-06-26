import pytest
import json
import time
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
from abyss.run.uos_publish_json import find_in_dict, publish
from abyss.mqtt.components.config_manager import ConfigurationManager


class TestFindInDict:
    """Simple tests for the find_in_dict function"""
    
    def test_find_key_in_simple_dict(self):
        """Test finding a key in a simple dictionary"""
        data = {"key1": "value1", "key2": "value2"}
        result = find_in_dict(data, "key1")
        assert result == ["value1"]
    
    def test_find_key_in_nested_dict(self):
        """Test finding a key in nested dictionaries"""
        data = {
            "level1": {
                "level2": {
                    "SourceTimestamp": "2023-01-01T00:00:00Z"
                }
            }
        }
        result = find_in_dict(data, "SourceTimestamp")
        assert result == ["2023-01-01T00:00:00Z"]
    
    def test_find_multiple_occurrences(self):
        """Test finding multiple occurrences of the same key"""
        data = {
            "section1": {"SourceTimestamp": "2023-01-01T00:00:00Z"},
            "section2": {"SourceTimestamp": "2023-01-01T00:00:00Z"}
        }
        result = find_in_dict(data, "SourceTimestamp")
        assert len(result) == 2
        assert "2023-01-01T00:00:00Z" in result


class TestPublish:
    """Tests for the publish function"""
    
    def test_publish_with_timestamp_replacement(self):
        """Test publishing with timestamp replacement"""
        mock_client = MagicMock()
        topic = "test/topic"
        payload = '{"timestamp": "2023-01-01T00:00:00Z", "data": "test"}'
        old_timestamp = "2023-01-01T00:00:00Z"
        new_timestamp = "2023-01-01T10:00:00Z"
        
        with patch('time.localtime') as mock_localtime, \
             patch('time.strftime') as mock_strftime, \
             patch('builtins.print'):
            
            mock_localtime.return_value = time.struct_time((2023, 1, 1, 10, 0, 0, 0, 1, 0))
            mock_strftime.return_value = "10:00:00"
            
            publish(mock_client, topic, payload, old_timestamp, new_timestamp)
            
            expected_payload = '{"timestamp": "2023-01-01T10:00:00Z", "data": "test"}'
            mock_client.publish.assert_called_once_with(topic, expected_payload)


class TestUosPublishJsonIntegration:
    """Integration tests for uos_publish_json functionality"""
    
    @pytest.fixture
    def test_data_directory(self):
        """Use the real test data directory"""
        data_path = "/home/windo/github/uos-drilling-wh/abyss/src/abyss/test_data/data_20250326"
        if not Path(data_path).exists():
            pytest.skip(f"Test data directory not found: {data_path}")
        return data_path
    
    @pytest.fixture
    def config_file_path(self):
        """Use the local config file"""
        config_path = "/home/windo/github/uos-drilling-wh/abyss/src/abyss/run/config/mqtt_conf_local.yaml"
        if not Path(config_path).exists():
            pytest.skip(f"Config file not found: {config_path}")
        return config_path
    
    def test_config_manager_loads_local_config(self, config_file_path):
        """Test that ConfigurationManager can load the local config"""
        config_manager = ConfigurationManager(config_file_path)
        
        # Verify basic configuration is loaded
        broker_config = config_manager.get_mqtt_broker_config()
        listener_config = config_manager.get_mqtt_listener_config()
        
        assert broker_config['host'] == 'localhost'
        assert broker_config['port'] == 1883
        assert listener_config['root'] == 'OPCPUBSUB'
        assert 'ResultManagement' in listener_config['result']
    
    def test_find_json_files_in_directory(self, test_data_directory):
        """Test that we can find all JSON files in the test directory"""
        data_path = Path(test_data_directory)
        json_files = list(data_path.glob("*.json"))
        
        # Should find the expected JSON files
        expected_files = ['ResultManagement.json', 'Trace.json', 'Heads.json']
        found_file_names = [f.name for f in json_files]
        
        for expected_file in expected_files:
            assert expected_file in found_file_names
    
    def test_find_in_dict_with_real_data(self, test_data_directory):
        """Test find_in_dict with real JSON data"""
        result_file = Path(test_data_directory) / "ResultManagement.json"
        
        with open(result_file) as f:
            data = json.load(f)
        
        # Test finding SourceTimestamp values
        timestamps = find_in_dict(data, 'SourceTimestamp')
        assert len(timestamps) > 0
        
        # All timestamps should be the same in this test file
        unique_timestamps = set(timestamps)
        assert len(unique_timestamps) == 1
    
    @patch('abyss.run.uos_publish_json.mqtt.Client')
    def test_mqtt_publishing_workflow(self, mock_mqtt_client, test_data_directory, config_file_path):
        """Test the MQTT publishing workflow with real data and config"""
        # Setup mock MQTT client
        mock_client_instance = MagicMock()
        mock_mqtt_client.return_value = mock_client_instance
        
        # Load real configuration
        config_manager = ConfigurationManager(config_file_path)
        broker_config = config_manager.get_mqtt_broker_config()
        listener_config = config_manager.get_mqtt_listener_config()
        
        # Find JSON files in directory
        data_path = Path(test_data_directory)
        json_files = list(data_path.glob("*.json"))
        
        # Simulate the publishing process
        with patch('time.strftime', return_value="2023-01-01T10:00:00Z"), \
             patch('builtins.print'):
            
            # Create MQTT client and connect (mocked)
            client = mock_mqtt_client()
            client.connect(broker_config['host'], broker_config['port'])
            
            # For each JSON file, read and publish
            for json_file in json_files:
                with open(json_file) as f:
                    file_content = f.read()
                    data = json.loads(file_content)
                
                # Find timestamps and replace them
                timestamps = find_in_dict(data, 'SourceTimestamp')
                if timestamps:
                    original_timestamp = timestamps[0]
                    new_timestamp = "2023-01-01T10:00:00Z"
                    
                    # Create topic based on file type
                    if 'ResultManagement' in json_file.name:
                        topic_suffix = listener_config['result']
                    elif 'Trace' in json_file.name:
                        topic_suffix = listener_config['trace']
                    elif 'Heads' in json_file.name:
                        topic_suffix = listener_config['heads']
                    else:
                        continue
                    
                    topic = f"{listener_config['root']}/test_toolbox/test_tool/{topic_suffix}"
                    
                    # Publish the message
                    publish(client, topic, file_content, original_timestamp, new_timestamp)
        
        # Verify MQTT client was used
        mock_mqtt_client.assert_called()
        mock_client_instance.connect.assert_called_with('localhost', 1883)
        
        # Should have published at least 3 messages (ResultManagement, Trace, Heads)
        assert mock_client_instance.publish.call_count >= 3
    
    def test_topic_construction(self, config_file_path):
        """Test proper MQTT topic construction from config"""
        config_manager = ConfigurationManager(config_file_path)
        listener_config = config_manager.get_mqtt_listener_config()
        
        toolbox_id = "ILLL502033771"
        tool_id = "setitec001"
        
        # Test different topic types
        result_topic = f"{listener_config['root']}/{toolbox_id}/{tool_id}/{listener_config['result']}"
        trace_topic = f"{listener_config['root']}/{toolbox_id}/{tool_id}/{listener_config['trace']}"
        heads_topic = f"{listener_config['root']}/{toolbox_id}/{tool_id}/{listener_config['heads']}"
        
        assert result_topic == "OPCPUBSUB/ILLL502033771/setitec001/ResultManagement"
        assert trace_topic == "OPCPUBSUB/ILLL502033771/setitec001/ResultManagement/Trace"
        assert heads_topic == "OPCPUBSUB/ILLL502033771/setitec001/AssetManagement/Head"


class TestFileDiscovery:
    """Tests for JSON file discovery functionality"""
    
    def test_discover_json_files_in_temp_directory(self):
        """Test discovering JSON files in a temporary directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test JSON files
            test_files = ['test1.json', 'test2.json', 'data.json']
            for filename in test_files:
                (temp_path / filename).write_text('{"test": "data"}')
            
            # Create non-JSON files (should be ignored)
            (temp_path / 'readme.txt').write_text('not json')
            (temp_path / 'data.xml').write_text('<xml></xml>')
            
            # Discover JSON files
            json_files = list(temp_path.glob("*.json"))
            json_filenames = [f.name for f in json_files]
            
            # Should find all JSON files and only JSON files
            assert len(json_files) == 3
            for test_file in test_files:
                assert test_file in json_filenames