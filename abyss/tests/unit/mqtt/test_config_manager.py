"""
Test suite for ConfigurationManager class.
"""

import pytest
import tempfile
import os
import yaml

from abyss.mqtt.components.config_manager import ConfigurationManager, ConfigurationError


class TestConfigurationManager:
    """Test cases for ConfigurationManager class."""
    
    # The sample_mqtt_config fixture is now provided by conftest.py
    # We can override it here if we need specific test data
    
    # The temp_config_file fixture is now provided by conftest.py
    # It automatically handles cleanup
    
    # The config_manager fixture is now provided by conftest.py
    
    def test_initialization_success(self, temp_config_file):
        """Test successful ConfigurationManager initialization."""
        config_manager = ConfigurationManager(temp_config_file)
        
        assert config_manager.config is not None
        assert config_manager.config_path == temp_config_file
        assert 'mqtt' in config_manager.config
    
    def test_initialization_file_not_found(self):
        """Test initialization with non-existent file."""
        with pytest.raises(ConfigurationError) as exc_info:
            ConfigurationManager('/nonexistent/config.yaml')
        
        assert 'Configuration file not found' in str(exc_info.value)
    
    def test_initialization_invalid_yaml(self):
        """Test initialization with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('invalid: yaml: content: [')
            invalid_config_path = f.name
        
        try:
            with pytest.raises(ConfigurationError) as exc_info:
                ConfigurationManager(invalid_config_path)
            
            assert 'Invalid YAML' in str(exc_info.value)
        finally:
            os.unlink(invalid_config_path)
    
    def test_validation_missing_mqtt_section(self):
        """Test validation with missing mqtt section."""
        config_dict = {'other_section': {}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            config_path = f.name
        
        try:
            with pytest.raises(ConfigurationError) as exc_info:
                ConfigurationManager(config_path)
            
            assert 'Missing required configuration sections' in str(exc_info.value)
        finally:
            os.unlink(config_path)
    
    def test_get_simple_value(self, config_manager):
        """Test getting simple configuration values."""
        assert config_manager.get('mqtt.broker.host') == 'localhost'
        assert config_manager.get('mqtt.broker.port') == 1883
        assert config_manager.get('mqtt.listener.root') == 'drilling/data'
    
    def test_get_with_default(self, config_manager):
        """Test getting configuration values with defaults."""
        assert config_manager.get('nonexistent.key', 'default_value') == 'default_value'
        assert config_manager.get('mqtt.broker.nonexistent', 9999) == 9999
    
    def test_get_nested_dict(self, config_manager):
        """Test getting nested dictionary values."""
        broker_config = config_manager.get('mqtt.broker')
        assert isinstance(broker_config, dict)
        assert broker_config['host'] == 'localhost'
        assert broker_config['port'] == 1883
    
    def test_get_mqtt_configs(self, config_manager):
        """Test specialized MQTT configuration getters."""
        broker_config = config_manager.get_mqtt_broker_config()
        assert broker_config['host'] == 'localhost'
        assert broker_config['port'] == 1883
        
        listener_config = config_manager.get_mqtt_listener_config()
        assert listener_config['root'] == 'drilling/data'
        assert listener_config['result'] == 'Result'
        
        estimation_config = config_manager.get_mqtt_estimation_config()
        assert estimation_config['keypoints'] == 'Keypoints'
    
    def test_get_default_values(self, config_manager):
        """Test getting default values for optional settings."""
        # These should return defaults from the sample config
        assert config_manager.get_time_window() == 30.0
        assert config_manager.get_cleanup_interval() == 60
        assert config_manager.get_max_buffer_size() == 10000
    
    def test_has_authentication(self, config_manager):
        """Test authentication detection."""
        assert config_manager.has_authentication() is True
        
        # Test with config without authentication
        config_dict = {
            'mqtt': {
                'broker': {'host': 'localhost', 'port': 1883},
                'listener': {'root': 'test', 'result': 'Result', 'trace': 'Trace'}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            config_path = f.name
        
        try:
            config_manager_no_auth = ConfigurationManager(config_path)
            assert config_manager_no_auth.has_authentication() is False
        finally:
            os.unlink(config_path)
    
    def test_get_topic_patterns(self, config_manager):
        """Test topic pattern generation."""
        patterns = config_manager.get_topic_patterns()
        
        assert 'result' in patterns
        assert 'trace' in patterns
        assert 'heads' in patterns
        
        assert patterns['result'] == 'drilling/data/+/+/Result'
        assert patterns['trace'] == 'drilling/data/+/+/Trace'
        assert patterns['heads'] == 'drilling/data/+/+/Heads'
    
    def test_validate_tool_identifiers(self, config_manager):
        """Test tool identifier validation."""
        # Valid identifiers
        assert config_manager.validate_tool_identifiers('toolbox1', 'tool1') is True
        assert config_manager.validate_tool_identifiers('TB_001', 'TOOL_A') is True
        
        # Invalid identifiers
        assert config_manager.validate_tool_identifiers('', 'tool1') is False
        assert config_manager.validate_tool_identifiers('toolbox1', '') is False
        assert config_manager.validate_tool_identifiers('tool+box', 'tool1') is False
        assert config_manager.validate_tool_identifiers('toolbox1', 'tool#1') is False
        assert config_manager.validate_tool_identifiers('tool/box', 'tool1') is False
    
    def test_get_config_summary(self, config_manager):
        """Test configuration summary generation."""
        summary = config_manager.get_config_summary()
        
        assert 'config_file' in summary
        assert 'broker_host' in summary
        assert 'broker_port' in summary
        assert 'has_authentication' in summary
        assert 'listener_root' in summary
        assert 'time_window' in summary
        
        assert summary['broker_host'] == 'localhost'
        assert summary['broker_port'] == 1883
        assert summary['has_authentication'] is True
        assert summary['listener_root'] == 'drilling/data'
    
    def test_reload_configuration(self, config_manager, sample_config_dict):
        """Test configuration reloading."""
        original_host = config_manager.get('mqtt.broker.host')
        assert original_host == 'localhost'
        
        # Modify the config file
        modified_config = sample_config_dict.copy()
        modified_config['mqtt']['broker']['host'] = 'modified_host'
        
        with open(config_manager.config_path, 'w') as f:
            yaml.dump(modified_config, f)
        
        # Reload and verify change
        config_manager.reload_configuration()
        assert config_manager.get('mqtt.broker.host') == 'modified_host'
    
    def test_get_raw_config(self, config_manager, sample_config_dict):
        """Test getting raw configuration."""
        raw_config = config_manager.get_raw_config()
        
        assert raw_config == sample_config_dict
        # Ensure it's a copy, not the original
        raw_config['test'] = 'modified'
        assert 'test' not in config_manager.config


if __name__ == '__main__':
    pytest.main([__file__])