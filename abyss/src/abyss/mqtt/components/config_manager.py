"""
Configuration Manager Module

Handles loading, validation, and access to configuration settings.
"""

import logging
import yaml
import os
from typing import Dict, Any, Optional, List
from pathlib import Path


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass


class ConfigurationManager:
    """
    Manages configuration loading, validation, and access.
    
    Responsibilities:
    - Load configuration from YAML files
    - Validate configuration structure and values
    - Provide typed access to configuration values
    - Handle configuration file errors gracefully
    """
    
    def __init__(self, config_path: str):
        """
        Initialize ConfigurationManager.
        
        Args:
            config_path: Path to the configuration file
            
        Raises:
            ConfigurationError: If configuration cannot be loaded or is invalid
        """
        self.config_path = config_path
        self.config = None
        self._load_configuration()
        self._validate_configuration()
    
    def _load_configuration(self):
        """Load configuration from YAML file."""
        try:
            if not os.path.exists(self.config_path):
                raise ConfigurationError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            
            if self.config is None:
                raise ConfigurationError("Configuration file is empty or invalid")
            
            logging.info("Configuration loaded successfully", extra={
                'config_path': self.config_path,
                'file_size': os.path.getsize(self.config_path)
            })
            
        except yaml.YAMLError as e:
            logging.error("YAML parsing error", extra={
                'config_path': self.config_path,
                'error_message': str(e)
            })
            raise ConfigurationError(f"Invalid YAML in configuration file: {e}") from e
        except Exception as e:
            logging.error("Error loading configuration", extra={
                'config_path': self.config_path,
                'error_type': type(e).__name__,
                'error_message': str(e)
            })
            raise ConfigurationError(f"Failed to load configuration: {e}") from e
    
    def _validate_configuration(self):
        """Validate configuration structure and required fields."""
        try:
            required_sections = ['mqtt']
            missing_sections = []
            
            for section in required_sections:
                if section not in self.config:
                    missing_sections.append(section)
            
            if missing_sections:
                raise ConfigurationError(f"Missing required configuration sections: {missing_sections}")
            
            # Validate MQTT configuration
            self._validate_mqtt_config()
            
            logging.info("Configuration validation completed successfully")
            
        except ConfigurationError:
            raise
        except Exception as e:
            logging.error("Configuration validation error", extra={
                'error_type': type(e).__name__,
                'error_message': str(e)
            })
            raise ConfigurationError(f"Configuration validation failed: {e}") from e
    
    def _validate_mqtt_config(self):
        """Validate MQTT-specific configuration."""
        mqtt_config = self.config.get('mqtt', {})
        
        # Validate broker configuration
        if 'broker' not in mqtt_config:
            raise ConfigurationError("Missing 'broker' section in MQTT configuration")
        
        broker_config = mqtt_config['broker']
        required_broker_fields = ['host', 'port']
        
        for field in required_broker_fields:
            if field not in broker_config:
                raise ConfigurationError(f"Missing required broker field: {field}")
        
        # Validate listener configuration
        if 'listener' not in mqtt_config:
            raise ConfigurationError("Missing 'listener' section in MQTT configuration")
        
        listener_config = mqtt_config['listener']
        required_listener_fields = ['root', 'result', 'trace']
        
        for field in required_listener_fields:
            if field not in listener_config:
                raise ConfigurationError(f"Missing required listener field: {field}")
        
        # Validate estimation configuration
        if 'estimation' in mqtt_config:
            estimation_config = mqtt_config['estimation']
            required_estimation_fields = ['keypoints', 'depth_estimation']
            
            for field in required_estimation_fields:
                if field not in estimation_config:
                    logging.warning(f"Missing estimation field: {field}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value (e.g., 'mqtt.broker.host')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        try:
            keys = key_path.split('.')
            value = self.config
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            
            return value
            
        except Exception as e:
            logging.warning("Error accessing configuration key", extra={
                'key_path': key_path,
                'error_message': str(e)
            })
            return default
    
    def get_mqtt_broker_config(self) -> Dict[str, Any]:
        """Get MQTT broker configuration."""
        return self.get('mqtt.broker', {})
    
    def get_mqtt_listener_config(self) -> Dict[str, Any]:
        """Get MQTT listener configuration."""
        return self.get('mqtt.listener', {})
    
    def get_mqtt_estimation_config(self) -> Dict[str, Any]:
        """Get MQTT estimation configuration."""
        return self.get('mqtt.estimation', {})
    
    def get_time_window(self) -> float:
        """Get message correlation time window."""
        return self.get('mqtt.listener.time_window', 30.0)
    
    def get_cleanup_interval(self) -> int:
        """Get buffer cleanup interval."""
        return self.get('mqtt.listener.cleanup_interval', 60)
    
    def get_max_buffer_size(self) -> int:
        """Get maximum buffer size."""
        return self.get('mqtt.listener.max_buffer_size', 10000)
    
    def has_authentication(self) -> bool:
        """Check if MQTT authentication is configured."""
        broker_config = self.get_mqtt_broker_config()
        return bool(broker_config.get('username') and broker_config.get('password'))
    
    def get_topic_patterns(self) -> Dict[str, str]:
        """Get pre-built topic patterns for message types."""
        listener_config = self.get_mqtt_listener_config()
        root = listener_config.get('root', '')
        
        return {
            'result': f"{root}/+/+/{listener_config.get('result', '')}",
            'trace': f"{root}/+/+/{listener_config.get('trace', '')}",
            'heads': f"{root}/+/+/{listener_config.get('heads', '')}"
        }
    
    def build_result_topic(self, toolbox_id: str, tool_id: str, result_type: str) -> str:
        """
        Build MQTT topic for result publishing.
        
        Args:
            toolbox_id: Toolbox identifier
            tool_id: Tool identifier
            result_type: Type of result ('keypoints' or 'depth_estimation')
            
        Returns:
            Complete MQTT topic for publishing results
            
        Raises:
            ConfigurationError: If required configuration is missing
        """
        try:
            listener_config = self.get_mqtt_listener_config()
            estimation_config = self.get_mqtt_estimation_config()
            
            root = listener_config.get('root')
            if not root:
                raise ConfigurationError("MQTT listener root not configured")
            
            endpoint = estimation_config.get(result_type)
            if not endpoint:
                raise ConfigurationError(f"MQTT estimation endpoint for '{result_type}' not configured")
            
            return f"{root}/{toolbox_id}/{tool_id}/{endpoint}"
            
        except Exception as e:
            raise ConfigurationError(f"Error building result topic: {e}")
    
    def get_raw_config(self) -> Dict[str, Any]:
        """Get the raw configuration dictionary."""
        return self.config.copy() if self.config else {}
    
    def reload_configuration(self):
        """Reload configuration from file."""
        logging.info("Reloading configuration", extra={
            'config_path': self.config_path
        })
        self._load_configuration()
        self._validate_configuration()
    
    def validate_tool_identifiers(self, toolbox_id: str, tool_id: str) -> bool:
        """
        Validate tool identifiers against configuration constraints.
        
        Args:
            toolbox_id: Toolbox identifier
            tool_id: Tool identifier
            
        Returns:
            True if identifiers are valid, False otherwise
        """
        try:
            # Basic validation - can be extended based on requirements
            if not toolbox_id or not tool_id:
                return False
            
            # Check for invalid characters that might break MQTT topics
            invalid_chars = ['+', '#', '/', '\x00']
            for char in invalid_chars:
                if char in toolbox_id or char in tool_id:
                    return False
            
            return True
            
        except Exception as e:
            logging.warning("Error validating tool identifiers", extra={
                'toolbox_id': toolbox_id,
                'tool_id': tool_id,
                'error_message': str(e)
            })
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get configuration summary for logging and debugging.
        
        Returns:
            Dictionary containing configuration summary
        """
        try:
            broker_config = self.get_mqtt_broker_config()
            listener_config = self.get_mqtt_listener_config()
            
            return {
                'config_file': self.config_path,
                'broker_host': broker_config.get('host'),
                'broker_port': broker_config.get('port'),
                'has_authentication': self.has_authentication(),
                'listener_root': listener_config.get('root'),
                'time_window': self.get_time_window(),
                'cleanup_interval': self.get_cleanup_interval(),
                'max_buffer_size': self.get_max_buffer_size(),
                'topic_patterns': list(self.get_topic_patterns().keys())
            }
            
        except Exception as e:
            logging.warning("Error generating config summary", extra={
                'error_message': str(e)
            })
            return {'error': str(e)}