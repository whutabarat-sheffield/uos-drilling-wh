"""
Lightweight configuration loader for MQTT publishers.

This module provides minimal configuration loading functionality
without any deep learning dependencies. It's designed specifically
for the MQTT publisher which only needs to read broker and topic
configurations from YAML files.
"""

import logging
import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


class PublisherConfigLoader:
    """
    Lightweight configuration loader for MQTT publishers.
    
    This is a minimal implementation that only loads what's needed
    for publishing MQTT messages, avoiding heavy dependencies.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the config loader.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = os.path.abspath(config_path)
        self.config = self._load_yaml()
        
    def _load_yaml(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            if config is None:
                raise ValueError("Configuration file is empty or invalid")
            
            logging.info(f"Configuration loaded from {self.config_path}")
            return config
            
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            raise
    
    def get_mqtt_broker_config(self) -> Dict[str, Any]:
        """Get MQTT broker configuration."""
        return self.config.get('mqtt', {}).get('broker', {})
    
    def get_mqtt_listener_config(self) -> Dict[str, Any]:
        """Get MQTT listener/topic configuration."""
        return self.config.get('mqtt', {}).get('listener', {})
    
    def get_publisher_config(self) -> Dict[str, Any]:
        """Get publisher-specific configuration."""
        return self.config.get('mqtt', {}).get('publisher', {})
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value
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
            
        except Exception:
            return default


def load_publisher_config(config_path: str) -> Dict[str, Any]:
    """
    Convenience function to load publisher configuration.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing loaded configuration
    """
    loader = PublisherConfigLoader(config_path)
    
    mqtt_config = {
        'broker': loader.get_mqtt_broker_config(),
        'listener': loader.get_mqtt_listener_config(),
        'publisher': loader.get_publisher_config()
    }
    
    # Build a simplified config structure for PublisherConfig.from_yaml
    return {
        'broker_host': mqtt_config['broker'].get('host', 'localhost'),
        'broker_port': mqtt_config['broker'].get('port', 1883),
        'username': mqtt_config['broker'].get('username'),
        'password': mqtt_config['broker'].get('password'),
        'root_topic': mqtt_config['listener'].get('root', 'OPCPUBSUB'),
        'result_suffix': mqtt_config['listener'].get('result', 'ResultManagement'),
        'trace_suffix': mqtt_config['listener'].get('trace', 'ResultManagement/Trace'),
        'heads_suffix': mqtt_config['listener'].get('heads', 'AssetManagement/Heads'),
    }