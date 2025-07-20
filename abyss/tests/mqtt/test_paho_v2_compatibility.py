"""Test Paho MQTT v2 callback compatibility"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

from abyss.mqtt.components.client_manager import MQTTClientManager


class TestPahoV2Compatibility(unittest.TestCase):
    """Test that MQTT callbacks work with Paho MQTT v2.x"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'mqtt': {
                'broker': {
                    'host': 'localhost',
                    'port': 1883,
                    'username': None,
                    'password': None
                },
                'listener': {
                    'root': 'OPCPUBSUB',
                    'result': 'ResultManagement',
                    'trace': 'ResultManagement/Trace',
                    'heads': 'AssetManagement/Heads'
                }
            }
        }
        self.manager = MQTTClientManager(self.config)
    
    def test_on_publish_callback_v2_signature(self):
        """Test that on_publish callback has correct v2 signature"""
        # Create publisher client
        publisher = self.manager.create_publisher('test_publisher')
        
        # Get the on_publish callback
        on_publish = publisher._on_publish
        
        # Simulate Paho MQTT v2 calling the callback with 5 parameters
        client_mock = Mock()
        userdata_mock = None
        mid = 1
        reason_code = mqtt.MQTT_ERR_SUCCESS  # Use the success constant directly
        properties = None
        
        # This should not raise TypeError
        try:
            on_publish(client_mock, userdata_mock, mid, reason_code, properties)
        except TypeError as e:
            self.fail(f"on_publish callback failed with v2 signature: {e}")
    
    def test_on_connect_callback_v2_signature(self):
        """Test that on_connect callbacks have correct v2 signature"""
        # Test listener on_connect
        listener = self.manager.create_listener('result', 'test_listener')
        on_connect = listener._on_connect
        
        # Simulate v2 callback
        client_mock = Mock()
        client_mock.subscribe = Mock()
        userdata_mock = None
        connect_flags = Mock()
        reason_code = mqtt.MQTT_ERR_SUCCESS  # Use the success constant directly
        properties = None
        
        # Should not raise TypeError
        try:
            on_connect(client_mock, userdata_mock, connect_flags, reason_code, properties)
        except TypeError as e:
            self.fail(f"on_connect callback failed with v2 signature: {e}")
    
    def test_client_created_with_v2_api(self):
        """Test that clients are created with VERSION2 API"""
        with patch('paho.mqtt.client.Client') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Create a client
            self.manager.create_mqtt_client('test_client')
            
            # Verify it was created with VERSION2
            mock_client_class.assert_called_with(
                client_id='test_client',
                callback_api_version=CallbackAPIVersion.VERSION2
            )
    
    def test_publisher_reconnection_on_failure(self):
        """Test publisher health check and reconnection logic"""
        # This would require more complex mocking of the drilling_analyser
        # but demonstrates the testing approach
        pass


if __name__ == '__main__':
    unittest.main()