"""
MQTT Client Manager Module

Handles MQTT client creation, connection management, and message routing.
Extracted from the original MQTTDrillingDataAnalyser class.
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Callable, Optional, Any
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

from ...uos_depth_est import TimestampedData, find_in_dict


class MQTTClientManager:
    """
    Manages MQTT client connections and message routing.
    
    Responsibilities:
    - Create and configure MQTT clients
    - Handle connection events and authentication
    - Route messages to appropriate handlers
    - Manage subscriptions and topics
    """
    
    def __init__(self, config: Dict[str, Any], message_handler: Optional[Callable] = None):
        """
        Initialize MQTT Client Manager.
        
        Args:
            config: Configuration dictionary containing MQTT settings
            message_handler: Optional callback for handling received messages
        """
        self.config = config
        self.broker = config['mqtt']['broker']
        self.message_handler = message_handler
        self.clients = {}
        self.topics = []
        
    def create_mqtt_client(self, client_id: str) -> mqtt.Client:
        """
        Create MQTT client with basic configuration.
        
        Args:
            client_id: Unique identifier for the MQTT client
            
        Returns:
            Configured MQTT client
        """
        client = mqtt.Client(client_id=client_id, callback_api_version=CallbackAPIVersion.VERSION2)
        
        if self.broker.get('username') and self.broker.get('password'):
            client.username_pw_set(
                self.broker['username'], 
                self.broker['password']
            )
            logging.debug("Configured MQTT client with username authentication", extra={
                'client_id': client_id,
                'username': self.broker['username']
            })
        
        return client
        
    def create_listener(self, listener_type: str, client_id: str) -> mqtt.Client:
        """
        Create a generic MQTT client for listening to specified data type.
        
        Args:
            listener_type: Type of listener ('result', 'trace', 'heads')
            client_id: Unique identifier for the client
            
        Returns:
            Configured MQTT client with callbacks
            
        Raises:
            ValueError: If listener_type is not valid
        """
        # Validate listener type
        valid_types = ['result', 'trace']
        if 'heads' in self.config['mqtt']['listener']:
            valid_types.append('heads')
        
        if listener_type not in valid_types:
            raise ValueError(f"Invalid listener type: {listener_type}. Must be one of {valid_types}")
        
        def on_connect(client, userdata, connect_flags, reason_code, properties):
            if reason_code == 0:  # Success
                logging.info(f"{listener_type.title()} Listener Connected", extra={
                    'reason_code': str(reason_code),
                    'listener_type': listener_type,
                    'client_id': client_id
                })
                topic = f"{self.config['mqtt']['listener']['root']}/+/+/{self.config['mqtt']['listener'][listener_type]}"
                client.subscribe(topic)
                self.topics.append(topic)
                logging.info(f"Subscribed to {topic}")
            else:
                logging.error("Failed to connect", extra={
                    'reason_code': str(reason_code),
                    'listener_type': listener_type,
                    'client_id': client_id
                })

        def on_message(client, userdata, msg):
            try:
                data = json.loads(msg.payload)
                logging.debug(f"{listener_type.title()} Message received", extra={
                    'topic': msg.topic,
                    'payload_size': len(msg.payload),
                    'listener_type': listener_type
                })
                
                # Extract timestamp
                st = find_in_dict(data, 'SourceTimestamp')
                if not st:
                    logging.warning("No SourceTimestamp found in message", extra={
                        'topic': msg.topic,
                        'listener_type': listener_type
                    })
                    return
                    
                logging.debug("Source Timestamp: %s", st[0])
                dt = datetime.strptime(st[0], "%Y-%m-%dT%H:%M:%SZ")
                unix_timestamp = dt.timestamp()
                
                # Create timestamped data
                tsdata = TimestampedData(
                    _timestamp=unix_timestamp,
                    _data=data,
                    _source=msg.topic
                )
                
                # Route to message handler if available
                if self.message_handler:
                    self.message_handler(tsdata)
                
                # Extract and log tool information
                topic_parts = msg.topic.split('/')
                if len(topic_parts) >= 3:
                    logging.debug("Message details", extra={
                        'unix_timestamp': unix_timestamp,
                        'toolbox_id': topic_parts[1],
                        'tool_id': topic_parts[2],
                        'listener_type': listener_type
                    })
                    
            except json.JSONDecodeError as e:
                logging.error("JSON decode error", extra={
                    'topic': msg.topic,
                    'payload_size': len(msg.payload),
                    'error_position': getattr(e, 'pos', None),
                    'error_message': str(e),
                    'listener_type': listener_type
                })
            except ValueError as e:
                logging.error("Timestamp parsing error", extra={
                    'topic': msg.topic,
                    'error_message': str(e),
                    'listener_type': listener_type
                })
            except Exception as e:
                logging.error("Message processing error", extra={
                    'topic': msg.topic,
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'listener_type': listener_type
                }, exc_info=True)

        def on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
            if reason_code != 0:
                logging.warning("Unexpected disconnection from MQTT broker", extra={
                    'reason_code': str(reason_code),
                    'listener_type': listener_type,
                    'client_id': client_id
                })
        
        # Create and configure client
        client = mqtt.Client(client_id=client_id, callback_api_version=CallbackAPIVersion.VERSION2)
        client.on_connect = on_connect
        client.on_message = on_message
        client.on_disconnect = on_disconnect
        
        # Apply authentication if configured
        if self.broker.get('username') and self.broker.get('password'):
            client.username_pw_set(
                self.broker['username'], 
                self.broker['password']
            )
        
        # Store client reference
        self.clients[listener_type] = client
        
        return client
        
    def create_result_listener(self) -> mqtt.Client:
        """Create an MQTT client for listening to Result data."""
        return self.create_listener('result', 'result_listener')
        
    def create_trace_listener(self) -> mqtt.Client:
        """Create an MQTT client for listening to Trace data."""
        return self.create_listener('trace', 'trace_listener')
        
    def create_heads_listener(self) -> mqtt.Client:
        """Create an MQTT client for listening to Heads data."""
        return self.create_listener('heads', 'heads_listener')
        
    def connect_all_clients(self) -> bool:
        """
        Connect all registered clients to the MQTT broker.
        
        Returns:
            True if all clients connected successfully, False otherwise
        """
        success = True
        
        for listener_type, client in self.clients.items():
            try:
                logging.info(f"Connecting {listener_type} client to broker", extra={
                    'broker_host': self.broker['host'],
                    'broker_port': self.broker['port'],
                    'listener_type': listener_type
                })
                
                client.connect(
                    self.broker['host'], 
                    self.broker['port'], 
                    self.broker.get('keepalive', 60)
                )
                client.loop_start()
                
            except Exception as e:
                logging.error(f"Failed to connect {listener_type} client", extra={
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'listener_type': listener_type,
                    'broker_host': self.broker['host'],
                    'broker_port': self.broker['port']
                })
                success = False
                
        return success
        
    def disconnect_all_clients(self):
        """Disconnect all registered clients from the MQTT broker."""
        for listener_type, client in self.clients.items():
            try:
                logging.info(f"Disconnecting {listener_type} client")
                client.loop_stop()
                client.disconnect()
            except Exception as e:
                logging.warning(f"Error disconnecting {listener_type} client", extra={
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'listener_type': listener_type
                })
                
    def get_client(self, listener_type: str) -> Optional[mqtt.Client]:
        """
        Get client by listener type.
        
        Args:
            listener_type: Type of listener to retrieve
            
        Returns:
            MQTT client if found, None otherwise
        """
        return self.clients.get(listener_type)
        
    def get_subscribed_topics(self) -> List[str]:
        """Get list of all subscribed topics."""
        return self.topics.copy()
        
    def set_message_handler(self, handler: Callable):
        """
        Set the message handler for processing received messages.
        
        Args:
            handler: Callable that accepts TimestampedData
        """
        self.message_handler = handler