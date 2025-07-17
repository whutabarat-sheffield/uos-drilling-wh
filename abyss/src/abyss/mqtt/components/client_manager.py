"""
MQTT Client Manager Module

Handles MQTT client creation, connection management, and message routing.
Extracted from the original MQTTDrillingDataAnalyser class.
"""

import logging
import json
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Callable, Optional, Any, Union
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

from ...uos_depth_est import TimestampedData, find_in_dict
from .config_manager import ConfigurationManager


class MQTTClientManager:
    """
    Manages MQTT client connections and message routing.
    
    Responsibilities:
    - Create and configure MQTT clients
    - Handle connection events and authentication
    - Route messages to appropriate handlers
    - Manage subscriptions and topics
    """
    
    def __init__(self, config: Union[Dict[str, Any], ConfigurationManager], message_handler: Optional[Callable] = None):
        """
        Initialize MQTT Client Manager.
        
        Args:
            config: Configuration dictionary or ConfigurationManager instance
            message_handler: Optional callback for handling received messages
        """
        # Handle both ConfigurationManager and raw config dict for backward compatibility
        if isinstance(config, ConfigurationManager):
            self.config_manager = config
            self.config = config.get_raw_config()
            self.broker = config.get_mqtt_broker_config()
        else:
            # Legacy support for raw config dictionary
            self.config_manager = None
            self.config = config
            self.broker = config['mqtt']['broker']
            
        self.message_handler = message_handler
        self.clients = {}
        self.topics = []
        
        # Error tracking for warnings
        self._connection_failures = defaultdict(int)  # client_type -> failure count
        self._message_errors = defaultdict(int)  # error_type -> count
        self._last_warning_time = {}  # warning_type -> timestamp
        self._error_window = 300  # 5 minutes
        
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
        if self.config_manager:
            # Use ConfigurationManager for typed access
            listener_config = self.config_manager.get_mqtt_listener_config()
        else:
            # Legacy raw config access
            listener_config = self.config['mqtt']['listener']
            
        if 'heads' in listener_config:
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
                topic = f"{listener_config['root']}/+/+/{listener_config[listener_type]}"
                client.subscribe(topic)
                self.topics.append(topic)
                logging.debug(f"Subscribed to {topic}")
                
                # Reset connection failure count on successful connection
                if listener_type in self._connection_failures:
                    self._connection_failures[listener_type] = 0
            else:
                logging.error("Failed to connect", extra={
                    'reason_code': str(reason_code),
                    'listener_type': listener_type,
                    'client_id': client_id
                })
                
                # Track connection failures
                self._connection_failures[listener_type] += 1
                self._check_connection_failures(listener_type)

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
                    self._track_message_error('missing_timestamp')
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
                self._track_message_error('json_decode_error')
            except ValueError as e:
                logging.error("Timestamp parsing error", extra={
                    'topic': msg.topic,
                    'error_message': str(e),
                    'listener_type': listener_type
                })
                self._track_message_error('timestamp_parse_error')
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
                logging.debug(f"Disconnecting {listener_type} client")
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
    
    def _check_connection_failures(self, client_type: str):
        """Check and warn about repeated connection failures."""
        failure_count = self._connection_failures[client_type]
        
        if failure_count >= 3:  # Warn after 3 consecutive failures
            self._log_rate_limited_warning(f'connection_failure_{client_type}', 300,
                "Repeated connection failures for MQTT client", {
                    'client_type': client_type,
                    'consecutive_failures': failure_count,
                    'broker_host': self.broker['host'],
                    'broker_port': self.broker['port'],
                    'possible_cause': 'Network issues, broker unavailable, or authentication problems'
                })
    
    def _track_message_error(self, error_type: str):
        """Track message errors and warn if rate is too high."""
        self._message_errors[error_type] += 1
        
        # Check error rates periodically
        current_time = time.time()
        last_check = self._last_warning_time.get('message_error_check', 0)
        
        if current_time - last_check >= 60:  # Check every minute
            self._last_warning_time['message_error_check'] = current_time
            
            # Check each error type
            for err_type, count in self._message_errors.items():
                if count > 50:  # More than 50 errors in the window
                    self._log_rate_limited_warning(f'message_error_{err_type}', 300,
                        "High rate of message processing errors", {
                            'error_type': err_type,
                            'error_count': count,
                            'time_window_minutes': self._error_window / 60,
                            'possible_cause': 'Malformed messages, schema changes, or timestamp format issues'
                        })
            
            # Reset counters after warning check
            self._message_errors.clear()
    
    def _log_rate_limited_warning(self, warning_type: str, min_interval: int, 
                                 message: str, extra: Dict[str, Any]):
        """Log warning with rate limiting to avoid spam."""
        current_time = time.time()
        last_warning = self._last_warning_time.get(warning_type, 0)
        
        if current_time - last_warning >= min_interval:
            logging.warning(message, extra=extra)
            self._last_warning_time[warning_type] = current_time