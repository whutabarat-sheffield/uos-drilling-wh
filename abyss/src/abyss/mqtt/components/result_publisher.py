"""
Result Publisher Module

Simplified MQTT publisher that delegates formatting and validation
to specialized components for better separation of concerns.
"""

import logging
import json
from typing import Dict, Optional, Any
import paho.mqtt.client as mqtt

from .message_processor import ProcessingResult
from .config_manager import ConfigurationManager
from .message_formatter import ResultMessageFormatter, ResultType
from .depth_validator import DepthValidator
from .exceptions import MQTTPublishError, wrap_exception


class ResultPublisher:
    """
    Publishes depth estimation results to MQTT broker.
    
    Simplified to focus only on publishing, with formatting
    and validation delegated to specialized components.
    """
    
    def __init__(self, mqtt_client: mqtt.Client, config: ConfigurationManager):
        """
        Initialize ResultPublisher.
        
        Args:
            mqtt_client: MQTT client for publishing results
            config: ConfigurationManager instance
        """
        self.mqtt_client = mqtt_client
        self.config_manager = config
        
        # Initialize components
        self.formatter = ResultMessageFormatter()
        self.validator = DepthValidator(self.config_manager)
        
        logging.info("ResultPublisher initialized with depth validation", extra={
            'behavior': self.config_manager.get('mqtt.depth_validation.negative_depth_behavior', 'publish')
        })
    
    def publish_processing_result(self, 
                                processing_result: ProcessingResult,
                                toolbox_id: str, 
                                tool_id: str, 
                                timestamp: float, 
                                algo_version: str) -> None:
        """
        Publish processing result to MQTT topics.
        
        Args:
            processing_result: Result from message processing
            toolbox_id: Toolbox identifier
            tool_id: Tool identifier  
            timestamp: Message timestamp
            algo_version: Algorithm version
            
        Raises:
            MQTTPublishError: If publishing to MQTT broker fails
        """
        try:
            # Validate the result
            validation = self.validator.validate(processing_result)
            
            # Log any warnings
            if validation.warnings:
                for warning in validation.warnings:
                    logging.warning(warning, extra={
                        'toolbox_id': toolbox_id,
                        'tool_id': tool_id,
                        'validation_action': validation.action
                    })
            
            # Take action based on validation
            if validation.action == 'skip':
                logging.info("Skipping publish based on validation", extra={
                    'reason': validation.reason,
                    'toolbox_id': toolbox_id,
                    'tool_id': tool_id
                })
                return
            
            # Determine result type and format messages
            if not processing_result.success or processing_result.error_message:
                result_type = ResultType.ERROR
                error_msg = processing_result.error_message or "Processing failed"
            elif (processing_result.keypoints is None or 
                  processing_result.depth_estimation is None or
                  (isinstance(processing_result.keypoints, list) and len(processing_result.keypoints) == 0) or
                  (isinstance(processing_result.depth_estimation, list) and len(processing_result.depth_estimation) == 0)):
                result_type = ResultType.INSUFFICIENT_DATA
                error_msg = None
            elif validation.action == 'publish_with_warning':
                result_type = ResultType.WARNING
                error_msg = None
            else:
                result_type = ResultType.SUCCESS
                error_msg = None
            
            # Format the messages
            messages = self.formatter.format_result(
                result=processing_result,
                timestamp=timestamp,
                algo_version=algo_version,
                result_type=result_type,
                error_message=error_msg
            )
            
            # Publish each message
            for topic_suffix, payload in messages.items():
                topic = self._build_topic(toolbox_id, tool_id, topic_suffix)
                logging.debug(f"Publishing to topic: {topic}", extra={
                    'topic_suffix': topic_suffix,
                    'toolbox_id': toolbox_id,
                    'tool_id': tool_id
                })
                self._publish_message(topic, payload)
            
            # Log success
            self._log_publish_success(result_type, toolbox_id, tool_id, processing_result)
            
        except MQTTPublishError:
            # Re-raise MQTT errors as-is
            raise
        except Exception as e:
            logging.error("Error publishing processing result", extra={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'toolbox_id': toolbox_id,
                'tool_id': tool_id
            }, exc_info=True)
            raise wrap_exception(e, MQTTPublishError, "Failed to publish processing result")
    
    def publish_custom_result(self, 
                            toolbox_id: str, 
                            tool_id: str,
                            result_type: str, 
                            data: Dict[str, Any]) -> None:
        """
        Publish a custom result to MQTT.
        
        Args:
            toolbox_id: Toolbox identifier
            tool_id: Tool identifier
            result_type: Type of result (used in topic)
            data: Custom data to publish
            
        Raises:
            MQTTPublishError: If message publishing fails
        """
        try:
            topic = self._build_topic(toolbox_id, tool_id, result_type)
            self._publish_message(topic, data)
            
            logging.debug("Published custom result", extra={
                'toolbox_id': toolbox_id,
                'tool_id': tool_id,
                'result_type': result_type
            })
            
        except Exception as e:
            raise wrap_exception(e, MQTTPublishError, "Failed to publish custom result")
    
    def _build_topic(self, toolbox_id: str, tool_id: str, result_type: str) -> str:
        """Build MQTT topic for publishing."""
        # Get the estimation path from config - these ARE used as topic suffixes
        # This maintains compatibility with the old system
        estimation_config = self.config_manager.get_mqtt_estimation_config()
        estimation_path = estimation_config.get(result_type)
        
        # Get listener config for root path
        listener_config = self.config_manager.get_mqtt_listener_config()
        root = listener_config.get('root', '')
        
        # Log what we're working with
        logging.debug(f"Building topic - root: {root}, toolbox: {toolbox_id}, tool: {tool_id}, "
                     f"result_type: {result_type}, estimation_path: {estimation_path}")
        
        if not estimation_path:
            # Fallback to simple suffix if not found in config
            result_suffix = listener_config.get('result', 'ResultManagement')
            topic = f"{root}/{toolbox_id}/{tool_id}/{result_suffix}/{result_type}"
            logging.debug(f"Using fallback topic: {topic}")
            return topic
        
        # Use the estimation path directly as in the old system
        topic = f"{root}/{toolbox_id}/{tool_id}/{estimation_path}"
        logging.debug(f"Built topic: {topic}")
        return topic
    
    def _publish_message(self, topic: str, data: Dict[str, Any]) -> None:
        """
        Publish message to MQTT broker.
        
        Args:
            topic: MQTT topic
            data: Message payload
            
        Raises:
            MQTTPublishError: If message publishing fails
        """
        try:
            json_data = json.dumps(data)
            info = self.mqtt_client.publish(topic, json_data)
            
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                raise MQTTPublishError(
                    f"MQTT publish failed with return code: {info.rc}"
                )
            
            logging.debug("Published message to MQTT", extra={
                'topic': topic,
                'payload_size': len(json_data)
            })
            
        except MQTTPublishError:
            raise
        except Exception as e:
            raise wrap_exception(e, MQTTPublishError, f"Failed to publish to topic '{topic}'")
    
    def _log_publish_success(self, 
                           result_type: ResultType,
                           toolbox_id: str, 
                           tool_id: str,
                           result: ProcessingResult):
        """Log successful publish with appropriate level."""
        if result_type == ResultType.SUCCESS:
            logging.debug("Successfully published depth estimation results", extra={
                'toolbox_id': toolbox_id,
                'tool_id': tool_id,
                'keypoints': result.keypoints,
                'depth_estimation': result.depth_estimation
            })
        elif result_type == ResultType.WARNING:
            logging.info("Published results with warnings", extra={
                'toolbox_id': toolbox_id,
                'tool_id': tool_id,
                'result_type': result_type.value
            })
        else:
            logging.warning(f"Published {result_type.value} result", extra={
                'toolbox_id': toolbox_id,
                'tool_id': tool_id
            })
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get current validation statistics."""
        return self.validator.get_validation_stats()