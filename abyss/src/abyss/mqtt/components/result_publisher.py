"""
Result Publisher Module

Handles publishing of depth estimation results to MQTT topics.
Extracted from the original MQTTDrillingDataAnalyser class.
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import paho.mqtt.client as mqtt

from .message_processor import ProcessingResult
from .config_manager import ConfigurationManager
from .exceptions import (
    MQTTPublishError,
    AbyssProcessingError,
    wrap_exception
)


class PublishResultType(Enum):
    """Enumeration of result publication types."""
    SUCCESS = "success"
    INSUFFICIENT_DATA = "insufficient_data"
    ERROR = "error"


class ResultPublisher:
    """
    Publishes depth estimation results to MQTT broker.
    
    Responsibilities:
    - Format result data for publishing
    - Publish keypoints and depth estimation results
    - Handle insufficient data scenarios
    - Manage MQTT client for result publishing
    """
    
    def __init__(self, mqtt_client: mqtt.Client, config: Union[Dict[str, Any], ConfigurationManager]):
        """
        Initialize ResultPublisher.
        
        Args:
            mqtt_client: MQTT client for publishing results
            config: Configuration dictionary or ConfigurationManager instance
        """
        self.mqtt_client = mqtt_client
        
        if isinstance(config, ConfigurationManager):
            self.config_manager = config
            self.config = config.get_raw_config()
        else:
            # Legacy support for raw config dictionary
            self.config_manager = None
            self.config = config
    
    def publish_processing_result(self, processing_result: ProcessingResult,
                                toolbox_id: str, tool_id: str, 
                                timestamp: float, algo_version: str) -> None:
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
            AbyssProcessingError: If result processing fails
        """
        try:
            # Convert timestamp to string format
            dt_utc = datetime.fromtimestamp(timestamp)
            dt_string = datetime.strftime(dt_utc, "%Y-%m-%dT%H:%M:%SZ")
            
            if not processing_result.success or processing_result.error_message:
                self._publish_error_result(
                    processing_result, toolbox_id, tool_id, dt_string, 
                    processing_result.error_message or "Processing failed"
                )
                return
            
            if (processing_result.keypoints is None or 
                processing_result.depth_estimation is None):
                self._publish_insufficient_data_result(
                    processing_result, toolbox_id, tool_id, dt_string
                )
                return
            
            self._publish_successful_result(
                processing_result, toolbox_id, tool_id, 
                dt_string, algo_version
            )
            
        except Exception as e:
            logging.error("Error publishing processing result", extra={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'toolbox_id': toolbox_id,
                'tool_id': tool_id
            })
            raise wrap_exception(e, AbyssProcessingError, "Failed to publish processing result")
    
    def _publish_result_consolidated(self, 
                                   result_type: PublishResultType,
                                   processing_result: ProcessingResult,
                                   toolbox_id: str, 
                                   tool_id: str,
                                   dt_string: str,
                                   algo_version: Optional[str] = None,
                                   error_message: Optional[str] = None) -> None:
        """
        Consolidated method for publishing all types of results.
        
        Args:
            result_type: Type of result being published
            processing_result: Result from message processing
            toolbox_id: Toolbox identifier
            tool_id: Tool identifier
            dt_string: Formatted timestamp string
            algo_version: Algorithm version (for success results only)
            error_message: Error message (for error results only)
            
        Raises:
            MQTTPublishError: If publishing to MQTT broker fails
        """
        try:
            # Prepare topics
            keyp_topic = self._build_topic(toolbox_id, tool_id, 'keypoints')
            dest_topic = self._build_topic(toolbox_id, tool_id, 'depth_estimation')
            
            # Build base data structure
            base_data = {
                'SourceTimestamp': dt_string,
                'MachineId': processing_result.machine_id,
                'ResultId': processing_result.result_id,
                'HeadId': processing_result.head_id
            }
            
            # Customize data based on result type
            if result_type == PublishResultType.SUCCESS:
                keyp_data = {
                    **base_data,
                    'Value': processing_result.keypoints,
                    'AlgoVersion': algo_version
                }
                dest_data = {
                    **base_data,
                    'Value': processing_result.depth_estimation,
                    'AlgoVersion': algo_version
                }
                log_level = logging.INFO
                log_message = "Successfully published depth estimation results"
                log_extra = {
                    'toolbox_id': toolbox_id,
                    'tool_id': tool_id,
                    'keypoints': processing_result.keypoints,
                    'depth_estimation': processing_result.depth_estimation,
                    'machine_id': processing_result.machine_id,
                    'result_id': processing_result.result_id,
                    'head_id': processing_result.head_id
                }
                
            elif result_type == PublishResultType.INSUFFICIENT_DATA:
                keyp_data = {
                    **base_data,
                    'Value': 'Not enough steps to estimate keypoints'
                }
                dest_data = {
                    **base_data,
                    'Value': 'Not enough steps to estimate depth'
                }
                log_level = logging.WARNING
                log_message = "Published insufficient data result"
                log_extra = {
                    'toolbox_id': toolbox_id,
                    'tool_id': tool_id
                }
                
            elif result_type == PublishResultType.ERROR:
                keyp_data = {
                    **base_data,
                    'Value': f'Error in keypoint estimation: {error_message}',
                    'Error': True
                }
                dest_data = {
                    **base_data,
                    'Value': f'Error in depth estimation: {error_message}',
                    'Error': True
                }
                log_level = logging.ERROR
                log_message = "Published error result"
                log_extra = {
                    'toolbox_id': toolbox_id,
                    'tool_id': tool_id,
                    'error_message': error_message
                }
            
            else:
                raise ValueError(f"Unknown result type: {result_type}")
            
            # Publish results - will raise exceptions on failure
            self._publish_message(keyp_topic, keyp_data)
            self._publish_message(dest_topic, dest_data)
            
            # Log successful publishing
            logging.log(log_level, log_message, extra=log_extra)
                
        except ValueError:
            # Re-raise ValueError for unknown result types
            raise
        except Exception as e:
            result_type_str = result_type.value if hasattr(result_type, 'value') else str(result_type)
            logging.error(f"Error publishing {result_type_str} result", extra={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'toolbox_id': toolbox_id,
                'tool_id': tool_id,
                'result_type': result_type_str
            })
            raise wrap_exception(e, MQTTPublishError, f"Failed to publish {result_type_str} result")
    
    def _publish_successful_result(self, processing_result: ProcessingResult,
                                 toolbox_id: str, tool_id: str,
                                 dt_string: str, algo_version: str) -> None:
        """Publish successful processing results using consolidated method."""
        self._publish_result_consolidated(
            PublishResultType.SUCCESS,
            processing_result,
            toolbox_id,
            tool_id,
            dt_string,
            algo_version=algo_version
        )
    
    def _publish_insufficient_data_result(self, processing_result: ProcessingResult,
                                        toolbox_id: str, tool_id: str,
                                        dt_string: str) -> None:
        """Publish result when there's insufficient data for estimation using consolidated method."""
        self._publish_result_consolidated(
            PublishResultType.INSUFFICIENT_DATA,
            processing_result,
            toolbox_id,
            tool_id,
            dt_string
        )
    
    def _publish_error_result(self, processing_result: ProcessingResult,
                            toolbox_id: str, tool_id: str,
                            dt_string: str, error_message: str) -> None:
        """Publish result when processing encountered an error using consolidated method."""
        self._publish_result_consolidated(
            PublishResultType.ERROR,
            processing_result,
            toolbox_id,
            tool_id,
            dt_string,
            error_message=error_message
        )
    
    def _build_topic(self, toolbox_id: str, tool_id: str, result_type: str) -> str:
        """Build MQTT topic for result publishing."""
        if self.config_manager:
            return self.config_manager.build_result_topic(toolbox_id, tool_id, result_type)
        else:
            # Legacy fallback
            root = self.config['mqtt']['listener']['root']
            endpoint = self.config['mqtt']['estimation'][result_type]
            return f"{root}/{toolbox_id}/{tool_id}/{endpoint}"
    
    def _publish_message(self, topic: str, data: Dict[str, Any]) -> None:
        """
        Publish a single message to MQTT broker.
        
        Args:
            topic: MQTT topic to publish to
            data: Data to publish
            
        Raises:
            MQTTPublishError: If message publishing fails
        """
        try:
            payload = json.dumps(data)
            result = self.mqtt_client.publish(topic, payload)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logging.debug("Message published successfully", extra={
                    'topic': topic,
                    'payload_size': len(payload)
                })
            else:
                logging.warning("MQTT publish failed", extra={
                    'topic': topic,
                    'result_code': result.rc,
                    'payload_size': len(payload)
                })
                raise MQTTPublishError(
                    f"MQTT publish failed for topic '{topic}' with result code {result.rc}"
                )
                
        except MQTTPublishError:
            # Re-raise our own exceptions
            raise
        except Exception as e:
            logging.error("Error publishing MQTT message", extra={
                'topic': topic,
                'error_type': type(e).__name__,
                'error_message': str(e)
            })
            raise wrap_exception(e, MQTTPublishError, f"Failed to publish to topic '{topic}'")
    
    def publish_custom_result(self, toolbox_id: str, tool_id: str,
                            result_type: str, value: Any,
                            timestamp: Optional[float] = None,
                            additional_fields: Optional[Dict[str, Any]] = None) -> None:
        """
        Publish custom result data.
        
        Args:
            toolbox_id: Toolbox identifier
            tool_id: Tool identifier
            result_type: Type of result ('keypoints' or 'depth_estimation')
            value: Result value to publish
            timestamp: Optional timestamp (uses current time if None)
            additional_fields: Optional additional fields to include
            
        Raises:
            MQTTPublishError: If message publishing fails
        """
        try:
            if timestamp is None:
                timestamp = datetime.now().timestamp()
            
            dt_utc = datetime.fromtimestamp(timestamp)
            dt_string = datetime.strftime(dt_utc, "%Y-%m-%dT%H:%M:%SZ")
            
            topic = self._build_topic(toolbox_id, tool_id, result_type)
            
            data = {
                'Value': value,
                'SourceTimestamp': dt_string
            }
            
            if additional_fields:
                data.update(additional_fields)
            
            self._publish_message(topic, data)
            
        except Exception as e:
            logging.error("Error publishing custom result", extra={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'toolbox_id': toolbox_id,
                'tool_id': tool_id,
                'result_type': result_type
            })
            raise wrap_exception(e, MQTTPublishError, "Failed to publish custom result")
    
    def get_publisher_stats(self) -> Dict[str, Any]:
        """
        Get publisher statistics.
        
        Returns:
            Dictionary containing publisher statistics
        """
        return {
            'mqtt_client_connected': self.mqtt_client.is_connected() if hasattr(self.mqtt_client, 'is_connected') else 'unknown',
            'config_loaded': self.config is not None,
            'publisher_ready': True
        }