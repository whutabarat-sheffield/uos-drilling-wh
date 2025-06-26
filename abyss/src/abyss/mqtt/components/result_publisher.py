"""
Result Publisher Module

Handles publishing of depth estimation results to MQTT topics.
Extracted from the original MQTTDrillingDataAnalyser class.
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import paho.mqtt.client as mqtt

from .message_processor import ProcessingResult


class ResultPublisher:
    """
    Publishes depth estimation results to MQTT broker.
    
    Responsibilities:
    - Format result data for publishing
    - Publish keypoints and depth estimation results
    - Handle insufficient data scenarios
    - Manage MQTT client for result publishing
    """
    
    def __init__(self, mqtt_client: mqtt.Client, config: Dict[str, Any]):
        """
        Initialize ResultPublisher.
        
        Args:
            mqtt_client: MQTT client for publishing results
            config: Configuration dictionary
        """
        self.mqtt_client = mqtt_client
        self.config = config
    
    def publish_processing_result(self, processing_result: ProcessingResult,
                                toolbox_id: str, tool_id: str, 
                                timestamp: float, algo_version: str) -> bool:
        """
        Publish processing result to MQTT topics.
        
        Args:
            processing_result: Result from message processing
            toolbox_id: Toolbox identifier
            tool_id: Tool identifier  
            timestamp: Message timestamp
            algo_version: Algorithm version
            
        Returns:
            True if publishing succeeded, False otherwise
        """
        try:
            # Convert timestamp to string format
            dt_utc = datetime.fromtimestamp(timestamp)
            dt_string = datetime.strftime(dt_utc, "%Y-%m-%dT%H:%M:%SZ")
            
            if not processing_result.success or processing_result.error_message:
                return self._publish_error_result(
                    processing_result, toolbox_id, tool_id, dt_string, 
                    processing_result.error_message or "Processing failed"
                )
            
            if (processing_result.keypoints is None or 
                processing_result.depth_estimation is None):
                return self._publish_insufficient_data_result(
                    processing_result, toolbox_id, tool_id, dt_string
                )
            
            return self._publish_successful_result(
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
            return False
    
    def _publish_successful_result(self, processing_result: ProcessingResult,
                                 toolbox_id: str, tool_id: str,
                                 dt_string: str, algo_version: str) -> bool:
        """Publish successful processing results."""
        try:
            # Prepare topics
            keyp_topic = self._build_topic(toolbox_id, tool_id, 'keypoints')
            dest_topic = self._build_topic(toolbox_id, tool_id, 'depth_estimation')
            
            # Prepare keypoints data
            keyp_data = {
                'Value': processing_result.keypoints,
                'SourceTimestamp': dt_string,
                'MachineId': processing_result.machine_id,
                'ResultId': processing_result.result_id,
                'HeadId': processing_result.head_id,
                'AlgoVersion': algo_version
            }
            
            # Prepare depth estimation data
            dest_data = {
                'Value': processing_result.depth_estimation,
                'SourceTimestamp': dt_string,
                'MachineId': processing_result.machine_id,
                'ResultId': processing_result.result_id,
                'HeadId': processing_result.head_id,
                'AlgoVersion': algo_version
            }
            
            # Publish results
            keyp_success = self._publish_message(keyp_topic, keyp_data)
            dest_success = self._publish_message(dest_topic, dest_data)
            
            if keyp_success and dest_success:
                logging.info("Successfully published depth estimation results", extra={
                    'toolbox_id': toolbox_id,
                    'tool_id': tool_id,
                    'keypoints': processing_result.keypoints,
                    'depth_estimation': processing_result.depth_estimation,
                    'machine_id': processing_result.machine_id,
                    'result_id': processing_result.result_id,
                    'head_id': processing_result.head_id
                })
                return True
            else:
                logging.warning("Partial failure in result publishing", extra={
                    'keypoints_published': keyp_success,
                    'depth_published': dest_success
                })
                return False
                
        except Exception as e:
            logging.error("Error publishing successful result", extra={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'toolbox_id': toolbox_id,
                'tool_id': tool_id
            })
            return False
    
    def _publish_insufficient_data_result(self, processing_result: ProcessingResult,
                                        toolbox_id: str, tool_id: str,
                                        dt_string: str) -> bool:
        """Publish result when there's insufficient data for estimation."""
        try:
            # Prepare topics
            keyp_topic = self._build_topic(toolbox_id, tool_id, 'keypoints')
            dest_topic = self._build_topic(toolbox_id, tool_id, 'depth_estimation')
            
            # Prepare insufficient data messages
            keyp_data = {
                'Value': 'Not enough steps to estimate keypoints',
                'SourceTimestamp': dt_string,
                'MachineId': processing_result.machine_id,
                'ResultId': processing_result.result_id,
                'HeadId': processing_result.head_id
            }
            
            dest_data = {
                'Value': 'Not enough steps to estimate depth',
                'SourceTimestamp': dt_string,
                'MachineId': processing_result.machine_id,
                'ResultId': processing_result.result_id,
                'HeadId': processing_result.head_id
            }
            
            # Publish messages
            keyp_success = self._publish_message(keyp_topic, keyp_data)
            dest_success = self._publish_message(dest_topic, dest_data)
            
            logging.warning("Published insufficient data result", extra={
                'toolbox_id': toolbox_id,
                'tool_id': tool_id,
                'keypoints_published': keyp_success,
                'depth_published': dest_success
            })
            
            return keyp_success and dest_success
            
        except Exception as e:
            logging.error("Error publishing insufficient data result", extra={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'toolbox_id': toolbox_id,
                'tool_id': tool_id
            })
            return False
    
    def _publish_error_result(self, processing_result: ProcessingResult,
                            toolbox_id: str, tool_id: str,
                            dt_string: str, error_message: str) -> bool:
        """Publish result when processing encountered an error."""
        try:
            # Prepare topics
            keyp_topic = self._build_topic(toolbox_id, tool_id, 'keypoints')
            dest_topic = self._build_topic(toolbox_id, tool_id, 'depth_estimation')
            
            # Prepare error messages
            keyp_data = {
                'Value': f'Error in keypoint estimation: {error_message}',
                'SourceTimestamp': dt_string,
                'MachineId': processing_result.machine_id,
                'ResultId': processing_result.result_id,
                'HeadId': processing_result.head_id,
                'Error': True
            }
            
            dest_data = {
                'Value': f'Error in depth estimation: {error_message}',
                'SourceTimestamp': dt_string,
                'MachineId': processing_result.machine_id,
                'ResultId': processing_result.result_id,
                'HeadId': processing_result.head_id,
                'Error': True
            }
            
            # Publish messages
            keyp_success = self._publish_message(keyp_topic, keyp_data)
            dest_success = self._publish_message(dest_topic, dest_data)
            
            logging.error("Published error result", extra={
                'toolbox_id': toolbox_id,
                'tool_id': tool_id,
                'error_message': error_message,
                'keypoints_published': keyp_success,
                'depth_published': dest_success
            })
            
            return keyp_success and dest_success
            
        except Exception as e:
            logging.error("Error publishing error result", extra={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'toolbox_id': toolbox_id,
                'tool_id': tool_id,
                'original_error': error_message
            })
            return False
    
    def _build_topic(self, toolbox_id: str, tool_id: str, result_type: str) -> str:
        """Build MQTT topic for result publishing."""
        root = self.config['mqtt']['listener']['root']
        endpoint = self.config['mqtt']['estimation'][result_type]
        return f"{root}/{toolbox_id}/{tool_id}/{endpoint}"
    
    def _publish_message(self, topic: str, data: Dict[str, Any]) -> bool:
        """
        Publish a single message to MQTT broker.
        
        Args:
            topic: MQTT topic to publish to
            data: Data to publish
            
        Returns:
            True if publish succeeded, False otherwise
        """
        try:
            payload = json.dumps(data)
            result = self.mqtt_client.publish(topic, payload)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logging.debug("Message published successfully", extra={
                    'topic': topic,
                    'payload_size': len(payload)
                })
                return True
            else:
                logging.warning("MQTT publish failed", extra={
                    'topic': topic,
                    'result_code': result.rc,
                    'payload_size': len(payload)
                })
                return False
                
        except Exception as e:
            logging.error("Error publishing MQTT message", extra={
                'topic': topic,
                'error_type': type(e).__name__,
                'error_message': str(e)
            })
            return False
    
    def publish_custom_result(self, toolbox_id: str, tool_id: str,
                            result_type: str, value: Any,
                            timestamp: Optional[float] = None,
                            additional_fields: Optional[Dict[str, Any]] = None) -> bool:
        """
        Publish custom result data.
        
        Args:
            toolbox_id: Toolbox identifier
            tool_id: Tool identifier
            result_type: Type of result ('keypoints' or 'depth_estimation')
            value: Result value to publish
            timestamp: Optional timestamp (uses current time if None)
            additional_fields: Optional additional fields to include
            
        Returns:
            True if publish succeeded, False otherwise
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
            
            return self._publish_message(topic, data)
            
        except Exception as e:
            logging.error("Error publishing custom result", extra={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'toolbox_id': toolbox_id,
                'tool_id': tool_id,
                'result_type': result_type
            })
            return False
    
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