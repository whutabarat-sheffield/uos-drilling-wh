"""
Message Formatter Module

Handles formatting of processing results into MQTT message payloads.
Extracted from ResultPublisher to separate formatting concerns.
"""

from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from enum import Enum

from .message_processor import ProcessingResult


class ResultType(Enum):
    """Types of results that can be formatted."""
    SUCCESS = "success"
    INSUFFICIENT_DATA = "insufficient_data"
    ERROR = "error"
    WARNING = "warning"  # For cases like negative depths with warning behavior


class ResultMessageFormatter:
    """
    Formats processing results into MQTT message payloads.
    
    Responsibilities:
    - Convert ProcessingResult objects to MQTT payloads
    - Format timestamps consistently
    - Structure messages for different result types
    - Maintain message schema compatibility
    """
    
    def format_result(self, 
                     result: ProcessingResult,
                     timestamp: float,
                     algo_version: str,
                     result_type: ResultType = ResultType.SUCCESS,
                     error_message: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Format a processing result into MQTT message payloads.
        
        Args:
            result: Processing result to format
            timestamp: Unix timestamp
            algo_version: Algorithm version string
            result_type: Type of result being formatted
            error_message: Optional error message for error results
            
        Returns:
            Dictionary mapping topic suffixes to message payloads
            Keys: 'keypoints', 'depth_estimation'
        """
        # Convert timestamp to ISO format
        dt_string = self._format_timestamp(timestamp)
        
        # Build base metadata
        base_data = self._build_base_data(result, dt_string)
        
        # Format based on result type
        if result_type == ResultType.SUCCESS:
            return self._format_success(result, base_data, algo_version)
        elif result_type == ResultType.INSUFFICIENT_DATA:
            return self._format_insufficient_data(base_data)
        elif result_type == ResultType.ERROR:
            return self._format_error(base_data, error_message)
        elif result_type == ResultType.WARNING:
            return self._format_warning(result, base_data, algo_version)
        else:
            raise ValueError(f"Unknown result type: {result_type}")
    
    def _format_timestamp(self, timestamp: float) -> str:
        """Convert Unix timestamp to ISO 8601 format."""
        dt_utc = datetime.fromtimestamp(timestamp)
        return datetime.strftime(dt_utc, "%Y-%m-%dT%H:%M:%SZ")
    
    def _build_base_data(self, result: ProcessingResult, dt_string: str) -> Dict[str, Any]:
        """Build base data structure common to all message types."""
        return {
            'SourceTimestamp': dt_string,
            'MachineId': result.machine_id,
            'ResultId': result.result_id,
            'HeadId': result.head_id
        }
    
    def _format_success(self, 
                       result: ProcessingResult, 
                       base_data: Dict[str, Any],
                       algo_version: str) -> Dict[str, Dict[str, Any]]:
        """Format successful processing result."""
        return {
            'keypoints': {
                **base_data,
                'Value': result.keypoints,
                'AlgoVersion': algo_version
            },
            'depth_estimation': {
                **base_data,
                'Value': result.depth_estimation,
                'AlgoVersion': algo_version
            }
        }
    
    def _format_insufficient_data(self, base_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Format result when insufficient data for processing."""
        return {
            'keypoints': {
                **base_data,
                'Value': 'Not enough steps to estimate keypoints'
            },
            'depth_estimation': {
                **base_data,
                'Value': 'Not enough steps to estimate depth'
            }
        }
    
    def _format_error(self, 
                     base_data: Dict[str, Any],
                     error_message: Optional[str]) -> Dict[str, Dict[str, Any]]:
        """Format error result."""
        error_msg = error_message or "Processing error occurred"
        return {
            'keypoints': {
                **base_data,
                'Value': f'Error in keypoint estimation: {error_msg}',
                'Error': True
            },
            'depth_estimation': {
                **base_data,
                'Value': f'Error in depth estimation: {error_msg}',
                'Error': True
            }
        }
    
    def _format_warning(self,
                       result: ProcessingResult,
                       base_data: Dict[str, Any],
                       algo_version: str) -> Dict[str, Dict[str, Any]]:
        """Format result with warning (e.g., negative depths)."""
        # Similar to success but could add warning flags
        messages = self._format_success(result, base_data, algo_version)
        
        # Add warning indicators
        messages['keypoints']['Warning'] = 'Negative depth values detected'
        messages['depth_estimation']['Warning'] = 'Negative depth values detected'
        
        return messages
    
    def format_custom(self,
                     topic_suffix: str,
                     data: Dict[str, Any],
                     timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Format a custom message payload.
        
        Args:
            topic_suffix: Topic suffix for the message
            data: Custom data to publish
            timestamp: Optional timestamp (uses current time if None)
            
        Returns:
            Formatted message payload
        """
        if timestamp:
            data['SourceTimestamp'] = self._format_timestamp(timestamp)
        
        return {topic_suffix: data}