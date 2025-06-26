"""
Message Processor Module

Handles the processing of matched MQTT messages for drilling data analysis.
Extracted from the original MQTTDrillingDataAnalyser class.
"""

import logging
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

# Import the TimestampedData from the main module
from ...uos_depth_est import TimestampedData, MessageProcessingError
from ...uos_depth_est_utils import reduce_dict


@dataclass
class ProcessingResult:
    """Result of message processing operation"""
    success: bool
    keypoints: Optional[List[float]] = None
    depth_estimation: Optional[List[float]] = None
    machine_id: Optional[str] = None
    result_id: Optional[str] = None
    head_id: Optional[str] = None  # Head identifier from heads message
    error_message: Optional[str] = None


class MessageProcessor:
    """
    Processes matched MQTT messages to perform depth estimation.
    
    Responsibilities:
    - Extract and validate message types (result, trace, heads)
    - Convert messages to DataFrame format
    - Perform depth estimation using ML models
    - Handle insufficient data scenarios
    """
    
    def __init__(self, depth_inference, data_converter, config: Dict[str, Any], algo_version: str = "1.0"):
        """
        Initialize MessageProcessor.
        
        Args:
            depth_inference: Depth inference model instance
            data_converter: Data converter instance
            config: Configuration dictionary
            algo_version: Algorithm version string
        """
        self.depth_inference = depth_inference
        self.data_converter = data_converter
        self.config = config
        self.algo_version = algo_version
        self.machine_id = None
        self.result_id = None
        self.head_id = None
    
    def process_matching_messages(self, matches: List[TimestampedData]) -> ProcessingResult:
        """
        Process messages that have matching timestamps.
        
        Args:
            matches: List of timestamped messages to process
            
        Returns:
            ProcessingResult with processing outcome and results
        """
        try:
            # Separate message types
            result_msg, trace_msg, heads_msg = self._separate_message_types(matches)
            logging.info(f"Heads message is {heads_msg}")
            
            # Extract head_id directly from heads_msg if available
            extracted_head_id = self._extract_head_id_simple(heads_msg)
            if extracted_head_id is not None:
                self.head_id = extracted_head_id
            
            if not result_msg or not trace_msg:
                return ProcessingResult(
                    success=False,
                    head_id=self.head_id,
                    error_message="Missing required result or trace message"
                )
            
            # Extract tool information
            toolbox_id, tool_id = self._extract_tool_info(result_msg)
            
            self._log_processing_info(result_msg, trace_msg, heads_msg, toolbox_id, tool_id)
            
            # Convert messages to DataFrame
            df = self.data_converter.convert_messages_to_df(result_msg, trace_msg, heads_msg)
            
            if df is None or df.empty:
                return ProcessingResult(
                    success=False,
                    head_id=self.head_id,
                    error_message="Failed to convert messages to DataFrame"
                )
            
            # Extract machine and result IDs
            self.machine_id = str(df.iloc[0]['HOLE_ID'])
            self.result_id = str(df.iloc[0]['local'])
            
            logging.info(f"Machine ID: {self.machine_id}")
            logging.info(f"Result ID: {self.result_id}")
            logging.info(f"Head ID: {self.head_id}")
            
            # Debug file output if enabled
            self._write_debug_files(result_msg, trace_msg, heads_msg, toolbox_id, tool_id, df)
            
            # Perform depth estimation
            return self._perform_depth_estimation(df)
            
        except MessageProcessingError as e:
            logging.error("Message processing error", extra={
                'error_message': str(e),
                'error_type': type(e).__name__
            })
            return ProcessingResult(
                success=False, 
                head_id=self.head_id,
                error_message=str(e)
            )
        except Exception as e:
            logging.error("Unexpected error in message processing", extra={
                'error_type': type(e).__name__,
                'error_message': str(e)
            }, exc_info=True)
            return ProcessingResult(
                success=False, 
                head_id=self.head_id,
                error_message=str(e)
            )
    
    def _separate_message_types(self, matches: List[TimestampedData]) -> tuple:
        """Separate messages by type (result, trace, heads)."""
        result_msg = next((m for m in matches if self.config['mqtt']['listener']['result'] in m.source), None)
        trace_msg = next((m for m in matches if self.config['mqtt']['listener']['trace'] in m.source), None)
        heads_msg = next((m for m in matches if self.config['mqtt']['listener']['heads'] in m.source), None)
        
        return result_msg, trace_msg, heads_msg
    
    def _extract_tool_info(self, result_msg: TimestampedData) -> tuple:
        """Extract toolbox and tool IDs from message source."""
        parts = result_msg.source.split('/')
        if len(parts) < 3:
            raise MessageProcessingError("Invalid message source format")
        
        return parts[1], parts[2]  # toolbox_id, tool_id
    
    def _log_processing_info(self, result_msg, trace_msg, heads_msg, toolbox_id, tool_id):
        """Log processing information."""
        logging.info("Processing matched messages", extra={
            'toolbox_id': toolbox_id,
            'tool_id': tool_id,
            'timestamp': datetime.fromtimestamp(result_msg.timestamp),
            'time_difference': abs(result_msg.timestamp - trace_msg.timestamp),
            'has_heads_message': heads_msg is not None
        })
    
    def _write_debug_files(self, result_msg, trace_msg, heads_msg, toolbox_id, tool_id, df):
        """Write debug files if debug logging is enabled."""
        if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
            timestamp_str = str(result_msg.timestamp)
            
            # Write debug text file
            debug_filename = f"{timestamp_str}.txt"
            with open(debug_filename, 'w') as file:
                file.write(f"Toolbox ID: {toolbox_id}\n")
                file.write(f"Tool ID: {tool_id}\n")
                file.write(f"Timestamp: {datetime.fromtimestamp(result_msg.timestamp)}\n\n")
                file.write(f"RESULT:\n\n{result_msg.data}\n\n")
                file.write(f"TRACE:\n\n{trace_msg.data}\n\n")
                if heads_msg:
                    file.write(f"HEADS:\n\n{heads_msg.data}\n\n")
            
            logging.debug(f"Stored matched data into {debug_filename}")
            
            # Write DataFrame to CSV
            csv_filename = f"{timestamp_str}.csv"
            df.to_csv(csv_filename, index=False)
            logging.debug(f"Stored matched dataframe into {csv_filename}")
    
    def _perform_depth_estimation(self, df) -> ProcessingResult:
        """Perform depth estimation on the DataFrame."""
        # Check if we have enough steps
        if 'Step (nb)' not in df.columns or len(df['Step (nb)'].unique()) < 2:
            logging.warning("Insufficient steps for depth estimation", extra={
                'available_steps': len(df['Step (nb)'].unique()) if 'Step (nb)' in df.columns else 0,
                'required_steps': 2
            })
            return ProcessingResult(
                success=True,
                keypoints=None,
                depth_estimation=None,
                machine_id=self.machine_id,
                result_id=self.result_id,
                head_id=self.head_id,
                error_message="Not enough steps to estimate depth"
            )
        
        # Perform depth estimation
        try:
            # Use 3-point estimation method
            keypoints = self.depth_inference.infer3_common(df)
            depth_estimation = [keypoints[i+1] - keypoints[i] for i in range(len(keypoints)-1)]
            
            logging.info("Depth estimation completed", extra={
                'keypoints': keypoints,
                'depth_estimation': depth_estimation,
                'machine_id': self.machine_id,
                'result_id': self.result_id
            })
            
            return ProcessingResult(
                success=True,
                keypoints=keypoints,
                depth_estimation=depth_estimation,
                machine_id=self.machine_id,
                result_id=self.result_id,
                head_id=self.head_id
            )
            
        except Exception as e:
            logging.error("Depth estimation failed", extra={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'machine_id': self.machine_id,
                'result_id': self.result_id
            })
            return ProcessingResult(
                success=False,
                machine_id=self.machine_id,
                result_id=self.result_id,
                head_id=self.head_id,
                error_message=f"Depth estimation failed: {str(e)}"
            )
    
    def _extract_head_id_simple(self, heads_msg: Optional[TimestampedData]) -> Optional[str]:
        """
        Extract head_id from heads message using find_in_dict.
        
        Args:
            heads_msg: Optional heads message data
            
        Returns:
            Extracted head_id or None if not available
        """
        try:
            if not heads_msg or not heads_msg.data:
                return None
            
            # Handle both string and dict data
            if isinstance(heads_msg.data, str):
                heads_data = json.loads(heads_msg.data)
            else:
                heads_data = heads_msg.data
            
            # Use find_in_dict to extract head_id using the last part of the config path
            head_id_path = self.config['mqtt']['data_ids']['head_id']
            # Extract the final key from the path: 'AssetManagement.Assets.Heads.0.Identification.SerialNumber' -> 'SerialNumber'
            heads_data = heads_data["Messages"]["Payload"]
            head_id = reduce_dict(heads_data, head_id_path)
            
            # Ensure we return a string or None
            if head_id is not None:
                return str(head_id) if not isinstance(head_id, list) else str(head_id[0]) if head_id else None
            return None
            
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            return None