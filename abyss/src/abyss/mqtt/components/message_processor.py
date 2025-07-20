"""
Message Processor Module

Handles the processing of matched MQTT messages for drilling data analysis.
Extracted from the original MQTTDrillingDataAnalyser class.
"""

import logging
import json
import time
from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
import pandas as pd

# Import the TimestampedData from the main module
from ...uos_depth_est import TimestampedData, MessageProcessingError
from ...uos_depth_est_utils import reduce_dict
from .config_manager import ConfigurationManager


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
    
    def __init__(self, depth_inference, data_converter, config: Union[Dict[str, Any], ConfigurationManager], algo_version: str = "1.0"):
        """
        Initialize MessageProcessor.
        
        Args:
            depth_inference: Depth inference model instance
            data_converter: Data converter instance
            config: Configuration dictionary or ConfigurationManager instance
            algo_version: Algorithm version string
        """
        self.depth_inference = depth_inference
        self.data_converter = data_converter
        self.algo_version = algo_version
        self.machine_id = None
        self.result_id = None
        self.head_id = None
        
        # Handle both ConfigurationManager and raw config dict for backward compatibility
        if isinstance(config, ConfigurationManager):
            self.config_manager = config
            self.config = config.get_raw_config()
        else:
            # Legacy support for raw config dictionary
            self.config_manager = None
            self.config = config
        
        # Error tracking for warnings
        self._processing_failures = defaultdict(int)  # tool_key -> failure count
        self._validation_failures = defaultdict(int)  # error_type -> count
        self._last_warning_time = {}  # warning_type -> timestamp
    
    def process_matching_messages(self, matches: List[TimestampedData]) -> ProcessingResult:
        """
        Process messages that have matching timestamps.
        
        Args:
            matches: List of timestamped messages to process
            
        Returns:
            ProcessingResult with processing outcome and results
        """
        try:
            # Reset state at start of processing cycle
            self.head_id = None
            self.machine_id = None
            self.result_id = None
            
            # Separate message types
            result_msg, trace_msg, heads_msg = self._separate_message_types(matches)
            logging.debug(f"Heads message is {heads_msg}")
            
            # Extract head_id directly from heads_msg if available
            self.head_id = self._extract_head_id(heads_msg)
            
            # Enhanced message validation
            validation_error = self._validate_messages(result_msg, trace_msg)
            if validation_error:
                self._track_validation_failure(validation_error)
                return ProcessingResult(
                    success=False,
                    head_id=self.head_id,
                    error_message=validation_error
                )
            
            # Extract tool information
            toolbox_id, tool_id = self._extract_tool_info(result_msg)
            
            self._log_processing_info(result_msg, trace_msg, heads_msg, toolbox_id, tool_id)
            
            # Convert messages to DataFrame
            try:
                df = self.data_converter.convert_messages_to_df(result_msg, trace_msg, heads_msg)
            except Exception as e:
                return ProcessingResult(
                    success=False,
                    head_id=self.head_id,
                    error_message=f"DataFrame conversion failed: {str(e)}"
                )
            
            # Enhanced DataFrame validation
            validation_error = self._validate_dataframe(df)
            if validation_error:
                return ProcessingResult(
                    success=False,
                    head_id=self.head_id,
                    error_message=validation_error
                )
            
            # Extract machine and result IDs with defensive programming
            try:
                self.machine_id = str(df.iloc[0]['HOLE_ID'])
                self.result_id = str(df.iloc[0]['local'])
            except (KeyError, IndexError) as e:
                return ProcessingResult(
                    success=False,
                    head_id=self.head_id,
                    error_message=f"Failed to extract IDs from DataFrame: {str(e)}"
                )
            
            logging.debug(f"Machine ID: {self.machine_id}, Result ID: {self.result_id}, Head ID: {self.head_id}")
            
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
        if self.config_manager:
            # Use ConfigurationManager for typed access
            listener_config = self.config_manager.get_mqtt_listener_config()
        else:
            # Legacy raw config access
            listener_config = self.config['mqtt']['listener']
            
        result_msg = next((m for m in matches if listener_config['result'] in m.source), None)
        trace_msg = next((m for m in matches if listener_config['trace'] in m.source), None)
        heads_msg = next((m for m in matches if listener_config['heads'] in m.source), None)
        
        return result_msg, trace_msg, heads_msg
    
    def _extract_tool_info(self, result_msg: TimestampedData) -> tuple:
        """Extract toolbox and tool IDs from message source with enhanced validation."""
        if not result_msg or not result_msg.source:
            raise MessageProcessingError("Message source is empty or None")
        
        parts = result_msg.source.split('/')
        if len(parts) < 3:
            raise MessageProcessingError(f"Invalid message source format: '{result_msg.source}'. Expected format: 'prefix/toolbox_id/tool_id/...'")
        
        toolbox_id = parts[1].strip()
        tool_id = parts[2].strip()
        
        if not toolbox_id or not tool_id:
            raise MessageProcessingError(f"Empty toolbox_id or tool_id in source: '{result_msg.source}'")
        
        return toolbox_id, tool_id
    
    def _log_processing_info(self, result_msg, trace_msg, heads_msg, toolbox_id, tool_id):
        """Log processing information."""
        logging.debug("Processing matched messages", extra={
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
            
            # Format depth estimation results for display
            keypoints_str = [f"{kp:.3f}" for kp in keypoints]
            depth_str = [f"{d:.3f}" for d in depth_estimation]
            
            logging.info(
                f"Depth estimation completed - "
                f"Machine ID: {self.machine_id}, "
                f"Result ID: {self.result_id} - "
                f"Keypoints: {keypoints_str}, "
                f"Depths: {depth_str} mm", 
                extra={
                    'keypoints': keypoints,
                    'depth_estimation': depth_estimation,
                    'machine_id': self.machine_id,
                    'result_id': self.result_id,
                    'head_id': self.head_id
                }
            )
            
            # Reset failure count on successful estimation
            tool_key = self._get_tool_key_from_machine_id(self.machine_id)
            if tool_key in self._processing_failures:
                self._processing_failures[tool_key] = 0
            
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
            
            # Track failure for warning detection
            tool_key = self._get_tool_key_from_machine_id(self.machine_id)
            if tool_key:
                self._processing_failures[tool_key] += 1
                self._check_processing_failures(tool_key)
            
            return ProcessingResult(
                success=False,
                machine_id=self.machine_id,
                result_id=self.result_id,
                head_id=self.head_id,
                error_message=f"Depth estimation failed: {str(e)}"
            )
    
    def _extract_head_id(self, heads_msg: Optional[TimestampedData]) -> Optional[str]:
        """
        Extract head_id from heads message using reduce_dict with full config path.
        
        Args:
            heads_msg: Optional heads message data
            
        Returns:
            Extracted head_id as string or None if not available
        """
        if not heads_msg or not heads_msg.data:
            logging.debug("Heads message is None or empty")
            return None
        
        try:
            # Parse JSON data if it's a string
            if isinstance(heads_msg.data, str):
                logging.debug("Parsing heads message JSON data")
                heads_data = json.loads(heads_msg.data)
            else:
                logging.debug("Using heads message dictionary data")
                heads_data = heads_msg.data
            
            # Extract payload data
            if "Messages" not in heads_data or "Payload" not in heads_data["Messages"]:
                logging.warning("Heads message missing required Messages.Payload structure")
                return None
            
            payload_data = heads_data["Messages"]["Payload"]
            
            # Use full config path with reduce_dict
            if self.config_manager:
                # Use ConfigurationManager for typed access
                head_id_path = self.config_manager.get('mqtt.data_ids.head_id')
            else:
                # Legacy raw config access
                head_id_path = self.config['mqtt']['data_ids']['head_id']
            logging.debug(f"Extracting head_id using path: {head_id_path}")
            
            head_id = reduce_dict(payload_data, head_id_path)
            
            # Validate and normalize the result
            return self._validate_and_normalize_head_id(head_id)
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse heads message JSON: {e}")
            return None
        except KeyError as e:
            logging.error(f"Missing required key in heads message: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error extracting head_id: {e}")
            return None
    
    def _validate_and_normalize_head_id(self, head_id: Any) -> Optional[str]:
        """
        Validate and normalize head_id to ensure consistent string output.
        
        Args:
            head_id: Raw head_id value from message
            
        Returns:
            Normalized head_id as string or None if invalid
        """
        if head_id is None:
            logging.info("Head ID not found in heads message")
            return None
        
        # Handle list values - take first non-empty element
        if isinstance(head_id, list):
            if not head_id:
                logging.warning("Head ID list is empty")
                return None
            head_id = head_id[0]
        
        # Convert to string and validate
        try:
            head_id_str = str(head_id).strip()
            if not head_id_str:
                logging.warning("Head ID is empty after normalization")
                return None
            
            # Basic validation - ensure it's reasonable length and format
            if len(head_id_str) > 100:
                logging.warning(f"Head ID unusually long ({len(head_id_str)} chars): {head_id_str[:50]}...")
            
            logging.debug(f"Extracted and validated head_id: {head_id_str}")
            return head_id_str
            
        except Exception as e:
            logging.error(f"Failed to normalize head_id {head_id}: {e}")
            return None
    
    def _validate_messages(self, result_msg: Optional[TimestampedData], trace_msg: Optional[TimestampedData]) -> Optional[str]:
        """
        Validate message structure and content before processing.
        
        Args:
            result_msg: Result message to validate
            trace_msg: Trace message to validate
            
        Returns:
            Error message if validation fails, None if valid
        """
        # Check message presence
        if not result_msg:
            return "Missing required result message"
        if not trace_msg:
            return "Missing required trace message"
        
        # Check message content
        if not result_msg.data:
            return "Result message has no data content"
        if not trace_msg.data:
            return "Trace message has no data content"
        
        # Validate JSON structure for result message
        try:
            if isinstance(result_msg.data, str):
                result_data = json.loads(result_msg.data)
            else:
                result_data = result_msg.data
            
            if not isinstance(result_data, dict):
                return "Result message data is not a valid dictionary"
            
            # Check for required structure
            if "Messages" not in result_data or "Payload" not in result_data["Messages"]:
                return "Result message missing required Messages.Payload structure"
                
        except json.JSONDecodeError as e:
            return f"Invalid JSON in result message: {e}"
        except Exception as e:
            return f"Error validating result message structure: {e}"
        
        # Validate JSON structure for trace message
        try:
            if isinstance(trace_msg.data, str):
                trace_data = json.loads(trace_msg.data)
            else:
                trace_data = trace_msg.data
            
            if not isinstance(trace_data, dict):
                return "Trace message data is not a valid dictionary"
            
            # Check for required structure
            if "Messages" not in trace_data or "Payload" not in trace_data["Messages"]:
                return "Trace message missing required Messages.Payload structure"
                
        except json.JSONDecodeError as e:
            return f"Invalid JSON in trace message: {e}"
        except Exception as e:
            return f"Error validating trace message structure: {e}"
        
        # Check timestamp consistency
        time_diff = abs(result_msg.timestamp - trace_msg.timestamp)
        max_time_diff = 30.0  # 30 seconds tolerance
        if time_diff > max_time_diff:
            logging.warning(f"Large time difference between result and trace messages: {time_diff} seconds")
        
        return None
    
    def _validate_dataframe(self, df: Optional[pd.DataFrame]) -> Optional[str]:
        """
        Validate DataFrame structure and content.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Error message if validation fails, None if valid
        """
        if df is None:
            return "DataFrame is None"
        
        if df.empty:
            return "DataFrame is empty"
        
        # Check for required columns
        required_columns = ['HOLE_ID', 'local']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return f"DataFrame missing required columns: {missing_columns}"
        
        # Check for data in required columns
        try:
            if df.iloc[0]['HOLE_ID'] is None or pd.isna(df.iloc[0]['HOLE_ID']):
                return "HOLE_ID is None or NaN"
            if df.iloc[0]['local'] is None or pd.isna(df.iloc[0]['local']):
                return "local (result_id) is None or NaN"
        except IndexError:
            return "DataFrame has no rows"
        
        # Validate step data consistency if available
        if 'Step (nb)' in df.columns:
            unique_steps = df['Step (nb)'].unique()
            if len(unique_steps) < 1:
                return "No valid step data found"
            
            # Check for reasonable step values
            if any(step < 0 for step in unique_steps if pd.notna(step)):
                logging.warning("Found negative step values in DataFrame")
        
        return None
    
    def _get_tool_key_from_machine_id(self, machine_id: str) -> Optional[str]:
        """Extract tool key from machine ID for tracking."""
        try:
            if machine_id and '/' in machine_id:
                parts = machine_id.split('/')
                if len(parts) >= 2:
                    return f"{parts[0]}/{parts[1]}"
        except Exception:
            pass
        return None
    
    def _track_validation_failure(self, error_message: str):
        """Track validation failures and warn if rate is too high."""
        # Categorize error type
        error_type = 'unknown'
        if 'Missing required' in error_message:
            error_type = 'missing_message'
        elif 'no data content' in error_message:
            error_type = 'empty_data'
        elif 'Invalid JSON' in error_message:
            error_type = 'invalid_json'
        elif 'missing required Messages.Payload' in error_message:
            error_type = 'invalid_structure'
        
        self._validation_failures[error_type] += 1
        
        # Check if we should warn
        current_time = time.time()
        last_check = self._last_warning_time.get('validation_check', 0)
        
        if current_time - last_check >= 300:  # Check every 5 minutes
            self._last_warning_time['validation_check'] = current_time
            
            total_failures = sum(self._validation_failures.values())
            if total_failures > 20:
                self._log_rate_limited_warning('high_validation_failures', 600,
                    "High rate of message validation failures", {
                        'total_failures': total_failures,
                        'failure_breakdown': dict(self._validation_failures),
                        'time_window_minutes': 5,
                        'possible_cause': 'Schema changes, malformed messages, or data quality issues'
                    })
            
            # Reset counters after check
            self._validation_failures.clear()
    
    def _check_processing_failures(self, tool_key: str):
        """Check and warn about repeated processing failures for a tool."""
        failure_count = self._processing_failures[tool_key]
        
        if failure_count >= 5:  # Warn after 5 consecutive failures
            self._log_rate_limited_warning(f'processing_failure_{tool_key}', 600,
                "Repeated depth estimation failures for tool", {
                    'tool_key': tool_key,
                    'consecutive_failures': failure_count,
                    'possible_cause': 'Insufficient data, corrupted messages, or algorithm issues'
                })
    
    def _log_rate_limited_warning(self, warning_type: str, min_interval: int, 
                                 message: str, extra: Dict[str, Any]):
        """Log warning with rate limiting to avoid spam."""
        current_time = time.time()
        last_warning = self._last_warning_time.get(warning_type, 0)
        
        if current_time - last_warning >= min_interval:
            logging.warning(message, extra=extra)
            self._last_warning_time[warning_type] = current_time