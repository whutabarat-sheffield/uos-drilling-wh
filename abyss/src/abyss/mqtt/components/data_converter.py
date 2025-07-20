"""
Data Converter Module

Handles conversion of MQTT messages to DataFrame format for analysis.
Extracted from the original MQTTDrillingDataAnalyser class.
"""

import logging
import json
import re
from typing import Dict, Any, Optional
import pandas as pd

from ...uos_depth_est import TimestampedData, MessageProcessingError
from ...uos_depth_est_utils import convert_mqtt_to_df
from .config_manager import ConfigurationManager


class DataFrameConverter:
    """
    Converts MQTT messages to pandas DataFrame format for drilling data analysis.
    
    Responsibilities:
    - Convert result and trace messages to DataFrame
    - Enhance DataFrame with heads message data
    - Extract and clean heads data fields
    - Handle conversion errors gracefully
    """
    
    def __init__(self, config: ConfigurationManager):
        """
        Initialize DataFrameConverter.
        
        Args:
            config: ConfigurationManager instance
        """
        self.config_manager = config
        self.config = config.get_raw_config()
    
    def convert_messages_to_df(self, result_msg: TimestampedData, 
                             trace_msg: TimestampedData,
                             heads_msg: Optional[TimestampedData] = None) -> pd.DataFrame:
        """
        Convert MQTT messages to DataFrame, including heads data if available.
        
        Args:
            result_msg: Result message data
            trace_msg: Trace message data
            heads_msg: Optional heads message data
            
        Returns:
            pandas DataFrame with converted message data
            
        Raises:
            MessageProcessingError: If conversion fails
        """
        try:
            logging.debug("Starting message to DataFrame conversion", extra={
                'has_result': result_msg is not None,
                'has_trace': trace_msg is not None,
                'has_heads': heads_msg is not None
            })
            
            # Validate input messages
            if not result_msg or not trace_msg:
                raise MessageProcessingError("Both result and trace messages are required")
            
            if not result_msg.data or not trace_msg.data:
                raise MessageProcessingError("Message data cannot be empty")
            
            # Validate array lengths before conversion
            self._validate_array_lengths(result_msg.data, trace_msg.data)
            
            # Base conversion using existing utility function
            df = convert_mqtt_to_df(
                json.dumps(result_msg.data), 
                json.dumps(trace_msg.data), 
                conf=self.config
            )
            
            if df is None:
                raise MessageProcessingError("DataFrame conversion returned None")
            
            if df.empty:
                raise MessageProcessingError("DataFrame conversion returned empty DataFrame")
            
            logging.debug("Base DataFrame conversion completed", extra={
                'dataframe_shape': df.shape,
                'columns': list(df.columns)
            })
            
            # Enhance with heads data if available
            if heads_msg and heads_msg.data:
                df = self._enhance_with_heads_data(df, heads_msg)
            
            return df
            
        except MessageProcessingError:
            # Re-raise MessageProcessingError as-is
            raise
        except Exception as e:
            logging.error("Error converting messages to DataFrame", extra={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'result_msg_type': type(result_msg).__name__ if result_msg else None,
                'trace_msg_type': type(trace_msg).__name__ if trace_msg else None
            })
            raise MessageProcessingError(f"DataFrame conversion failed: {e}") from e
    
    def _enhance_with_heads_data(self, df: pd.DataFrame, 
                               heads_msg: TimestampedData) -> pd.DataFrame:
        """
        Enhance DataFrame with heads message data.
        
        Args:
            df: Base DataFrame to enhance
            heads_msg: Heads message containing additional data
            
        Returns:
            Enhanced DataFrame with heads data
        """
        try:
            heads_data = self._extract_heads_data(heads_msg.data)
            
            if heads_data:
                original_columns = len(df.columns)
                
                # Add heads data as additional columns
                for key, value in heads_data.items():
                    column_name = f'heads_{key}'
                    
                    # Handle different value types appropriately
                    if isinstance(value, (list, tuple)):
                        # For array-like values, repeat to match DataFrame length
                        if len(value) == len(df):
                            df[column_name] = value
                        else:
                            # If length doesn't match, use first value or convert to string
                            df[column_name] = str(value)
                    else:
                        # For scalar values, broadcast to all rows
                        df[column_name] = value
                
                added_columns = len(df.columns) - original_columns
                logging.info("Enhanced DataFrame with heads data", extra={
                    'original_columns': original_columns,
                    'added_columns': added_columns,
                    'heads_fields': list(heads_data.keys())
                })
            else:
                logging.debug("No usable heads data found for enhancement")
            
            return df
            
        except Exception as e:
            logging.warning("Failed to enhance DataFrame with heads data", extra={
                'error_type': type(e).__name__,
                'error_message': str(e)
            })
            # Return original DataFrame if enhancement fails
            return df
    
    def _extract_heads_data(self, heads_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant data from heads message.
        
        Args:
            heads_data: Raw heads message data
            
        Returns:
            Dictionary of extracted and cleaned heads data
        """
        try:
            extracted = {}
            
            if not isinstance(heads_data, dict):
                logging.warning("Heads data is not a dictionary", extra={
                    'heads_data_type': type(heads_data).__name__
                })
                return extracted
            
            # Extract head_id using configuration path first, then fallback to simple search
            head_id = self._extract_head_id_from_config(heads_data)
            if head_id is None:
                head_id = self._find_in_dict(heads_data, 'head_id')
            
            if head_id is not None:
                extracted['head_id'] = head_id
                logging.debug("Extracted head_id", extra={
                    'head_id': head_id
                })
            
            # Extract from Payload if present
            if 'Payload' in heads_data and isinstance(heads_data['Payload'], dict):
                payload = heads_data['Payload']
                extracted.update(self._extract_from_payload(payload))
            
            # Extract from other common top-level fields
            for field in ['SourceTimestamp', 'Value', 'StatusCode']:
                if field in heads_data:
                    extracted[field.lower()] = heads_data[field]
            
            # Extract from nested structures
            if 'Extensions' in heads_data and isinstance(heads_data['Extensions'], dict):
                extensions = heads_data['Extensions']
                extracted.update(self._extract_from_extensions(extensions))
            
            logging.debug("Extracted heads data", extra={
                'extracted_fields': list(extracted.keys()),
                'field_count': len(extracted)
            })
            
            return extracted
            
        except Exception as e:
            logging.warning("Failed to extract heads data", extra={
                'error_type': type(e).__name__,
                'error_message': str(e)
            })
            return {}
    
    def _find_in_dict(self, data: Any, target_key: str) -> Optional[str]:
        """
        Recursively search for a key in nested dictionary/list structures.
        
        Args:
            data: Data structure to search in
            target_key: Key to search for
            
        Returns:
            First matching value found or None
        """
        try:
            if isinstance(data, dict):
                # Check if target key exists at this level
                if target_key in data:
                    return str(data[target_key])
                
                # Search recursively in all values
                for value in data.values():
                    result = self._find_in_dict(value, target_key)
                    if result is not None:
                        return result
            
            elif isinstance(data, list):
                # Search recursively in all list items
                for item in data:
                    result = self._find_in_dict(item, target_key)
                    if result is not None:
                        return result
            
            return None
            
        except Exception as e:
            logging.debug("Error in recursive search", extra={
                'target_key': target_key,
                'error_message': str(e)
            })
            return None
    
    def _extract_from_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from payload section of heads message."""
        extracted = {}
        
        try:
            for key, value_obj in payload.items():
                # Handle structured value objects
                if isinstance(value_obj, dict):
                    if 'Value' in value_obj:
                        # Clean up the key name for column naming
                        clean_key = self._clean_key_name(key)
                        extracted[clean_key] = value_obj['Value']
                    
                    # Extract other useful fields from value object
                    for sub_field in ['StatusCode', 'SourceTimestamp']:
                        if sub_field in value_obj:
                            clean_key = f"{self._clean_key_name(key)}_{sub_field.lower()}"
                            extracted[clean_key] = value_obj[sub_field]
                else:
                    # Handle direct values
                    clean_key = self._clean_key_name(key)
                    extracted[clean_key] = value_obj
                    
        except Exception as e:
            logging.debug("Error extracting from payload", extra={
                'error_message': str(e)
            })
        
        return extracted
    
    def _extract_from_extensions(self, extensions: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from extensions section of heads message."""
        extracted = {}
        
        try:
            # Handle common extension patterns
            for key, value in extensions.items():
                clean_key = f"ext_{self._clean_key_name(key)}"
                
                if isinstance(value, dict) and 'Value' in value:
                    extracted[clean_key] = value['Value']
                else:
                    extracted[clean_key] = value
                    
        except Exception as e:
            logging.debug("Error extracting from extensions", extra={
                'error_message': str(e)
            })
        
        return extracted
    
    def _clean_key_name(self, key: str) -> str:
        """
        Clean up key names for use as DataFrame column names.
        
        Args:
            key: Raw key name
            
        Returns:
            Cleaned key name suitable for DataFrame columns
        """
        # Take the last part if it's a dotted path
        if '.' in key:
            key = key.split('.')[-1]
        
        # Replace common problematic characters
        key = key.replace(' ', '_').replace('-', '_').replace('/', '_')
        
        # Remove special characters and make lowercase
        import re
        key = re.sub(r'[^a-zA-Z0-9_]', '', key).lower()
        
        # Ensure it doesn't start with a number
        if key and key[0].isdigit():
            key = f"field_{key}"
        
        return key
    
    def _validate_array_lengths(self, result_data: Any, trace_data: Any) -> None:
        """
        Validate that result and trace message arrays have consistent lengths.
        
        Args:
            result_data: Result message data
            trace_data: Trace message data
            
        Raises:
            MessageProcessingError: If array lengths are inconsistent
        """
        try:
            # Parse JSON data if needed
            if isinstance(result_data, str):
                result_data = json.loads(result_data)
            if isinstance(trace_data, str):
                trace_data = json.loads(trace_data)
            
            # Extract payload data
            result_payload = result_data.get("Messages", {}).get("Payload", {})
            trace_payload = trace_data.get("Messages", {}).get("Payload", {})
            
            # Find step arrays in both messages
            result_steps = self._extract_array_from_payload(result_payload, "StepNumber")
            trace_steps = self._extract_array_from_payload(trace_payload, "StepNumber")
            
            # Extract trace arrays
            position_array = self._extract_array_from_payload(trace_payload, "PositionTrace")
            torque_array = self._extract_array_from_payload(trace_payload, "IntensityTorqueTrace")
            thrust_array = self._extract_array_from_payload(trace_payload, "IntensityThrustTrace")
            
            # Check array lengths
            arrays_info = {
                'result_steps': len(result_steps) if result_steps else 0,
                'trace_steps': len(trace_steps) if trace_steps else 0,
                'position': len(position_array) if position_array else 0,
                'torque': len(torque_array) if torque_array else 0,
                'thrust': len(thrust_array) if thrust_array else 0
            }
            
            # Log array lengths for debugging
            logging.debug("Array lengths validation", extra=arrays_info)
            
            # Check if trace arrays have consistent lengths
            trace_arrays = [arr for arr in [trace_steps, position_array, torque_array, thrust_array] if arr]
            if trace_arrays:
                first_length = len(trace_arrays[0])
                for i, arr in enumerate(trace_arrays[1:], 1):
                    if len(arr) != first_length:
                        logging.warning(f"Inconsistent trace array lengths: first={first_length}, array_{i}={len(arr)}")
            
            # Check if we have reasonable data amounts
            if trace_steps and len(trace_steps) == 0:
                raise MessageProcessingError("No trace step data found")
            if result_steps and len(result_steps) == 0:
                raise MessageProcessingError("No result step data found")
            
            # Warn about large array length differences
            if trace_steps and position_array:
                if abs(len(trace_steps) - len(position_array)) > 10:
                    logging.warning(f"Large difference in array lengths: steps={len(trace_steps)}, position={len(position_array)}")
            
        except json.JSONDecodeError as e:
            raise MessageProcessingError(f"Invalid JSON in message data: {e}")
        except Exception as e:
            logging.warning(f"Array length validation warning: {e}")
            # Don't fail conversion for array length issues, just warn
    
    def _extract_array_from_payload(self, payload: Dict[str, Any], search_term: str) -> Optional[list]:
        """
        Extract array data from payload by searching for keys containing the search term.
        
        Args:
            payload: Payload dictionary to search in
            search_term: Term to search for in keys
            
        Returns:
            List of values if found, None otherwise
        """
        try:
            for key, value_obj in payload.items():
                if search_term in key and isinstance(value_obj, dict):
                    if 'Value' in value_obj and isinstance(value_obj['Value'], list):
                        return value_obj['Value']
            return None
        except Exception as e:
            logging.debug(f"Error extracting array for {search_term}: {e}")
            return None
    
    def _extract_head_id_from_config(self, heads_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract head_id using the configured path from data_ids.
        
        Args:
            heads_data: Raw heads message data
            
        Returns:
            Head ID if found, None otherwise
        """
        try:
            # Check if we have a configured path for head_id
            if (self.config and 
                'mqtt' in self.config and 
                'data_ids' in self.config['mqtt'] and 
                'head_id' in self.config['mqtt']['data_ids']):
                
                head_id_path = self.config['mqtt']['data_ids']['head_id']
                return self._extract_value_by_path(heads_data, head_id_path)
            
            return None
            
        except Exception as e:
            logging.debug("Error extracting head_id from config", extra={
                'error_message': str(e)
            })
            return None
    
    def _extract_value_by_path(self, data: Any, path: str) -> Optional[str]:
        """
        Extract a value from nested dictionary using dot notation path.
        
        Args:
            data: Data structure to search in
            path: Dot-separated path (e.g., "AssetManagement.Assets.Heads.0.Identification.SerialNumber")
            
        Returns:
            String value if found, None otherwise
        """
        try:
            if not path or not isinstance(data, dict):
                return None
            
            # Split the path into components
            path_parts = path.split('.')
            current = data
            
            for part in path_parts:
                if not isinstance(current, (dict, list)):
                    return None
                
                # Handle array indices
                if part.isdigit():
                    index = int(part)
                    if isinstance(current, list):
                        if 0 <= index < len(current):
                            current = current[index]
                        else:
                            return None
                    else:
                        return None
                else:
                    # Handle dictionary keys
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        return None
            
            # Convert result to string
            return str(current) if current is not None else None
            
        except (KeyError, IndexError, ValueError, TypeError) as e:
            logging.debug("Error extracting value by path", extra={
                'path': path,
                'error_message': str(e)
            })
            return None
        except Exception as e:
            logging.debug("Unexpected error extracting value by path", extra={
                'path': path,
                'error_type': type(e).__name__,
                'error_message': str(e)
            })
            return None

    def get_conversion_stats(self) -> Dict[str, Any]:
        """
        Get conversion statistics for monitoring.
        
        Returns:
            Dictionary containing conversion statistics
        """
        # This could be extended to track conversion metrics
        return {
            'config_loaded': self.config is not None,
            'converter_ready': True
        }