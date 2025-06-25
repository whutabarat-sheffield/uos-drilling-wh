"""
Data Converter Module

Handles conversion of MQTT messages to DataFrame format for analysis.
Extracted from the original MQTTDrillingDataAnalyser class.
"""

import logging
import json
from typing import Dict, Any, Optional
import pandas as pd

from ...uos_depth_est import TimestampedData, MessageProcessingError
from ...uos_depth_est_utils import convert_mqtt_to_df


class DataFrameConverter:
    """
    Converts MQTT messages to pandas DataFrame format for drilling data analysis.
    
    Responsibilities:
    - Convert result and trace messages to DataFrame
    - Enhance DataFrame with heads message data
    - Extract and clean heads data fields
    - Handle conversion errors gracefully
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataFrameConverter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
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
            
            logging.info("Base DataFrame conversion completed", extra={
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