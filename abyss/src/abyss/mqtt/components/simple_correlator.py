"""
Simple Message Correlator Module - Exact Match Implementation

This implementation uses exact (tool_key, source_timestamp) matching.
No time windows, no fuzzy matching - just pulls completed matches from the buffer.
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Any, Callable, Optional, Union

from ...uos_depth_est import TimestampedData
from .config_manager import ConfigurationManager


class SimpleMessageCorrelator:
    """
    Exact-match correlator. No time windows, no fuzzy matching.
    The buffer already does the matching, we just process the results.
    Maintains same public API for compatibility.
    """
    
    def __init__(self, config: Union[Dict[str, Any], ConfigurationManager], time_window: float = 30.0):
        """
        Initialize SimpleMessageCorrelator.
        
        Args:
            config: Configuration dictionary or ConfigurationManager instance
            time_window: Time window parameter (ignored but kept for API compatibility)
        """
        if isinstance(config, ConfigurationManager):
            self.config_manager = config
            self.config = config.get_raw_config()
        else:
            self.config_manager = None
            self.config = config
        
        # time_window parameter ignored in exact-match mode
        self.time_window = time_window  # Kept for API compatibility
        
        # Metrics
        self._metrics = {
            'matches_processed': 0,
            'processing_errors': 0,
            'total_messages_seen': 0,
            'exact_matches_found': 0
        }
        
        # Always require heads messages for complete match
        listener_config = self._get_listener_config()
        self.require_heads = True  # Always require heads messages
        
        logging.info("SimpleMessageCorrelator initialized with exact matching", extra={
            'require_heads': self.require_heads,
            'correlation_approach': 'exact_match_only'
        })
    
    def find_and_process_matches(self, buffers: Dict[str, List[TimestampedData]], 
                                message_processor: Callable) -> bool:
        """
        Process completed matches from the buffer.
        
        In exact-match mode, the buffer has already done the matching.
        We just need to identify complete groups and process them.
        
        Args:
            buffers: Dictionary of message buffers by topic pattern
            message_processor: Callback function to process matched messages
            
        Returns:
            True if any matches were found and processed
        """
        start_time = time.time()
        matches_found = False
        
        try:
            logging.debug("Starting exact message correlation", extra={
                'buffer_count': len(buffers),
                'total_messages': sum(len(msgs) for msgs in buffers.values())
            })
            
            # Group messages by exact key
            exact_groups = defaultdict(lambda: defaultdict(list))
            
            for topic_pattern, messages in buffers.items():
                self._metrics['total_messages_seen'] += len(messages)
                
                for msg in messages:
                    # Extract exact key components
                    tool_key = self._extract_tool_key(msg.source)
                    source_timestamp = self._extract_source_timestamp(msg)
                    
                    if tool_key and source_timestamp:
                        exact_key = (tool_key, source_timestamp)
                        msg_type = self._get_message_type(msg.source)
                        exact_groups[exact_key][msg_type].append(msg)
                        
                        # Mark as processed so buffer knows we've seen it
                        msg.processed = True
                    else:
                        logging.warning("Failed to extract key components", extra={
                            'source': msg.source,
                            'has_tool_key': tool_key is not None,
                            'has_timestamp': source_timestamp is not None
                        })
            
            # Process complete groups
            for exact_key, msg_types in exact_groups.items():
                if self._is_complete_group(msg_types):
                    try:
                        # Flatten messages for processor
                        messages = []
                        for msg_list in msg_types.values():
                            messages.extend(msg_list)
                        
                        logging.debug("Processing exact match", extra={
                            'tool_key': exact_key[0],
                            'timestamp': exact_key[1],
                            'message_count': len(messages),
                            'result_ids': [
                                m.data.get('ResultId') for m in msg_types.get('result', [])
                            ]
                        })
                        
                        # Call the processor
                        message_processor(messages)
                        
                        self._metrics['matches_processed'] += 1
                        self._metrics['exact_matches_found'] += 1
                        matches_found = True
                        
                    except Exception as e:
                        logging.error("Error processing match", extra={
                            'exact_key': exact_key,
                            'error_type': type(e).__name__,
                            'error_message': str(e)
                        }, exc_info=True)
                        self._metrics['processing_errors'] += 1
                else:
                    # Log incomplete groups for debugging
                    logging.debug("Incomplete message group", extra={
                        'tool_key': exact_key[0],
                        'timestamp': exact_key[1],
                        'has_result': 'result' in msg_types,
                        'has_trace': 'trace' in msg_types,
                        'has_heads': 'heads' in msg_types,
                        'require_heads': self.require_heads
                    })
            
            # Log correlation summary
            processing_time = time.time() - start_time
            logging.debug("Correlation completed", extra={
                'matches_found': matches_found,
                'exact_groups_checked': len(exact_groups),
                'matches_processed': self._metrics['matches_processed'],
                'processing_time_ms': processing_time * 1000
            })
            
            return matches_found
            
        except Exception as e:
            logging.error("Unexpected error in message correlation", extra={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'buffer_count': len(buffers)
            }, exc_info=True)
            return False
    
    def get_correlation_stats(self, buffers: Dict[str, List[TimestampedData]]) -> Dict[str, Any]:
        """
        Get correlation statistics (API compatibility).
        
        Args:
            buffers: Dictionary of message buffers by topic pattern
            
        Returns:
            Dictionary containing correlation statistics
        """
        stats = self._metrics.copy()
        
        # Count messages by state
        total_messages = sum(len(msgs) for msgs in buffers.values())
        processed_messages = sum(
            1 for msgs in buffers.values() 
            for msg in msgs 
            if getattr(msg, 'processed', False)
        )
        
        # Group by exact key to count potential matches
        exact_groups = defaultdict(lambda: defaultdict(int))
        for topic_pattern, messages in buffers.items():
            for msg in messages:
                tool_key = self._extract_tool_key(msg.source)
                source_timestamp = self._extract_source_timestamp(msg)
                
                if tool_key and source_timestamp:
                    exact_key = (tool_key, source_timestamp)
                    msg_type = self._get_message_type(msg.source)
                    exact_groups[exact_key][msg_type] += 1
        
        # Count complete vs incomplete groups
        complete_groups = sum(
            1 for msg_types in exact_groups.values()
            if self._is_complete_group(msg_types)
        )
        
        stats.update({
            'total_messages': total_messages,
            'processed_messages': processed_messages,
            'unprocessed_messages': total_messages - processed_messages,
            'correlation_approach': 'exact_match_only',
            'exact_groups': len(exact_groups),
            'complete_groups': complete_groups,
            'incomplete_groups': len(exact_groups) - complete_groups,
            'time_window': 'N/A (exact match only)'
        })
        
        return stats
    
    # Private helper methods
    
    def _is_complete_group(self, msg_types: Dict[str, List]) -> bool:
        """Check if we have required message types."""
        if self.require_heads:
            return all(msg_types.get(t) for t in ['result', 'trace', 'heads'])
        return 'result' in msg_types and 'trace' in msg_types
    
    def _extract_tool_key(self, source: str) -> Optional[str]:
        """Extract tool key from source topic."""
        try:
            parts = source.split('/')
            if len(parts) >= 3:
                return f"{parts[1]}/{parts[2]}"
        except Exception as e:
            logging.debug(f"Failed to extract tool key from {source}: {e}")
        return None
    
    def _extract_source_timestamp(self, msg: TimestampedData) -> Optional[str]:
        """Extract SourceTimestamp from message data."""
        if isinstance(msg.data, dict):
            return self._find_in_dict(msg.data, 'SourceTimestamp')
        return None
    
    def _find_in_dict(self, data: dict, key: str, depth: int = 0) -> Optional[str]:
        """Recursively search for key in nested dict."""
        if depth > 10:  # Prevent infinite recursion
            return None
            
        for k, v in data.items():
            if k == key and isinstance(v, str):
                return v
            elif isinstance(v, dict):
                # First check if key is directly in this dict
                if key in v and isinstance(v[key], str):
                    return v[key]
                # Otherwise recurse
                result = self._find_in_dict(v, key, depth + 1)
                if result:
                    return result
        return None
    
    def _get_message_type(self, source: str) -> str:
        """Determine message type from source topic."""
        if 'ResultManagement' in source:
            return 'result'
        elif 'Trace' in source:
            return 'trace'
        elif 'AssetManagement' in source or 'Heads' in source:
            return 'heads'
        return 'unknown'
    
    def _get_listener_config(self) -> dict:
        """Get listener configuration."""
        if self.config_manager:
            return self.config_manager.get_mqtt_listener_config()
        else:
            return self.config.get('mqtt', {}).get('listener', {})
    
    # Compatibility methods (for potential future use)
    
    def _check_correlation_health(self, buffers: Dict[str, List[TimestampedData]], 
                                 matches_found: bool, messages_processed: int):
        """
        Check correlation health (kept for API compatibility).
        In exact-match mode, this is mostly informational.
        """
        # Count unprocessed messages
        unprocessed = sum(
            1 for msgs in buffers.values()
            for msg in msgs
            if not getattr(msg, 'processed', False)
        )
        
        if unprocessed > 100:
            logging.warning("High number of unprocessed messages", extra={
                'unprocessed_count': unprocessed,
                'correlation_mode': 'exact_match',
                'possible_cause': 'Messages with non-matching timestamps or missing pairs'
            })