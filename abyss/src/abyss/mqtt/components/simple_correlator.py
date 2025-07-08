"""
Simple Message Correlator Module

Handles correlation and matching of MQTT messages based on tool keys and timestamps.
Simplified approach replacing complex time-bucket grouping with direct key matching.
"""

import logging
from typing import Dict, List, Set, Tuple, Any, Callable, Optional, Union

from ...uos_depth_est import TimestampedData
from .config_manager import ConfigurationManager


class SimpleMessageCorrelator:
    """
    Simplified correlator that matches MQTT messages based on tool keys and timestamps.
    
    Responsibilities:
    - Match messages by tool keys (toolbox_id/tool_id)
    - Simple timestamp-based correlation within time window
    - Handle message processing and cleanup
    - Provide correlation statistics
    """
    
    def __init__(self, config: Union[Dict[str, Any], ConfigurationManager], time_window: float = 30.0):
        """
        Initialize SimpleMessageCorrelator.
        
        Args:
            config: Configuration dictionary or ConfigurationManager instance
            time_window: Time window for message correlation (seconds)
        """
        if isinstance(config, ConfigurationManager):
            self.config_manager = config
            self.time_window = time_window if time_window != 30.0 else config.get_time_window()
            # Get topic patterns from ConfigurationManager
            self._topic_patterns = config.get_topic_patterns()
        else:
            # Legacy support
            self.config_manager = None
            self.config = config
            self.time_window = time_window
            
            # Pre-compute topic patterns (legacy method)
            listener_config = self.config['mqtt']['listener']
            self._topic_patterns = {
                'result': f"{listener_config['root']}/+/+/{listener_config['result']}",
                'trace': f"{listener_config['root']}/+/+/{listener_config['trace']}"
            }
            
            # Add heads pattern only if configured
            if 'heads' in listener_config:
                self._topic_patterns['heads'] = f"{listener_config['root']}/+/+/{listener_config['heads']}"
                logging.debug("Heads topic configured, adding pattern")
                logging.debug(f"Topic patterns: {self._topic_patterns['heads']}")
    
    def find_and_process_matches(self, buffers: Dict[str, List[TimestampedData]], 
                                message_processor: Callable) -> bool:
        """
        Find and process messages with matching tool keys and timestamps.
        
        Args:
            buffers: Dictionary of message buffers by topic
            message_processor: Callback function to process matched messages
            
        Returns:
            True if any matches were found and processed, False otherwise
        """
        try:
            logging.debug("Starting simple message correlation")
            
            # Get unprocessed messages by type
            result_messages, trace_messages, heads_messages = self._get_unprocessed_messages(buffers)
            
            logging.debug("Message counts for correlation", extra={
                'result_messages': len(result_messages),
                'trace_messages': len(trace_messages),
                'heads_messages': len(heads_messages)
            })
            
            if not result_messages or not trace_messages:
                logging.debug("Insufficient messages for correlation")
                return False
            
            # Find matches using simple key-based approach
            matches_found, processed_messages = self._find_key_based_matches(
                result_messages, trace_messages, heads_messages, message_processor
            )
            
            # Mark processed messages
            self._mark_messages_as_processed(processed_messages)
            
            logging.info("Correlation process completed", extra={
                'matches_found': matches_found,
                'messages_processed': len(processed_messages)
            })
            
            return matches_found
            
        except Exception as e:
            logging.error("Unexpected error in simple message correlation", extra={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'buffer_sizes': {k: len(v) for k, v in buffers.items()}
            }, exc_info=True)
            return False
    
    def _get_unprocessed_messages(self, buffers: Dict[str, List[TimestampedData]]) -> Tuple[List, List, List]:
        """Get unprocessed messages from buffers by type."""
        # Count duplicates for logging
        result_total = len(buffers.get(self._topic_patterns['result'], []))
        result_messages = [
            msg for msg in buffers.get(self._topic_patterns['result'], []) 
            if not getattr(msg, 'processed', False)
        ]
        result_duplicates = result_total - len(result_messages)
        
        trace_total = len(buffers.get(self._topic_patterns['trace'], []))
        trace_messages = [
            msg for msg in buffers.get(self._topic_patterns['trace'], []) 
            if not getattr(msg, 'processed', False)
        ]
        trace_duplicates = trace_total - len(trace_messages)
        
        heads_messages = []
        heads_duplicates = 0
        if 'heads' in self._topic_patterns:
            heads_total = len(buffers.get(self._topic_patterns['heads'], []))
            heads_messages = [
                msg for msg in buffers.get(self._topic_patterns['heads'], []) 
                if not getattr(msg, 'processed', False)
            ]
            heads_duplicates = heads_total - len(heads_messages)
        
        # Log duplicate detection details
        if result_duplicates > 0 or trace_duplicates > 0 or heads_duplicates > 0:
            self._log_duplicate_messages(buffers, result_duplicates, trace_duplicates, heads_duplicates)
        
        return result_messages, trace_messages, heads_messages
    
    def _log_duplicate_messages(self, buffers: Dict[str, List[TimestampedData]], 
                               result_duplicates: int, trace_duplicates: int, heads_duplicates: int):
        """Log details about duplicate messages being ignored."""
        total_duplicates = result_duplicates + trace_duplicates + heads_duplicates
        logging.info(f"Duplicate messages detected and ignored [result_duplicates={result_duplicates}, trace_duplicates={trace_duplicates}, heads_duplicates={heads_duplicates}, total_duplicates={total_duplicates}]")
        
        # Log specific details about each duplicate message type
        if result_duplicates > 0:
            processed_results = [
                msg for msg in buffers.get(self._topic_patterns['result'], []) 
                if getattr(msg, 'processed', False)
            ]
            for msg in processed_results:
                tool_key = self._extract_tool_key(msg.source)
                logging.info(f"Duplicate RESULT message ignored [message_type=result, source_topic={msg.source}, tool_key={tool_key}]")
        
        if trace_duplicates > 0:
            processed_traces = [
                msg for msg in buffers.get(self._topic_patterns['trace'], []) 
                if getattr(msg, 'processed', False)
            ]
            for msg in processed_traces:
                tool_key = self._extract_tool_key(msg.source)
                logging.info(f"Duplicate TRACE message ignored [message_type=trace, source_topic={msg.source}, tool_key={tool_key}]")
        
        if heads_duplicates > 0:
            processed_heads = [
                msg for msg in buffers.get(self._topic_patterns['heads'], []) 
                if getattr(msg, 'processed', False)
            ]
            for msg in processed_heads:
                tool_key = self._extract_tool_key(msg.source)
                logging.info(f"Duplicate HEADS message ignored [message_type=heads, source_topic={msg.source}, tool_key={tool_key}]")
    
    def _find_key_based_matches(self, result_messages: List[TimestampedData],
                               trace_messages: List[TimestampedData],
                               heads_messages: List[TimestampedData],
                               message_processor: Callable) -> Tuple[bool, Set]:
        """
        Find matches using simple key-based approach.
        
        This is much simpler than the time-bucket approach:
        1. Group messages by tool key
        2. For each tool key, match messages within time window
        3. Process matches immediately
        """
        matches_found = False
        processed_messages = set()
        
        # Group messages by tool key
        result_by_key = self._group_by_tool_key(result_messages)
        trace_by_key = self._group_by_tool_key(trace_messages)
        heads_by_key = self._group_by_tool_key(heads_messages)
        
        # Find matches for each tool key
        for tool_key in result_by_key.keys():
            if tool_key in trace_by_key:
                key_matches = self._process_key_matches(
                    tool_key,
                    result_by_key[tool_key],
                    trace_by_key[tool_key],
                    heads_by_key.get(tool_key, []),
                    processed_messages,
                    message_processor
                )
                matches_found = matches_found or key_matches
        
        return matches_found, processed_messages
    
    def _group_by_tool_key(self, messages: List[TimestampedData]) -> Dict[str, List[TimestampedData]]:
        """Group messages by tool key (toolbox_id/tool_id)."""
        grouped = {}
        for msg in messages:
            tool_key = self._extract_tool_key(msg.source)
            if tool_key:
                if tool_key not in grouped:
                    grouped[tool_key] = []
                grouped[tool_key].append(msg)
        return grouped
    
    def _process_key_matches(self, tool_key: str,
                           result_msgs: List[TimestampedData],
                           trace_msgs: List[TimestampedData],
                           heads_msgs: List[TimestampedData],
                           processed_messages: Set[TimestampedData],
                           message_processor: Callable) -> bool:
        """Process matches for a specific tool key."""
        matches_found = False
        
        # For each result message, find matching trace (and heads) within time window
        for result_msg in result_msgs:
            if result_msg in processed_messages:
                continue
                
            # Find matching trace message within time window
            for trace_msg in trace_msgs:
                if (trace_msg not in processed_messages and
                    abs(result_msg.timestamp - trace_msg.timestamp) <= self.time_window):
                    
                    # Find matching heads message (optional)
                    heads_msg = self._find_matching_heads_message(
                        result_msg, heads_msgs, processed_messages
                    )
                    
                    # Create match group
                    match_group = [result_msg, trace_msg]
                    if heads_msg:
                        match_group.append(heads_msg)
                    
                    try:
                        logging.info("Found matching message pair", extra={
                            'tool_key': tool_key,
                            'result_timestamp': result_msg.timestamp,
                            'trace_timestamp': trace_msg.timestamp,
                            'time_difference': abs(result_msg.timestamp - trace_msg.timestamp),
                            'has_heads_message': heads_msg is not None
                        })
                        
                        # Process the match
                        message_processor(match_group)
                        matches_found = True
                        
                        # Mark messages as processed
                        processed_messages.add(result_msg)
                        processed_messages.add(trace_msg)
                        if heads_msg:
                            processed_messages.add(heads_msg)
                        
                    except Exception as e:
                        logging.error("Error processing message match", extra={
                            'tool_key': tool_key,
                            'error_type': type(e).__name__,
                            'error_message': str(e)
                        })
                    
                    break  # Only match each result message once
        
        return matches_found
    
    def _find_matching_heads_message(self, result_msg: TimestampedData,
                                   heads_msgs: List[TimestampedData],
                                   processed_messages: Set[TimestampedData]) -> Optional[TimestampedData]:
        """Find heads message that matches the result message timestamp."""
        for heads_msg in heads_msgs:
            if (heads_msg not in processed_messages and
                abs(result_msg.timestamp - heads_msg.timestamp) <= self.time_window):
                return heads_msg
        return None
    
    def _extract_tool_key(self, source: str) -> Optional[str]:
        """Extract tool key from message source topic."""
        try:
            parts = source.split('/')
            if len(parts) >= 3:
                return f"{parts[1]}/{parts[2]}"  # toolbox_id/tool_id
        except Exception as e:
            logging.warning("Failed to extract tool key", extra={
                'source': source,
                'error_message': str(e)
            })
        return None
    
    def _mark_messages_as_processed(self, processed_messages: Set[TimestampedData]):
        """Mark messages as processed to avoid reprocessing."""
        for msg in processed_messages:
            msg.processed = True
    
    def get_correlation_stats(self, buffers: Dict[str, List[TimestampedData]]) -> Dict[str, Any]:
        """
        Get correlation statistics for monitoring and debugging.
        
        Args:
            buffers: Dictionary of message buffers by topic
            
        Returns:
            Dictionary containing correlation statistics
        """
        total_messages = sum(len(buffer) for buffer in buffers.values())
        unprocessed_messages = {}
        
        for pattern, topic in self._topic_patterns.items():
            buffer = buffers.get(topic, [])
            unprocessed = len([msg for msg in buffer if not getattr(msg, 'processed', False)])
            unprocessed_messages[pattern] = unprocessed
        
        return {
            'total_messages': total_messages,
            'unprocessed_messages': unprocessed_messages,
            'time_window': self.time_window,
            'correlation_approach': 'simple_key_based'
        }