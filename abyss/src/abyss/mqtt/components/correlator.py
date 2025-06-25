"""
Message Correlator Module

Handles correlation and matching of MQTT messages based on timestamps and tool IDs.
Extracted from the original MQTTDrillingDataAnalyser class.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Callable

from ...uos_depth_est import TimestampedData


class MessageCorrelator:
    """
    Correlates MQTT messages based on timestamps and tool identifiers.
    
    Responsibilities:
    - Group messages by timestamp buckets
    - Match messages by tool keys within time windows
    - Handle message processing and cleanup
    - Provide correlation statistics
    """
    
    def __init__(self, config: Dict[str, Any], time_window: float = 30.0):
        """
        Initialize MessageCorrelator.
        
        Args:
            config: Configuration dictionary
            time_window: Time window for message correlation (seconds)
        """
        self.config = config
        self.time_window = time_window
        
        # Pre-compute topic patterns
        listener_config = self.config['mqtt']['listener']
        self._topic_patterns = {
            'result': f"{listener_config['root']}/+/+/{listener_config['result']}",
            'trace': f"{listener_config['root']}/+/+/{listener_config['trace']}"
        }
        
        # Add heads pattern only if configured
        if 'heads' in listener_config:
            self._topic_patterns['heads'] = f"{listener_config['root']}/+/+/{listener_config['heads']}"
    
    def find_and_process_matches(self, buffers: Dict[str, List[TimestampedData]], 
                                message_processor: Callable) -> bool:
        """
        Find and process messages with matching timestamps and tool IDs.
        
        Args:
            buffers: Dictionary of message buffers by topic
            message_processor: Callback function to process matched messages
            
        Returns:
            True if any matches were found and processed, False otherwise
        """
        try:
            logging.debug("Starting message correlation process")
            
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
            
            # Group messages by timestamp buckets
            result_by_time, trace_by_time, heads_by_time = self._group_messages_by_time_bucket(
                result_messages, trace_messages, heads_messages
            )
            
            # Find and process matches
            matches_found, processed_messages = self._find_matches_in_time_buckets(
                result_by_time, trace_by_time, heads_by_time, message_processor
            )
            
            # Mark processed messages
            self._mark_messages_as_processed(processed_messages)
            
            logging.info("Correlation process completed", extra={
                'matches_found': matches_found,
                'messages_processed': len(processed_messages)
            })
            
            return matches_found
            
        except ValueError as e:
            logging.error("Value error in message correlation", extra={
                'error_message': str(e),
                'buffer_sizes': {k: len(v) for k, v in buffers.items()},
                'time_window': self.time_window
            })
            return False
        except KeyError as e:
            logging.error("Configuration key missing", extra={
                'missing_key': str(e),
                'config_section': 'mqtt.listener',
                'available_keys': list(self.config.get('mqtt', {}).get('listener', {}).keys())
            })
            return False
        except Exception as e:
            logging.error("Unexpected error in message correlation", extra={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'buffer_sizes': {k: len(v) for k, v in buffers.items()}
            }, exc_info=True)
            return False
    
    def _get_unprocessed_messages(self, buffers: Dict[str, List[TimestampedData]]) -> Tuple[List, List, List]:
        """Get unprocessed messages from buffers by type."""
        result_messages = [
            msg for msg in buffers.get(self._topic_patterns['result'], []) 
            if not getattr(msg, 'processed', False)
        ]
        trace_messages = [
            msg for msg in buffers.get(self._topic_patterns['trace'], []) 
            if not getattr(msg, 'processed', False)
        ]
        heads_messages = []
        if 'heads' in self._topic_patterns:
            heads_messages = [
                msg for msg in buffers.get(self._topic_patterns['heads'], []) 
                if not getattr(msg, 'processed', False)
            ]
        
        return result_messages, trace_messages, heads_messages
    
    def _group_messages_by_time_bucket(self, result_messages: List[TimestampedData],
                                     trace_messages: List[TimestampedData],
                                     heads_messages: List[TimestampedData]) -> Tuple[Dict, Dict, Dict]:
        """Group messages by timestamp buckets for efficient correlation."""
        
        def get_time_bucket(timestamp: float) -> float:
            """Convert timestamp to time bucket."""
            return round(timestamp / self.time_window) * self.time_window
        
        result_by_time = defaultdict(list)
        trace_by_time = defaultdict(list)
        heads_by_time = defaultdict(list)
        
        # Group result messages
        for msg in result_messages:
            bucket = get_time_bucket(msg.timestamp)
            result_by_time[bucket].append(msg)
        
        # Group trace messages
        for msg in trace_messages:
            bucket = get_time_bucket(msg.timestamp)
            trace_by_time[bucket].append(msg)
        
        # Group heads messages
        for msg in heads_messages:
            bucket = get_time_bucket(msg.timestamp)
            heads_by_time[bucket].append(msg)
        
        logging.debug("Time bucket grouping completed", extra={
            'result_buckets': len(result_by_time),
            'trace_buckets': len(trace_by_time),
            'heads_buckets': len(heads_by_time)
        })
        
        return result_by_time, trace_by_time, heads_by_time
    
    def _find_matches_in_time_buckets(self, result_by_time: Dict, trace_by_time: Dict,
                                    heads_by_time: Dict, message_processor: Callable) -> Tuple[bool, Set]:
        """Find and process matches within time buckets."""
        matches_found = False
        processed_messages = set()
        
        # Check each result time bucket against trace buckets
        for time_bucket in result_by_time.keys():
            # Check current bucket and adjacent buckets (±1)
            for bucket_offset in [-1, 0, 1]:
                check_bucket = time_bucket + (bucket_offset * self.time_window)
                
                if check_bucket in trace_by_time:
                    bucket_matches = self._process_time_bucket_matches(
                        result_by_time[time_bucket],
                        trace_by_time[check_bucket],
                        heads_by_time.get(check_bucket, []),
                        processed_messages,
                        message_processor
                    )
                    matches_found = matches_found or bucket_matches
        
        return matches_found, processed_messages
    
    def _process_time_bucket_matches(self, result_msgs: List[TimestampedData],
                                   trace_msgs: List[TimestampedData],
                                   heads_msgs: List[TimestampedData],
                                   processed_messages: Set[TimestampedData],
                                   message_processor: Callable) -> bool:
        """Process matches within a specific time bucket."""
        matches_found = False
        
        # Group by tool key within the time bucket
        result_by_tool = defaultdict(list)
        trace_by_tool = defaultdict(list)
        
        # Group result messages by tool key
        for msg in result_msgs:
            if msg not in processed_messages:
                tool_key = self._extract_tool_key(msg.source)
                if tool_key:
                    result_by_tool[tool_key].append(msg)
        
        # Group trace messages by tool key
        for msg in trace_msgs:
            if msg not in processed_messages:
                tool_key = self._extract_tool_key(msg.source)
                if tool_key:
                    trace_by_tool[tool_key].append(msg)
        
        # Match result and trace messages by tool key
        for tool_key in result_by_tool.keys():
            if tool_key in trace_by_tool:
                tool_matches = self._match_messages_for_tool(
                    tool_key,
                    result_by_tool[tool_key],
                    trace_by_tool[tool_key],
                    heads_msgs,
                    processed_messages,
                    message_processor
                )
                matches_found = matches_found or tool_matches
        
        return matches_found
    
    def _match_messages_for_tool(self, tool_key: str,
                               result_msgs: List[TimestampedData],
                               trace_msgs: List[TimestampedData],
                               heads_msgs: List[TimestampedData],
                               processed_messages: Set[TimestampedData],
                               message_processor: Callable) -> bool:
        """Match messages for a specific tool within the time window."""
        matches_found = False
        
        for result_msg in result_msgs:
            if result_msg in processed_messages:
                continue
                
            for trace_msg in trace_msgs:
                if trace_msg in processed_messages:
                    continue
                
                # Check if messages are within time window
                time_diff = abs(result_msg.timestamp - trace_msg.timestamp)
                if time_diff <= self.time_window:
                    
                    # Find matching heads message by timestamp
                    matching_heads = self._find_matching_heads_message(
                        result_msg, heads_msgs, processed_messages
                    )
                    
                    logging.info("Found matching message pair", extra={
                        'tool_key': tool_key,
                        'time_difference': time_diff,
                        'has_heads_message': matching_heads is not None,
                        'result_timestamp': result_msg.timestamp,
                        'trace_timestamp': trace_msg.timestamp
                    })
                    
                    # Create match group
                    match_group = [result_msg, trace_msg]
                    if matching_heads:
                        match_group.append(matching_heads)
                        processed_messages.add(matching_heads)
                    
                    # Process the match
                    try:
                        message_processor(match_group)
                        matches_found = True
                        
                        # Mark messages as processed
                        processed_messages.add(result_msg)
                        processed_messages.add(trace_msg)
                        
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
                                   processed_messages: Set[TimestampedData]) -> TimestampedData:
        """Find heads message that matches the result message timestamp."""
        for heads_msg in heads_msgs:
            if (heads_msg not in processed_messages and
                abs(result_msg.timestamp - heads_msg.timestamp) <= self.time_window):
                return heads_msg
        return None
    
    def _extract_tool_key(self, source: str) -> str:
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
            buffers: Message buffers to analyze
            
        Returns:
            Dictionary containing correlation statistics
        """
        stats = {
            'time_window': self.time_window,
            'total_unprocessed_messages': 0,
            'unprocessed_by_type': {},
            'tool_distribution': defaultdict(int),
            'timestamp_range': {}
        }
        
        # Analyze each message type
        for msg_type, topic_pattern in self._topic_patterns.items():
            messages = buffers.get(topic_pattern, [])
            unprocessed = [msg for msg in messages if not getattr(msg, 'processed', False)]
            
            stats['unprocessed_by_type'][msg_type] = len(unprocessed)
            stats['total_unprocessed_messages'] += len(unprocessed)
            
            # Analyze tool distribution
            for msg in unprocessed:
                tool_key = self._extract_tool_key(msg.source)
                if tool_key:
                    stats['tool_distribution'][tool_key] += 1
            
            # Analyze timestamp range
            if unprocessed:
                timestamps = [msg.timestamp for msg in unprocessed]
                stats['timestamp_range'][msg_type] = {
                    'oldest': min(timestamps),
                    'newest': max(timestamps),
                    'span_seconds': max(timestamps) - min(timestamps)
                }
        
        return stats