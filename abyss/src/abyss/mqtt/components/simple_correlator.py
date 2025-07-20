"""
Simple Message Correlator Module

Handles correlation and matching of MQTT messages based on tool keys and timestamps.
Simplified approach replacing complex time-bucket grouping with direct key matching.
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Callable, Optional

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
    
    def __init__(self, config: ConfigurationManager, time_window: float = 30.0):
        """
        Initialize SimpleMessageCorrelator.
        
        Args:
            config: ConfigurationManager instance
            time_window: Time window for message correlation (seconds)
        """
        self.config_manager = config
        self.time_window = time_window if time_window != 30.0 else config.get_time_window()
        # Get topic patterns from ConfigurationManager
        self._topic_patterns = config.get_topic_patterns()
        # Get debug mode setting
        self.debug_mode = config.get_correlation_debug_mode()
        
        # Track correlation failures for warning detection
        self._correlation_failures = defaultdict(int)  # tool_key -> failure count
        self._last_warning_time = {}  # warning_type -> timestamp
        self._unprocessed_threshold = 100  # Warn when unprocessed messages exceed this
        
        # Track correlation successes
        self._correlation_successes = 0  # Total successful correlations
        self._correlation_attempts = 0  # Total correlation attempts
        
        # Heads message cleanup settings
        self.heads_cleanup_timeout = 60.0  # Clean up heads messages after 60 seconds
    
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
                
                # Debug mode: log more details about why correlation can't proceed
                if self.debug_mode:
                    logging.debug("Correlation debug - insufficient messages", extra={
                        'has_result': len(result_messages) > 0,
                        'has_trace': len(trace_messages) > 0,
                        'result_count': len(result_messages),
                        'trace_count': len(trace_messages),
                        'buffer_state': {
                            topic: len(msgs) for topic, msgs in buffers.items()
                        }
                    })
                
                return False
            
            # Increment correlation attempts
            self._correlation_attempts += 1
            
            # Find matches using simple key-based approach
            matches_found, processed_messages = self._find_key_based_matches(
                result_messages, trace_messages, heads_messages, message_processor
            )
            
            # Mark processed messages
            self._mark_messages_as_processed(processed_messages)
            
            # Clean up orphaned heads messages
            self._cleanup_orphaned_heads(heads_messages, processed_messages, buffers)
            
            logging.debug("Correlation process completed", extra={
                'matches_found': matches_found,
                'messages_processed': len(processed_messages)
            })
            
            # Check for correlation issues and warn if needed
            self._check_correlation_health(buffers, matches_found, len(processed_messages))
            
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
        logging.debug(f"Duplicate messages detected and ignored [result_duplicates={result_duplicates}, trace_duplicates={trace_duplicates}, heads_duplicates={heads_duplicates}, total_duplicates={total_duplicates}]")
        
        # Log specific details about each duplicate message type
        if result_duplicates > 0:
            processed_results = [
                msg for msg in buffers.get(self._topic_patterns['result'], []) 
                if getattr(msg, 'processed', False)
            ]
            for msg in processed_results:
                tool_key = self._extract_tool_key(msg.source)
                logging.debug(f"Duplicate RESULT message ignored [message_type=result, source_topic={msg.source}, tool_key={tool_key}]")
        
        if trace_duplicates > 0:
            processed_traces = [
                msg for msg in buffers.get(self._topic_patterns['trace'], []) 
                if getattr(msg, 'processed', False)
            ]
            for msg in processed_traces:
                tool_key = self._extract_tool_key(msg.source)
                logging.debug(f"Duplicate TRACE message ignored [message_type=trace, source_topic={msg.source}, tool_key={tool_key}]")
        
        if heads_duplicates > 0:
            processed_heads = [
                msg for msg in buffers.get(self._topic_patterns['heads'], []) 
                if getattr(msg, 'processed', False)
            ]
            for msg in processed_heads:
                tool_key = self._extract_tool_key(msg.source)
                logging.debug(f"Duplicate HEADS message ignored [message_type=heads, source_topic={msg.source}, tool_key={tool_key}]")
    
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
                if self.debug_mode:
                    logging.debug(f"Correlation debug - attempting to match tool_key: {tool_key}", extra={
                        'result_msgs': len(result_by_key[tool_key]),
                        'trace_msgs': len(trace_by_key[tool_key]),
                        'heads_msgs': len(heads_by_key.get(tool_key, []))
                    })
                
                key_matches = self._process_key_matches(
                    tool_key,
                    result_by_key[tool_key],
                    trace_by_key[tool_key],
                    heads_by_key.get(tool_key, []),
                    processed_messages,
                    message_processor
                )
                matches_found = matches_found or key_matches
            elif self.debug_mode:
                logging.debug(f"Correlation debug - no trace messages for tool_key: {tool_key}")
        
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
                if trace_msg not in processed_messages:
                    time_diff = abs(result_msg.timestamp - trace_msg.timestamp)
                    if time_diff <= self.time_window:
                        
                        # Find matching heads message (optional)
                        heads_msg = self._find_matching_heads_message(
                            result_msg, heads_msgs, processed_messages
                        )
                        
                        # Create match group
                        match_group = [result_msg, trace_msg]
                        if heads_msg:
                            match_group.append(heads_msg)
                        
                        try:
                            logging.debug("Found matching message pair", extra={
                                'tool_key': tool_key,
                                'result_timestamp': result_msg.timestamp,
                                'trace_timestamp': trace_msg.timestamp,
                                'time_difference': abs(result_msg.timestamp - trace_msg.timestamp),
                                'has_heads_message': heads_msg is not None
                            })
                        
                            # Reset failure count for successful correlation
                            if tool_key in self._correlation_failures:
                                self._correlation_failures[tool_key] = 0
                            
                            # Process the match
                            message_processor(match_group)
                            matches_found = True
                            
                            # Increment success counter
                            self._correlation_successes += 1
                            
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
                
                elif self.debug_mode:
                    # Debug: log why messages didn't match
                    logging.debug("Correlation debug - messages outside time window", extra={
                        'tool_key': tool_key,
                        'result_timestamp': result_msg.timestamp,
                        'trace_timestamp': trace_msg.timestamp,
                        'time_difference': time_diff,
                        'time_window': self.time_window
                    })
        
        # Track correlation failures for this tool
        if not matches_found and tool_key:
            self._correlation_failures[tool_key] += 1
            self._check_tool_correlation_failures(tool_key)
        
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
            'correlation_approach': 'simple_key_based',
            'correlation_attempts': self._correlation_attempts,
            'correlation_successes': self._correlation_successes,
            'correlation_success_rate': (self._correlation_successes / self._correlation_attempts * 100 
                                       if self._correlation_attempts > 0 else 0.0)
        }
    
    def _check_correlation_health(self, buffers: Dict[str, List[TimestampedData]], 
                                 matches_found: bool, messages_processed: int):
        """Check correlation health and warn about issues."""
        # Check for high unprocessed message count
        total_unprocessed = 0
        unprocessed_by_type = {}
        
        for pattern, topic in self._topic_patterns.items():
            buffer = buffers.get(topic, [])
            unprocessed = len([msg for msg in buffer if not getattr(msg, 'processed', False)])
            unprocessed_by_type[pattern] = unprocessed
            total_unprocessed += unprocessed
        
        # Warn if unprocessed messages are accumulating
        if total_unprocessed > self._unprocessed_threshold:
            self._log_rate_limited_warning('high_unprocessed', 60,
                "High number of unprocessed messages - correlation may be falling behind", {
                    'total_unprocessed': total_unprocessed,
                    'unprocessed_by_type': unprocessed_by_type,
                    'threshold': self._unprocessed_threshold,
                    'possible_causes': 'Message arrival rate exceeds processing rate or timing mismatches'
                })
        
        # Warn if no matches found but messages exist
        if not matches_found and total_unprocessed > 10:
            self._log_rate_limited_warning('no_matches', 120,
                "No message matches found despite pending messages", {
                    'unprocessed_messages': total_unprocessed,
                    'time_window': self.time_window,
                    'possible_cause': 'Messages may be arriving outside correlation time window'
                })
    
    def _check_tool_correlation_failures(self, tool_key: str):
        """Check and warn about repeated correlation failures for a specific tool."""
        failure_count = self._correlation_failures[tool_key]
        
        if failure_count >= 5:  # Warn after 5 consecutive failures
            self._log_rate_limited_warning(f'tool_failure_{tool_key}', 300,
                "Repeated correlation failures for tool", {
                    'tool_key': tool_key,
                    'consecutive_failures': failure_count,
                    'possible_cause': 'Tool may be sending incomplete message sets or timing issues'
                })
    
    def _log_rate_limited_warning(self, warning_type: str, min_interval: int, 
                                 message: str, extra: Dict[str, Any]):
        """Log warning with rate limiting to avoid spam."""
        current_time = time.time()
        last_warning = self._last_warning_time.get(warning_type, 0)
        
        if current_time - last_warning >= min_interval:
            logging.warning(message, extra=extra)
            self._last_warning_time[warning_type] = current_time
    
    def _cleanup_orphaned_heads(self, heads_messages: List[TimestampedData], 
                               processed_messages: Set[TimestampedData],
                               buffers: Dict[str, List[TimestampedData]]):
        """
        Clean up heads messages that are too old to correlate.
        
        Args:
            heads_messages: List of heads messages from buffer
            processed_messages: Set of already processed messages  
            buffers: Message buffers to update
        """
        current_time = time.time()
        heads_cleaned = 0
        
        # Get the heads buffer
        heads_topic = self._topic_patterns.get('heads')
        if not heads_topic or heads_topic not in buffers:
            return
            
        heads_buffer = buffers[heads_topic]
        
        # Check each heads message
        for heads_msg in heads_messages:
            if heads_msg not in processed_messages:
                age = current_time - heads_msg.timestamp
                if age > self.heads_cleanup_timeout:
                    # Mark as processed to prevent further correlation attempts
                    heads_msg.processed = True
                    heads_cleaned += 1
                    
                    if self.debug_mode:
                        tool_key = self._extract_tool_key(heads_msg.source)
                        logging.info(f"Cleaned up orphaned heads message", extra={
                            'tool_key': tool_key,
                            'age_seconds': age,
                            'timestamp': heads_msg.timestamp,
                            'source': heads_msg.source
                        })
        
        if heads_cleaned > 0:
            logging.info(f"Cleaned up {heads_cleaned} orphaned heads messages older than {self.heads_cleanup_timeout}s")