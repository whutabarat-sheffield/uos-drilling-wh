"""
Message Buffer Module

Handles buffering and cleanup of timestamped MQTT messages.
Extracted from the original MQTTDrillingDataAnalyser class.
"""

import logging
import threading
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Any, Optional, Deque
import time as time_module

from ...uos_depth_est import TimestampedData, ConfigurationError
from .config_manager import ConfigurationManager


class DuplicateMessageError(Exception):
    """Exception raised when a duplicate message is detected and error handling is configured."""
    pass


class MessageBuffer:
    """
    Manages buffering of timestamped MQTT messages with automatic cleanup.
    
    Responsibilities:
    - Buffer messages by topic type
    - Validate incoming messages
    - Perform time-based and size-based cleanup
    - Provide buffer statistics and monitoring
    """
    
    def __init__(self, config: ConfigurationManager, cleanup_interval: int = 60, 
                 max_buffer_size: int = 10000, max_age_seconds: int = 300, 
                 hysteresis_factor: float = 0.8, throughput_monitor=None):
        """
        Initialize MessageBuffer.
        
        Args:
            config: ConfigurationManager instance
            cleanup_interval: Interval between automatic cleanups (seconds)
            max_buffer_size: Maximum number of messages per buffer
            max_age_seconds: Maximum age of messages before cleanup (seconds)
            hysteresis_factor: Cleanup target size as fraction of max (0.8 = 80%)
            throughput_monitor: Optional SimpleThroughputMonitor instance for tracking message arrivals
        """
        self.config_manager = config
        
        # Use ConfigurationManager methods for typed access
        self.duplicate_handling = config.get_duplicate_handling()
        listener_config = config.get_mqtt_listener_config()
        
        self.buffers: Dict[str, Deque[TimestampedData]] = defaultdict(deque)
        self._buffer_lock = threading.RLock()  # Use RLock to prevent deadlocks
        self.cleanup_interval = cleanup_interval
        self.max_buffer_size = max_buffer_size
        self.max_age_seconds = max_age_seconds
        self.hysteresis_factor = hysteresis_factor  # Cleanup target size as fraction of max
        self.cleanup_target_size = int(max_buffer_size * hysteresis_factor)
        self.last_cleanup = datetime.now().timestamp()
        self.last_cleanup_by_topic: Dict[str, float] = {}  # Track cleanup times per topic
        
        # Buffer-specific metrics only
        self._metrics = {
            'messages_received': 0,  # Total received
            'messages_dropped': 0,   # Total dropped
            'duplicate_messages': 0, # Duplicates detected
            'cleanup_cycles': 0,     # Number of cleanups
            'last_warning_time': {}, # Track last warning time by type
            'messages_dropped_by_topic': defaultdict(int),  # Drops per topic
            'messages_dropped_age': defaultdict(int),       # Drops due to age
            'messages_dropped_size': defaultdict(int)       # Drops due to size limit
        }
        self._metrics_lock = threading.RLock()  # Use RLock to prevent deadlocks
        
        # Store throughput monitor reference
        self.throughput_monitor = throughput_monitor
        
        # Pre-compute topic patterns for efficiency
        self._topic_patterns = {
            'trace': f"{listener_config.get('root', '')}/+/+/{listener_config.get('trace', '')}",
            'result': f"{listener_config.get('root', '')}/+/+/{listener_config.get('result', '')}",
            'heads': f"{listener_config.get('root', '')}/+/+/{listener_config.get('heads', '')}"
        }
        
    def add_message(self, data: TimestampedData) -> bool:
        """
        Add a message to the buffer with improved error handling.
        
        Args:
            data: Timestamped message data
            
        Returns:
            True if message was added successfully, False otherwise
        """
        try:
            # Validation
            if not data or not data.source:
                logging.warning("Message validation failed", extra={
                    'reason': 'empty_source',
                    'has_data': data is not None
                })
                return False
            
            # Determine message type and topic pattern
            matching_topic = self._get_matching_topic(data.source)
            if not matching_topic:
                logging.debug("Non-essential topic received", extra={
                    'source': data.source
                })
                return False

            # Record message arrival for throughput monitoring
            if self.throughput_monitor:
                self.throughput_monitor.record_arrival()
            
            # Increment arrival counter for workflow statistics
            if hasattr(self, '_analyser_ref') and self._analyser_ref:
                self._analyser_ref._workflow_stats['messages_arrived'] += 1

            # Log buffer operation
            buffer_size_before = len(self.buffers[matching_topic])
            logging.debug("Adding message to buffer", extra={
                'topic_pattern': matching_topic,
                'source': data.source,
                'buffer_size_before': buffer_size_before,
                'timestamp': data.timestamp
            })
            
            # Thread-safe buffer operations
            with self._buffer_lock:
                # Check for duplicates before adding
                existing_messages = self.buffers[matching_topic]
                duplicate_found, duplicate_index = self._check_for_duplicate(data, existing_messages)
                
                if duplicate_found:
                    # Track duplicate in metrics
                    with self._metrics_lock:
                        self._metrics['duplicate_messages'] += 1
                        
                    if self.duplicate_handling == 'ignore':
                        logging.debug("Duplicate message ignored per configuration", extra={
                            'duplicate_handling': self.duplicate_handling,
                            'source': data.source,
                            'timestamp': data.timestamp
                        })
                        return False
                    elif self.duplicate_handling == 'replace':
                        # Replace the existing duplicate message
                        if duplicate_index is not None:
                            self.buffers[matching_topic][duplicate_index] = data
                            logging.debug("Duplicate message replaced per configuration", extra={
                                'duplicate_handling': self.duplicate_handling,
                                'source': data.source,
                                'timestamp': data.timestamp,
                                'replaced_index': duplicate_index
                            })
                            return True
                    elif self.duplicate_handling == 'error':
                        logging.error("Duplicate message detected and error raised per configuration", extra={
                            'duplicate_handling': self.duplicate_handling,
                            'source': data.source,
                            'timestamp': data.timestamp
                        })
                        raise DuplicateMessageError(f"Duplicate message detected: {data.source} at {data.timestamp}")
                
                # Add message to buffer (only if not a duplicate or handling is not 'ignore')
                self.buffers[matching_topic].append(data)
                
                # Track received messages after successful addition
                with self._metrics_lock:
                    self._metrics['messages_received'] += 1
                
                buffer_size_after = len(self.buffers[matching_topic])
                
                # Check buffer capacity with progressive warnings
                buffer_usage_percent = (buffer_size_after / self.max_buffer_size) * 100
                
                # Progressive warnings with rate limiting and cleanup awareness
                current_time = time_module.time()
                time_since_cleanup = current_time - self.last_cleanup_by_topic.get(matching_topic, 0)
                suppress_warnings = time_since_cleanup < 30  # Suppress warnings for 30s after cleanup
                
                if not suppress_warnings and buffer_usage_percent >= 90:
                    self._log_rate_limited_warning('buffer_critical', 30, 
                        "CRITICAL: Buffer at critical capacity - data loss imminent", {
                        'topic': matching_topic,
                        'buffer_usage_percent': buffer_usage_percent,
                        'buffer_size': buffer_size_after,
                        'max_size': self.max_buffer_size,
                        'cleanup_target': self.cleanup_target_size
                    })
                elif not suppress_warnings and buffer_usage_percent >= 80:
                    self._log_rate_limited_warning('buffer_high', 60,
                        "Buffer at high capacity - system under stress", {
                        'topic': matching_topic,
                        'buffer_usage_percent': buffer_usage_percent,
                        'buffer_size': buffer_size_after,
                        'max_size': self.max_buffer_size
                    })
                elif not suppress_warnings and buffer_usage_percent >= 60:
                    self._log_rate_limited_warning('buffer_moderate', 300,
                        "Buffer usage elevated - monitor system load", {
                        'topic': matching_topic,
                        'buffer_size': buffer_size_after,
                        'max_size': self.max_buffer_size,
                        'usage_percent': buffer_usage_percent
                    })
            logging.debug("Message added to buffer", extra={
                'buffer_size_after': buffer_size_after,
                'all_buffer_sizes': {k: len(v) for k, v in self.buffers.items()},
                'duplicate_detected': duplicate_found,
                'duplicate_handling': self.duplicate_handling
            })
            
            # Trigger cleanup if needed
            self._check_cleanup_triggers()
            
            # Note: Processing time tracking removed - this is buffer insertion time, not processing
            # Actual processing happens in the ProcessingPool
            
            return True
            
        except ValueError as e:
            logging.warning("Invalid message data", extra={
                'error_message': str(e),
                'source': getattr(data, 'source', 'unknown') if data else 'no_data'
            })
            # Track dropped message
            with self._metrics_lock:
                self._metrics['messages_dropped'] += 1
                # Can't determine topic for validation failures
            self._check_drop_rate_warning()
            return False
        except DuplicateMessageError:
            # Re-raise duplicate message errors (don't catch them)
            raise
        except ConfigurationError as e:
            logging.error("Configuration error in add_message", extra={
                'error_message': str(e),
                'config_section': 'mqtt.listener'
            })
            # Track dropped message
            with self._metrics_lock:
                self._metrics['messages_dropped'] += 1
            self._check_drop_rate_warning()
            return False
        except Exception as e:
            logging.error("Unexpected error in add_message", extra={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'source': getattr(data, 'source', 'unknown') if data else 'no_data'
            })
            # Track dropped message
            with self._metrics_lock:
                self._metrics['messages_dropped'] += 1
                # Try to determine topic for metrics
                if data and hasattr(data, 'source'):
                    try:
                        matching_topic = self._get_matching_topic(data.source)
                        if matching_topic:
                            self._metrics['messages_dropped_by_topic'][matching_topic] += 1
                    except:
                        pass  # Don't let metrics tracking cause additional errors
            self._check_drop_rate_warning()
            return False
    
    def _check_for_duplicate(self, new_message: TimestampedData, existing_messages: Deque[TimestampedData]) -> tuple[bool, Optional[int]]:
        """Check if the new message is a duplicate of any existing message.
        
        Returns:
            Tuple of (is_duplicate, index_of_duplicate)
        """
        try:
            # Get configurable time window for duplicate detection
            if self.config_manager:
                # Use ConfigurationManager for typed access
                time_window = self.config_manager.get('mqtt.listener.duplicate_time_window', 1.0)
            else:
                # Legacy raw config access
                time_window = self.config.get('mqtt', {}).get('listener', {}).get('duplicate_time_window', 1.0)
            
            for index, existing in enumerate(existing_messages):
                # Check if this is a potential duplicate based on source and timestamp
                if (existing.source == new_message.source and 
                    abs(existing.timestamp - new_message.timestamp) < time_window):
                    
                    # Check if data content is identical
                    if self._compare_message_data(existing.data, new_message.data):
                        self._log_duplicate_detection(new_message, existing)
                        return True, index
            
            return False, None
            
        except Exception as e:
            logging.debug(f"Error checking for duplicates: {e}")
            return False, None
    
    def _compare_message_data(self, data1: Any, data2: Any) -> bool:
        """Compare two message data objects for equality with improved accuracy."""
        try:
            # Handle None cases
            if data1 is None and data2 is None:
                return True
            if data1 is None or data2 is None:
                return False
            
            # Direct equality check first (handles most cases efficiently)
            if data1 == data2:
                return True
            
            # For dictionaries, use more reliable comparison
            if isinstance(data1, dict) and isinstance(data2, dict):
                return self._compare_dicts(data1, data2)
            
            # For lists, compare element by element
            if isinstance(data1, list) and isinstance(data2, list):
                if len(data1) != len(data2):
                    return False
                return all(self._compare_message_data(item1, item2) 
                          for item1, item2 in zip(data1, data2))
            
            # For numeric types, handle potential precision issues
            if isinstance(data1, (int, float)) and isinstance(data2, (int, float)):
                # Allow small differences for floating point comparison
                return abs(float(data1) - float(data2)) < 1e-10
            
            # Fall back to string comparison for other types
            str1 = str(data1)
            str2 = str(data2)
            return str1 == str2
            
        except Exception as e:
            logging.debug(f"Error comparing message data: {e}")
            return False
    
    def _compare_dicts(self, dict1: dict, dict2: dict) -> bool:
        """Compare two dictionaries recursively."""
        try:
            if set(dict1.keys()) != set(dict2.keys()):
                return False
            
            for key in dict1:
                if not self._compare_message_data(dict1[key], dict2[key]):
                    return False
            
            return True
        except Exception:
            return False
    
    def _log_duplicate_detection(self, new_message: TimestampedData, existing_message: TimestampedData):
        """Log details about duplicate message detection."""
        # Extract tool key for identification
        tool_key = "unknown"
        try:
            parts = new_message.source.split('/')
            if len(parts) >= 3:
                tool_key = f"{parts[1]}/{parts[2]}"
        except Exception:
            pass
        
        # Determine message type
        message_type = "unknown"
        if "ResultManagement" in new_message.source:
            message_type = "result"
        elif "Trace" in new_message.source:
            message_type = "trace"
        elif "AssetManagement" in new_message.source:
            message_type = "heads"
        
        logging.debug("Duplicate message detected at buffer level", extra={
            'message_type': message_type,
            'source_topic': new_message.source,
            'tool_key': tool_key,
            'new_timestamp': new_message.timestamp,
            'existing_timestamp': existing_message.timestamp,
            'timestamp_diff': abs(new_message.timestamp - existing_message.timestamp),
            'data_preview': str(new_message.data)[:200] + '...' if len(str(new_message.data)) > 200 else str(new_message.data),
            'duplicate_source': 'message_buffer',
            'duplicate_handling': self.duplicate_handling
        })
    
    def _get_matching_topic(self, source: str) -> Optional[str]:
        """
        Determine which topic pattern matches the message source.
        
        Args:
            source: Message source topic
            
        Returns:
            Matching topic pattern or None if no match
        """
        # Use ConfigurationManager for typed access
        listener_config = self.config_manager.get_mqtt_listener_config()
        
        if listener_config.get('trace', '') and listener_config['trace'] in source:
            logging.debug("Trace message identified")
            return self._topic_patterns['trace']
        elif listener_config.get('result', '') and listener_config['result'] in source:
            logging.debug("Result message identified")
            return self._topic_patterns['result']
        elif listener_config.get('heads', '') and listener_config['heads'] in source:
            logging.debug("Heads message identified")
            return self._topic_patterns['heads']
        
        return None
    
    def _check_cleanup_triggers(self):
        """Check if cleanup should be triggered based on size or time."""
        cleanup_needed = False
        
        # Check for buffer size overflow
        with self._buffer_lock:
            for topic, buffer in self.buffers.items():
                if len(buffer) >= self.max_buffer_size:
                    self._log_rate_limited_warning('buffer_overflow', 30,
                        "Buffer at maximum capacity - cleanup triggered", {
                        'topic': topic,
                        'buffer_size': len(buffer),
                        'max_size': self.max_buffer_size,
                        'action': 'Removing oldest messages'
                    })
                    cleanup_needed = True
                    break
        
        # Check for time-based cleanup
        current_time = datetime.now().timestamp()
        if current_time - self.last_cleanup >= self.cleanup_interval:
            logging.debug("Time-based cleanup triggered", extra={
                'elapsed_seconds': current_time - self.last_cleanup,
                'cleanup_interval': self.cleanup_interval
            })
            cleanup_needed = True
        
        # Perform cleanup if needed
        if cleanup_needed:
            self.cleanup_old_messages()
            self.last_cleanup = current_time
            
        # Check for old messages (warning only, doesn't trigger cleanup)
        self._check_message_age_warnings()
    
    def cleanup_old_messages(self):
        """
        Improved cleanup with hysteresis and deque-optimized operations.
        """
        try:
            current_time = datetime.now().timestamp()
            total_removed = 0
            cleanup_warnings = []
            
            # Track cleanup cycle
            with self._metrics_lock:
                self._metrics['cleanup_cycles'] += 1
            
            with self._buffer_lock:
                for topic in list(self.buffers.keys()):
                    if topic not in self.buffers:
                        continue
                        
                    original_length = len(self.buffers[topic])
                    if original_length == 0:
                        continue
                    
                    # First pass: Remove old messages beyond time window
                    # For deque, we need to rebuild without old messages
                    age_removed = 0
                    new_buffer = deque()
                    for msg in self.buffers[topic]:
                        if current_time - msg.timestamp <= self.max_age_seconds:
                            new_buffer.append(msg)
                        else:
                            age_removed += 1
                            with self._metrics_lock:
                                self._metrics['messages_dropped_age'][topic] += 1
                    
                    self.buffers[topic] = new_buffer
                    
                    # If still over size limit, implement hysteresis
                    size_removed = 0
                    if len(self.buffers[topic]) > self.max_buffer_size:
                        # Sort messages by timestamp (newest first)
                        sorted_messages = sorted(self.buffers[topic], key=lambda msg: msg.timestamp, reverse=True)
                        # Keep only up to cleanup_target_size (hysteresis)
                        messages_to_keep = sorted_messages[:self.cleanup_target_size]
                        size_removed = len(self.buffers[topic]) - len(messages_to_keep)
                        
                        # Track size-based drops
                        with self._metrics_lock:
                            self._metrics['messages_dropped_size'][topic] += size_removed
                        
                        # Rebuild deque with kept messages in original order
                        self.buffers[topic] = deque(sorted(messages_to_keep, key=lambda msg: msg.timestamp))
                    
                    # Track cleanup time for this topic
                    if age_removed > 0 or size_removed > 0:
                        self.last_cleanup_by_topic[topic] = current_time
                    
                    total_topic_removed = age_removed + size_removed
                    total_removed += total_topic_removed
                    
                    # Check if we're removing too many messages (indicates falling behind)
                    removal_percent = (total_topic_removed / original_length) * 100 if original_length > 0 else 0
                    if removal_percent > 50:
                        cleanup_warnings.append({
                            'topic': topic,
                            'removal_percent': removal_percent,
                            'messages_removed': total_topic_removed
                        })
                    
                    if total_topic_removed > 0:
                        logging.debug("Buffer cleanup completed", extra={
                            'topic': topic,
                            'original_size': original_length,
                            'final_size': len(self.buffers[topic]),
                            'age_removed': age_removed,
                            'size_removed': size_removed,
                            'total_removed': total_topic_removed
                        })
            
            # Log warnings for excessive cleanup
            for warning in cleanup_warnings:
                logging.warning("Excessive buffer cleanup indicates system falling behind", extra={
                    'topic': warning['topic'],
                    'removal_percent': warning['removal_percent'],
                    'messages_removed': warning['messages_removed'],
                    'possible_cause': 'Processing slower than message arrival rate'
                })
            
            if total_removed > 0:
                logging.debug("Global cleanup summary", extra={
                    'total_messages_removed': total_removed,
                    'buffer_sizes': {k: len(v) for k, v in self.buffers.items()}
                })
            
        except Exception as e:
            logging.error("Error in cleanup_old_messages", extra={
                'error_type': type(e).__name__,
                'error_message': str(e)
            }, exc_info=True)
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive buffer statistics.
        
        Returns:
            Dictionary containing buffer statistics
        """
        current_time = datetime.now().timestamp()
        with self._buffer_lock:
            stats = {
                'total_buffers': len(self.buffers),
                'total_messages': sum(len(buffer) for buffer in self.buffers.values()),
                'buffer_sizes': {topic: len(buffer) for topic, buffer in self.buffers.items()},
                'oldest_message_age': None,
                'newest_message_age': None,
                'last_cleanup_seconds_ago': current_time - self.last_cleanup,
                'message_type_distribution': {},
                'age_distribution': {
                    '0-10s': 0,
                    '10-30s': 0,
                    '30-60s': 0,
                    '60-120s': 0,
                    '120-300s': 0,
                    '>300s': 0
                }
            }
            
            # Calculate message type distribution
            for topic_pattern, buffer in self.buffers.items():
                # Extract message type from topic pattern (result/trace/heads)
                msg_type = topic_pattern.split('/')[-1] if '/' in topic_pattern else topic_pattern
                stats['message_type_distribution'][msg_type] = len(buffer)
            
            # Find oldest and newest messages and calculate age distribution
            all_timestamps = []
            for buffer in self.buffers.values():
                for msg in buffer:
                    timestamp = msg.timestamp
                    all_timestamps.append(timestamp)
                    
                    # Calculate age and categorize
                    age = current_time - timestamp
                    if age <= 10:
                        stats['age_distribution']['0-10s'] += 1
                    elif age <= 30:
                        stats['age_distribution']['10-30s'] += 1
                    elif age <= 60:
                        stats['age_distribution']['30-60s'] += 1
                    elif age <= 120:
                        stats['age_distribution']['60-120s'] += 1
                    elif age <= 300:
                        stats['age_distribution']['120-300s'] += 1
                    else:
                        stats['age_distribution']['>300s'] += 1
            
            if all_timestamps:
                oldest_timestamp = min(all_timestamps)
                newest_timestamp = max(all_timestamps)
                stats['oldest_message_age'] = current_time - oldest_timestamp
                stats['newest_message_age'] = current_time - newest_timestamp
                stats['average_message_age'] = current_time - (sum(all_timestamps) / len(all_timestamps))
        
        return stats
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get buffer performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        with self._metrics_lock:
            metrics = self._metrics.copy()
            
            # Calculate derived metrics
            total_messages = metrics['messages_received'] + metrics['messages_dropped']
            if total_messages > 0:
                metrics['drop_rate'] = metrics['messages_dropped'] / total_messages
                metrics['duplicate_rate'] = metrics['duplicate_messages'] / total_messages
            else:
                metrics['drop_rate'] = 0
                metrics['duplicate_rate'] = 0
            
            # Add placeholder for metrics that aren't tracked yet
            metrics['messages_processed'] = metrics['messages_received']  # Assume all received are processed
            metrics['avg_processing_time_ms'] = 0  # Not tracked in current implementation
            
        return metrics
    
    def get_messages_by_topic(self, topic_pattern: str) -> List[TimestampedData]:
        """
        Get all messages for a specific topic pattern.
        
        Args:
            topic_pattern: Topic pattern to retrieve
            
        Returns:
            List of messages for the topic
        """
        with self._buffer_lock:
            buffer = self.buffers.get(topic_pattern, deque())
            return list(buffer)
    
    def get_all_buffers(self) -> Dict[str, List[TimestampedData]]:
        """
        Get all buffer contents.
        
        Returns:
            Dictionary mapping topic patterns to message lists
        """
        with self._buffer_lock:
            return {topic: list(buffer) for topic, buffer in self.buffers.items()}
    
    def clear_buffer(self, topic_pattern: Optional[str] = None):
        """
        Clear buffer(s).
        
        Args:
            topic_pattern: Specific topic to clear, or None to clear all
        """
        with self._buffer_lock:
            if topic_pattern:
                if topic_pattern in self.buffers:
                    cleared_count = len(self.buffers[topic_pattern])
                    self.buffers[topic_pattern].clear()
                    logging.info(f"Cleared buffer for {topic_pattern}", extra={
                        'topic': topic_pattern,
                        'messages_cleared': cleared_count
                    })
            else:
                total_cleared = sum(len(buffer) for buffer in self.buffers.values())
                self.buffers.clear()
                logging.info("Cleared all buffers", extra={
                    'total_messages_cleared': total_cleared
                })
    
    def _log_rate_limited_warning(self, warning_type: str, min_interval_seconds: int, 
                                 message: str, extra: Dict[str, Any]):
        """
        Log warning with rate limiting to avoid spam.
        
        Args:
            warning_type: Type of warning for tracking
            min_interval_seconds: Minimum seconds between warnings of this type
            message: Warning message
            extra: Extra logging context
        """
        current_time = time_module.time()
        with self._metrics_lock:
            last_warning = self._metrics['last_warning_time'].get(warning_type, 0)
            if current_time - last_warning >= min_interval_seconds:
                logging.warning(message, extra=extra)
                self._metrics['last_warning_time'][warning_type] = current_time
    
    def _check_drop_rate_warning(self):
        """Check and warn if message drop rate is too high."""
        with self._metrics_lock:
            total = self._metrics['messages_received'] + self._metrics['messages_dropped']
            if total > 100:  # Only check after reasonable sample size
                drop_rate = self._metrics['messages_dropped'] / total
                if drop_rate > 0.05:  # 5% drop rate
                    self._log_rate_limited_warning('high_drop_rate', 60,
                        "High message drop rate detected", {
                            'drop_rate_percent': drop_rate * 100,
                            'messages_dropped': self._metrics['messages_dropped'],
                            'messages_received': self._metrics['messages_received']
                        })
    
    def _check_message_age_warnings(self):
        """Check for old messages in buffers and warn if needed."""
        current_time = datetime.now().timestamp()
        with self._buffer_lock:
            for topic, buffer in self.buffers.items():
                if not buffer:
                    continue
                    
                # Check oldest message age
                oldest_msg = min(buffer, key=lambda m: m.timestamp)
                age_seconds = current_time - oldest_msg.timestamp
                
                if age_seconds > 240:  # 4 minutes
                    self._log_rate_limited_warning(f'old_messages_{topic}', 120,
                        "Old messages detected in buffer - processing may be falling behind", {
                            'topic': topic,
                            'oldest_message_age_seconds': age_seconds,
                            'buffer_size': len(buffer),
                            'oldest_timestamp': datetime.fromtimestamp(oldest_msg.timestamp).isoformat()
                        })
    
