"""
Message Buffer Module

Handles buffering and cleanup of timestamped MQTT messages.
Extracted from the original MQTTDrillingDataAnalyser class.
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any

from ...uos_depth_est import TimestampedData, ConfigurationError


class MessageBuffer:
    """
    Manages buffering of timestamped MQTT messages with automatic cleanup.
    
    Responsibilities:
    - Buffer messages by topic type
    - Validate incoming messages
    - Perform time-based and size-based cleanup
    - Provide buffer statistics and monitoring
    """
    
    def __init__(self, config: Dict[str, Any], cleanup_interval: int = 60, 
                 max_buffer_size: int = 10000, max_age_seconds: int = 300):
        """
        Initialize MessageBuffer.
        
        Args:
            config: Configuration dictionary
            cleanup_interval: Interval between automatic cleanups (seconds)
            max_buffer_size: Maximum number of messages per buffer
            max_age_seconds: Maximum age of messages before cleanup (seconds)
        """
        self.config = config
        self.buffers = defaultdict(list)
        self.cleanup_interval = cleanup_interval
        self.max_buffer_size = max_buffer_size
        self.max_age_seconds = max_age_seconds
        self.last_cleanup = datetime.now().timestamp()
        
        # Pre-compute topic patterns for efficiency
        listener_config = self.config['mqtt']['listener']
        self._topic_patterns = {
            'trace': f"{listener_config['root']}/+/+/{listener_config['trace']}",
            'result': f"{listener_config['root']}/+/+/{listener_config['result']}"
        }
        
        # Add heads pattern only if configured
        if 'heads' in listener_config:
            self._topic_patterns['heads'] = f"{listener_config['root']}/+/+/{listener_config['heads']}"
    
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

            # Log buffer operation
            buffer_size_before = len(self.buffers[matching_topic])
            logging.info("Adding message to buffer", extra={
                'topic_pattern': matching_topic,
                'source': data.source,
                'buffer_size_before': buffer_size_before,
                'timestamp': data.timestamp
            })
            
            # Add message to buffer
            self.buffers[matching_topic].append(data)
            
            buffer_size_after = len(self.buffers[matching_topic])
            logging.info("Message added to buffer", extra={
                'buffer_size_after': buffer_size_after,
                'all_buffer_sizes': {k: len(v) for k, v in self.buffers.items()}
            })
            
            # Trigger cleanup if needed
            self._check_cleanup_triggers()
            
            return True
            
        except ValueError as e:
            logging.warning("Invalid message data", extra={
                'error_message': str(e),
                'source': getattr(data, 'source', 'unknown') if data else 'no_data'
            })
            return False
        except ConfigurationError as e:
            logging.error("Configuration error in add_message", extra={
                'error_message': str(e),
                'config_section': 'mqtt.listener'
            })
            return False
        except Exception as e:
            logging.error("Unexpected error in add_message", extra={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'source': getattr(data, 'source', 'unknown') if data else 'no_data'
            })
            return False
    
    def _get_matching_topic(self, source: str) -> str:
        """
        Determine which topic pattern matches the message source.
        
        Args:
            source: Message source topic
            
        Returns:
            Matching topic pattern or None if no match
        """
        listener_config = self.config['mqtt']['listener']
        
        if listener_config['trace'] in source:
            logging.debug("Trace message identified")
            return self._topic_patterns['trace']
        elif listener_config['result'] in source:
            logging.debug("Result message identified")
            return self._topic_patterns['result']
        elif 'heads' in listener_config and listener_config['heads'] in source:
            logging.debug("Heads message identified")
            return self._topic_patterns['heads']
        
        return None
    
    def _check_cleanup_triggers(self):
        """Check if cleanup should be triggered based on size or time."""
        # Check for buffer size overflow
        for topic, buffer in self.buffers.items():
            if len(buffer) >= self.max_buffer_size:
                logging.info("Buffer size cleanup triggered", extra={
                    'topic': topic,
                    'buffer_size': len(buffer),
                    'max_size': self.max_buffer_size
                })
                self.cleanup_old_messages()
                break
        
        # Check for time-based cleanup
        current_time = datetime.now().timestamp()
        if current_time - self.last_cleanup >= self.cleanup_interval:
            logging.debug("Time-based cleanup triggered", extra={
                'elapsed_seconds': current_time - self.last_cleanup,
                'cleanup_interval': self.cleanup_interval
            })
            self.cleanup_old_messages()
            self.last_cleanup = current_time
    
    def cleanup_old_messages(self):
        """
        Improved cleanup with time-based sliding window approach.
        """
        try:
            current_time = datetime.now().timestamp()
            total_removed = 0
            
            for topic in list(self.buffers.keys()):
                if topic not in self.buffers:
                    continue
                    
                original_length = len(self.buffers[topic])
                if original_length == 0:
                    continue
                
                # Remove old messages beyond time window
                self.buffers[topic] = [
                    msg for msg in self.buffers[topic] 
                    if current_time - msg.timestamp <= self.max_age_seconds
                ]
                age_removed = original_length - len(self.buffers[topic])
                
                # If still over size limit, keep only newest messages
                size_removed = 0
                if len(self.buffers[topic]) > self.max_buffer_size:
                    self.buffers[topic].sort(key=lambda msg: msg.timestamp, reverse=True)
                    size_removed = len(self.buffers[topic]) - self.max_buffer_size
                    self.buffers[topic] = self.buffers[topic][:self.max_buffer_size]
                
                total_topic_removed = age_removed + size_removed
                total_removed += total_topic_removed
                
                if total_topic_removed > 0:
                    logging.info("Buffer cleanup completed", extra={
                        'topic': topic,
                        'original_size': original_length,
                        'final_size': len(self.buffers[topic]),
                        'age_removed': age_removed,
                        'size_removed': size_removed,
                        'total_removed': total_topic_removed
                    })
            
            if total_removed > 0:
                logging.info("Global cleanup summary", extra={
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
        stats = {
            'total_buffers': len(self.buffers),
            'total_messages': sum(len(buffer) for buffer in self.buffers.values()),
            'buffer_sizes': {topic: len(buffer) for topic, buffer in self.buffers.items()},
            'oldest_message_age': None,
            'newest_message_age': None,
            'last_cleanup_seconds_ago': current_time - self.last_cleanup
        }
        
        # Find oldest and newest messages
        all_timestamps = []
        for buffer in self.buffers.values():
            all_timestamps.extend([msg.timestamp for msg in buffer])
        
        if all_timestamps:
            oldest_timestamp = min(all_timestamps)
            newest_timestamp = max(all_timestamps)
            stats['oldest_message_age'] = current_time - oldest_timestamp
            stats['newest_message_age'] = current_time - newest_timestamp
        
        return stats
    
    def get_messages_by_topic(self, topic_pattern: str) -> List[TimestampedData]:
        """
        Get all messages for a specific topic pattern.
        
        Args:
            topic_pattern: Topic pattern to retrieve
            
        Returns:
            List of messages for the topic
        """
        return self.buffers.get(topic_pattern, []).copy()
    
    def get_all_buffers(self) -> Dict[str, List[TimestampedData]]:
        """
        Get all buffer contents.
        
        Returns:
            Dictionary mapping topic patterns to message lists
        """
        return {topic: buffer.copy() for topic, buffer in self.buffers.items()}
    
    def clear_buffer(self, topic_pattern: str = None):
        """
        Clear buffer(s).
        
        Args:
            topic_pattern: Specific topic to clear, or None to clear all
        """
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