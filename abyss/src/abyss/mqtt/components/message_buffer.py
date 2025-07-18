"""
Message Buffer Module - Exact Match Implementation

This implementation uses exact (tool_key, source_timestamp) matching instead of fuzzy time windows.
Maintains the same public API as the original MessageBuffer for compatibility.
"""

import logging
import threading
import time
from collections import defaultdict
from queue import Queue, Empty
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple, Union

from ...uos_depth_est import TimestampedData
from .config_manager import ConfigurationManager


@dataclass
class MessageSet:
    """Holds messages for an exact (tool_key, timestamp) match."""
    tool_key: str
    source_timestamp: str
    created_at: float = field(default_factory=time.time)
    
    # Messages by type
    result_messages: List[TimestampedData] = field(default_factory=list)
    trace_messages: List[TimestampedData] = field(default_factory=list)
    heads_messages: List[TimestampedData] = field(default_factory=list)
    
    def add(self, msg_type: str, message: TimestampedData):
        """Add message to appropriate list."""
        if msg_type == 'result':
            self.result_messages.append(message)
        elif msg_type == 'trace':
            self.trace_messages.append(message)
        elif msg_type == 'heads':
            self.heads_messages.append(message)
    
    def is_complete(self, require_heads: bool = False) -> bool:
        """Check if we have minimum required messages."""
        if require_heads:
            return (len(self.result_messages) > 0 and 
                    len(self.trace_messages) > 0 and 
                    len(self.heads_messages) > 0)
        return len(self.result_messages) > 0 and len(self.trace_messages) > 0
    
    def get_all_messages(self) -> List[TimestampedData]:
        """Get all messages in this set."""
        return self.result_messages + self.trace_messages + self.heads_messages
    
    def get_all_by_type(self) -> Dict[str, List[TimestampedData]]:
        """Get messages organized by type."""
        return {
            'result': self.result_messages,
            'trace': self.trace_messages,
            'heads': self.heads_messages
        }
    
    def age_seconds(self) -> float:
        """Age of this message set."""
        return time.time() - self.created_at


class MessageBuffer:
    """
    Exact-match message buffer using (tool_key, timestamp_string) as keys.
    Maintains same public API for compatibility with existing code.
    """
    
    def __init__(self, config: Union[Dict[str, Any], ConfigurationManager], 
                 cleanup_interval: int = 60, max_buffer_size: int = 10000, 
                 max_age_seconds: int = 300):
        """
        Initialize MessageBuffer with exact matching.
        
        Args:
            config: Configuration dictionary or ConfigurationManager instance
            cleanup_interval: Interval between cleanups (seconds)
            max_buffer_size: Maximum messages per buffer (ignored in exact-match mode)
            max_age_seconds: Maximum age before expiry (seconds)
        """
        # Handle both ConfigurationManager and raw config dict
        if isinstance(config, ConfigurationManager):
            self.config_manager = config
            self.config = config.get_raw_config()
        else:
            self.config_manager = None
            self.config = config
        
        self.cleanup_interval = cleanup_interval
        self.max_buffer_size = max_buffer_size  # Kept for API compatibility
        self.max_age_seconds = max_age_seconds
        
        # Core data structure: Dict[ExactKey, MessageSet]
        # ExactKey = (tool_key, source_timestamp_string)
        self._exact_matches = {}  # type: Dict[Tuple[str, str], MessageSet]
        self._exact_matches_lock = threading.RLock()
        
        # Reverse index for topic pattern queries
        self._by_topic_pattern = defaultdict(set)  # topic_pattern -> set of exact_keys
        
        # Processing queue for completed matches
        self._completed_matches = Queue()
        
        # Always require heads messages for complete match
        listener_config = self._get_listener_config()
        self.require_heads = True  # Always require heads messages
        
        # Topic patterns for backward compatibility
        self._topic_patterns = self._build_topic_patterns(listener_config)
        
        # Metrics
        self._metrics = {
            'messages_received': 0,
            'exact_matches_completed': 0,
            'messages_expired': 0,
            'messages_dropped': 0,
            'current_active_keys': 0,
            'extraction_failures': 0
        }
        self._metrics_lock = threading.Lock()
        
        # Start cleanup thread
        self._running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        
        logging.info("MessageBuffer initialized with exact matching", extra={
            'max_age_seconds': max_age_seconds,
            'cleanup_interval': cleanup_interval,
            'require_heads': self.require_heads
        })
    
    def add_message(self, data: TimestampedData) -> bool:
        """
        Add message using exact matching.
        
        Args:
            data: Timestamped message data
            
        Returns:
            True if message was added successfully
        """
        try:
            with self._metrics_lock:
                self._metrics['messages_received'] += 1
            
            # Validate input
            if not data or not data.source:
                logging.warning("Invalid message: missing source")
                return False
            
            # Extract exact matching key components
            tool_key = self._extract_tool_key(data.source)
            source_timestamp = self._extract_source_timestamp(data)
            
            if not tool_key or not source_timestamp:
                logging.error("Cannot extract key components", extra={
                    'source': data.source,
                    'has_tool_key': tool_key is not None,
                    'has_timestamp': source_timestamp is not None,
                    'data_type': type(data.data).__name__
                })
                with self._metrics_lock:
                    self._metrics['extraction_failures'] += 1
                return False
            
            exact_key = (tool_key, source_timestamp)
            topic_pattern = self._get_topic_pattern(data.source)
            message_type = self._get_message_type(data.source)
            
            with self._exact_matches_lock:
                # Get or create message set
                if exact_key not in self._exact_matches:
                    self._exact_matches[exact_key] = MessageSet(tool_key, source_timestamp)
                    self._by_topic_pattern[topic_pattern].add(exact_key)
                
                message_set = self._exact_matches[exact_key]
                
                # Add message to set
                message_set.add(message_type, data)
                
                logging.debug("Added message to exact match set", extra={
                    'tool_key': tool_key,
                    'timestamp': source_timestamp,
                    'message_type': message_type,
                    'result_count': len(message_set.result_messages),
                    'trace_count': len(message_set.trace_messages),
                    'heads_count': len(message_set.heads_messages)
                })
                
                # Check if complete
                if message_set.is_complete(self.require_heads):
                    self._completed_matches.put(message_set)
                    with self._metrics_lock:
                        self._metrics['exact_matches_completed'] += 1
                    
                    # Log successful match
                    logging.info("Exact match completed", extra={
                        'tool_key': tool_key,
                        'timestamp': source_timestamp,
                        'result_ids': [m.data.get('ResultId') for m in message_set.result_messages]
                    })
                    
                    # Remove from active tracking
                    del self._exact_matches[exact_key]
                    self._by_topic_pattern[topic_pattern].discard(exact_key)
            
            return True
            
        except Exception as e:
            logging.error("Error adding message", extra={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'source': getattr(data, 'source', 'unknown')
            }, exc_info=True)
            with self._metrics_lock:
                self._metrics['messages_dropped'] += 1
            return False
    
    def get_all_buffers(self) -> Dict[str, List[TimestampedData]]:
        """
        Return buffers in format expected by correlator.
        Adapts our internal structure to the old API.
        """
        buffers = defaultdict(list)
        
        with self._exact_matches_lock:
            # Add active (unmatched) messages
            for exact_key, message_set in self._exact_matches.items():
                for msg in message_set.get_all_messages():
                    topic_pattern = self._get_topic_pattern(msg.source)
                    buffers[topic_pattern].append(msg)
        
        # Add completed matches from queue (for backward compatibility)
        # Note: This is a bit of a hack to maintain the old API
        completed = []
        try:
            while True:
                match = self._completed_matches.get_nowait()
                completed.append(match)
                for msg in match.get_all_messages():
                    topic_pattern = self._get_topic_pattern(msg.source)
                    buffers[topic_pattern].append(msg)
        except Empty:
            pass
        
        # Put completed matches back for correlator to process
        for match in completed:
            self._completed_matches.put(match)
        
        return dict(buffers)
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics (API compatibility)."""
        with self._exact_matches_lock:
            stats = {
                'total_buffers': len(self._topic_patterns),  # For compatibility
                'total_messages': sum(
                    len(ms.get_all_messages()) 
                    for ms in self._exact_matches.values()
                ),
                'buffer_sizes': {},  # Will be populated below
                'oldest_message_age': None,
                'newest_message_age': None,
                'last_cleanup_seconds_ago': 0,  # Not tracked in new implementation
                
                # New exact-match specific stats
                'total_active_keys': len(self._exact_matches),
                'messages_by_type': defaultdict(int),
                'completed_matches_pending': self._completed_matches.qsize()
            }
            
            # Count messages by topic pattern for compatibility
            for pattern in self._topic_patterns.values():
                stats['buffer_sizes'][pattern] = len(self.get_messages_by_topic(pattern))
            
            # Count messages by type
            for message_set in self._exact_matches.values():
                stats['messages_by_type']['result'] += len(message_set.result_messages)
                stats['messages_by_type']['trace'] += len(message_set.trace_messages)
                stats['messages_by_type']['heads'] += len(message_set.heads_messages)
            
            # Find oldest and newest
            if self._exact_matches:
                ages = [ms.age_seconds() for ms in self._exact_matches.values()]
                stats['oldest_message_age'] = max(ages)
                stats['newest_message_age'] = min(ages)
        
        # Add metrics
        with self._metrics_lock:
            stats.update(self._metrics)
        
        return stats
    
    def get_messages_by_topic(self, topic_pattern: str) -> List[TimestampedData]:
        """Get messages for a specific topic pattern (API compatibility)."""
        messages = []
        
        with self._exact_matches_lock:
            # Get exact keys for this topic pattern
            exact_keys = self._by_topic_pattern.get(topic_pattern, set())
            
            for key in exact_keys:
                if key in self._exact_matches:
                    # Only return messages that match this specific topic pattern
                    message_set = self._exact_matches[key]
                    for msg in message_set.get_all_messages():
                        if self._get_topic_pattern(msg.source) == topic_pattern:
                            messages.append(msg)
        
        return messages
    
    def clear_buffer(self, topic_pattern: Optional[str] = None):
        """Clear buffers (API compatibility)."""
        with self._exact_matches_lock:
            if topic_pattern:
                # Clear specific topic
                exact_keys = list(self._by_topic_pattern.get(topic_pattern, set()))
                for key in exact_keys:
                    if key in self._exact_matches:
                        del self._exact_matches[key]
                self._by_topic_pattern[topic_pattern].clear()
                
                logging.info(f"Cleared buffer for {topic_pattern}")
            else:
                # Clear everything
                self._exact_matches.clear()
                self._by_topic_pattern.clear()
                
                logging.info("Cleared all buffers")
        
        # Clear completed matches queue
        while not self._completed_matches.empty():
            try:
                self._completed_matches.get_nowait()
            except Empty:
                break
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics (API compatibility)."""
        with self._metrics_lock:
            metrics = self._metrics.copy()
            
        # Calculate additional metrics
        if metrics['messages_received'] > 0:
            metrics['exact_match_rate'] = (
                metrics['exact_matches_completed'] / metrics['messages_received']
            )
            metrics['drop_rate'] = metrics['messages_dropped'] / metrics['messages_received']
            metrics['extraction_failure_rate'] = (
                metrics['extraction_failures'] / metrics['messages_received']
            )
        else:
            metrics['exact_match_rate'] = 0
            metrics['drop_rate'] = 0
            metrics['extraction_failure_rate'] = 0
        
        return metrics
    
    def cleanup_old_messages(self):
        """Manual cleanup trigger (API compatibility)."""
        self._cleanup_expired_messages()
    
    # Private methods
    
    def _cleanup_loop(self):
        """Background thread to remove expired messages."""
        while self._running:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_expired_messages()
            except Exception as e:
                logging.error("Error in cleanup loop", exc_info=True)
    
    def _cleanup_expired_messages(self):
        """Remove messages older than max_age_seconds."""
        current_time = time.time()
        expired_keys = []
        
        with self._exact_matches_lock:
            for exact_key, message_set in self._exact_matches.items():
                if message_set.age_seconds() > self.max_age_seconds:
                    expired_keys.append((exact_key, message_set))
            
            # Remove expired entries
            for exact_key, message_set in expired_keys:
                # Log what we're expiring for analysis
                logging.warning("Expiring unmatched message set", extra={
                    'tool_key': message_set.tool_key,
                    'source_timestamp': message_set.source_timestamp,
                    'age_seconds': message_set.age_seconds(),
                    'had_result': len(message_set.result_messages) > 0,
                    'had_trace': len(message_set.trace_messages) > 0,
                    'had_heads': len(message_set.heads_messages) > 0,
                    'result_ids': [m.data.get('ResultId') for m in message_set.result_messages],
                    'result_count': len(message_set.result_messages),
                    'trace_count': len(message_set.trace_messages),
                    'heads_count': len(message_set.heads_messages)
                })
                
                del self._exact_matches[exact_key]
                
                # Clean up reverse index
                for msg in message_set.get_all_messages():
                    topic_pattern = self._get_topic_pattern(msg.source)
                    self._by_topic_pattern[topic_pattern].discard(exact_key)
                
                with self._metrics_lock:
                    self._metrics['messages_expired'] += len(message_set.get_all_messages())
            
            # Update active key count
            with self._metrics_lock:
                self._metrics['current_active_keys'] = len(self._exact_matches)
        
        if expired_keys:
            logging.info(f"Expired {len(expired_keys)} unmatched message sets")
    
    def _extract_tool_key(self, source: str) -> Optional[str]:
        """Extract toolbox_id/tool_id from topic."""
        try:
            parts = source.split('/')
            if len(parts) >= 3:
                return f"{parts[1]}/{parts[2]}"
        except Exception as e:
            logging.debug(f"Failed to extract tool key from {source}: {e}")
        return None
    
    def _extract_source_timestamp(self, data: TimestampedData) -> Optional[str]:
        """Extract original SourceTimestamp string from message data."""
        if isinstance(data.data, dict):
            return self._find_source_timestamp(data.data)
        return None
    
    def _find_source_timestamp(self, data: dict, depth: int = 0) -> Optional[str]:
        """Recursively find SourceTimestamp in nested dict."""
        if depth > 10:  # Prevent infinite recursion
            return None
            
        for key, value in data.items():
            if key == 'SourceTimestamp' and isinstance(value, str):
                return value
            elif isinstance(value, dict):
                # First check if SourceTimestamp is directly in this dict
                if 'SourceTimestamp' in value and isinstance(value['SourceTimestamp'], str):
                    return value['SourceTimestamp']
                # Otherwise recurse
                result = self._find_source_timestamp(value, depth + 1)
                if result:
                    return result
        return None
    
    def _get_message_type(self, source: str) -> str:
        """Determine message type from topic."""
        if 'ResultManagement' in source:
            return 'result'
        elif 'Trace' in source:
            return 'trace'
        elif 'AssetManagement' in source or 'Heads' in source:
            return 'heads'
        return 'unknown'
    
    def _get_topic_pattern(self, source: str) -> str:
        """Convert source topic to pattern for backward compatibility."""
        msg_type = self._get_message_type(source)
        return self._topic_patterns.get(msg_type, source)
    
    def _get_listener_config(self) -> dict:
        """Get listener configuration."""
        if self.config_manager:
            return self.config_manager.get_mqtt_listener_config()
        else:
            return self.config.get('mqtt', {}).get('listener', {})
    
    def _build_topic_patterns(self, listener_config: dict) -> dict:
        """Build topic patterns for backward compatibility."""
        patterns = {}
        root = listener_config.get('root', 'mqtt')
        
        if 'result' in listener_config:
            patterns['result'] = f"{root}/+/+/{listener_config['result']}"
        if 'trace' in listener_config:
            patterns['trace'] = f"{root}/+/+/{listener_config['trace']}"
        if 'heads' in listener_config:
            patterns['heads'] = f"{root}/+/+/{listener_config['heads']}"
            
        return patterns
    
    def __del__(self):
        """Cleanup on deletion."""
        self._running = False