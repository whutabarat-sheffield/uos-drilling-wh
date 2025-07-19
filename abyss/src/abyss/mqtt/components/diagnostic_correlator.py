"""
Diagnostic Correlator Module

Enhanced message correlator with diagnostic capabilities for monitoring
throughput and identifying correlation issues.
"""

import time
import logging
from collections import deque, defaultdict
from typing import Dict, List, Set, Tuple, Any, Callable, Optional, Union

from .simple_correlator import SimpleMessageCorrelator
from .throughput_monitor import SimpleThroughputMonitor
from ...uos_depth_est import TimestampedData


class DiagnosticCorrelator(SimpleMessageCorrelator):
    """
    Enhanced correlator with diagnostic and throughput monitoring capabilities.
    
    Extends SimpleMessageCorrelator to add:
    - Throughput monitoring
    - Correlation history tracking
    - Orphan message tracking
    - Queue growth monitoring
    """
    
    def __init__(self, config: Union[Dict[str, Any], Any], time_window: float = 30.0):
        """
        Initialize DiagnosticCorrelator.
        
        Args:
            config: Configuration dictionary or ConfigurationManager instance
            time_window: Time window for message correlation (seconds)
        """
        super().__init__(config, time_window)
        
        # Throughput monitoring
        self.throughput_monitor = SimpleThroughputMonitor(sample_rate=0.1)
        
        # Correlation history for diagnostics
        self.correlation_history = deque(maxlen=1000)
        
        # Orphan message tracking
        self.orphan_messages = defaultdict(list)
        self.orphan_cleanup_interval = 300  # 5 minutes
        self.last_orphan_cleanup = time.time()
        
        # Queue monitoring
        self.queue_depth_history = deque(maxlen=3600)  # 1 hour at 1 sample/sec
        self.last_queue_sample = time.time()
        
        # Metrics
        self._correlation_attempts = 0
        self._successful_correlations = 0
        self._orphaned_messages = 0
        
    def find_and_process_matches(self, buffers: Dict[str, List[TimestampedData]], 
                                message_processor: Callable) -> bool:
        """
        Find and process messages with enhanced diagnostics.
        
        Args:
            buffers: Dictionary of message buffers by topic
            message_processor: Callback function to process matched messages
            
        Returns:
            True if any matches were found and processed, False otherwise
        """
        start_time = time.time()
        self._correlation_attempts += 1
        
        # Track buffer depths before processing
        total_unprocessed = self._count_unprocessed_messages(buffers)
        self.throughput_monitor.record_arrival()  # Each correlation attempt represents arrivals
        self._record_queue_depth(total_unprocessed)
        
        # Track oldest unprocessed message age
        oldest_age = self._find_oldest_unprocessed_age(buffers)
        
        # Perform correlation using parent class method
        matches_found = super().find_and_process_matches(buffers, message_processor)
        
        # Track successful correlations
        if matches_found:
            self._successful_correlations += 1
            self.throughput_monitor.record_processing_complete(start_time)
        
        # Record correlation attempt in history
        correlation_time = time.time() - start_time
        history_entry = {
            'timestamp': time.time(),
            'success': matches_found,
            'buffer_sizes': {k: len(v) for k, v in buffers.items()},
            'unprocessed_count': total_unprocessed,
            'oldest_message_age': oldest_age,
            'correlation_time_ms': correlation_time * 1000,
            'orphan_count': sum(len(msgs) for msgs in self.orphan_messages.values())
        }
        self.correlation_history.append(history_entry)
        
        # Track unmatched messages as potential orphans
        self._track_orphans(buffers)
        
        # Periodic orphan cleanup
        self._cleanup_old_orphans()
        
        # Check throughput status
        self.throughput_monitor.check_and_log_status()
        
        return matches_found
    
    def _count_unprocessed_messages(self, buffers: Dict[str, List[TimestampedData]]) -> int:
        """Count total unprocessed messages across all buffers."""
        total = 0
        for buffer in buffers.values():
            total += sum(1 for msg in buffer if not getattr(msg, 'processed', False))
        return total
    
    def _find_oldest_unprocessed_age(self, buffers: Dict[str, List[TimestampedData]]) -> Optional[float]:
        """Find age of oldest unprocessed message."""
        current_time = time.time()
        oldest_timestamp = None
        
        for buffer in buffers.values():
            for msg in buffer:
                if not getattr(msg, 'processed', False):
                    if oldest_timestamp is None or msg.timestamp < oldest_timestamp:
                        oldest_timestamp = msg.timestamp
        
        if oldest_timestamp is not None:
            return current_time - oldest_timestamp
        return None
    
    def _record_queue_depth(self, depth: int):
        """Record queue depth for monitoring."""
        current_time = time.time()
        
        # Sample at most once per second
        if current_time - self.last_queue_sample >= 1.0:
            self.queue_depth_history.append({
                'timestamp': current_time,
                'depth': depth
            })
            self.last_queue_sample = current_time
    
    def _track_orphans(self, buffers: Dict[str, List[TimestampedData]]):
        """Track messages that haven't been correlated as potential orphans."""
        current_time = time.time()
        
        for topic_pattern, buffer in buffers.items():
            for msg in buffer:
                if not getattr(msg, 'processed', False):
                    # Check if message is old enough to be considered orphaned
                    message_age = current_time - msg.timestamp
                    if message_age > self.time_window * 2:  # Double the correlation window
                        tool_key = self._extract_tool_key(msg.source)
                        if tool_key:
                            # Check if already tracked
                            already_tracked = any(
                                orphan['message'].timestamp == msg.timestamp and 
                                orphan['message'].source == msg.source
                                for orphan in self.orphan_messages[tool_key]
                            )
                            
                            if not already_tracked:
                                self.orphan_messages[tool_key].append({
                                    'message': msg,
                                    'topic_pattern': topic_pattern,
                                    'orphaned_at': current_time,
                                    'message_age_when_orphaned': message_age
                                })
                                self._orphaned_messages += 1
                                
                                logging.debug("Message orphaned", extra={
                                    'tool_key': tool_key,
                                    'topic_pattern': topic_pattern,
                                    'message_age': message_age,
                                    'source': msg.source
                                })
    
    def _cleanup_old_orphans(self):
        """Remove orphans that are too old to ever be correlated."""
        current_time = time.time()
        
        if current_time - self.last_orphan_cleanup < self.orphan_cleanup_interval:
            return
        
        self.last_orphan_cleanup = current_time
        total_removed = 0
        
        for tool_key in list(self.orphan_messages.keys()):
            orphans = self.orphan_messages[tool_key]
            
            # Remove orphans older than cleanup interval
            orphans[:] = [
                orphan for orphan in orphans
                if current_time - orphan['orphaned_at'] < self.orphan_cleanup_interval
            ]
            
            removed = len(self.orphan_messages[tool_key]) - len(orphans)
            total_removed += removed
            
            # Remove empty entries
            if not orphans:
                del self.orphan_messages[tool_key]
        
        if total_removed > 0:
            logging.debug(f"Cleaned up {total_removed} old orphaned messages")
    
    def try_reconcile_orphan(self, new_message: TimestampedData) -> List[List[TimestampedData]]:
        """
        Try to match a new message with orphaned messages.
        
        Args:
            new_message: Newly arrived message
            
        Returns:
            List of reconciled message sets
        """
        reconciled = []
        tool_key = self._extract_tool_key(new_message.source)
        
        if not tool_key or tool_key not in self.orphan_messages:
            return reconciled
        
        # Determine message type
        message_type = self._get_message_type(new_message.source)
        if not message_type:
            return reconciled
        
        # Try to match with orphans
        orphans = self.orphan_messages[tool_key]
        
        for orphan_data in orphans[:]:  # Copy to allow modification
            orphan_msg = orphan_data['message']
            orphan_type = self._get_message_type(orphan_msg.source)
            
            # Don't match same type messages
            if orphan_type == message_type:
                continue
            
            # Check if within extended time window (3x normal)
            if abs(new_message.timestamp - orphan_msg.timestamp) <= self.time_window * 3:
                # Found a match!
                match_set = [new_message, orphan_msg]
                reconciled.append(match_set)
                
                # Remove from orphans
                orphans.remove(orphan_data)
                
                logging.info("Orphaned message reconciled", extra={
                    'tool_key': tool_key,
                    'orphan_age': time.time() - orphan_data['orphaned_at'],
                    'time_difference': abs(new_message.timestamp - orphan_msg.timestamp)
                })
        
        return reconciled
    
    def _get_message_type(self, source: str) -> Optional[str]:
        """Determine message type from source topic."""
        for msg_type, pattern in self._topic_patterns.items():
            if msg_type in ['result', 'trace', 'heads']:
                topic_suffix = pattern.split('/')[-1]
                if topic_suffix in source:
                    return msg_type
        return None
    
    def get_queue_growth_rate(self) -> Optional[float]:
        """
        Calculate queue growth rate over recent history.
        
        Returns:
            Growth rate in messages per second, or None if insufficient data
        """
        if len(self.queue_depth_history) < 10:
            return None
        
        # Use last 60 seconds of data
        recent_history = [
            entry for entry in self.queue_depth_history
            if entry['timestamp'] > time.time() - 60
        ]
        
        if len(recent_history) < 2:
            return None
        
        # Simple linear regression
        start_depth = recent_history[0]['depth']
        end_depth = recent_history[-1]['depth']
        time_span = recent_history[-1]['timestamp'] - recent_history[0]['timestamp']
        
        if time_span > 0:
            return (end_depth - start_depth) / time_span
        
        return 0.0
    
    def get_correlation_success_rate(self) -> float:
        """
        Get the correlation success rate.
        
        Returns:
            Success rate as percentage (0-100)
        """
        if self._correlation_attempts == 0:
            return 100.0
        
        return (self._successful_correlations / self._correlation_attempts) * 100
    
    def get_diagnostic_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive diagnostic metrics.
        
        Returns:
            Dictionary of diagnostic information
        """
        # Get throughput status
        throughput_status = self.throughput_monitor.get_status()
        
        # Calculate additional metrics
        recent_history = list(self.correlation_history)[-100:]  # Last 100 attempts
        if recent_history:
            avg_correlation_time = sum(h['correlation_time_ms'] for h in recent_history) / len(recent_history)
            avg_unprocessed = sum(h['unprocessed_count'] for h in recent_history) / len(recent_history)
        else:
            avg_correlation_time = 0
            avg_unprocessed = 0
        
        return {
            'throughput_status': throughput_status.status,
            'throughput_details': throughput_status.details,
            'correlation_success_rate': self.get_correlation_success_rate(),
            'total_correlation_attempts': self._correlation_attempts,
            'total_orphaned_messages': self._orphaned_messages,
            'current_orphan_count': sum(len(msgs) for msgs in self.orphan_messages.values()),
            'orphans_by_tool': {k: len(v) for k, v in self.orphan_messages.items()},
            'avg_correlation_time_ms': avg_correlation_time,
            'avg_unprocessed_count': avg_unprocessed,
            'queue_growth_rate': self.get_queue_growth_rate(),
            'correlation_history_size': len(self.correlation_history)
        }
    
    def get_correlation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent correlation history.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of correlation history entries
        """
        return list(self.correlation_history)[-limit:]
    
    def reset_diagnostics(self):
        """Reset diagnostic metrics while preserving configuration."""
        self.throughput_monitor.reset_metrics()
        self.correlation_history.clear()
        self.orphan_messages.clear()
        self.queue_depth_history.clear()
        self._correlation_attempts = 0
        self._successful_correlations = 0
        self._orphaned_messages = 0
        self.last_orphan_cleanup = time.time()
        self.last_queue_sample = time.time()