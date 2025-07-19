"""
Throughput Monitor Module

Provides simple monitoring of message processing throughput to determine
if the system is keeping up with message arrival rates.
"""

import time
import logging
from collections import deque
from typing import Dict, Any, Optional, Deque
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ThroughputStatus:
    """Status of the throughput monitoring."""
    status: str  # 'HEALTHY' or 'FALLING_BEHIND'
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class SimpleThroughputMonitor:
    """
    Simple monitor to track if processing is keeping up with message arrival rate.
    
    Provides a clear yes/no answer on system health with minimal overhead.
    """
    
    def __init__(self, sample_rate: float = 0.1, window_size: int = 3600):
        """
        Initialize SimpleThroughputMonitor.
        
        Args:
            sample_rate: Fraction of messages to sample (0.1 = 10%)
            window_size: Size of sliding window for rate calculations
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        
        # Metrics storage with sliding windows
        self.arrival_times: Deque[float] = deque(maxlen=window_size)
        self.processing_times: Deque[float] = deque(maxlen=window_size)
        self.processing_durations: Deque[float] = deque(maxlen=1000)
        
        # Counters for rate calculation
        self.total_arrived = 0
        self.total_processed = 0
        self.sample_counter = 0
        
        # Health tracking
        self.last_health_check = time.time()
        self.consecutive_falling_behind = 0
        self.health_check_interval = 5.0  # seconds
        
    def should_sample(self) -> bool:
        """Determine if current message should be sampled."""
        self.sample_counter += 1
        return (self.sample_counter % int(1 / self.sample_rate)) == 0
    
    def record_arrival(self, timestamp: Optional[float] = None):
        """
        Record message arrival.
        
        Args:
            timestamp: Optional timestamp, defaults to current time
        """
        self.total_arrived += 1
        
        if self.should_sample():
            self.arrival_times.append(timestamp or time.time())
    
    def record_processing_complete(self, start_time: float, timestamp: Optional[float] = None):
        """
        Record completion of message processing.
        
        Args:
            start_time: When processing started
            timestamp: When processing completed (defaults to current time)
        """
        self.total_processed += 1
        
        if self.should_sample():
            end_time = timestamp or time.time()
            self.processing_times.append(end_time)
            self.processing_durations.append(end_time - start_time)
    
    def get_arrival_rate(self) -> float:
        """
        Calculate current message arrival rate.
        
        Returns:
            Messages per second arrival rate
        """
        if len(self.arrival_times) < 2:
            return 0.0
        
        time_span = self.arrival_times[-1] - self.arrival_times[0]
        if time_span <= 0:
            return 0.0
        
        # Extrapolate from samples
        sampled_rate = len(self.arrival_times) / time_span
        return sampled_rate / self.sample_rate
    
    def get_processing_rate(self) -> float:
        """
        Calculate current message processing rate.
        
        Returns:
            Messages per second processing rate
        """
        if len(self.processing_times) < 2:
            return 0.0
        
        time_span = self.processing_times[-1] - self.processing_times[0]
        if time_span <= 0:
            return 0.0
        
        # Extrapolate from samples
        sampled_rate = len(self.processing_times) / time_span
        return sampled_rate / self.sample_rate
    
    def get_avg_processing_time(self) -> float:
        """
        Get average processing time per message.
        
        Returns:
            Average processing time in seconds
        """
        if not self.processing_durations:
            return 0.0
        
        return sum(self.processing_durations) / len(self.processing_durations)
    
    def get_status(self) -> ThroughputStatus:
        """
        Get current throughput status.
        
        Returns:
            ThroughputStatus with health assessment
        """
        arrival_rate = self.get_arrival_rate()
        processing_rate = self.get_processing_rate()
        
        # Calculate if keeping up (with 5% tolerance)
        keeping_up = processing_rate >= arrival_rate * 0.95
        
        # Calculate rates and capacity
        if arrival_rate > 0:
            capacity_ratio = processing_rate / arrival_rate
        else:
            capacity_ratio = 1.0  # No arrivals means we're keeping up
        
        # Create status
        if keeping_up:
            self.consecutive_falling_behind = 0
            
            if capacity_ratio > 1.5:
                headroom_pct = (capacity_ratio - 1) * 100
                status = ThroughputStatus(
                    status='HEALTHY',
                    message='Processing keeping up with arrivals',
                    details={
                        'headroom': f'{headroom_pct:.1f}% spare capacity',
                        'arrival_rate': f'{arrival_rate:.1f} msg/s',
                        'processing_rate': f'{processing_rate:.1f} msg/s',
                        'avg_processing_time_ms': f'{self.get_avg_processing_time() * 1000:.1f}'
                    }
                )
            else:
                headroom_pct = (capacity_ratio - 1) * 100
                status = ThroughputStatus(
                    status='HEALTHY',
                    message='Processing keeping up with arrivals (low headroom)',
                    details={
                        'headroom': f'{headroom_pct:.1f}% spare capacity',
                        'arrival_rate': f'{arrival_rate:.1f} msg/s',
                        'processing_rate': f'{processing_rate:.1f} msg/s',
                        'avg_processing_time_ms': f'{self.get_avg_processing_time() * 1000:.1f}',
                        'warning': 'Consider monitoring closely'
                    }
                )
        else:
            self.consecutive_falling_behind += 1
            deficit_pct = (1 - capacity_ratio) * 100
            
            # Recommend scaling after consistent falling behind
            if self.consecutive_falling_behind >= 3:
                recommendation = 'Add processing instance'
            else:
                recommendation = 'Monitor closely'
            
            status = ThroughputStatus(
                status='FALLING_BEHIND',
                message='Processing cannot keep up',
                details={
                    'deficit': f'{deficit_pct:.1f}% shortfall',
                    'arrival_rate': f'{arrival_rate:.1f} msg/s',
                    'processing_rate': f'{processing_rate:.1f} msg/s',
                    'avg_processing_time_ms': f'{self.get_avg_processing_time() * 1000:.1f}',
                    'consecutive_checks_behind': self.consecutive_falling_behind,
                    'recommendation': recommendation
                }
            )
        
        return status
    
    def check_and_log_status(self):
        """Check status and log if necessary."""
        current_time = time.time()
        
        if current_time - self.last_health_check >= self.health_check_interval:
            self.last_health_check = current_time
            status = self.get_status()
            
            if status.status == 'FALLING_BEHIND':
                logging.warning(
                    f"Throughput Monitor: {status.message}",
                    extra={
                        'status': status.status,
                        'details': status.details
                    }
                )
            elif 'warning' in status.details:
                logging.info(
                    f"Throughput Monitor: {status.message}",
                    extra={
                        'status': status.status,
                        'details': status.details
                    }
                )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics.
        
        Returns:
            Dictionary of current metrics
        """
        status = self.get_status()
        
        return {
            'status': status.status,
            'message': status.message,
            'arrival_rate': self.get_arrival_rate(),
            'processing_rate': self.get_processing_rate(),
            'avg_processing_time_ms': self.get_avg_processing_time() * 1000,
            'total_arrived': self.total_arrived,
            'total_processed': self.total_processed,
            'samples_collected': len(self.arrival_times),
            'sample_rate': self.sample_rate,
            **status.details
        }
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.arrival_times.clear()
        self.processing_times.clear()
        self.processing_durations.clear()
        self.total_arrived = 0
        self.total_processed = 0
        self.sample_counter = 0
        self.consecutive_falling_behind = 0
        self.last_health_check = time.time()