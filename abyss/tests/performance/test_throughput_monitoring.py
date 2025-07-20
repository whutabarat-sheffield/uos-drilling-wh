"""
Test suite for SimpleThroughputMonitor.
"""

import pytest
import time
from unittest.mock import patch

# Add the source directory to the path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from abyss.mqtt.components.throughput_monitor import SimpleThroughputMonitor, ThroughputStatus


class TestSimpleThroughputMonitor:
    """Test cases for SimpleThroughputMonitor."""
    
    @pytest.fixture
    def monitor(self):
        """Create a monitor with 100% sampling for testing."""
        return SimpleThroughputMonitor(sample_rate=1.0, window_size=100)
    
    def test_initialization(self):
        """Test monitor initialization."""
        monitor = SimpleThroughputMonitor(sample_rate=0.1, window_size=3600)
        
        assert monitor.sample_rate == 0.1
        assert monitor.window_size == 3600
        assert monitor.total_arrived == 0
        assert monitor.total_processed == 0
        assert len(monitor.arrival_times) == 0
        assert len(monitor.processing_times) == 0
    
    def test_sampling_logic(self):
        """Test that sampling works correctly."""
        monitor = SimpleThroughputMonitor(sample_rate=0.1)  # 10% sampling
        
        samples_taken = 0
        for i in range(100):
            if monitor.should_sample():
                samples_taken += 1
        
        # Should be approximately 10 samples (±2 for randomness)
        assert 8 <= samples_taken <= 12
    
    def test_record_arrival(self, monitor):
        """Test recording message arrivals."""
        # Record some arrivals
        for i in range(5):
            monitor.record_arrival(timestamp=100.0 + i)
        
        assert monitor.total_arrived == 5
        assert len(monitor.arrival_times) == 5
        assert monitor.arrival_times[0] == 100.0
        assert monitor.arrival_times[-1] == 104.0
    
    def test_record_processing_complete(self, monitor):
        """Test recording processing completion."""
        # Record some processing completions
        for i in range(5):
            start_time = 100.0 + i
            end_time = start_time + 0.1  # 100ms processing time
            monitor.record_processing_complete(start_time, timestamp=end_time)
        
        assert monitor.total_processed == 5
        assert len(monitor.processing_times) == 5
        assert len(monitor.processing_durations) == 5
        assert all(d == pytest.approx(0.1) for d in monitor.processing_durations)
    
    def test_arrival_rate_calculation(self, monitor):
        """Test arrival rate calculation."""
        # No arrivals yet
        assert monitor.get_arrival_rate() == 0.0
        
        # Record arrivals over 10 seconds (1 per second)
        for i in range(10):
            monitor.record_arrival(timestamp=100.0 + i)
        
        # Should be approximately 1 msg/s (10 messages over 9 seconds = 1.11, but close to 1.0)
        rate = monitor.get_arrival_rate()
        assert rate == pytest.approx(1.11, rel=0.1)
    
    def test_processing_rate_calculation(self, monitor):
        """Test processing rate calculation."""
        # No processing yet
        assert monitor.get_processing_rate() == 0.0
        
        # Record processing over 10 seconds (2 per second)
        for i in range(21):
            start_time = 100.0 + i * 0.5
            monitor.record_processing_complete(start_time, timestamp=start_time + 0.01)
        
        # Should be approximately 2 msg/s
        rate = monitor.get_processing_rate()
        assert rate == pytest.approx(2.0, rel=0.1)
    
    def test_avg_processing_time(self, monitor):
        """Test average processing time calculation."""
        # No processing yet
        assert monitor.get_avg_processing_time() == 0.0
        
        # Record various processing times
        processing_times = [0.01, 0.02, 0.015, 0.025, 0.02]  # seconds
        for i, duration in enumerate(processing_times):
            start_time = 100.0 + i
            monitor.record_processing_complete(start_time, timestamp=start_time + duration)
        
        avg_time = monitor.get_avg_processing_time()
        expected_avg = sum(processing_times) / len(processing_times)
        assert avg_time == pytest.approx(expected_avg)
    
    def test_status_healthy_high_headroom(self, monitor):
        """Test status when system is healthy with high headroom."""
        # Simulate 10 msg/s arrival, 20 msg/s processing
        base_time = 100.0
        
        # Arrivals: 10 per second for 5 seconds
        for i in range(50):
            monitor.record_arrival(timestamp=base_time + i * 0.1)
        
        # Processing: 20 per second for 5 seconds
        for i in range(100):
            start = base_time + i * 0.05
            monitor.record_processing_complete(start, timestamp=start + 0.01)
        
        status = monitor.get_status()
        
        assert status.status == 'HEALTHY'
        assert 'keeping up' in status.message
        # Allow for precision differences in the calculation
        assert 'spare capacity' in status.details['headroom']
        # Extract the percentage and check it's close to 100%
        headroom_text = status.details['headroom']
        headroom_pct = float(headroom_text.split('%')[0])
        assert headroom_pct >= 95.0  # Allow some tolerance for calculation precision
    
    def test_status_healthy_low_headroom(self, monitor):
        """Test status when system is healthy but with low headroom."""
        # Simulate 10 msg/s arrival, 11 msg/s processing (10% headroom)
        base_time = 100.0
        
        # Arrivals: 10 per second for 5 seconds
        for i in range(50):
            monitor.record_arrival(timestamp=base_time + i * 0.1)
        
        # Processing: 11 per second for 5 seconds
        for i in range(55):
            start = base_time + i * 0.0909
            monitor.record_processing_complete(start, timestamp=start + 0.01)
        
        status = monitor.get_status()
        
        assert status.status == 'HEALTHY'
        assert 'low headroom' in status.message
        assert 'warning' in status.details
    
    def test_status_falling_behind(self, monitor):
        """Test status when system is falling behind."""
        # Simulate 10 msg/s arrival, 5 msg/s processing
        base_time = 100.0
        
        # Arrivals: 10 per second for 5 seconds
        for i in range(50):
            monitor.record_arrival(timestamp=base_time + i * 0.1)
        
        # Processing: 5 per second for 5 seconds
        for i in range(25):
            start = base_time + i * 0.2
            monitor.record_processing_complete(start, timestamp=start + 0.01)
        
        status = monitor.get_status()
        
        assert status.status == 'FALLING_BEHIND'
        assert 'cannot keep up' in status.message
        # Allow for precision differences in the calculation
        assert 'shortfall' in status.details['deficit']
        # Extract the percentage and check it's close to 50%
        deficit_text = status.details['deficit']
        deficit_pct = float(deficit_text.split('%')[0])
        assert 45.0 <= deficit_pct <= 55.0  # Allow some tolerance for calculation precision
        assert status.details['recommendation'] == 'Monitor closely'
    
    def test_status_falling_behind_consistent(self, monitor):
        """Test recommendation changes after consistent falling behind."""
        # Simulate falling behind
        base_time = 100.0
        
        # Check status multiple times while falling behind
        for check in range(4):
            # Add more arrivals than processing
            for i in range(10):
                monitor.record_arrival(timestamp=base_time + check * 10 + i)
            
            for i in range(5):
                start = base_time + check * 10 + i * 2
                monitor.record_processing_complete(start, timestamp=start + 0.01)
            
            status = monitor.get_status()
            
            if check < 2:
                assert status.details['recommendation'] == 'Monitor closely'
            else:
                assert status.details['recommendation'] == 'Add processing instance'
    
    def test_check_and_log_status(self, monitor, caplog):
        """Test automatic status checking and logging."""
        # Set up falling behind scenario
        base_time = time.time()
        
        for i in range(20):
            monitor.record_arrival(timestamp=base_time + i * 0.1)
        
        for i in range(10):
            monitor.record_processing_complete(base_time + i * 0.2, timestamp=base_time + i * 0.2 + 0.01)
        
        # Should not log immediately
        monitor.check_and_log_status()
        assert len(caplog.records) == 0
        
        # Simulate time passing
        monitor.last_health_check = base_time - 10
        monitor.check_and_log_status()
        
        # Should have logged a warning
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == 'WARNING'
        assert 'cannot keep up' in caplog.records[0].message
    
    def test_get_metrics(self, monitor):
        """Test comprehensive metrics retrieval."""
        # Add some data
        for i in range(10):
            monitor.record_arrival(timestamp=100.0 + i)
            monitor.record_processing_complete(100.0 + i, timestamp=100.0 + i + 0.02)
        
        metrics = monitor.get_metrics()
        
        assert 'status' in metrics
        assert 'message' in metrics
        assert 'arrival_rate' in metrics
        assert 'processing_rate' in metrics
        assert 'avg_processing_time_ms' in metrics
        assert metrics['total_arrived'] == 10
        assert metrics['total_processed'] == 10
        assert metrics['sample_rate'] == 1.0
    
    def test_reset_metrics(self, monitor):
        """Test metrics reset functionality."""
        # Add some data
        for i in range(10):
            monitor.record_arrival()
            monitor.record_processing_complete(time.time())
        
        # Verify data exists
        assert monitor.total_arrived == 10
        assert monitor.total_processed == 10
        assert len(monitor.arrival_times) > 0
        
        # Reset
        monitor.reset_metrics()
        
        # Verify reset
        assert monitor.total_arrived == 0
        assert monitor.total_processed == 0
        assert len(monitor.arrival_times) == 0
        assert len(monitor.processing_times) == 0
        assert monitor.consecutive_falling_behind == 0


class TestThroughputMonitorWithSampling:
    """Test throughput monitor with realistic sampling rates."""
    
    def test_extrapolation_from_samples(self):
        """Test that rate calculations correctly extrapolate from samples."""
        monitor = SimpleThroughputMonitor(sample_rate=0.1)  # 10% sampling
        
        # Simulate 100 messages arriving at 10 msg/s
        base_time = 100.0
        for i in range(100):
            monitor.record_arrival(timestamp=base_time + i * 0.1)
        
        # Only ~10 should be sampled, but rate should still be ~10 msg/s
        assert len(monitor.arrival_times) < 20  # Some sampling variance
        
        rate = monitor.get_arrival_rate()
        assert rate == pytest.approx(10.0, rel=0.3)  # Allow 30% variance due to sampling
    
    def test_low_overhead_sampling(self):
        """Test that sampling reduces overhead."""
        # Use a smaller window for this test
        monitor_full = SimpleThroughputMonitor(sample_rate=1.0, window_size=100)
        monitor_sampled = SimpleThroughputMonitor(sample_rate=0.001, window_size=100)  # 0.1% sampling
        
        # Process many messages
        for i in range(10000):
            monitor_full.record_arrival()
            monitor_sampled.record_arrival()
        
        # Sampled monitor should have much fewer stored values
        assert len(monitor_full.arrival_times) == 100  # Limited by window
        assert len(monitor_sampled.arrival_times) <= 10  # Very few samples


if __name__ == '__main__':
    pytest.main([__file__, '-v'])