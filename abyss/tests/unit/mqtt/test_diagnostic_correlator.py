"""
Test suite for DiagnosticCorrelator.
"""

import pytest
import time
from unittest.mock import Mock, patch
from collections import deque

# Add the source directory to the path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from abyss.mqtt.components.diagnostic_correlator import DiagnosticCorrelator
from abyss.uos_depth_est import TimestampedData


class TestDiagnosticCorrelator:
    """Test cases for DiagnosticCorrelator."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'mqtt': {
                'listener': {
                    'root': 'OPCPUBSUB',
                    'result': 'ResultManagement',
                    'trace': 'ResultManagement/Trace',
                    'heads': 'AssetManagement/Heads',
                    'time_window': 30.0
                }
            }
        }
    
    @pytest.fixture
    def correlator(self, sample_config):
        """Create DiagnosticCorrelator instance."""
        return DiagnosticCorrelator(sample_config, time_window=30.0)
    
    @pytest.fixture
    def sample_messages(self):
        """Create sample messages for testing."""
        timestamp = time.time()
        return {
            'result': TimestampedData(
                timestamp,
                {'test': 'result_data'},
                'OPCPUBSUB/toolbox1/tool1/ResultManagement'
            ),
            'trace': TimestampedData(
                timestamp + 0.5,
                {'test': 'trace_data'},
                'OPCPUBSUB/toolbox1/tool1/ResultManagement/Trace'
            ),
            'heads': TimestampedData(
                timestamp + 0.3,
                {'test': 'heads_data'},
                'OPCPUBSUB/toolbox1/tool1/AssetManagement/Heads'
            )
        }
    
    def test_initialization(self, correlator):
        """Test correlator initialization."""
        assert correlator.throughput_monitor is not None
        assert isinstance(correlator.correlation_history, deque)
        assert len(correlator.orphan_messages) == 0
        assert correlator._correlation_attempts == 0
        assert correlator._successful_correlations == 0
    
    def test_correlation_history_recording(self, correlator, sample_messages):
        """Test that correlation attempts are recorded in history."""
        # Create buffers with messages
        buffers = {
            'OPCPUBSUB/+/+/ResultManagement': [sample_messages['result']],
            'OPCPUBSUB/+/+/ResultManagement/Trace': [sample_messages['trace']]
        }
        
        # Mock processor
        processor = Mock()
        processor.return_value = True
        
        # Initial history size
        initial_history_size = len(correlator.correlation_history)
        
        # Perform correlation
        result = correlator.find_and_process_matches(buffers, processor)
        
        # Check history was updated
        assert len(correlator.correlation_history) == initial_history_size + 1
        
        # Check history entry
        history_entry = correlator.correlation_history[-1]
        assert 'timestamp' in history_entry
        assert 'success' in history_entry
        assert history_entry['success'] == result
        assert 'buffer_sizes' in history_entry
        assert 'unprocessed_count' in history_entry
        assert 'correlation_time_ms' in history_entry
        assert 'orphan_count' in history_entry
    
    def test_orphan_tracking(self, correlator):
        """Test that unmatched old messages are tracked as orphans."""
        # Create old unmatched messages
        old_timestamp = time.time() - 70  # Older than 2x time window
        
        old_result = TimestampedData(
            old_timestamp,
            {'test': 'old_result'},
            'OPCPUBSUB/toolbox1/tool1/ResultManagement'
        )
        old_result.processed = False
        
        buffers = {
            'OPCPUBSUB/+/+/ResultManagement': [old_result],
            'OPCPUBSUB/+/+/ResultManagement/Trace': []  # No matching trace
        }
        
        # Process to trigger orphan tracking
        processor = Mock()
        correlator.find_and_process_matches(buffers, processor)
        
        # Check orphan was tracked
        assert 'toolbox1/tool1' in correlator.orphan_messages
        assert len(correlator.orphan_messages['toolbox1/tool1']) == 1
        assert correlator._orphaned_messages == 1
    
    def test_orphan_reconciliation(self, correlator):
        """Test that orphans can be matched with late-arriving messages."""
        # Add an orphaned trace message
        old_trace = TimestampedData(
            time.time() - 10,
            {'test': 'orphan_trace'},
            'OPCPUBSUB/toolbox1/tool1/ResultManagement/Trace'
        )
        
        correlator.orphan_messages['toolbox1/tool1'].append({
            'message': old_trace,
            'topic_pattern': 'OPCPUBSUB/+/+/ResultManagement/Trace',
            'orphaned_at': time.time() - 5,
            'message_age_when_orphaned': 5
        })
        
        # Create matching result message
        new_result = TimestampedData(
            time.time() - 9,  # Close in time to orphan
            {'test': 'new_result'},
            'OPCPUBSUB/toolbox1/tool1/ResultManagement'
        )
        
        # Try reconciliation
        reconciled = correlator.try_reconcile_orphan(new_result)
        
        assert len(reconciled) == 1
        assert len(reconciled[0]) == 2
        assert new_result in reconciled[0]
        assert old_trace in reconciled[0]
        
        # Orphan should be removed
        assert len(correlator.orphan_messages['toolbox1/tool1']) == 0
    
    def test_queue_growth_monitoring(self, correlator):
        """Test queue depth monitoring and growth rate calculation."""
        # Record queue depths over time
        base_time = time.time() - 10
        
        for i in range(10):
            correlator.queue_depth_history.append({
                'timestamp': base_time + i,
                'depth': 100 + i * 5  # Growing queue
            })
            correlator.last_queue_sample = base_time + i
        
        # Calculate growth rate
        growth_rate = correlator.get_queue_growth_rate()
        
        assert growth_rate is not None
        assert growth_rate > 0  # Queue is growing
        assert growth_rate == pytest.approx(5.0, rel=0.1)  # ~5 messages/second
    
    def test_throughput_monitoring_integration(self, correlator, sample_messages):
        """Test integration with throughput monitor."""
        # Create matched messages
        buffers = {
            'OPCPUBSUB/+/+/ResultManagement': [sample_messages['result']],
            'OPCPUBSUB/+/+/ResultManagement/Trace': [sample_messages['trace']]
        }
        
        # Process successfully
        processor = Mock()
        correlator.find_and_process_matches(buffers, processor)
        
        # Check throughput monitor was updated
        metrics = correlator.throughput_monitor.get_metrics()
        assert metrics['total_arrived'] > 0
        assert metrics['total_processed'] > 0
    
    def test_correlation_success_rate(self, correlator):
        """Test correlation success rate calculation."""
        # Simulate some successful and failed correlations
        correlator._correlation_attempts = 10
        correlator._successful_correlations = 7
        
        success_rate = correlator.get_correlation_success_rate()
        assert success_rate == 70.0
        
        # Test with no attempts
        correlator._correlation_attempts = 0
        success_rate = correlator.get_correlation_success_rate()
        assert success_rate == 100.0
    
    def test_diagnostic_metrics(self, correlator):
        """Test comprehensive diagnostic metrics."""
        # Add some test data
        correlator._correlation_attempts = 50
        correlator._successful_correlations = 45
        correlator._orphaned_messages = 5
        
        # Add some correlation history
        for i in range(10):
            correlator.correlation_history.append({
                'timestamp': time.time() - i,
                'success': True,
                'correlation_time_ms': 10 + i,
                'unprocessed_count': 20 - i
            })
        
        metrics = correlator.get_diagnostic_metrics()
        
        assert 'throughput_status' in metrics
        assert 'correlation_success_rate' in metrics
        assert metrics['correlation_success_rate'] == 90.0
        assert metrics['total_correlation_attempts'] == 50
        assert metrics['total_orphaned_messages'] == 5
        assert 'avg_correlation_time_ms' in metrics
        assert 'avg_unprocessed_count' in metrics
    
    def test_orphan_cleanup(self, correlator):
        """Test automatic cleanup of old orphans."""
        # Add some old orphans
        very_old_time = time.time() - 400  # Older than cleanup interval
        recent_time = time.time() - 100
        
        correlator.orphan_messages['tool1'] = [
            {
                'message': Mock(),
                'orphaned_at': very_old_time,
                'topic_pattern': 'test',
                'message_age_when_orphaned': 60
            },
            {
                'message': Mock(),
                'orphaned_at': recent_time,
                'topic_pattern': 'test',
                'message_age_when_orphaned': 60
            }
        ]
        
        # Force cleanup
        correlator.last_orphan_cleanup = 0
        correlator._cleanup_old_orphans()
        
        # Only recent orphan should remain
        assert len(correlator.orphan_messages['tool1']) == 1
        assert correlator.orphan_messages['tool1'][0]['orphaned_at'] == recent_time
    
    def test_get_correlation_history(self, correlator):
        """Test retrieving correlation history with limit."""
        # Add more entries than limit
        for i in range(150):
            correlator.correlation_history.append({
                'timestamp': time.time() - i,
                'success': i % 2 == 0
            })
        
        # Get limited history
        history = correlator.get_correlation_history(limit=50)
        
        assert len(history) == 50
        # Should get most recent entries
        assert history[-1]['timestamp'] > history[0]['timestamp']
    
    def test_reset_diagnostics(self, correlator):
        """Test resetting diagnostic data."""
        # Add some data
        correlator._correlation_attempts = 100
        correlator._successful_correlations = 90
        correlator.correlation_history.append({'test': 'data'})
        correlator.orphan_messages['tool1'] = [{'test': 'orphan'}]
        
        # Reset
        correlator.reset_diagnostics()
        
        # Verify reset
        assert correlator._correlation_attempts == 0
        assert correlator._successful_correlations == 0
        assert len(correlator.correlation_history) == 0
        assert len(correlator.orphan_messages) == 0
        assert correlator.throughput_monitor.total_arrived == 0
    
    def test_unprocessed_message_counting(self, correlator, sample_messages):
        """Test counting of unprocessed messages."""
        # Create buffers with mix of processed and unprocessed
        sample_messages['result'].processed = False
        sample_messages['trace'].processed = True
        
        buffers = {
            'OPCPUBSUB/+/+/ResultManagement': [sample_messages['result']],
            'OPCPUBSUB/+/+/ResultManagement/Trace': [sample_messages['trace']]
        }
        
        count = correlator._count_unprocessed_messages(buffers)
        assert count == 1  # Only result is unprocessed
    
    def test_oldest_message_age_calculation(self, correlator, sample_messages):
        """Test finding age of oldest unprocessed message."""
        current_time = time.time()
        
        # Set message timestamps
        sample_messages['result'].timestamp = current_time - 60  # 60 seconds old
        sample_messages['result'].processed = False
        sample_messages['trace'].timestamp = current_time - 30  # 30 seconds old
        sample_messages['trace'].processed = False
        
        buffers = {
            'OPCPUBSUB/+/+/ResultManagement': [sample_messages['result']],
            'OPCPUBSUB/+/+/ResultManagement/Trace': [sample_messages['trace']]
        }
        
        age = correlator._find_oldest_unprocessed_age(buffers)
        
        assert age is not None
        assert age == pytest.approx(60, rel=0.1)


class TestDiagnosticCorrelatorIntegration:
    """Integration tests for DiagnosticCorrelator."""
    
    def test_full_correlation_flow_with_diagnostics(self, sample_config):
        """Test complete correlation flow with diagnostic tracking."""
        correlator = DiagnosticCorrelator(sample_config)
        
        # Create messages at different times
        base_time = time.time()
        messages = []
        
        # Add 10 message pairs
        for i in range(10):
            result = TimestampedData(
                base_time + i * 2,
                {'index': i, 'type': 'result'},
                f'OPCPUBSUB/toolbox1/tool{i}/ResultManagement'
            )
            trace = TimestampedData(
                base_time + i * 2 + 0.1,
                {'index': i, 'type': 'trace'},
                f'OPCPUBSUB/toolbox1/tool{i}/ResultManagement/Trace'
            )
            messages.extend([result, trace])
        
        # Create buffers
        buffers = {
            'OPCPUBSUB/+/+/ResultManagement': [m for m in messages if 'ResultManagement' in m.source and not m.source.endswith('Trace')],
            'OPCPUBSUB/+/+/ResultManagement/Trace': [m for m in messages if m.source.endswith('Trace')]
        }
        
        # Process multiple times
        processor = Mock()
        for _ in range(5):
            correlator.find_and_process_matches(buffers, processor)
            time.sleep(0.1)
        
        # Check diagnostics
        metrics = correlator.get_diagnostic_metrics()
        
        assert metrics['correlation_success_rate'] > 0
        assert metrics['total_correlation_attempts'] == 5
        assert 'throughput_status' in metrics
        assert len(correlator.correlation_history) == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])