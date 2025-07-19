"""
Test suite for BottleneckProfiler.
"""

import pytest
import time
import json
from unittest.mock import Mock, MagicMock

# Add the source directory to the path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from abyss.mqtt.components.bottleneck_profiler import BottleneckProfiler, BottleneckReport


class TestBottleneckProfiler:
    """Test cases for BottleneckProfiler."""
    
    @pytest.fixture
    def profiler(self):
        """Create a profiler with 100% sampling for testing."""
        return BottleneckProfiler(sample_rate=1.0)
    
    def test_initialization(self):
        """Test profiler initialization."""
        profiler = BottleneckProfiler(sample_rate=0.1)
        
        assert profiler.sample_rate == 0.1
        assert profiler.sample_counter == 0
        assert profiler.profiling_enabled is False
        assert len(profiler.stage_timings) == 0
    
    def test_sampling_logic(self):
        """Test that sampling works correctly."""
        profiler = BottleneckProfiler(sample_rate=0.1)  # 10% sampling
        profiler.profiling_enabled = True
        
        samples_taken = 0
        for i in range(100):
            if profiler.should_profile():
                samples_taken += 1
        
        # Should be approximately 10 samples
        assert 8 <= samples_taken <= 12
    
    def test_profile_section_basic(self, profiler):
        """Test basic section profiling."""
        profiler.profiling_enabled = True
        
        with profiler.profile_section('test_section'):
            time.sleep(0.01)  # 10ms operation
        
        assert 'test_section' in profiler.stage_timings
        assert len(profiler.stage_timings['test_section']) == 1
        assert profiler.stage_timings['test_section'][0] >= 0.01
    
    def test_profile_section_nested(self, profiler):
        """Test nested section profiling."""
        profiler.profiling_enabled = True
        
        with profiler.profile_section('outer'):
            time.sleep(0.01)
            with profiler.profile_section('inner'):
                time.sleep(0.01)
        
        assert 'outer' in profiler.stage_timings
        assert 'inner' in profiler.stage_timings
        assert profiler.stage_timings['outer'][0] >= 0.02
        assert profiler.stage_timings['inner'][0] >= 0.01
    
    def test_profile_section_disabled(self, profiler):
        """Test that profiling doesn't run when disabled."""
        profiler.profiling_enabled = False
        
        with profiler.profile_section('test_section'):
            time.sleep(0.01)
        
        assert len(profiler.stage_timings) == 0
    
    def test_profile_individual_stage(self, profiler):
        """Test profiling individual function calls."""
        profiler.profiling_enabled = True
        
        def slow_function(duration):
            time.sleep(duration)
            return 'result'
        
        result = profiler.profile_individual_stage('slow_op', slow_function, 0.01)
        
        assert result == 'result'
        assert 'slow_op' in profiler.stage_timings
        assert profiler.stage_timings['slow_op'][0] >= 0.01
    
    def test_generate_profile_report_no_data(self, profiler):
        """Test report generation with no data."""
        report = profiler.generate_profile_report()
        
        assert 'error' in report
        assert 'No profiling data' in report['error']
    
    def test_generate_profile_report_with_data(self, profiler):
        """Test report generation with profiling data."""
        profiler.profiling_enabled = True
        
        # Simulate different stage timings
        stages = {
            'json_parsing': 0.001,    # 1ms
            'correlation': 0.005,     # 5ms
            'depth_estimation': 0.020, # 20ms (bottleneck)
            'mqtt_publish': 0.004     # 4ms
        }
        
        # Profile each stage multiple times
        for _ in range(10):
            for stage, duration in stages.items():
                with profiler.profile_section(stage):
                    time.sleep(duration)
        
        report = profiler.generate_profile_report()
        
        assert report['primary_bottleneck'] == 'depth_estimation'
        assert report['bottleneck_percentage'] > 60  # Should be ~66%
        assert 'depth_estimation' in report['stage_timings_ms']
        assert report['stage_timings_ms']['depth_estimation'] >= 20
        assert len(report['recommendations']) > 0
        assert report['confidence'] == 0.1  # 10 samples / 100
    
    def test_recommendations_for_depth_estimation_bottleneck(self, profiler):
        """Test recommendations when depth estimation is bottleneck."""
        profiler.profiling_enabled = True
        
        # Make depth estimation the clear bottleneck
        with profiler.profile_section('depth_estimation'):
            time.sleep(0.150)  # 150ms - slow
        
        with profiler.profile_section('correlation'):
            time.sleep(0.010)
        
        report = profiler.generate_profile_report()
        recommendations = report['recommendations']
        
        assert any('Depth estimation is the bottleneck' in r for r in recommendations)
        assert any('GPU acceleration' in r for r in recommendations)
    
    def test_recommendations_for_mqtt_bottleneck(self, profiler):
        """Test recommendations when MQTT is bottleneck."""
        profiler.profiling_enabled = True
        
        # Make MQTT the bottleneck
        with profiler.profile_section('mqtt_publish'):
            time.sleep(0.100)
        
        with profiler.profile_section('depth_estimation'):
            time.sleep(0.010)
        
        report = profiler.generate_profile_report()
        recommendations = report['recommendations']
        
        assert any('MQTT publishing is the bottleneck' in r for r in recommendations)
        assert any('network latency' in r for r in recommendations)
    
    def test_identify_bottleneck_from_report(self, profiler):
        """Test BottleneckReport creation."""
        profile_data = {
            'primary_bottleneck': 'correlation',
            'bottleneck_percentage': 45.5,
            'stage_timings_ms': {'correlation': 45.5, 'depth_estimation': 30.0},
            'recommendations': ['Test recommendation'],
            'confidence': 0.8
        }
        
        report = profiler.identify_bottleneck(profile_data)
        
        assert isinstance(report, BottleneckReport)
        assert report.primary_bottleneck == 'correlation'
        assert report.bottleneck_percentage == 45.5
        assert report.confidence == 0.8
    
    def test_profile_processing_pipeline_mock(self, profiler):
        """Test profiling complete pipeline with mocks."""
        # Create mocks
        correlator = Mock()
        correlator.find_matches.return_value = [['msg1', 'msg2']]
        
        processor = Mock()
        processor.process_matching_messages.return_value = {'result': 'data'}
        
        publisher = Mock()
        
        # Profile pipeline
        messages = ['{"test": "data1"}', '{"test": "data2"}']
        results = profiler.profile_processing_pipeline(
            messages, correlator, processor, publisher
        )
        
        # Verify all stages were profiled
        assert 'primary_bottleneck' in results
        assert 'stage_timings_ms' in results
        assert all(stage in results['stage_timings_ms'] 
                  for stage in ['json_parsing', 'correlation', 'depth_estimation', 'mqtt_publish'])
    
    def test_reset_profiling_data(self, profiler):
        """Test resetting profiling data."""
        profiler.profiling_enabled = True
        
        # Add some data
        with profiler.profile_section('test'):
            time.sleep(0.001)
        
        assert len(profiler.stage_timings) > 0
        
        # Reset
        profiler.reset_profiling_data()
        
        assert len(profiler.stage_timings) == 0
        assert profiler.sample_counter == 0
    
    def test_get_summary_stats(self, profiler):
        """Test getting summary statistics."""
        profiler.profiling_enabled = True
        
        # Add some profiling data
        with profiler.profile_section('stage1'):
            pass
        with profiler.profile_section('stage2'):
            pass
        with profiler.profile_section('stage2'):
            pass
        
        stats = profiler.get_summary_stats()
        
        assert set(stats['stages_profiled']) == {'stage1', 'stage2'}
        assert stats['samples_collected']['stage1'] == 1
        assert stats['samples_collected']['stage2'] == 2
        assert stats['sample_rate'] == 1.0
    
    def test_thread_safety(self, profiler):
        """Test thread-safe operation of profiler."""
        import threading
        
        profiler.profiling_enabled = True
        results = []
        
        def profile_in_thread(thread_id):
            with profiler.profile_section(f'thread_{thread_id}'):
                time.sleep(0.01)
            results.append(thread_id)
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=profile_in_thread, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All threads should have completed
        assert len(results) == 5
        
        # Check that timings were recorded for each thread
        for i in range(5):
            assert f'thread_{i}' in profiler.stage_timings


class TestBottleneckProfilerIntegration:
    """Integration tests for BottleneckProfiler."""
    
    def test_realistic_pipeline_profiling(self):
        """Test profiling with realistic timing patterns."""
        profiler = BottleneckProfiler(sample_rate=1.0)
        profiler.profiling_enabled = True
        
        # Simulate realistic processing pattern
        for i in range(20):
            # JSON parsing - fast
            with profiler.profile_section('json_parsing'):
                time.sleep(0.0005)  # 0.5ms
            
            # Correlation - moderate  
            with profiler.profile_section('correlation'):
                time.sleep(0.002)   # 2ms
            
            # Depth estimation - slow (bottleneck)
            with profiler.profile_section('depth_estimation'):
                # Simulate variance in processing time
                base_time = 0.015
                variance = 0.005 if i % 3 == 0 else 0
                time.sleep(base_time + variance)  # 15-20ms
            
            # Publishing - moderate
            with profiler.profile_section('mqtt_publish'):
                time.sleep(0.003)   # 3ms
        
        report = profiler.generate_profile_report()
        
        # Verify analysis
        assert report['primary_bottleneck'] == 'depth_estimation'
        assert report['bottleneck_percentage'] > 70
        assert report['confidence'] >= 0.2  # 20 samples
        assert 'Scale horizontally' in ' '.join(report['recommendations'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])