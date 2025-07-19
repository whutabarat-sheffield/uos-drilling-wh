"""
Realistic load pattern tests for drilling system.

Tests the system under realistic drilling operation patterns including:
- Active drilling periods
- Bit changes (no data)
- Calibration bursts
- Normal operation
"""

import pytest
import time
import random
import threading
from collections import deque
from typing import List, Generator

# Add the source directory to the path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from abyss.mqtt.components.diagnostic_correlator import DiagnosticCorrelator
from abyss.mqtt.components.throughput_monitor import SimpleThroughputMonitor
from abyss.mqtt.components.message_buffer import MessageBuffer
from abyss.uos_depth_est import TimestampedData


class RealisticLoadGenerator:
    """Generate realistic drilling message patterns."""
    
    def __init__(self):
        self.patterns = {
            'normal_operation': {
                'rate': lambda: random.gauss(5, 1),  # 5±1 msg/s
                'duration': 300,  # 5 minutes
                'description': 'Normal drilling operation'
            },
            'active_drilling': {
                'rate': lambda: random.gauss(10, 2),  # 10±2 msg/s
                'duration': 600,  # 10 minutes
                'description': 'Active drilling phase'
            },
            'bit_change': {
                'rate': lambda: 0,  # No messages
                'duration': 180,  # 3 minutes
                'description': 'Bit change - no data'
            },
            'calibration': {
                'rate': lambda: random.gauss(50, 10),  # 50±10 msg/s burst
                'duration': 60,  # 1 minute
                'description': 'Sensor calibration burst'
            },
            'ramp_up': {
                'rate': lambda t: min(20, 2 + t/10),  # Gradual increase
                'duration': 120,  # 2 minutes
                'description': 'Ramping up after bit change'
            }
        }
    
    def generate_drilling_sequence(self) -> List[tuple]:
        """Generate a realistic drilling sequence."""
        # Typical drilling sequence over ~30 minutes
        sequence = [
            ('normal_operation', 300),    # 5 min normal
            ('active_drilling', 600),     # 10 min drilling
            ('bit_change', 180),         # 3 min bit change
            ('ramp_up', 120),           # 2 min ramp up
            ('calibration', 60),        # 1 min calibration
            ('active_drilling', 600),    # 10 min drilling
            ('normal_operation', 180)    # 3 min normal
        ]
        return sequence
    
    def generate_messages_for_phase(self, phase: str, duration: int, 
                                  start_time: float) -> Generator[TimestampedData, None, None]:
        """Generate messages for a specific drilling phase."""
        pattern = self.patterns[phase]
        end_time = start_time + duration
        current_time = start_time
        elapsed = 0
        
        while current_time < end_time:
            # Get rate for this phase
            if phase == 'ramp_up':
                rate = pattern['rate'](elapsed)
            else:
                rate = pattern['rate']()
            
            if rate <= 0:
                # No messages in this phase
                time.sleep(1)
                current_time = time.time()
                elapsed = current_time - start_time
                continue
            
            # Calculate inter-message delay
            delay = 1.0 / rate if rate > 0 else 1.0
            
            # Generate message pair (result + trace)
            tool_id = f"tool{random.randint(1, 5)}"
            
            yield TimestampedData(
                current_time,
                {
                    'phase': phase,
                    'value': random.uniform(0, 100),
                    'tool': tool_id
                },
                f'OPCPUBSUB/toolbox1/{tool_id}/ResultManagement'
            )
            
            yield TimestampedData(
                current_time + random.uniform(0.01, 0.1),  # Small delay for trace
                {
                    'phase': phase,
                    'trace': random.uniform(0, 100),
                    'tool': tool_id
                },
                f'OPCPUBSUB/toolbox1/{tool_id}/ResultManagement/Trace'
            )
            
            time.sleep(delay)
            current_time = time.time()
            elapsed = current_time - start_time


class TestRealisticLoadPatterns:
    """Test system under realistic load patterns."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'mqtt': {
                'listener': {
                    'root': 'OPCPUBSUB',
                    'result': 'ResultManagement',
                    'trace': 'ResultManagement/Trace',
                    'duplicate_handling': 'ignore',
                    'time_window': 30.0
                }
            }
        }
    
    @pytest.fixture
    def test_system(self, sample_config):
        """Create test system components."""
        return {
            'buffer': MessageBuffer(sample_config, max_buffer_size=1000),
            'correlator': DiagnosticCorrelator(sample_config),
            'monitor': SimpleThroughputMonitor(sample_rate=0.1),
            'generator': RealisticLoadGenerator()
        }
    
    @pytest.mark.slow
    def test_drilling_sequence_simulation(self, test_system):
        """Test system with complete drilling sequence."""
        buffer = test_system['buffer']
        correlator = test_system['correlator']
        generator = test_system['generator']
        
        # Track performance metrics
        phase_metrics = {}
        
        # Generate drilling sequence
        sequence = generator.generate_drilling_sequence()
        
        # Process each phase
        for phase_name, duration in sequence[:3]:  # Limit for test speed
            print(f"\n--- Phase: {phase_name} ({duration}s) ---")
            
            phase_start = time.time()
            messages_in_phase = 0
            
            # Generate and process messages for this phase
            for message in generator.generate_messages_for_phase(phase_name, 
                                                               min(duration, 10),  # Limit duration for tests
                                                               phase_start):
                # Add to buffer
                buffer.add_message(message)
                messages_in_phase += 1
                
                # Periodically process correlations
                if messages_in_phase % 10 == 0:
                    buffers = buffer.get_all_buffers()
                    
                    def mock_processor(matches):
                        # Simulate processing time
                        time.sleep(random.uniform(0.01, 0.02))
                    
                    correlator.find_and_process_matches(buffers, mock_processor)
            
            # Get phase metrics
            phase_end = time.time()
            phase_duration = phase_end - phase_start
            
            metrics = correlator.get_diagnostic_metrics()
            phase_metrics[phase_name] = {
                'duration': phase_duration,
                'messages': messages_in_phase,
                'rate': messages_in_phase / phase_duration if phase_duration > 0 else 0,
                'correlation_success_rate': metrics['correlation_success_rate'],
                'orphan_count': metrics['current_orphan_count']
            }
            
            print(f"Messages: {messages_in_phase}, Rate: {phase_metrics[phase_name]['rate']:.1f} msg/s")
            print(f"Correlation success: {metrics['correlation_success_rate']:.1f}%")
        
        # Verify system handled different phases
        assert 'normal_operation' in phase_metrics
        assert 'active_drilling' in phase_metrics
        assert 'bit_change' in phase_metrics
        
        # Normal operation should have moderate rate
        assert 3 <= phase_metrics['normal_operation']['rate'] <= 7
        
        # Active drilling should have higher rate
        assert phase_metrics['active_drilling']['rate'] > phase_metrics['normal_operation']['rate']
        
        # Bit change should have zero or very low rate
        assert phase_metrics['bit_change']['rate'] < 1
    
    def test_burst_handling(self, test_system):
        """Test system handles calibration bursts."""
        buffer = test_system['buffer']
        correlator = test_system['correlator']
        generator = test_system['generator']
        
        # Generate calibration burst
        burst_start = time.time()
        burst_messages = 0
        
        for message in generator.generate_messages_for_phase('calibration', 5, burst_start):
            buffer.add_message(message)
            burst_messages += 1
            
            if burst_messages >= 100:  # Limit for test
                break
        
        burst_duration = time.time() - burst_start
        burst_rate = burst_messages / burst_duration if burst_duration > 0 else 0
        
        # Verify burst characteristics
        assert burst_rate > 20  # Should be high rate
        assert buffer.get_buffer_stats()['total_messages'] >= 100
        
        # Process and check system didn't fall behind catastrophically
        buffers = buffer.get_all_buffers()
        correlator.find_and_process_matches(buffers, lambda x: None)
        
        metrics = correlator.get_diagnostic_metrics()
        assert metrics['correlation_success_rate'] > 50  # Should still correlate most messages
    
    def test_queue_growth_during_phases(self, test_system):
        """Test queue growth patterns during different phases."""
        buffer = test_system['buffer']
        correlator = test_system['correlator']
        generator = test_system['generator']
        
        queue_growth_by_phase = {}
        
        # Test active drilling (should see queue growth if processing can't keep up)
        phase_start = time.time()
        initial_depth = 0
        
        # Add messages faster than we process
        message_count = 0
        for message in generator.generate_messages_for_phase('active_drilling', 5, phase_start):
            buffer.add_message(message)
            message_count += 1
            
            # Process every 20 messages with simulated slow processing
            if message_count % 20 == 0:
                buffers = buffer.get_all_buffers()
                
                def slow_processor(matches):
                    time.sleep(0.05)  # Simulate slow processing
                
                correlator.find_and_process_matches(buffers, slow_processor)
                
                # Check queue growth
                current_depth = sum(len(b) for b in buffers.values())
                if initial_depth == 0:
                    initial_depth = current_depth
            
            if message_count >= 100:
                break
        
        final_depth = sum(len(buffer.get_all_buffers()[k]) for k in buffer.get_all_buffers())
        queue_growth = final_depth - initial_depth
        
        # Should see some queue growth with slow processing
        assert queue_growth > 0
        
        # Check correlator detected the growth
        growth_rate = correlator.get_queue_growth_rate()
        if growth_rate is not None:
            assert growth_rate >= 0  # Queue should be growing or stable
    
    def test_recovery_after_bit_change(self, test_system):
        """Test system recovery after bit change period."""
        buffer = test_system['buffer']
        correlator = test_system['correlator']
        generator = test_system['generator']
        
        # Process some normal messages first
        for message in generator.generate_messages_for_phase('normal_operation', 3, time.time()):
            buffer.add_message(message)
        
        initial_metrics = correlator.get_diagnostic_metrics()
        
        # Simulate bit change (no messages)
        time.sleep(2)
        
        # Then ramp up
        ramp_messages = 0
        for message in generator.generate_messages_for_phase('ramp_up', 5, time.time()):
            buffer.add_message(message)
            ramp_messages += 1
            
            if ramp_messages % 10 == 0:
                buffers = buffer.get_all_buffers()
                correlator.find_and_process_matches(buffers, lambda x: None)
            
            if ramp_messages >= 50:
                break
        
        final_metrics = correlator.get_diagnostic_metrics()
        
        # System should maintain or improve correlation rate after recovery
        assert final_metrics['correlation_success_rate'] >= initial_metrics['correlation_success_rate'] - 10
    
    def test_throughput_monitoring_with_patterns(self, test_system):
        """Test throughput monitoring across different drilling patterns."""
        monitor = test_system['monitor']
        generator = test_system['generator']
        
        phase_throughput = {}
        
        # Test each phase
        for phase_name, pattern in generator.patterns.items():
            # Reset monitor for clean measurement
            monitor.reset_metrics()
            
            # Simulate message arrival for this phase
            phase_start = time.time()
            message_count = 0
            
            for i in range(50):  # Fixed number of messages per phase
                monitor.record_arrival()
                
                # Simulate processing with realistic delay
                processing_start = time.time()
                time.sleep(random.uniform(0.005, 0.015))  # 5-15ms processing
                monitor.record_processing_complete(processing_start)
                
                message_count += 1
                
                # Simulate inter-arrival delay based on phase rate
                if phase_name == 'bit_change':
                    time.sleep(1)  # Long delay during bit change
                elif phase_name == 'calibration':
                    time.sleep(0.02)  # Fast during calibration
                else:
                    time.sleep(0.1)  # Normal delay
            
            # Get throughput status for this phase
            status = monitor.get_status()
            phase_throughput[phase_name] = {
                'status': status.status,
                'arrival_rate': monitor.get_arrival_rate(),
                'processing_rate': monitor.get_processing_rate()
            }
        
        # Verify different phases have different characteristics
        assert phase_throughput['calibration']['arrival_rate'] > phase_throughput['normal_operation']['arrival_rate']
        assert phase_throughput['bit_change']['arrival_rate'] < phase_throughput['normal_operation']['arrival_rate']


class TestConcurrentLoadPatterns:
    """Test system under concurrent load from multiple tools."""
    
    def test_multi_tool_concurrent_load(self, sample_config):
        """Test handling messages from multiple tools concurrently."""
        buffer = MessageBuffer(sample_config, max_buffer_size=5000)
        correlator = DiagnosticCorrelator(sample_config)
        
        # Track messages by tool
        messages_sent = defaultdict(int)
        stop_flag = threading.Event()
        
        def generate_tool_messages(tool_id: str):
            """Generate messages for a specific tool."""
            while not stop_flag.is_set():
                timestamp = time.time()
                
                # Generate result
                buffer.add_message(TimestampedData(
                    timestamp,
                    {'tool': tool_id, 'data': random.random()},
                    f'OPCPUBSUB/toolbox1/{tool_id}/ResultManagement'
                ))
                
                # Generate trace with small delay
                buffer.add_message(TimestampedData(
                    timestamp + 0.01,
                    {'tool': tool_id, 'trace': random.random()},
                    f'OPCPUBSUB/toolbox1/{tool_id}/ResultManagement/Trace'
                ))
                
                messages_sent[tool_id] += 2
                time.sleep(random.uniform(0.05, 0.15))  # Variable rate per tool
        
        # Start threads for multiple tools
        threads = []
        for i in range(5):  # 5 concurrent tools
            t = threading.Thread(target=generate_tool_messages, args=(f'tool{i}',))
            t.start()
            threads.append(t)
        
        # Let them run for a bit
        time.sleep(5)
        
        # Process correlations periodically
        for _ in range(10):
            buffers = buffer.get_all_buffers()
            correlator.find_and_process_matches(buffers, lambda x: None)
            time.sleep(0.5)
        
        # Stop generation
        stop_flag.set()
        for t in threads:
            t.join()
        
        # Verify handling
        total_sent = sum(messages_sent.values())
        metrics = correlator.get_diagnostic_metrics()
        
        assert total_sent > 100  # Should have generated many messages
        assert metrics['correlation_success_rate'] > 70  # Should correlate most
        assert len(messages_sent) == 5  # All tools generated messages


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '-k', 'not slow'])