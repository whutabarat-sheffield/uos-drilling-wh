"""
Load tests for MessageBuffer to verify robustness under heavy load.
"""

import pytest
import time
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch, MagicMock
import logging

# Import the classes we need to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from abyss.mqtt.components.message_buffer import MessageBuffer
from abyss.uos_depth_est import TimestampedData


class TestMessageBufferLoadTests:
    """Load tests for MessageBuffer under various stress conditions."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'mqtt': {
                'listener': {
                    'root': 'test/root',
                    'result': 'Result',
                    'trace': 'Trace',
                    'heads': 'Heads',
                    'duplicate_handling': 'ignore'
                }
            }
        }
    
    @pytest.fixture
    def load_test_buffer(self, sample_config):
        """Create MessageBuffer configured for load testing."""
        return MessageBuffer(
            config=sample_config,
            cleanup_interval=60,
            max_buffer_size=5000,
            max_age_seconds=300
        )
    
    def create_test_message(self, msg_type: str, tool_id: int, msg_id: int, 
                          timestamp: float = None) -> TimestampedData:
        """Create a test message with specified parameters."""
        if timestamp is None:
            timestamp = time.time()
        
        return TimestampedData(
            _timestamp=timestamp,
            _data={
                'test_id': msg_id,
                'tool_id': tool_id,
                'type': msg_type,
                'random_data': ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=100))
            },
            _source=f'test/root/toolbox{tool_id}/tool{tool_id}/{msg_type}'
        )
    
    def test_high_volume_concurrent_writes(self, load_test_buffer):
        """Test buffer under high volume of concurrent writes."""
        num_threads = 20
        messages_per_thread = 500
        
        def write_messages(thread_id):
            """Write messages from a single thread."""
            success_count = 0
            for i in range(messages_per_thread):
                msg_type = random.choice(['Result', 'Trace', 'Heads'])
                tool_id = random.randint(1, 10)
                msg = self.create_test_message(msg_type, tool_id, thread_id * 1000 + i)
                
                if load_test_buffer.add_message(msg):
                    success_count += 1
                    
                # Simulate realistic message arrival rate
                if i % 10 == 0:
                    time.sleep(0.001)
                    
            return success_count
        
        # Execute concurrent writes
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(write_messages, i) for i in range(num_threads)]
            
            total_success = 0
            for future in as_completed(futures):
                total_success += future.result()
        
        # Verify results
        stats = load_test_buffer.get_buffer_stats()
        metrics = load_test_buffer.get_metrics()
        
        print(f"Total messages attempted: {num_threads * messages_per_thread}")
        print(f"Total messages successful: {total_success}")
        print(f"Messages in buffer: {stats['total_messages']}")
        print(f"Drop rate: {metrics['drop_rate']:.2%}")
        
        # Assert reasonable success rate (allowing for some drops due to overflow)
        assert total_success >= num_threads * messages_per_thread * 0.8
        assert stats['total_messages'] <= load_test_buffer.max_buffer_size
    
    def test_sustained_load_with_cleanup(self, load_test_buffer):
        """Test buffer behavior under sustained load with cleanup cycles."""
        duration_seconds = 5
        write_threads = 10
        cleanup_interval = 0.5
        
        stop_event = threading.Event()
        message_counts = {}
        
        def continuous_writer(thread_id):
            """Continuously write messages until stopped."""
            count = 0
            while not stop_event.is_set():
                msg_type = random.choice(['Result', 'Trace'])
                tool_id = thread_id % 5 + 1
                msg = self.create_test_message(msg_type, tool_id, count)
                
                if load_test_buffer.add_message(msg):
                    count += 1
                
                # Vary the rate
                time.sleep(random.uniform(0.0001, 0.002))
            
            message_counts[thread_id] = count
            return count
        
        def cleanup_runner():
            """Run periodic cleanup."""
            while not stop_event.is_set():
                load_test_buffer.cleanup_old_messages()
                time.sleep(cleanup_interval)
        
        # Start threads
        with ThreadPoolExecutor(max_workers=write_threads + 1) as executor:
            # Start writers
            writer_futures = [
                executor.submit(continuous_writer, i) 
                for i in range(write_threads)
            ]
            
            # Start cleanup thread
            cleanup_future = executor.submit(cleanup_runner)
            
            # Run for specified duration
            time.sleep(duration_seconds)
            
            # Stop all threads
            stop_event.set()
            
            # Wait for completion
            for future in writer_futures:
                future.result()
            cleanup_future.result()
        
        # Analyze results
        total_written = sum(message_counts.values())
        stats = load_test_buffer.get_buffer_stats()
        metrics = load_test_buffer.get_metrics()
        
        print(f"Total messages written: {total_written}")
        print(f"Messages per second: {total_written / duration_seconds:.0f}")
        print(f"Final buffer size: {stats['total_messages']}")
        print(f"Cleanup cycles: {metrics['cleanup_cycles']}")
        
        # Verify system remained stable
        assert stats['total_messages'] <= load_test_buffer.max_buffer_size
        assert metrics['cleanup_cycles'] > 0
    
    def test_burst_traffic_handling(self, load_test_buffer):
        """Test handling of burst traffic patterns."""
        burst_size = 1000
        num_bursts = 5
        pause_between_bursts = 0.5
        
        @patch('logging.warning')
        def run_burst_test(mock_warning):
            for burst_num in range(num_bursts):
                # Send burst of messages
                with ThreadPoolExecutor(max_workers=20) as executor:
                    futures = []
                    for i in range(burst_size):
                        msg = self.create_test_message(
                            'Result', 
                            burst_num + 1, 
                            burst_num * 1000 + i
                        )
                        future = executor.submit(load_test_buffer.add_message, msg)
                        futures.append(future)
                    
                    # Wait for burst to complete
                    success_count = sum(1 for f in as_completed(futures) if f.result())
                
                print(f"Burst {burst_num + 1}: {success_count}/{burst_size} messages")
                
                # Pause between bursts
                time.sleep(pause_between_bursts)
            
            # Check if warnings were triggered
            warning_calls = [call for call in mock_warning.call_args_list 
                           if 'capacity' in str(call)]
            return warning_calls
        
        warning_calls = run_burst_test()
        
        # Verify appropriate warnings were generated
        stats = load_test_buffer.get_buffer_stats()
        if stats['total_messages'] > load_test_buffer.max_buffer_size * 0.6:
            assert len(warning_calls) > 0, "Expected capacity warnings during burst traffic"
    
    def test_memory_pressure_simulation(self, load_test_buffer):
        """Test buffer behavior under memory pressure with large messages."""
        # Create large messages
        large_data_size = 10000  # 10KB per message
        num_large_messages = 100
        
        def create_large_message(msg_id):
            """Create a message with large payload."""
            return TimestampedData(
                _timestamp=time.time(),
                _data={
                    'test_id': msg_id,
                    'large_data': 'x' * large_data_size,
                    'nested_data': {
                        'level1': {
                            'level2': {
                                'data': list(range(1000))
                            }
                        }
                    }
                },
                _source=f'test/root/toolbox1/tool1/Result'
            )
        
        # Track memory-related metrics
        success_count = 0
        start_time = time.time()
        
        for i in range(num_large_messages):
            msg = create_large_message(i)
            if load_test_buffer.add_message(msg):
                success_count += 1
            
            # Check if cleanup is triggered
            if i % 20 == 0:
                stats = load_test_buffer.get_buffer_stats()
                if stats['total_messages'] >= load_test_buffer.max_buffer_size * 0.9:
                    print(f"Buffer near capacity at message {i}")
        
        elapsed_time = time.time() - start_time
        
        # Verify system handled large messages appropriately
        stats = load_test_buffer.get_buffer_stats()
        metrics = load_test_buffer.get_metrics()
        
        print(f"Large messages processed: {success_count}/{num_large_messages}")
        print(f"Processing time: {elapsed_time:.2f}s")
        print(f"Average time per message: {elapsed_time/num_large_messages*1000:.1f}ms")
        
        assert success_count > 0
        assert stats['total_messages'] <= load_test_buffer.max_buffer_size
    
    def test_concurrent_read_write_stress(self, load_test_buffer):
        """Test concurrent reads and writes under stress."""
        duration_seconds = 3
        write_threads = 5
        read_threads = 5
        
        stop_event = threading.Event()
        operation_counts = {'writes': 0, 'reads': 0}
        lock = threading.Lock()
        
        def writer_thread():
            """Continuously write messages."""
            local_count = 0
            while not stop_event.is_set():
                msg = self.create_test_message(
                    random.choice(['Result', 'Trace']),
                    random.randint(1, 5),
                    local_count
                )
                if load_test_buffer.add_message(msg):
                    local_count += 1
                time.sleep(0.0001)
            
            with lock:
                operation_counts['writes'] += local_count
        
        def reader_thread():
            """Continuously read buffer state."""
            local_count = 0
            while not stop_event.is_set():
                # Perform various read operations
                operations = [
                    lambda: load_test_buffer.get_buffer_stats(),
                    lambda: load_test_buffer.get_all_buffers(),
                    lambda: load_test_buffer.get_metrics(),
                    lambda: load_test_buffer.get_messages_by_topic('test/root/+/+/Result')
                ]
                
                random.choice(operations)()
                local_count += 1
                time.sleep(0.0005)
            
            with lock:
                operation_counts['reads'] += local_count
        
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=write_threads + read_threads) as executor:
            futures = []
            
            # Start writers
            for _ in range(write_threads):
                futures.append(executor.submit(writer_thread))
            
            # Start readers
            for _ in range(read_threads):
                futures.append(executor.submit(reader_thread))
            
            # Run for duration
            time.sleep(duration_seconds)
            stop_event.set()
            
            # Wait for all threads
            for future in futures:
                future.result()
        
        # Verify no deadlocks or data corruption
        stats = load_test_buffer.get_buffer_stats()
        
        print(f"Total writes: {operation_counts['writes']}")
        print(f"Total reads: {operation_counts['reads']}")
        print(f"Writes per second: {operation_counts['writes'] / duration_seconds:.0f}")
        print(f"Reads per second: {operation_counts['reads'] / duration_seconds:.0f}")
        
        assert operation_counts['writes'] > 0
        assert operation_counts['reads'] > 0
        assert stats['total_messages'] >= 0
    
    def test_error_recovery_under_load(self, load_test_buffer):
        """Test error recovery mechanisms under load."""
        num_threads = 10
        messages_per_thread = 100
        error_injection_rate = 0.1  # 10% of messages will be invalid
        
        def write_with_errors(thread_id):
            """Write messages with some invalid ones."""
            success_count = 0
            error_count = 0
            
            for i in range(messages_per_thread):
                if random.random() < error_injection_rate:
                    # Create invalid message
                    msg = None if random.random() < 0.5 else TimestampedData(
                        _timestamp=time.time(),
                        _data=None,
                        _source=''
                    )
                else:
                    # Create valid message
                    msg = self.create_test_message('Result', thread_id, i)
                
                try:
                    if load_test_buffer.add_message(msg):
                        success_count += 1
                    else:
                        error_count += 1
                except Exception:
                    error_count += 1
            
            return success_count, error_count
        
        # Run test with error injection
        total_success = 0
        total_errors = 0
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(write_with_errors, i) for i in range(num_threads)]
            
            for future in as_completed(futures):
                success, errors = future.result()
                total_success += success
                total_errors += errors
        
        # Verify system remained stable despite errors
        stats = load_test_buffer.get_buffer_stats()
        metrics = load_test_buffer.get_metrics()
        
        print(f"Successful messages: {total_success}")
        print(f"Failed messages: {total_errors}")
        print(f"Drop rate: {metrics['drop_rate']:.2%}")
        print(f"Buffer integrity maintained: {stats['total_messages'] <= load_test_buffer.max_buffer_size}")
        
        # System should handle errors gracefully
        assert total_success > 0
        assert stats['total_messages'] <= load_test_buffer.max_buffer_size
        assert metrics['messages_dropped'] > 0  # Some messages should have been dropped