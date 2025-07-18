"""
Load tests for MessageBuffer to verify robustness under heavy load with exact matching.
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
    """Load tests for MessageBuffer under various stress conditions with exact matching."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'mqtt': {
                'listener': {
                    'root': 'test/root',
                    'result': 'ResultManagement',
                    'trace': 'Trace',
                    'heads': 'AssetManagement',
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
            max_buffer_size=5000,  # Max message sets
            max_age_seconds=300
        )
    
    def create_test_message(self, msg_type: str, tool_id: int, msg_id: int, 
                          source_timestamp: str, timestamp: float = None) -> TimestampedData:
        """Create a test message with exact matching fields."""
        if timestamp is None:
            timestamp = time.time()
        
        data = {
            'SourceTimestamp': source_timestamp,
            'toolboxID': f'toolbox{tool_id}',
            'toolID': f'tool{tool_id}',
            'test_id': msg_id,
            'random_data': ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=100))
        }
        
        # Add type-specific fields
        if msg_type == 'ResultManagement':
            data['ResultId'] = msg_id
        elif msg_type == 'AssetManagement':
            data['HeadsId'] = f'HEADS{msg_id}'
        
        return TimestampedData(
            _timestamp=timestamp,
            _data=data,
            _source=f'test/root/toolbox{tool_id}/tool{tool_id}/{msg_type}'
        )
    
    def test_high_volume_concurrent_writes(self, load_test_buffer):
        """Test buffer under high volume of concurrent complete message sets."""
        num_threads = 20
        message_sets_per_thread = 100  # Complete message sets
        
        def write_message_sets(thread_id):
            """Write complete message sets from a single thread."""
            success_count = 0
            for i in range(message_sets_per_thread):
                tool_id = random.randint(1, 10)
                msg_id = thread_id * 1000 + i
                source_timestamp = f'2024-01-18T{thread_id:02d}:{i//60:02d}:{i%60:02d}Z'
                
                # Create complete message set
                result_msg = self.create_test_message('ResultManagement', tool_id, msg_id, source_timestamp)
                trace_msg = self.create_test_message('Trace', tool_id, msg_id, source_timestamp)
                heads_msg = self.create_test_message('AssetManagement', tool_id, msg_id, source_timestamp)
                
                # Add all three messages (complete set)
                if (load_test_buffer.add_message(result_msg) and 
                    load_test_buffer.add_message(trace_msg) and
                    load_test_buffer.add_message(heads_msg)):
                    success_count += 1
                    
                # Simulate realistic message arrival rate
                if i % 10 == 0:
                    time.sleep(0.001)
                    
            return success_count
        
        # Execute concurrent writes
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(write_message_sets, i) for i in range(num_threads)]
            
            total_success = 0
            for future in as_completed(futures):
                total_success += future.result()
        
        # Verify results
        stats = load_test_buffer.get_buffer_stats()
        
        print(f"Total message sets attempted: {num_threads * message_sets_per_thread}")
        print(f"Total message sets successful: {total_success}")
        print(f"Exact matches completed: {stats['exact_matches_completed']}")
        print(f"Active keys: {stats['total_active_keys']}")
        print(f"Messages dropped: {stats['messages_dropped']}")
        
        # Assert reasonable success rate (allowing for some drops due to overflow)
        assert total_success >= num_threads * message_sets_per_thread * 0.8
        assert stats['exact_matches_completed'] >= total_success * 0.8
        assert stats['total_active_keys'] <= load_test_buffer.max_buffer_size
    
    def test_sustained_load_with_cleanup(self, load_test_buffer):
        """Test buffer behavior under sustained load with cleanup cycles."""
        duration_seconds = 5
        write_threads = 10
        cleanup_interval = 0.5
        
        stop_event = threading.Event()
        message_set_counts = {}
        
        def continuous_writer(thread_id):
            """Continuously write incomplete message sets until stopped."""
            count = 0
            while not stop_event.is_set():
                tool_id = thread_id % 5 + 1
                source_timestamp = f'2024-01-18T{thread_id:02d}:{count//60:02d}:{count%60:02d}Z'
                
                # Create incomplete message set (missing heads to stay in buffer)
                result_msg = self.create_test_message('ResultManagement', tool_id, count, source_timestamp)
                trace_msg = self.create_test_message('Trace', tool_id, count, source_timestamp)
                
                if (load_test_buffer.add_message(result_msg) and 
                    load_test_buffer.add_message(trace_msg)):
                    count += 1
                
                # Vary the rate
                time.sleep(random.uniform(0.0001, 0.002))
            
            message_set_counts[thread_id] = count
            return count
        
        def cleanup_runner():
            """Run periodic cleanup."""
            while not stop_event.is_set():
                load_test_buffer._cleanup_old_incomplete_sets()
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
        total_written = sum(message_set_counts.values())
        stats = load_test_buffer.get_buffer_stats()
        
        print(f"Total incomplete message sets written: {total_written}")
        print(f"Message sets per second: {total_written / duration_seconds:.0f}")
        print(f"Final active keys: {stats['total_active_keys']}")
        print(f"Messages dropped: {stats['messages_dropped']}")
        
        # Verify system remained stable
        assert stats['total_active_keys'] <= load_test_buffer.max_buffer_size
    
    def test_burst_traffic_handling(self, load_test_buffer):
        """Test handling of burst traffic patterns with complete message sets."""
        burst_size = 100  # Complete message sets
        num_bursts = 5
        pause_between_bursts = 0.5
        
        @patch('logging.warning')
        def run_burst_test(mock_warning):
            for burst_num in range(num_bursts):
                # Send burst of complete message sets
                with ThreadPoolExecutor(max_workers=20) as executor:
                    futures = []
                    for i in range(burst_size):
                        source_timestamp = f'2024-01-18T{burst_num:02d}:{i//60:02d}:{i%60:02d}Z'
                        msg_id = burst_num * 1000 + i
                        
                        # Create complete message set
                        result_msg = self.create_test_message('ResultManagement', 1, msg_id, source_timestamp)
                        trace_msg = self.create_test_message('Trace', 1, msg_id, source_timestamp)
                        heads_msg = self.create_test_message('AssetManagement', 1, msg_id, source_timestamp)
                        
                        # Submit all three messages
                        futures.append(executor.submit(load_test_buffer.add_message, result_msg))
                        futures.append(executor.submit(load_test_buffer.add_message, trace_msg))
                        futures.append(executor.submit(load_test_buffer.add_message, heads_msg))
                    
                    # Wait for burst to complete
                    success_count = sum(1 for f in as_completed(futures) if f.result())
                
                print(f"Burst {burst_num + 1}: {success_count}/{burst_size * 3} messages")
                
                # Pause between bursts
                time.sleep(pause_between_bursts)
            
            # Check if warnings were triggered
            warning_calls = [call for call in mock_warning.call_args_list 
                           if 'capacity' in str(call)]
            return warning_calls
        
        warning_calls = run_burst_test()
        
        # Verify appropriate warnings were generated
        stats = load_test_buffer.get_buffer_stats()
        if stats['total_active_keys'] > load_test_buffer.max_buffer_size * 0.6:
            assert len(warning_calls) > 0, "Expected capacity warnings during burst traffic"
    
    def test_memory_pressure_simulation(self, load_test_buffer):
        """Test buffer behavior under memory pressure with large message sets."""
        # Create large messages
        large_data_size = 10000  # 10KB per message
        num_large_message_sets = 50
        
        def create_large_message(msg_type, msg_id, source_timestamp):
            """Create a message with large payload."""
            data = {
                'SourceTimestamp': source_timestamp,
                'toolboxID': 'toolbox1',
                'toolID': 'tool1',
                'test_id': msg_id,
                'large_data': 'x' * large_data_size,
                'nested_data': {
                    'level1': {
                        'level2': {
                            'data': list(range(1000))
                        }
                    }
                }
            }
            
            if msg_type == 'ResultManagement':
                data['ResultId'] = msg_id
            elif msg_type == 'AssetManagement':
                data['HeadsId'] = f'HEADS{msg_id}'
            
            return TimestampedData(
                _timestamp=time.time(),
                _data=data,
                _source=f'test/root/toolbox1/tool1/{msg_type}'
            )
        
        # Track memory-related metrics
        success_count = 0
        start_time = time.time()
        
        for i in range(num_large_message_sets):
            source_timestamp = f'2024-01-18T10:{i//60:02d}:{i%60:02d}Z'
            
            # Create complete large message set
            result_msg = create_large_message('ResultManagement', i, source_timestamp)
            trace_msg = create_large_message('Trace', i, source_timestamp)
            heads_msg = create_large_message('AssetManagement', i, source_timestamp)
            
            if (load_test_buffer.add_message(result_msg) and
                load_test_buffer.add_message(trace_msg) and
                load_test_buffer.add_message(heads_msg)):
                success_count += 1
            
            # Check if cleanup is triggered
            if i % 10 == 0:
                stats = load_test_buffer.get_buffer_stats()
                if stats['total_active_keys'] >= load_test_buffer.max_buffer_size * 0.9:
                    print(f"Buffer near capacity at message set {i}")
        
        elapsed_time = time.time() - start_time
        
        # Verify system handled large message sets appropriately
        stats = load_test_buffer.get_buffer_stats()
        
        print(f"Large message sets processed: {success_count}/{num_large_message_sets}")
        print(f"Processing time: {elapsed_time:.2f}s")
        print(f"Average time per message set: {elapsed_time/num_large_message_sets*1000:.1f}ms")
        print(f"Exact matches completed: {stats['exact_matches_completed']}")
        
        assert success_count > 0
        assert stats['total_active_keys'] <= load_test_buffer.max_buffer_size
    
    def test_concurrent_read_write_stress(self, load_test_buffer):
        """Test concurrent reads and writes under stress with exact matching."""
        duration_seconds = 3
        write_threads = 5
        read_threads = 5
        
        stop_event = threading.Event()
        operation_counts = {'writes': 0, 'reads': 0}
        lock = threading.Lock()
        
        def writer_thread():
            """Continuously write complete message sets."""
            local_count = 0
            while not stop_event.is_set():
                tool_id = random.randint(1, 5)
                source_timestamp = f'2024-01-18T10:30:{local_count%60:02d}Z'
                
                # Create complete message set
                result_msg = self.create_test_message('ResultManagement', tool_id, local_count, source_timestamp)
                trace_msg = self.create_test_message('Trace', tool_id, local_count, source_timestamp)
                heads_msg = self.create_test_message('AssetManagement', tool_id, local_count, source_timestamp)
                
                if (load_test_buffer.add_message(result_msg) and
                    load_test_buffer.add_message(trace_msg) and
                    load_test_buffer.add_message(heads_msg)):
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
                    lambda: load_test_buffer.get_messages_by_topic('test/root/+/+/ResultManagement')
                ]
                
                try:
                    random.choice(operations)()
                    local_count += 1
                except Exception:
                    pass  # Ignore errors during stress testing
                    
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
        print(f"Exact matches completed: {stats['exact_matches_completed']}")
        
        assert operation_counts['writes'] > 0
        assert operation_counts['reads'] > 0
        assert stats['total_active_keys'] >= 0
    
    def test_error_recovery_under_load(self, load_test_buffer):
        """Test error recovery mechanisms under load with exact matching."""
        num_threads = 10
        message_sets_per_thread = 50
        error_injection_rate = 0.1  # 10% of message sets will be invalid
        
        def write_with_errors(thread_id):
            """Write message sets with some invalid ones."""
            success_count = 0
            error_count = 0
            
            for i in range(message_sets_per_thread):
                source_timestamp = f'2024-01-18T{thread_id:02d}:{i//60:02d}:{i%60:02d}Z'
                
                if random.random() < error_injection_rate:
                    # Create invalid message (None or malformed)
                    msg = None if random.random() < 0.5 else TimestampedData(
                        _timestamp=time.time(),
                        _data=None,
                        _source=''
                    )
                    
                    try:
                        load_test_buffer.add_message(msg)
                        error_count += 1
                    except Exception:
                        error_count += 1
                else:
                    # Create valid complete message set
                    try:
                        result_msg = self.create_test_message('ResultManagement', thread_id, i, source_timestamp)
                        trace_msg = self.create_test_message('Trace', thread_id, i, source_timestamp)
                        heads_msg = self.create_test_message('AssetManagement', thread_id, i, source_timestamp)
                        
                        if (load_test_buffer.add_message(result_msg) and
                            load_test_buffer.add_message(trace_msg) and
                            load_test_buffer.add_message(heads_msg)):
                            success_count += 1
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
        
        print(f"Successful message sets: {total_success}")
        print(f"Failed operations: {total_errors}")
        print(f"Exact matches completed: {stats['exact_matches_completed']}")
        print(f"Buffer integrity maintained: {stats['total_active_keys'] <= load_test_buffer.max_buffer_size}")
        
        # System should handle errors gracefully
        assert total_success > 0
        assert stats['total_active_keys'] <= load_test_buffer.max_buffer_size
        assert total_errors > 0  # Some operations should have failed due to error injection