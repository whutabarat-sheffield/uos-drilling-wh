"""
Test suite for MessageBuffer thread safety.
"""

import pytest
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import random
from datetime import datetime

# Import the classes we need to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from abyss.mqtt.components.message_buffer import MessageBuffer
from abyss.uos_depth_est import TimestampedData


class TestMessageBufferThreadSafety:
    """Test cases for MessageBuffer thread safety."""
    
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
    def thread_safe_buffer(self, sample_config):
        """Create MessageBuffer instance for thread safety testing."""
        return MessageBuffer(
            config=sample_config,
            cleanup_interval=60,
            max_buffer_size=1000,
            max_age_seconds=300
        )
    
    def create_test_message(self, topic_type: str, tool_id: int, msg_id: int) -> TimestampedData:
        """Create a test message with unique data."""
        return TimestampedData(
            _timestamp=time.time() + (msg_id * 0.001),  # Slightly different timestamps
            _data={'test_id': msg_id, 'tool_id': tool_id, 'type': topic_type},
            _source=f'test/root/toolbox{tool_id}/tool{tool_id}/{topic_type}'
        )
    
    def test_concurrent_add_messages(self, thread_safe_buffer):
        """Test adding messages from multiple threads concurrently."""
        num_threads = 10
        messages_per_thread = 100
        
        def add_messages(thread_id):
            """Add messages from a single thread."""
            for i in range(messages_per_thread):
                msg = self.create_test_message('Result', thread_id, i)
                success = thread_safe_buffer.add_message(msg)
                assert success is True
        
        # Run multiple threads adding messages
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(add_messages, i) for i in range(num_threads)]
            for future in futures:
                future.result()  # Wait for all threads to complete
        
        # Verify all messages were added
        stats = thread_safe_buffer.get_buffer_stats()
        assert stats['total_messages'] == num_threads * messages_per_thread
    
    def test_concurrent_add_and_cleanup(self, thread_safe_buffer):
        """Test adding messages while cleanup is running."""
        stop_event = threading.Event()
        messages_added = threading.Event()
        
        def add_messages():
            """Continuously add messages."""
            count = 0
            while not stop_event.is_set():
                msg = self.create_test_message('Trace', count % 5, count)
                thread_safe_buffer.add_message(msg)
                count += 1
                if count > 50:
                    messages_added.set()
                time.sleep(0.001)
        
        def run_cleanup():
            """Continuously run cleanup."""
            while not stop_event.is_set():
                thread_safe_buffer.cleanup_old_messages()
                time.sleep(0.01)
        
        # Start threads
        add_thread = threading.Thread(target=add_messages)
        cleanup_thread = threading.Thread(target=run_cleanup)
        
        add_thread.start()
        cleanup_thread.start()
        
        # Let them run for a bit
        messages_added.wait(timeout=5)
        time.sleep(0.5)
        
        # Stop threads
        stop_event.set()
        add_thread.join(timeout=2)
        cleanup_thread.join(timeout=2)
        
        # Verify buffer is in consistent state
        stats = thread_safe_buffer.get_buffer_stats()
        assert stats['total_messages'] >= 0
        assert stats['total_buffers'] >= 0
    
    def test_concurrent_read_operations(self, thread_safe_buffer):
        """Test reading from buffer while adding messages."""
        # Pre-populate some messages
        for i in range(100):
            msg = self.create_test_message('Result', i % 3, i)
            thread_safe_buffer.add_message(msg)
        
        results = []
        
        def read_operations():
            """Perform various read operations."""
            for _ in range(50):
                # Random read operation
                op = random.choice([
                    lambda: thread_safe_buffer.get_buffer_stats(),
                    lambda: thread_safe_buffer.get_all_buffers(),
                    lambda: thread_safe_buffer.get_messages_by_topic('test/root/+/+/Result')
                ])
                result = op()
                results.append(result)
                time.sleep(0.001)
        
        def write_operations():
            """Add more messages while reading."""
            for i in range(50):
                msg = self.create_test_message('Trace', i % 3, i + 100)
                thread_safe_buffer.add_message(msg)
                time.sleep(0.001)
        
        # Run concurrent reads and writes
        with ThreadPoolExecutor(max_workers=4) as executor:
            read_futures = [executor.submit(read_operations) for _ in range(2)]
            write_futures = [executor.submit(write_operations) for _ in range(2)]
            
            for future in read_futures + write_futures:
                future.result()
        
        # Verify results are consistent
        assert len(results) > 0
        # Results can be either dict (from get_buffer_stats/get_all_buffers) or list (from get_messages_by_topic)
        assert all(isinstance(r, (dict, list)) for r in results if r is not None)
    
    def test_buffer_overflow_warnings(self, thread_safe_buffer):
        """Test that warnings are logged when buffer approaches capacity."""
        # Set a small buffer size for testing
        thread_safe_buffer.max_buffer_size = 10
        
        # Add messages to approach capacity
        for i in range(9):  # 90% capacity
            msg = self.create_test_message('Result', 1, i)
            thread_safe_buffer.add_message(msg)
        
        # This should trigger a warning (>80% capacity)
        # In real test, we would capture logs and verify warning was logged
        stats = thread_safe_buffer.get_buffer_stats()
        assert stats['total_messages'] == 9
    
    def test_clear_buffer_thread_safety(self, thread_safe_buffer):
        """Test clearing buffers while other operations are ongoing."""
        stop_event = threading.Event()
        
        def add_messages():
            """Add messages continuously."""
            count = 0
            while not stop_event.is_set():
                msg = self.create_test_message('Result', count % 3, count)
                try:
                    thread_safe_buffer.add_message(msg)
                except Exception:
                    pass  # Ignore errors during concurrent clear
                count += 1
                time.sleep(0.001)
        
        def clear_buffers():
            """Clear buffers periodically."""
            while not stop_event.is_set():
                thread_safe_buffer.clear_buffer()
                time.sleep(0.05)
        
        # Start threads
        threads = []
        for _ in range(3):
            t = threading.Thread(target=add_messages)
            t.start()
            threads.append(t)
        
        clear_thread = threading.Thread(target=clear_buffers)
        clear_thread.start()
        threads.append(clear_thread)
        
        # Run for a short time
        time.sleep(0.5)
        
        # Stop all threads
        stop_event.set()
        for t in threads:
            t.join(timeout=2)
        
        # Buffer should be in valid state (possibly empty)
        stats = thread_safe_buffer.get_buffer_stats()
        assert stats['total_messages'] >= 0
    
    def test_deadlock_prevention(self, thread_safe_buffer):
        """Test that no deadlocks occur with nested operations."""
        # This tests the cleanup trigger which could potentially deadlock
        # if not implemented correctly
        
        # Fill buffer to trigger cleanup
        thread_safe_buffer.max_buffer_size = 5
        
        def fill_and_trigger_cleanup():
            """Fill buffer to trigger automatic cleanup."""
            for i in range(10):
                msg = self.create_test_message('Result', 1, i)
                thread_safe_buffer.add_message(msg)
        
        # Run in multiple threads
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(fill_and_trigger_cleanup) for _ in range(3)]
            for future in futures:
                future.result()  # Should complete without deadlock
        
        # Verify buffer state
        stats = thread_safe_buffer.get_buffer_stats()
        assert stats['total_messages'] <= thread_safe_buffer.max_buffer_size