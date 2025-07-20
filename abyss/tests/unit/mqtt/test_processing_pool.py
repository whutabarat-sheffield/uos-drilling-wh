"""Unit tests for ProcessingPool component.

These tests focus on fast, isolated testing of ProcessingPool functionality
without requiring actual message processing or external dependencies.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from concurrent.futures import Future
import time
import os

from abyss.mqtt.components.processing_pool import SimpleProcessingPool, _init_worker, _process_messages


class TestProcessingPool:
    """Test the SimpleProcessingPool class"""
    
    @pytest.fixture
    def mock_config_path(self, tmp_path):
        """Create a temporary config path"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("mqtt:\n  processing:\n    model_id: 4\n")
        return str(config_file)
    
    @pytest.fixture
    def processing_pool(self, mock_config_path):
        """Create a ProcessingPool with mocked executor"""
        with patch('abyss.mqtt.components.processing_pool.ProcessPoolExecutor') as mock_executor:
            pool = SimpleProcessingPool(max_workers=5, config_path=mock_config_path)
            pool.executor = mock_executor.return_value
            yield pool
    
    def test_initialization(self, mock_config_path):
        """Test pool initialization"""
        with patch('abyss.mqtt.components.processing_pool.ProcessPoolExecutor') as mock_executor:
            pool = SimpleProcessingPool(max_workers=10, config_path=mock_config_path)
            
            # Verify executor was created with correct parameters
            mock_executor.assert_called_once_with(
                max_workers=10,
                initializer=_init_worker,
                initargs=(mock_config_path,)
            )
            
            # Verify initial counters
            assert pool.submitted == 0
            assert pool.completed == 0
            assert pool.failed == 0
            assert pool.max_workers == 10
            assert pool.config_path == mock_config_path
    
    def test_submit_success(self, processing_pool):
        """Test successful task submission"""
        messages = [Mock(), Mock()]
        callback = Mock()
        
        # Mock the executor submit
        future = Mock(spec=Future)
        processing_pool.executor.submit.return_value = future
        
        # Submit task
        result = processing_pool.submit(messages, callback)
        
        assert result is True
        assert processing_pool.submitted == 1
        assert len(processing_pool.pending_futures) == 1
        
        # Verify executor was called correctly
        processing_pool.executor.submit.assert_called_once_with(
            _process_messages, 
            (messages, processing_pool.config_path)
        )
        
        # Verify callback was registered
        future.add_done_callback.assert_called_once()
    
    def test_submit_backpressure(self, processing_pool):
        """Test task rejection when pool is full"""
        # Fill up the pending futures
        for i in range(processing_pool.max_workers * 2):
            future = Mock(spec=Future)
            processing_pool.pending_futures[future] = time.time()
        
        # Try to submit another task
        result = processing_pool.submit([Mock()], None)
        
        assert result is False
        assert processing_pool.submitted == 0
    
    def test_submit_exception_handling(self, processing_pool):
        """Test exception handling during submission"""
        processing_pool.executor.submit.side_effect = RuntimeError("Pool shutdown")
        
        result = processing_pool.submit([Mock()], None)
        
        assert result is False
        assert processing_pool.submitted == 0
    
    def test_collect_completed_success(self, processing_pool):
        """Test collecting completed successful tasks"""
        # Create mock futures
        future1 = Mock(spec=Future)
        future1.done.return_value = True
        future1.result.return_value = {'success': True}
        
        future2 = Mock(spec=Future)
        future2.done.return_value = False
        
        future3 = Mock(spec=Future)
        future3.done.return_value = True
        future3.result.return_value = {'success': False, 'error_message': 'Test error'}
        
        # Add to pending
        processing_pool.pending_futures[future1] = time.time()
        processing_pool.pending_futures[future2] = time.time()
        processing_pool.pending_futures[future3] = time.time()
        
        # Collect completed
        completed = processing_pool.collect_completed()
        
        assert completed == 2
        assert processing_pool.completed == 1
        assert processing_pool.failed == 1
        assert len(processing_pool.pending_futures) == 1
        assert future2 in processing_pool.pending_futures
    
    def test_collect_completed_with_exception(self, processing_pool):
        """Test collecting tasks that raised exceptions"""
        future = Mock(spec=Future)
        future.done.return_value = True
        future.result.side_effect = Exception("Processing failed")
        
        processing_pool.pending_futures[future] = time.time()
        
        completed = processing_pool.collect_completed()
        
        assert completed == 1
        assert processing_pool.failed == 1
        assert len(processing_pool.pending_futures) == 0
    
    def test_get_queue_depth(self, processing_pool):
        """Test queue depth reporting"""
        # Add some pending futures
        for i in range(3):
            future = Mock(spec=Future)
            processing_pool.pending_futures[future] = time.time()
        
        assert processing_pool.get_queue_depth() == 3
    
    def test_get_stats(self, processing_pool):
        """Test statistics reporting"""
        # Set up some stats
        processing_pool.submitted = 10
        processing_pool.completed = 7
        processing_pool.failed = 2
        processing_pool.start_time = time.time() - 10  # 10 seconds ago
        
        # Add pending future
        processing_pool.pending_futures[Mock(spec=Future)] = time.time()
        
        stats = processing_pool.get_stats()
        
        assert stats['workers'] == 5
        assert stats['queue_depth'] == 1
        assert stats['submitted'] == 10
        assert stats['completed'] == 7
        assert stats['failed'] == 2
        assert stats['success_rate'] == 0.7
        assert 0.6 <= stats['avg_throughput'] <= 0.8  # ~0.7 msg/s
    
    def test_shutdown(self, processing_pool):
        """Test pool shutdown"""
        processing_pool.shutdown(wait=True)
        
        processing_pool.executor.shutdown.assert_called_once_with(wait=True)
    
    def test_shutdown_with_pending_tasks(self, processing_pool):
        """Test shutdown with pending tasks logs warning"""
        # Add pending futures
        for i in range(3):
            processing_pool.pending_futures[Mock(spec=Future)] = time.time()
        
        with patch('logging.warning') as mock_warning:
            processing_pool.shutdown(wait=True)
            
            mock_warning.assert_called_once()
            assert "3 pending tasks" in mock_warning.call_args[0][0]
    
    def test_callback_error_handling(self, processing_pool):
        """Test that callback errors are handled gracefully"""
        messages = [Mock()]
        callback = Mock(side_effect=Exception("Callback failed"))
        
        # Mock the executor and future
        future = Mock(spec=Future)
        future.result.return_value = {'success': True}
        processing_pool.executor.submit.return_value = future
        
        # Submit task
        result = processing_pool.submit(messages, callback)
        assert result is True
        
        # Get the actual callback function that was registered
        safe_callback = future.add_done_callback.call_args[0][0]
        
        # Call it and verify error is logged but doesn't propagate
        with patch('logging.error') as mock_error:
            safe_callback(future)
            mock_error.assert_called_once()
            assert "Callback error" in mock_error.call_args[0][0]




