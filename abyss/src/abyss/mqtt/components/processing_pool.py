"""
Processing Pool Module

Provides parallel processing of matched MQTT messages using ProcessPoolExecutor
to handle CPU-intensive depth inference operations.
"""

from concurrent.futures import ProcessPoolExecutor, Future
from typing import Dict, List, Optional, Callable, Any
import logging
import time
import os

# Global model and config cache per process
_model_cache = None
_config_cache = None
_converter_cache = None
_model_initialized = False


def _init_worker(config_path: str):
    """Initialize model and configuration in each worker process."""
    global _model_cache, _config_cache, _converter_cache, _model_initialized
    
    if _model_initialized:
        return
        
    try:
        from ...uos_depth_est_core import DepthInference
        from .config_manager import ConfigurationManager
        from .data_converter import DataFrameConverter
        
        # Initialize configuration once per worker
        _config_cache = ConfigurationManager(config_path)
        _converter_cache = DataFrameConverter(_config_cache)
        
        # Initialize model
        model_id = _config_cache.get('mqtt.processing.model_id', 4)
        _model_cache = DepthInference(n_model=model_id)
        _model_initialized = True
        
        logging.info(f"Worker {os.getpid()} initialized with model {model_id} and cached configuration")
    except Exception as e:
        logging.error(f"Failed to initialize worker {os.getpid()}: {e}")
        raise


def _process_messages(args: tuple) -> Dict[str, Any]:
    """
    Process messages using cached model and configuration.
    
    Args:
        args: Tuple of (matched_messages, config_path)
        
    Returns:
        Processing result dictionary
    """
    matched_messages, config_path = args
    
    try:
        from .message_processor import MessageProcessor
        
        # Ensure worker is initialized (in case of new worker)
        if not _model_initialized:
            _init_worker(config_path)
        
        # Use cached components - no more disk I/O per message!
        processor = MessageProcessor(_model_cache, _converter_cache, _config_cache, "0.2.5")
        
        # Process the messages
        result = processor.process_matching_messages(matched_messages)
        
        # Convert result to serializable format
        return {
            'success': result.success,
            'keypoints': result.keypoints,
            'depth_estimation': result.depth_estimation,
            'machine_id': result.machine_id,
            'result_id': result.result_id,
            'head_id': result.head_id,
            'error_message': result.error_message
        }
        
    except Exception as e:
        logging.error(f"Worker processing error: {e}", exc_info=True)
        return {
            'success': False,
            'error_message': str(e)
        }


class SimpleProcessingPool:
    """
    Minimal processing pool with basic monitoring.
    
    Uses ProcessPoolExecutor to parallelize depth inference across multiple
    worker processes, each with its own DepthInference model instance.
    """
    
    def __init__(self, max_workers: int = 10, config_path: str = 'mqtt_conf.yaml'):
        """
        Initialize the processing pool.
        
        Args:
            max_workers: Maximum number of worker processes
            config_path: Path to configuration file
        """
        self.max_workers = max_workers
        self.config_path = config_path
        
        # Set model ID environment variable for workers
        if 'MODEL_ID' not in os.environ:
            os.environ['MODEL_ID'] = '4'
        
        self.executor = ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(config_path,)
        )
        self.pending_futures: Dict[Future, float] = {}
        
        # Simple counters for monitoring
        self.submitted = 0
        self.completed = 0
        self.failed = 0
        self.start_time = time.time()
        
        logging.info(f"Initialized processing pool with {max_workers} workers")
        
    def submit(self, matched_messages: List[Any], callback: Optional[Callable] = None) -> bool:
        """
        Submit messages for processing.
        
        Args:
            matched_messages: Messages to process
            callback: Optional callback for when processing completes
            
        Returns:
            True if submitted successfully, False if pool is full
        """
        # Simple backpressure: reject if too many pending
        if len(self.pending_futures) >= self.max_workers * 2:
            return False
            
        try:
            future = self.executor.submit(_process_messages, (matched_messages, self.config_path))
            self.pending_futures[future] = time.time()
            
            if callback:
                # Wrap callback to handle exceptions
                def safe_callback(fut):
                    try:
                        result = fut.result()
                        callback(result)
                    except Exception as e:
                        logging.error(f"Callback error: {e}")
                        
                future.add_done_callback(safe_callback)
                
            self.submitted += 1
            return True
            
        except Exception as e:
            logging.error(f"Failed to submit task: {e}")
            return False
        
    def collect_completed(self) -> int:
        """
        Collect completed futures and update counters.
        
        Returns:
            Number of completed tasks collected
        """
        completed = []
        
        for future in list(self.pending_futures.keys()):
            if future.done():
                try:
                    result = future.result()
                    if result.get('success', False):
                        self.completed += 1
                    else:
                        self.failed += 1
                        
                except Exception as e:
                    logging.error(f"Task failed with exception: {e}")
                    self.failed += 1
                    
                completed.append(future)
                
        # Remove completed futures
        for future in completed:
            submit_time = self.pending_futures.pop(future, None)
            if submit_time:
                processing_time = time.time() - submit_time
                if processing_time > 2.0:  # Log slow tasks
                    logging.debug(f"Slow task completed: {processing_time:.2f}s")
            
        return len(completed)
        
    def get_queue_depth(self) -> int:
        """Get number of pending tasks."""
        return len(self.pending_futures)
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get pool statistics.
        
        Returns:
            Dictionary of pool metrics
        """
        elapsed = time.time() - self.start_time
        
        return {
            'workers': self.max_workers,
            'queue_depth': self.get_queue_depth(),
            'submitted': self.submitted,
            'completed': self.completed,
            'failed': self.failed,
            'success_rate': self.completed / self.submitted if self.submitted > 0 else 0,
            'avg_throughput': self.completed / elapsed if elapsed > 0 else 0
        }
        
    def shutdown(self, wait: bool = True):
        """
        Shutdown the pool gracefully.
        
        Args:
            wait: Whether to wait for pending tasks to complete
        """
        logging.info("Shutting down processing pool")
        self.executor.shutdown(wait=wait)
        
        if self.pending_futures and wait:
            logging.warning(f"Shutdown with {len(self.pending_futures)} pending tasks")