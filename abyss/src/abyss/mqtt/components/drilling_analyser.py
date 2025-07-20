"""
Simplified MQTT Drilling Data Analyser Orchestrator

This is the main orchestrator class that coordinates all the refactored components
to replicate the functionality of the original MQTTDrillingDataAnalyser.
"""

import logging
import time
import threading
from typing import List, Optional, Dict, Any
from importlib.metadata import version

from .config_manager import ConfigurationManager
from .client_manager import MQTTClientManager
from .message_buffer import MessageBuffer
from .simple_correlator import SimpleMessageCorrelator
from .message_processor import MessageProcessor
from .data_converter import DataFrameConverter
from .result_publisher import ResultPublisher
from .processing_pool import SimpleProcessingPool
from .throughput_monitor import SimpleThroughputMonitor

from ...uos_depth_est import TimestampedData
from ...uos_depth_est_core import DepthInference


class DrillingDataAnalyser:
    """
    Simplified orchestrator for MQTT drilling data analysis.
    
    This class coordinates all the refactored components to provide
    the same functionality as the original monolithic class, but with
    better separation of concerns and maintainability.
    """
    
    def __init__(self, config_path: str = 'mqtt_conf.yaml'):
        """
        Initialize the drilling data analyser.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.processing_active = False
        self.processing_thread = None
        
        # Algorithm version and IDs
        try:
            self.ALGO_VERSION = version('abyss')
        except Exception:
            self.ALGO_VERSION = "0.2.5"  # Fallback version
        self.MACHINE_ID = "MACHINE_ID"
        self.RESULT_ID = "RESULT_ID"
        
        # Error tracking for warnings
        self._processing_errors = []
        self._max_error_history = 100
        self._error_warning_threshold = 10  # Warn if 10 errors in 60 seconds
        
        # Processing pool and monitoring
        self.processing_pool = None
        self.throughput_monitor = None
        self.last_status_log_time = 0
        
        # Initialize components
        self._initialize_components()
        
        logging.info("DrillingDataAnalyser initialized successfully", extra={
            'config_path': config_path,
            'algo_version': self.ALGO_VERSION
        })
    
    def _initialize_components(self):
        """Initialize all the refactored components."""
        try:
            # Configuration management
            self.config_manager = ConfigurationManager(self.config_path)
            config = self.config_manager.get_raw_config()
            
            # Initialize throughput monitor first so it can be passed to other components
            self.throughput_monitor = SimpleThroughputMonitor(sample_rate=0.1)
            
            # Message buffering
            self.message_buffer = MessageBuffer(
                config=self.config_manager,  # Pass ConfigurationManager instead of raw config
                cleanup_interval=self.config_manager.get_cleanup_interval(),
                max_buffer_size=self.config_manager.get_max_buffer_size(),
                max_age_seconds=300,  # 5 minutes
                throughput_monitor=self.throughput_monitor
            )
            
            # Message correlation
            self.message_correlator = SimpleMessageCorrelator(
                config=self.config_manager,
                time_window=self.config_manager.get_time_window()
            )
            
            # Data conversion
            self.data_converter = DataFrameConverter(config=self.config_manager)
            
            # Depth inference
            self.depth_inference = DepthInference()
            
            # Message processing
            self.message_processor = MessageProcessor(
                depth_inference=self.depth_inference,
                data_converter=self.data_converter,
                config=self.config_manager,
                algo_version=self.ALGO_VERSION
            )
            
            # MQTT client management
            self.client_manager = MQTTClientManager(
                config=self.config_manager,
                message_handler=self.message_buffer.add_message
            )
            
            # Result publishing (will be set up after clients are created)
            self.result_publisher = None
            
            # Initialize processing pool
            num_workers = self.config_manager.get('mqtt.processing.workers', 10)
            self.processing_pool = SimpleProcessingPool(max_workers=num_workers, config_path=self.config_path)
            
            logging.info("All components initialized successfully", extra={
                'processing_workers': num_workers
            })
            
        except Exception as e:
            logging.error("Failed to initialize components", extra={
                'error_type': type(e).__name__,
                'error_message': str(e)
            }, exc_info=True)
            raise
    
    def _setup_result_publisher(self):
        """Set up result publisher with a dedicated publisher client."""
        try:
            # Create a dedicated publisher client
            publisher_client = self.client_manager.create_publisher()
            if publisher_client:
                self.result_publisher = ResultPublisher(
                    mqtt_client=publisher_client,
                    config=self.config_manager
                )
                logging.info("Result publisher initialized with dedicated publisher client")
            else:
                logging.warning("Failed to create publisher client")
        except Exception as e:
            logging.error("Failed to setup result publisher", extra={
                'error_type': type(e).__name__,
                'error_message': str(e)
            })
    
    def continuous_processing(self):
        """
        Continuously check for and process matches in a separate thread.
        
        This replicates the continuous_processing method from the original class.
        """
        logging.info("Starting continuous processing thread with worker pool")
        
        log_interval = 30.0  # Log status every 30 seconds
        
        while self.processing_active:
            try:
                # Get all buffer contents
                buffers = self.message_buffer.get_all_buffers()
                
                # Find and process matches
                matches_found = self.message_correlator.find_and_process_matches(
                    buffers=buffers,
                    message_processor=self._submit_to_pool  # Changed to pool submission
                )
                
                # Collect completed results from pool
                self.processing_pool.collect_completed()
                
                # Log simple status every 30 seconds
                current_time = time.time()
                if current_time - self.last_status_log_time >= log_interval:
                    self._log_simple_status()
                    self.last_status_log_time = current_time
                    # Also check publisher health periodically
                    self._check_publisher_health()
                
                if not matches_found:
                    # Sleep if no matches found to avoid CPU spinning
                    time.sleep(0.1)
                else:
                    # Brief pause even when processing to allow other operations
                    time.sleep(0.01)
                    
            except Exception as e:
                logging.error("Error in continuous processing", extra={
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }, exc_info=True)
                
                # Track errors for warning detection
                self._track_processing_error(e)
                
                time.sleep(1)  # Longer sleep on error to prevent rapid error loops
        
        # Shutdown pool when stopping
        if self.processing_pool:
            self.processing_pool.shutdown()
            
        logging.info("Continuous processing thread stopped")
    
    def _process_matched_messages(self, matches: List[TimestampedData]):
        """
        Process matched messages and publish results.
        
        Args:
            matches: List of matched timestamped messages
        """
        try:
            # Process the messages
            processing_result = self.message_processor.process_matching_messages(matches)
            
            if not processing_result.success:
                logging.warning("Message processing failed", extra={
                    'error_message': processing_result.error_message
                })
                return
            
            # Extract tool information from the first message
            first_message = matches[0]
            parts = first_message.source.split('/')
            if len(parts) >= 3:
                toolbox_id = parts[1]
                tool_id = parts[2]
                
                # Publish results if publisher is available
                if self.result_publisher:
                    try:
                        self.result_publisher.publish_processing_result(
                            processing_result=processing_result,
                            toolbox_id=toolbox_id,
                            tool_id=tool_id,
                            timestamp=first_message.timestamp,
                            algo_version=self.ALGO_VERSION
                        )
                        
                        logging.debug("Results published successfully", extra={
                            'toolbox_id': toolbox_id,
                            'tool_id': tool_id,
                            'has_keypoints': processing_result.keypoints is not None,
                            'has_depth_estimation': processing_result.depth_estimation is not None
                        })
                    except Exception as e:
                        logging.warning("Failed to publish results", extra={
                            'toolbox_id': toolbox_id,
                            'tool_id': tool_id,
                            'error_type': type(e).__name__,
                            'error_message': str(e)
                        })
                else:
                    logging.warning("No result publisher available for publishing results")
            else:
                logging.error("Invalid message source format", extra={
                    'source': first_message.source
                })
                
        except Exception as e:
            logging.error("Error processing matched messages", extra={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'message_count': len(matches)
            }, exc_info=True)
    
    def _submit_to_pool(self, matches: List[TimestampedData]):
        """
        Submit matched messages to processing pool.
        
        Args:
            matches: List of matched timestamped messages
        """
        start_time = time.time()
        
        # Try to submit to pool
        submitted = self.processing_pool.submit(
            matches,
            callback=lambda result: self._handle_pool_result(result, matches, start_time)
        )
        
        if not submitted:
            logging.warning("Processing pool full - applying backpressure")
            # Buffer will handle overflow
    
    def _handle_pool_result(self, result: Dict[str, Any], matches: List[TimestampedData], start_time: float):
        """
        Handle result from pool processing.
        
        Args:
            result: Processing result from pool
            matches: Original matched messages
            start_time: When processing started
        """
        # Record completion for monitoring
        self.throughput_monitor.record_processing_complete(start_time)
        
        # Convert result dict back to ProcessingResult-like object
        class ProcessingResult:
            def __init__(self, result_dict):
                self.success = result_dict.get('success', False)
                self.keypoints = result_dict.get('keypoints')
                self.depth_estimation = result_dict.get('depth_estimation')
                self.machine_id = result_dict.get('machine_id')
                self.result_id = result_dict.get('result_id')
                self.head_id = result_dict.get('head_id')
                self.error_message = result_dict.get('error_message')
        
        processing_result = ProcessingResult(result)
        
        # Existing result handling logic
        if not processing_result.success:
            logging.warning("Message processing failed", extra={
                'error_message': processing_result.error_message
            })
            return
        
        # Extract tool information from the first message
        first_message = matches[0]
        parts = first_message.source.split('/')
        if len(parts) >= 3:
            toolbox_id = parts[1]
            tool_id = parts[2]
            
            # Publish results if publisher is available
            if self.result_publisher:
                try:
                    # Log what we're attempting to publish
                    logging.info(f"Attempting to publish results for {toolbox_id}/{tool_id}")
                    
                    self.result_publisher.publish_processing_result(
                        processing_result=processing_result,
                        toolbox_id=toolbox_id,
                        tool_id=tool_id,
                        timestamp=first_message.timestamp,
                        algo_version=self.ALGO_VERSION
                    )
                    
                    logging.debug("Results published successfully", extra={
                        'toolbox_id': toolbox_id,
                        'tool_id': tool_id,
                        'has_keypoints': processing_result.keypoints is not None,
                        'has_depth_estimation': processing_result.depth_estimation is not None
                    })
                except Exception as e:
                    logging.warning("Failed to publish results", extra={
                        'toolbox_id': toolbox_id,
                        'tool_id': tool_id,
                        'error_type': type(e).__name__,
                        'error_message': str(e)
                    })
                    # Check if publisher client is still connected
                    self._check_publisher_health()
    
    def _check_publisher_health(self):
        """Check and recover publisher client if needed."""
        try:
            publisher_client = self.client_manager.get_client('publisher')
            if publisher_client and not publisher_client.is_connected():
                logging.warning("Publisher client disconnected, attempting to reconnect")
                broker_config = self.config_manager.get_mqtt_broker_config()
                try:
                    publisher_client.reconnect()
                    logging.info("Publisher client reconnected successfully")
                except Exception as e:
                    logging.error("Failed to reconnect publisher client", extra={
                        'error_type': type(e).__name__,
                        'error_message': str(e)
                    })
                    # Try to create a new publisher
                    self._setup_result_publisher()
        except Exception as e:
            logging.error("Error checking publisher health", extra={
                'error_type': type(e).__name__,
                'error_message': str(e)
            })
    
    def _log_simple_status(self):
        """Log simple status every 30 seconds."""
        try:
            # Get comprehensive metrics
            metrics = self.get_comprehensive_metrics()
            
            # Get throughput status
            status = self.throughput_monitor.get_status()
            
            # Extract key values
            messages = metrics.get('messages', {})
            buffer = metrics.get('buffer', {})
            processing = metrics.get('processing', {})
            correlation = metrics.get('correlation', {})
            
            # Build buffer breakdown string
            buffer_by_topic = buffer.get('by_topic', {})
            buffer_breakdown = ", ".join([f"{k}: {v}" for k, v in buffer_by_topic.items()])
            
            # Build detailed status message
            status_parts = [
                f"Arrivals: {status.details.get('arrival_rate', 'N/A')}",
                f"Processing: {status.details.get('processing_rate', 'N/A')}",
                f"Correlated: {messages.get('correlated', 0)}/{correlation.get('attempts', 0)}",
                f"Queue: {processing.get('queue_depth', 0)}",
                f"Buffer: {buffer.get('total_messages', 0)} ({buffer_breakdown})"
            ]
            
            # Add oldest message age if buffer has messages
            if buffer.get('total_messages', 0) > 0:
                oldest_age = buffer.get('oldest_message_age', 0.0)
                status_parts.append(f"Oldest: {oldest_age:.1f}s")
            
            # Log appropriate level based on status
            if status.status == 'FALLING_BEHIND':
                logging.warning(f"System falling behind: {' | '.join(status_parts)}")
                
                # Add detailed diagnostics for falling behind
                unprocessed = correlation.get('unprocessed_messages', {})
                if any(v > 0 for v in unprocessed.values()):
                    logging.warning(f"Unprocessed messages: {unprocessed}")
                    
            else:
                logging.info(f"System status: {' | '.join(status_parts)}")
            
            # Alert if queue is getting too deep
            queue_depth = processing.get('queue_depth', 0)
            if queue_depth > 15:
                logging.error(f"Processing queue critical: {queue_depth} pending")
                
        except Exception as e:
            logging.debug(f"Error logging status: {e}")
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics from all components.
        
        Returns:
            Dictionary containing all system metrics
        """
        try:
            # Throughput metrics
            throughput_metrics = self.throughput_monitor.get_metrics()
            
            # Buffer metrics
            buffer_stats = self.message_buffer.get_buffer_stats()
            buffer_metrics = self.message_buffer._metrics.copy()  # Get internal metrics
            
            # Correlation metrics
            correlation_stats = self.message_correlator.get_correlation_stats(
                self.message_buffer.get_all_buffers()
            )
            
            # Processing pool metrics
            pool_stats = self.processing_pool.get_stats()
            
            # Calculate derived metrics
            messages_buffered = buffer_metrics.get('messages_received', 0) - buffer_metrics.get('duplicate_messages', 0)
            
            return {
                'messages': {
                    'received': buffer_metrics.get('messages_received', 0),
                    'buffered': messages_buffered,
                    'duplicates': buffer_metrics.get('duplicate_messages', 0),
                    'dropped': buffer_metrics.get('messages_dropped', 0),
                    'correlated': correlation_stats.get('correlation_successes', 0),
                    'processed': pool_stats.get('completed', 0),
                    'failed': pool_stats.get('failed', 0),
                    'expired': sum(buffer_metrics.get('messages_dropped_age', {}).values())
                },
                'rates': {
                    'arrival_rate': throughput_metrics.get('arrival_rate', 0.0),
                    'processing_rate': throughput_metrics.get('processing_rate', 0.0),
                    'correlation_success_rate': correlation_stats.get('correlation_success_rate', 0.0)
                },
                'buffer': {
                    'total_messages': buffer_stats.get('total_messages', 0),
                    'by_topic': buffer_stats.get('buffer_sizes', {}),
                    'oldest_message_age': buffer_stats.get('oldest_message_age', 0.0),
                    'average_message_age': buffer_stats.get('average_message_age', 0.0),
                    'message_type_distribution': buffer_stats.get('message_type_distribution', {}),
                    'age_distribution': buffer_stats.get('age_distribution', {})
                },
                'processing': {
                    'queue_depth': pool_stats.get('queue_depth', 0),
                    'workers': pool_stats.get('workers', 0),
                    'avg_processing_time_ms': throughput_metrics.get('avg_processing_time_ms', 0.0)
                },
                'correlation': {
                    'attempts': correlation_stats.get('correlation_attempts', 0),
                    'successes': correlation_stats.get('correlation_successes', 0),
                    'unprocessed_messages': correlation_stats.get('unprocessed_messages', {})
                }
            }
        except Exception as e:
            logging.error("Error collecting comprehensive metrics", extra={
                'error_type': type(e).__name__,
                'error_message': str(e)
            })
            return {}
    
    def run(self):
        """
        Main method to set up MQTT clients and start listening.
        
        This replicates the functionality of the original run() method.
        """
        try:
            logging.info("Starting drilling data analyser")
            
            # Create MQTT clients
            logging.info("Creating MQTT clients")
            result_client = self.client_manager.create_result_listener()
            trace_client = self.client_manager.create_trace_listener()
            
            # Create heads client only if configured
            heads_client = None
            if 'heads' in self.config_manager.get_mqtt_listener_config():
                heads_client = self.client_manager.create_heads_listener()
            
            # Setup result publisher now that we have the result client
            self._setup_result_publisher()
            
            # Connect all clients
            logging.info("Connecting to MQTT broker")
            broker_config = self.config_manager.get_mqtt_broker_config()
            
            result_client.connect(broker_config['host'], broker_config['port'])
            trace_client.connect(broker_config['host'], broker_config['port'])
            if heads_client:
                heads_client.connect(broker_config['host'], broker_config['port'])
            
            # Connect publisher client if it was created
            publisher_client = self.client_manager.get_client('publisher')
            if publisher_client:
                publisher_client.connect(broker_config['host'], broker_config['port'])
                logging.info("Publisher client connected")
            
            # Start client loops
            logging.info("Starting MQTT client loops")
            result_client.loop_start()
            trace_client.loop_start()
            if heads_client:
                heads_client.loop_start()
            if publisher_client:
                publisher_client.loop_start()
            
            # Start continuous processing thread
            logging.info("Starting continuous processing")
            self.processing_active = True
            self.processing_thread = threading.Thread(target=self.continuous_processing)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            # Verify thread started successfully
            time.sleep(0.1)
            if not self.processing_thread.is_alive():
                logging.warning("Processing thread failed to start", extra={
                    'thread_status': 'not_alive',
                    'processing_active': self.processing_active
                })
            
            # Log startup summary
            config_summary = self.config_manager.get_config_summary()
            buffer_stats = self.message_buffer.get_buffer_stats()
            
            logging.info("Drilling data analyser started successfully", extra={
                'broker_host': config_summary.get('broker_host'),
                'broker_port': config_summary.get('broker_port'),
                'time_window': config_summary.get('time_window'),
                'buffer_stats': buffer_stats,
                'subscribed_topics': self.client_manager.get_subscribed_topics()
            })
            
            # Main loop - keep running until interrupted
            try:
                while True:
                    time.sleep(1)
                    
                    # Periodically log system status
                    if int(time.time()) % 60 == 0:  # Every minute
                        self._log_system_status()
                        
            except KeyboardInterrupt:
                logging.info("Keyboard interrupt received, shutting down...")
                self.shutdown()
                
        except Exception as e:
            logging.error("Error in main run method", extra={
                'error_type': type(e).__name__,
                'error_message': str(e)
            }, exc_info=True)
            self.shutdown()
            raise
    
    def _log_system_status(self):
        """Log periodic system status information."""
        try:
            buffer_stats = self.message_buffer.get_buffer_stats()
            correlation_stats = self.message_correlator.get_correlation_stats(
                self.message_buffer.get_all_buffers()
            )
            
            logging.debug("System status", extra={
                'processing_active': self.processing_active,
                'processing_thread_alive': self.processing_thread.is_alive() if self.processing_thread else False,
                'buffer_stats': buffer_stats,
                'correlation_stats': correlation_stats
            })
        except Exception as e:
            logging.debug("Error logging system status", extra={
                'error_message': str(e)
            })
    
    def shutdown(self):
        """
        Gracefully shutdown the analyser.
        
        This replicates the shutdown logic from the original run() method.
        """
        try:
            logging.info("Shutting down drilling data analyser")
            
            # Stop processing thread
            if self.processing_active:
                logging.info("Stopping continuous processing")
                self.processing_active = False
                
                if self.processing_thread and self.processing_thread.is_alive():
                    self.processing_thread.join(timeout=2.0)
                    if self.processing_thread.is_alive():
                        logging.warning("Processing thread did not stop within timeout")
            
            # Disconnect MQTT clients
            if hasattr(self, 'client_manager'):
                logging.info("Disconnecting MQTT clients")
                
                client_types = ['result', 'trace', 'publisher']
                if 'heads' in self.config_manager.get_mqtt_listener_config():
                    client_types.append('heads')
                
                for client_type in client_types:
                    client = self.client_manager.get_client(client_type)
                    if client:
                        try:
                            logging.debug(f"Stopping {client_type} client loop")
                            client.loop_stop()
                            
                            logging.debug(f"Disconnecting {client_type} client")
                            client.disconnect()
                            
                        except Exception as e:
                            logging.warning(f"Error disconnecting {client_type} client", extra={
                                'error_message': str(e)
                            })
            
            # Log final statistics
            if hasattr(self, 'message_buffer'):
                final_stats = self.message_buffer.get_buffer_stats()
                logging.info("Final buffer statistics", extra=final_stats)
            
            logging.info("Drilling data analyser shutdown complete")
            
        except Exception as e:
            logging.error("Error during shutdown", extra={
                'error_type': type(e).__name__,
                'error_message': str(e)
            }, exc_info=True)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the analyser.
        
        Returns:
            Dictionary containing status information
        """
        try:
            status = {
                'processing_active': self.processing_active,
                'processing_thread_alive': self.processing_thread.is_alive() if self.processing_thread else False,
                'algo_version': self.ALGO_VERSION,
                'config_path': self.config_path
            }
            
            if hasattr(self, 'message_buffer'):
                status['buffer_stats'] = self.message_buffer.get_buffer_stats()
            
            if hasattr(self, 'config_manager'):
                status['config_summary'] = self.config_manager.get_config_summary()
            
            if hasattr(self, 'client_manager'):
                status['subscribed_topics'] = self.client_manager.get_subscribed_topics()
            
            return status
            
        except Exception as e:
            logging.error("Error getting status", extra={
                'error_type': type(e).__name__,
                'error_message': str(e)
            })
            return {'error': str(e)}
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False
    
    def _track_processing_error(self, error: Exception):
        """Track processing errors and warn if rate is too high."""
        current_time = time.time()
        self._processing_errors.append(current_time)
        
        # Keep only recent errors
        if len(self._processing_errors) > self._max_error_history:
            self._processing_errors = self._processing_errors[-self._max_error_history:]
        
        # Check errors in last 60 seconds
        recent_errors = [t for t in self._processing_errors if current_time - t < 60]
        
        if len(recent_errors) >= self._error_warning_threshold:
            logging.warning("High rate of processing errors detected", extra={
                'errors_in_last_minute': len(recent_errors),
                'threshold': self._error_warning_threshold,
                'latest_error_type': type(error).__name__,
                'latest_error_message': str(error),
                'possible_cause': 'System may be overloaded or experiencing connectivity issues'
            })