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
            
            # Message buffering
            self.message_buffer = MessageBuffer(
                config=self.config_manager,  # Pass ConfigurationManager instead of raw config
                cleanup_interval=self.config_manager.get_cleanup_interval(),
                max_buffer_size=self.config_manager.get_max_buffer_size(),
                max_age_seconds=300  # 5 minutes
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
            
            logging.info("All components initialized successfully")
            
        except Exception as e:
            logging.error("Failed to initialize components", extra={
                'error_type': type(e).__name__,
                'error_message': str(e)
            }, exc_info=True)
            raise
    
    def _setup_result_publisher(self):
        """Set up result publisher with the result client."""
        try:
            result_client = self.client_manager.get_client('result')
            if result_client:
                self.result_publisher = ResultPublisher(
                    mqtt_client=result_client,
                    config=self.config_manager
                )
                logging.info("Result publisher initialized")
            else:
                logging.warning("No result client available for result publisher")
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
        logging.info("Starting continuous processing thread")
        
        while self.processing_active:
            try:
                # Get all buffer contents
                buffers = self.message_buffer.get_all_buffers()
                
                # Find and process matches
                matches_found = self.message_correlator.find_and_process_matches(
                    buffers=buffers,
                    message_processor=self._process_matched_messages
                )
                
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
            
            # Start client loops
            logging.info("Starting MQTT client loops")
            result_client.loop_start()
            trace_client.loop_start()
            if heads_client:
                heads_client.loop_start()
            
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
                
                client_types = ['result', 'trace']
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