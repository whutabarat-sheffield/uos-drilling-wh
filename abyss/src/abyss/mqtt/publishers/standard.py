"""
Standard MQTT publisher for normal operation mode.
"""

import json
import logging
import random
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional

from .base import BasePublisher, PublisherConfig
from .patterns import DrillPattern, PatternGenerator


class StandardPublisher(BasePublisher):
    """Standard MQTT publisher with realistic drilling patterns."""
    
    def __init__(self, config: PublisherConfig, use_patterns: bool = True):
        super().__init__(config)
        self.use_patterns = use_patterns
        self.pattern_generator = PatternGenerator() if use_patterns else None
        self._running = True
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle termination signals gracefully."""
        self.logger.info("\nTermination signal received. Exiting gracefully...")
        self._running = False
    
    def run(self):
        """Main publishing loop."""
        # Set up client
        self.setup_client()
        self._start_time = time.time()
        
        # Find test data folders
        try:
            data_folders = self.find_test_data_folders()
        except ValueError as e:
            self.logger.error(e)
            return
        
        # Initialize signal tracking CSV if enabled
        if self.config.track_signals:
            with open(self.config.signal_log, 'w', newline='') as f:
                import csv
                writer = csv.writer(f)
                writer.writerow(['signal_id', 'timestamp', 'toolbox_id', 'tool_id'])
        
        # Main publishing loop
        repetition = 0
        while self._running and (self.config.repetitions == 0 or repetition < self.config.repetitions):
            repetition += 1
            
            if self.config.repetitions > 0:
                self.logger.info(f"Starting repetition {repetition}/{self.config.repetitions}")
            
            # Select random data folder
            data_folder = random.choice(data_folders)
            self.logger.debug(f"Selected data folder: {data_folder}")
            
            try:
                # Load test data
                result_data, trace_data, heads_data, original_timestamp = self.load_test_data(data_folder)
                
                # Generate new timestamp
                new_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.localtime())
                
                # Select random toolbox and tool IDs
                toolbox_id = random.choice(self.config.toolbox_ids)
                tool_id = random.choice(self.config.tool_ids)
                
                # Handle signal tracking
                signal_id = None
                if self.config.track_signals:
                    signal_id = str(uuid.uuid4())
                    self.log_signal(signal_id, toolbox_id, tool_id)
                    
                    # Inject signal ID into data
                    result_data = self.inject_signal_tracking(result_data, signal_id)
                    trace_data = self.inject_signal_tracking(trace_data, signal_id)
                    heads_data = self.inject_signal_tracking(heads_data, signal_id)
                    
                    self.logger.debug(f"Signal tracking enabled - ID: {signal_id}")
                
                # Build topics
                topic_result = f"{self.config.root_topic}/{toolbox_id}/{tool_id}/{self.config.result_suffix}"
                topic_trace = f"{self.config.root_topic}/{toolbox_id}/{tool_id}/{self.config.trace_suffix}"
                topic_heads = f"{self.config.root_topic}/{toolbox_id}/{tool_id}/{self.config.heads_suffix}"
                
                # Prepare messages for publishing in random order
                messages = [
                    (topic_result, result_data),
                    (topic_trace, trace_data),
                    (topic_heads, heads_data)
                ]
                random.shuffle(messages)
                
                # Publish messages
                for topic, payload in messages:
                    if not self._running:
                        break
                        
                    self.publish_message(topic, payload, original_timestamp, new_timestamp)
                    
                    # Get sleep interval
                    if self.use_patterns and self.pattern_generator:
                        interval = self.pattern_generator.get_next_interval()
                        pattern_info = self.pattern_generator.get_current_pattern_info()
                        self.logger.debug(f"Pattern: {pattern_info['name']}, interval: {interval:.2f}s")
                    else:
                        interval = random.uniform(self.config.sleep_min, self.config.sleep_max)
                    
                    time.sleep(interval)
                
                # Log current stats periodically
                if repetition % 10 == 0:
                    stats = self.get_stats()
                    self.logger.info(
                        f"Progress: {stats['signals_sent']} signals sent, "
                        f"rate: {stats['rate']:.2f} signals/sec"
                    )
            
            except FileNotFoundError as e:
                self.logger.error(f"File not found: {e}. Skipping this data folder.")
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error: {e}. Skipping this data folder.")
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}. Skipping this data folder.")
        
        # Final statistics
        stats = self.get_stats()
        self.logger.info(
            f"\nPublishing completed:\n"
            f"  Total signals: {stats['signals_sent']}\n"
            f"  Total time: {stats['elapsed_time']:.2f}s\n"
            f"  Average rate: {stats['rate']:.2f} signals/sec"
        )
        
        # Disconnect
        self.disconnect()