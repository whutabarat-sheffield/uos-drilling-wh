"""
High-performance MQTT stress test publisher using threading.
"""

import json
import logging
import queue
import random
import signal
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base import BasePublisher, PublisherConfig


@dataclass
class StressTestConfig(PublisherConfig):
    """Extended configuration for stress testing."""
    target_rate: int = 1000  # messages per second
    duration: int = 60  # seconds
    num_threads: int = 10
    batch_size: int = 100
    queue_size: int = 10000


class StressTestPublisher(BasePublisher):
    """High-performance MQTT publisher for stress testing."""
    
    def __init__(self, config: StressTestConfig):
        super().__init__(config)
        self.config: StressTestConfig = config
        self._running = True
        self._message_queue = queue.Queue(maxsize=config.queue_size)
        self._stats_lock = threading.Lock()
        self._thread_stats: Dict[int, Dict] = {}
        
        # Pre-generate message batches for performance
        self._message_batches: List[List[Tuple[str, str]]] = []
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle termination signals gracefully."""
        self.logger.info("\nTermination signal received. Stopping stress test...")
        self._running = False
    
    def _prepare_message_batches(self):
        """Pre-generate message batches for maximum performance."""
        self.logger.info("Preparing message batches...")
        
        # Find test data folders
        data_folders = self.find_test_data_folders()
        
        # Pre-load all test data
        all_data = []
        for folder in data_folders:
            try:
                result_data, trace_data, heads_data, original_timestamp = self.load_test_data(folder)
                all_data.append((result_data, trace_data, heads_data, original_timestamp))
            except Exception as e:
                self.logger.warning(f"Failed to load data from {folder}: {e}")
        
        if not all_data:
            raise ValueError("No valid test data could be loaded")
        
        # Generate batches
        batch_count = max(10, self.config.duration)  # At least 10 batches
        
        for _ in range(batch_count):
            batch = []
            
            for _ in range(self.config.batch_size):
                # Select random data set
                result_data, trace_data, heads_data, original_timestamp = random.choice(all_data)
                
                # Generate identifiers
                toolbox_id = random.choice(self.config.toolbox_ids)
                tool_id = random.choice(self.config.tool_ids)
                new_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.localtime())
                
                # Handle signal tracking if enabled
                if self.config.track_signals:
                    signal_id = str(uuid.uuid4())
                    self.log_signal(signal_id, toolbox_id, tool_id)
                    
                    result_data = self.inject_signal_tracking(result_data, signal_id)
                    trace_data = self.inject_signal_tracking(trace_data, signal_id)
                    heads_data = self.inject_signal_tracking(heads_data, signal_id)
                
                # Replace timestamps
                result_data = result_data.replace(original_timestamp, new_timestamp)
                trace_data = trace_data.replace(original_timestamp, new_timestamp)
                heads_data = heads_data.replace(original_timestamp, new_timestamp)
                
                # Build topics
                topic_result = f"{self.config.root_topic}/{toolbox_id}/{tool_id}/{self.config.result_suffix}"
                topic_trace = f"{self.config.root_topic}/{toolbox_id}/{tool_id}/{self.config.trace_suffix}"
                topic_heads = f"{self.config.root_topic}/{toolbox_id}/{tool_id}/{self.config.heads_suffix}"
                
                # Add to batch (3 messages per signal)
                batch.extend([
                    (topic_result, result_data),
                    (topic_trace, trace_data),
                    (topic_heads, heads_data)
                ])
            
            self._message_batches.append(batch)
        
        self.logger.info(f"Prepared {len(self._message_batches)} batches with {self.config.batch_size} signals each")
    
    def _publisher_worker(self, worker_id: int):
        """Worker thread for publishing messages."""
        thread_stats = {
            'messages_sent': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
        while self._running:
            try:
                # Get message from queue with timeout
                topic, payload = self._message_queue.get(timeout=1.0)
                
                # Publish message
                self.client.publish(topic, payload)
                thread_stats['messages_sent'] += 1
                self._signal_counter += 1
                
                self._message_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                thread_stats['errors'] += 1
                self.logger.error(f"Worker {worker_id} error: {e}")
        
        # Store final stats
        with self._stats_lock:
            self._thread_stats[worker_id] = thread_stats
    
    def _producer_worker(self):
        """Producer thread that feeds messages to the queue."""
        messages_per_second = self.config.target_rate * 3  # 3 messages per signal
        interval = 1.0 / (messages_per_second / self.config.batch_size)
        
        batch_index = 0
        
        while self._running:
            start_time = time.time()
            
            # Get next batch
            batch = self._message_batches[batch_index % len(self._message_batches)]
            batch_index += 1
            
            # Add messages to queue
            for topic, payload in batch:
                if not self._running:
                    break
                    
                try:
                    self._message_queue.put((topic, payload), timeout=1.0)
                except queue.Full:
                    self.logger.warning("Message queue full, dropping messages")
                    break
            
            # Sleep to maintain target rate
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def run(self):
        """Run the stress test."""
        # Set up client
        self.setup_client()
        self._start_time = time.time()
        
        # Initialize signal tracking CSV if enabled
        if self.config.track_signals:
            with open(self.config.signal_log, 'w', newline='') as f:
                import csv
                writer = csv.writer(f)
                writer.writerow(['signal_id', 'timestamp', 'toolbox_id', 'tool_id'])
        
        # Prepare message batches
        try:
            self._prepare_message_batches()
        except ValueError as e:
            self.logger.error(e)
            return
        
        self.logger.info(
            f"Starting stress test:\n"
            f"  Target rate: {self.config.target_rate} signals/sec\n"
            f"  Duration: {self.config.duration} seconds\n"
            f"  Threads: {self.config.num_threads}\n"
            f"  Batch size: {self.config.batch_size}"
        )
        
        # Start publisher threads
        with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
            # Start publisher workers
            publisher_futures = []
            for i in range(self.config.num_threads):
                future = executor.submit(self._publisher_worker, i)
                publisher_futures.append(future)
            
            # Start producer thread
            producer_future = executor.submit(self._producer_worker)
            
            # Monitor progress
            start_time = time.time()
            last_count = 0
            
            while self._running and (time.time() - start_time) < self.config.duration:
                time.sleep(1.0)
                
                # Calculate and display rate
                current_count = self._signal_counter
                rate = (current_count - last_count) / 3  # Convert messages to signals
                last_count = current_count
                
                elapsed = time.time() - start_time
                avg_rate = (current_count / 3) / elapsed if elapsed > 0 else 0
                
                self.logger.info(
                    f"[{elapsed:.1f}s] Current: {rate:.1f} signals/sec, "
                    f"Average: {avg_rate:.1f} signals/sec, "
                    f"Total: {current_count // 3} signals, "
                    f"Queue: {self._message_queue.qsize()}"
                )
            
            # Stop test
            self._running = False
            
            # Wait for threads to finish
            self.logger.info("Waiting for threads to complete...")
            for future in publisher_futures:
                future.result()
            producer_future.result()
        
        # Final statistics
        stats = self.get_stats()
        total_signals = stats['signals_sent'] // 3
        
        self.logger.info(
            f"\nStress test completed:\n"
            f"  Total signals: {total_signals}\n"
            f"  Total messages: {stats['signals_sent']}\n"
            f"  Duration: {stats['elapsed_time']:.2f}s\n"
            f"  Average rate: {total_signals / stats['elapsed_time']:.2f} signals/sec\n"
            f"  Target rate: {self.config.target_rate} signals/sec\n"
            f"  Achievement: {(total_signals / stats['elapsed_time']) / self.config.target_rate * 100:.1f}%"
        )
        
        # Thread statistics
        if self._thread_stats:
            self.logger.info("\nThread statistics:")
            for worker_id, thread_stats in self._thread_stats.items():
                self.logger.info(
                    f"  Worker {worker_id}: {thread_stats['messages_sent']} messages, "
                    f"{thread_stats['errors']} errors"
                )
        
        # Disconnect
        self.disconnect()