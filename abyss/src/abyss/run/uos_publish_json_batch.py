#!/usr/bin/env python3
"""
Batch-optimized MQTT Publisher for Maximum Throughput

This implementation batches multiple messages together to reduce overhead
and achieve higher throughput rates.
"""

import argparse
import json
import logging
import signal
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import deque
import random

import paho.mqtt.client as mqtt
import yaml

from abyss.uos_depth_est_utils import setup_logging


class BatchPublisher(StressTestPublisher):
    """Batch-optimized MQTT publisher"""
    
    def __init__(self, config: Dict[str, Any], args: argparse.Namespace):
        super().__init__(config, args)
        self.batch_queue = deque()
        self.batch_lock = threading.Lock()
        self.batch_size = args.batch_size or 10
        self.batch_timeout = 0.1  # 100ms timeout for batching
    
    def batch_publish_signals(self, client: mqtt.Client, batch: List[Tuple[str, str, Dict[str, str]]]) -> int:
        """Publish a batch of signals efficiently"""
        messages_published = 0
        
        # Pre-generate timestamp for the entire batch
        new_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.localtime())
        
        # Prepare all messages first
        messages_to_publish = []
        
        for toolbox_id, tool_id, data in batch:
            try:
                # Build topics
                mqtt_root = self.config['mqtt']['listener']['root']
                topic_result = f"{mqtt_root}/{toolbox_id}/{tool_id}/{self.config['mqtt']['listener']['result']}"
                topic_trace = f"{mqtt_root}/{toolbox_id}/{tool_id}/{self.config['mqtt']['listener']['trace']}"
                topic_heads = f"{mqtt_root}/{toolbox_id}/{tool_id}/{self.config['mqtt']['listener']['heads']}"
                
                # Update payloads
                result_payload = data['result'].replace(data['result_timestamp'], new_timestamp)
                trace_payload = data['trace'].replace(data['result_timestamp'], new_timestamp)
                heads_payload = data['heads'].replace(data['result_timestamp'], new_timestamp)
                
                messages_to_publish.extend([
                    (topic_result, result_payload),
                    (topic_trace, trace_payload),
                    (topic_heads, heads_payload)
                ])
                
            except Exception as e:
                logging.error(f"Error preparing batch message: {e}")
                with self.lock:
                    self.metrics.messages_failed += 3
        
        # Publish all messages in rapid succession
        start_time = time.perf_counter()
        
        for topic, payload in messages_to_publish:
            try:
                info = client.publish(topic, payload, qos=0)
                if info.rc != mqtt.MQTT_ERR_SUCCESS:
                    with self.lock:
                        self.metrics.messages_failed += 1
                else:
                    with self.lock:
                        self.metrics.messages_sent += 1
                        self.metrics.bytes_sent += len(payload)
                    messages_published += 1
            except Exception as e:
                logging.error(f"Error publishing message: {e}")
                with self.lock:
                    self.metrics.messages_failed += 1
        
        # Track batch latency
        latency = time.perf_counter() - start_time
        with self.lock:
            self.metrics.latencies.append(latency / len(batch) if batch else 0)
        
        return messages_published
    
    def batch_worker_thread(self, client_id: int, client: mqtt.Client, data_folders: List[Path], 
                           toolbox_ids: List[str], tool_ids: List[str]):
        """Worker thread that publishes in batches"""
        logging.debug(f"Batch worker {client_id} started")
        
        local_batch = []
        last_batch_time = time.time()
        
        # Pre-select data folders
        folder_keys = [str(f) for f in data_folders if str(f) in self.data_cache]
        
        while self.running:
            try:
                # Generate signal data
                folder_key = random.choice(folder_keys)
                data = self.data_cache[folder_key]
                toolbox_id = random.choice(toolbox_ids)
                tool_id = random.choice(tool_ids)
                
                # Add to batch
                local_batch.append((toolbox_id, tool_id, data))
                
                # Check if we should publish the batch
                current_time = time.time()
                should_publish = (
                    len(local_batch) >= self.batch_size or 
                    (current_time - last_batch_time) >= self.batch_timeout
                )
                
                if should_publish and local_batch:
                    # Publish the batch
                    self.batch_publish_signals(client, local_batch)
                    local_batch = []
                    last_batch_time = current_time
                
            except Exception as e:
                logging.error(f"Batch worker {client_id} error: {e}")
                time.sleep(0.001)
        
        # Publish any remaining messages
        if local_batch:
            self.batch_publish_signals(client, local_batch)
    
    def run_batch_stress_test(self, data_folders: List[Path], duration: int):
        """Run the batch-optimized stress test"""
        logging.info(f"Starting batch stress test: {self.target_rate} signals/sec for {duration}s")
        logging.info(f"Batch size: {self.batch_size}")
        
        # Constants
        TOOLBOX_IDS = [
            'ILLL502033771', 'ILLL502033772', 'ILLL502033773', 
            'ILLL502033774', 'ILLL502033775'
        ] * 2  # Duplicate for more variety
        
        TOOL_IDS = [
            'setitec001', 'setitec002', 'setitec003', 
            'setitec004', 'setitec005'
        ] * 2
        
        # Pre-load all data
        logging.info("Pre-loading data into cache...")
        self.data_cache = self.load_and_cache_data(data_folders)
        if not self.data_cache:
            logging.error("No data loaded, exiting")
            return
        
        # Create MQTT clients with even more aggressive settings
        logging.info(f"Creating {self.args.concurrent_publishers} MQTT clients...")
        self.clients = []
        for i in range(self.args.concurrent_publishers):
            try:
                # Try new API
                client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, f"batch_publisher_{i}")
            except (AttributeError, TypeError):
                # Fall back to old API
                client = mqtt.Client(f"batch_publisher_{i}")
            
            # Very aggressive settings
            client.max_inflight_messages_set(500)  # Much higher
            client.max_queued_messages_set(5000)   # Much higher
            
            # Minimal callbacks
            client.on_connect = lambda c, u, f, rc: logging.debug(f"Client connected: {rc}")
            
            # Connect
            try:
                client.connect(
                    self.config['mqtt']['broker']['host'], 
                    self.config['mqtt']['broker']['port']
                )
                client.loop_start()
                self.clients.append(client)
            except Exception as e:
                logging.error(f"Failed to create client {i}: {e}")
        
        if not self.clients:
            logging.error("No MQTT clients created, exiting")
            return
        
        # Wait for connections
        time.sleep(0.5)
        
        # Create thread pool
        self.executor = ThreadPoolExecutor(max_workers=self.args.concurrent_publishers)
        
        # Start batch workers
        futures = []
        for i, client in enumerate(self.clients):
            future = self.executor.submit(
                self.batch_worker_thread, i, client, data_folders, TOOLBOX_IDS, TOOL_IDS
            )
            futures.append(future)
        
        # Monitor progress
        start_time = time.time()
        report_interval = 5
        last_report = start_time
        
        try:
            while self.running and (time.time() - start_time) < duration:
                current_time = time.time()
                
                if current_time - last_report >= report_interval:
                    self.print_metrics()
                    last_report = current_time
                
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            logging.info("Batch stress test interrupted by user")
        
        finally:
            # Shutdown
            logging.info("Shutting down batch stress test...")
            self.running = False
            
            if self.executor:
                self.executor.shutdown(wait=True)
            
            for client in self.clients:
                client.loop_stop()
                client.disconnect()
            
            # Final report
            self.print_metrics()
            
            # Summary with batch statistics
            elapsed = time.time() - start_time
            total_signals = self.metrics.messages_sent / 3
            avg_rate = total_signals / elapsed if elapsed > 0 else 0
            
            print(f"\n{'='*60}")
            print(f"BATCH STRESS TEST COMPLETE")
            print(f"{'='*60}")
            print(f"Duration:          {elapsed:.1f} seconds")
            print(f"Total Signals:     {int(total_signals):,}")
            print(f"Average Rate:      {avg_rate:.1f} signals/second")
            print(f"Target Rate:       {self.target_rate} signals/second")
            print(f"Achievement:       {(avg_rate/self.target_rate*100):.1f}%")
            print(f"Batch Size:        {self.batch_size}")
            print(f"{'='*60}")


# Import base class
from uos_publish_json_stress import StressTestPublisher, setup_argument_parser, validate_and_get_data_folders


def main():
    """Main function for batch publisher"""
    # Parse arguments
    parser = setup_argument_parser()
    parser.add_argument(
        "--batch-mode", 
        action="store_true",
        help="Use batch publishing mode"
    )
    args = parser.parse_args()
    
    # Force stress test mode
    args.stress_test = True
    
    # Configure logging
    setup_logging(getattr(logging, args.log_level.upper()))
    
    # Load configuration
    try:
        with open(args.conf) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file '{args.conf}' not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        sys.exit(1)
    
    # Validate path
    try:
        data_folders = validate_and_get_data_folders(args.path)
        logging.info(f"Found {len(data_folders)} data folder(s) to process")
    except ValueError as e:
        logging.error(e)
        sys.exit(1)
    
    # Run batch publisher
    publisher = BatchPublisher(config, args)
    publisher.run_batch_stress_test(data_folders, args.duration)


if __name__ == "__main__":
    main()