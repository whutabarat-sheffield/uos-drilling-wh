"""
MQTT JSON Data Publisher - Stress Testing Version

A high-performance MQTT publisher for stress testing that can publish JSON data 
at rates up to 1000 signals per second using concurrent publishing.
"""

import argparse
import asyncio
import json
import logging
import random
import signal
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import deque, defaultdict

import paho.mqtt.client as mqtt
import yaml

from abyss.uos_depth_est_utils import setup_logging


@dataclass
class PublishMetrics:
    """Track publishing performance metrics"""
    messages_sent: int = 0
    messages_failed: int = 0
    bytes_sent: int = 0
    start_time: float = 0
    last_report_time: float = 0
    latencies: deque = None
    
    def __post_init__(self):
        if self.latencies is None:
            self.latencies = deque(maxlen=1000)  # Keep last 1000 latencies
        self.start_time = time.time()
        self.last_report_time = self.start_time


class StressTestPublisher:
    """High-performance MQTT publisher for stress testing"""
    
    def __init__(self, config: Dict[str, Any], args: argparse.Namespace):
        self.config = config
        self.args = args
        self.metrics = PublishMetrics()
        self.lock = threading.Lock()
        self.clients = []
        self.executor = None
        self.running = True
        self.data_cache = {}  # Cache parsed JSON data
        
        # Performance settings
        self.batch_size = args.batch_size
        self.target_rate = args.rate
        self.target_interval = 1.0 / self.target_rate if self.target_rate > 0 else 0
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals gracefully"""
        logging.info("\nShutdown signal received...")
        self.running = False
    
    def setup_mqtt_clients(self, num_clients: int) -> List[mqtt.Client]:
        """Create and configure multiple MQTT clients for concurrent publishing"""
        clients = []
        
        for i in range(num_clients):
            # Handle different paho-mqtt versions
            try:
                # Try new API (paho-mqtt >= 2.0)
                client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, f"stress_test_publisher_{i}")
            except (AttributeError, TypeError):
                # Fall back to old API (paho-mqtt < 2.0)
                client = mqtt.Client(f"stress_test_publisher_{i}")
            
            # Optimize client settings for high throughput
            client.max_inflight_messages_set(100)  # Increase from default 20
            client.max_queued_messages_set(1000)   # Increase queue size
            
            # Set callbacks
            client.on_connect = self._on_connect
            client.on_disconnect = self._on_disconnect
            client.on_publish = self._on_publish
            
            # Connect to broker
            try:
                client.connect(
                    self.config['mqtt']['broker']['host'], 
                    self.config['mqtt']['broker']['port']
                )
                client.loop_start()
                clients.append(client)
                logging.debug(f"Created MQTT client {i}")
            except Exception as e:
                logging.error(f"Failed to create client {i}: {e}")
        
        return clients
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback for MQTT connection"""
        if rc == 0:
            logging.debug(f"Client {client._client_id} connected successfully")
        else:
            logging.error(f"Client {client._client_id} connection failed with code {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback for MQTT disconnection"""
        if rc != 0:
            logging.warning(f"Client {client._client_id} disconnected unexpectedly")
    
    def _on_publish(self, client, userdata, mid):
        """Callback for successful publish"""
        # Update metrics
        with self.lock:
            self.metrics.messages_sent += 1
    
    def load_and_cache_data(self, data_folders: List[Path]) -> Dict[str, Dict[str, str]]:
        """Pre-load and cache all JSON data to reduce I/O during stress test"""
        cache = {}
        
        for folder in data_folders:
            try:
                folder_data = {}
                
                # Read and parse JSON files
                with open(folder / "ResultManagement.json") as f:
                    result_content = f.read()
                    result_json = json.loads(result_content)
                    folder_data['result'] = result_content
                    folder_data['result_timestamp'] = self._extract_timestamp(result_json)
                
                with open(folder / "Trace.json") as f:
                    folder_data['trace'] = f.read()
                
                with open(folder / "Heads.json") as f:
                    folder_data['heads'] = f.read()
                
                cache[str(folder)] = folder_data
                logging.debug(f"Cached data from {folder}")
                
            except Exception as e:
                logging.error(f"Failed to cache data from {folder}: {e}")
        
        return cache
    
    def _extract_timestamp(self, data: dict) -> str:
        """Extract timestamp from JSON data"""
        timestamps = self._find_in_dict(data, 'SourceTimestamp')
        return timestamps[0] if timestamps else ""
    
    def _find_in_dict(self, data: dict, target_key: str) -> list:
        """Recursively search for a key in a nested dictionary"""
        results = []
        
        def _search(current_dict):
            for key, value in current_dict.items():
                if key == target_key:
                    results.append(value)
                if isinstance(value, dict):
                    _search(value)
        
        _search(data)
        return results
    
    def publish_signal_batch(self, client: mqtt.Client, signals: List[Tuple[str, Dict[str, str]]]) -> int:
        """Publish a batch of signals (each signal = 3 messages)"""
        messages_published = 0
        
        for toolbox_id, tool_id, data in signals:
            try:
                # Build topics
                mqtt_root = self.config['mqtt']['listener']['root']
                topic_result = f"{mqtt_root}/{toolbox_id}/{tool_id}/{self.config['mqtt']['listener']['result']}"
                topic_trace = f"{mqtt_root}/{toolbox_id}/{tool_id}/{self.config['mqtt']['listener']['trace']}"
                topic_heads = f"{mqtt_root}/{toolbox_id}/{tool_id}/{self.config['mqtt']['listener']['heads']}"
                
                # Generate new timestamp
                new_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.localtime())
                
                # Publish messages (no sleep in stress mode)
                start_time = time.time()
                
                # Update timestamps and publish
                result_payload = data['result'].replace(data['result_timestamp'], new_timestamp)
                trace_payload = data['trace'].replace(data['result_timestamp'], new_timestamp)
                heads_payload = data['heads'].replace(data['result_timestamp'], new_timestamp)
                
                # Use QoS 0 for maximum throughput
                qos = 0 if self.args.stress_test else 1
                
                info = client.publish(topic_result, result_payload, qos=qos)
                if info.rc != mqtt.MQTT_ERR_SUCCESS:
                    with self.lock:
                        self.metrics.messages_failed += 1
                else:
                    with self.lock:
                        self.metrics.bytes_sent += len(result_payload)
                
                info = client.publish(topic_trace, trace_payload, qos=qos)
                if info.rc != mqtt.MQTT_ERR_SUCCESS:
                    with self.lock:
                        self.metrics.messages_failed += 1
                else:
                    with self.lock:
                        self.metrics.bytes_sent += len(trace_payload)
                
                info = client.publish(topic_heads, heads_payload, qos=qos)
                if info.rc != mqtt.MQTT_ERR_SUCCESS:
                    with self.lock:
                        self.metrics.messages_failed += 1
                else:
                    with self.lock:
                        self.metrics.bytes_sent += len(heads_payload)
                
                # Track latency
                latency = time.time() - start_time
                with self.lock:
                    self.metrics.latencies.append(latency)
                
                messages_published += 3
                
            except Exception as e:
                logging.error(f"Error publishing signal: {e}")
                with self.lock:
                    self.metrics.messages_failed += 3
        
        return messages_published
    
    def worker_thread(self, client_id: int, client: mqtt.Client, data_folders: List[Path], 
                     toolbox_ids: List[str], tool_ids: List[str]):
        """Worker thread for concurrent publishing"""
        logging.debug(f"Worker {client_id} started")
        
        # Calculate signals per worker
        signals_per_second_per_worker = self.target_rate / self.args.concurrent_publishers
        interval_per_signal = 1.0 / signals_per_second_per_worker if signals_per_second_per_worker > 0 else 0
        
        last_publish_time = time.time()
        
        while self.running:
            try:
                # Select random data
                folder_key = str(random.choice(data_folders))
                data = self.data_cache[folder_key]
                toolbox_id = random.choice(toolbox_ids)
                tool_id = random.choice(tool_ids)
                
                # Publish signal
                self.publish_signal_batch(client, [(toolbox_id, tool_id, data)])
                
                # Rate limiting (only if not in full stress mode)
                if not self.args.no_sleep and interval_per_signal > 0:
                    elapsed = time.time() - last_publish_time
                    sleep_time = interval_per_signal - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                last_publish_time = time.time()
                
            except Exception as e:
                logging.error(f"Worker {client_id} error: {e}")
                time.sleep(0.1)  # Brief pause on error
    
    def print_metrics(self):
        """Print current performance metrics"""
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.metrics.start_time
            interval = current_time - self.metrics.last_report_time
            self.metrics.last_report_time = current_time
            
            if elapsed > 0:
                # Calculate rates
                overall_rate = self.metrics.messages_sent / elapsed
                signals_per_second = overall_rate / 3  # 3 messages per signal
                
                # Calculate latency stats
                latencies = list(self.metrics.latencies)
                avg_latency = sum(latencies) / len(latencies) if latencies else 0
                max_latency = max(latencies) if latencies else 0
                min_latency = min(latencies) if latencies else 0
                
                # Calculate throughput
                mb_sent = self.metrics.bytes_sent / (1024 * 1024)
                throughput_mbps = (mb_sent * 8) / elapsed  # Megabits per second
                
                print(f"\n{'='*60}")
                print(f"Stress Test Metrics - Elapsed: {elapsed:.1f}s")
                print(f"{'='*60}")
                print(f"Messages Sent:     {self.metrics.messages_sent:,}")
                print(f"Messages Failed:   {self.metrics.messages_failed:,}")
                print(f"Success Rate:      {(self.metrics.messages_sent / (self.metrics.messages_sent + self.metrics.messages_failed) * 100):.1f}%")
                print(f"Signals/Second:    {signals_per_second:.1f} (target: {self.target_rate})")
                print(f"Messages/Second:   {overall_rate:.1f}")
                print(f"Throughput:        {throughput_mbps:.2f} Mbps")
                print(f"Avg Latency:       {avg_latency*1000:.2f} ms")
                print(f"Min/Max Latency:   {min_latency*1000:.2f} / {max_latency*1000:.2f} ms")
                print(f"{'='*60}")
    
    def run_stress_test(self, data_folders: List[Path], duration: int):
        """Run the stress test"""
        logging.info(f"Starting stress test: {self.target_rate} signals/sec for {duration}s")
        
        # Constants for simulation
        TOOLBOX_IDS = [
            'ILLL502033771', 'ILLL502033772', 'ILLL502033773', 
            'ILLL502033774', 'ILLL502033775'
        ]
        TOOL_IDS = [
            'setitec001', 'setitec002', 'setitec003', 
            'setitec004', 'setitec005'
        ]
        
        # Pre-load all data
        logging.info("Pre-loading data into cache...")
        self.data_cache = self.load_and_cache_data(data_folders)
        if not self.data_cache:
            logging.error("No data loaded, exiting")
            return
        
        # Create MQTT clients
        logging.info(f"Creating {self.args.concurrent_publishers} MQTT clients...")
        self.clients = self.setup_mqtt_clients(self.args.concurrent_publishers)
        if not self.clients:
            logging.error("No MQTT clients created, exiting")
            return
        
        # Wait for connections
        time.sleep(1)
        
        # Create thread pool
        self.executor = ThreadPoolExecutor(max_workers=self.args.concurrent_publishers)
        
        # Start worker threads
        futures = []
        for i, client in enumerate(self.clients):
            future = self.executor.submit(
                self.worker_thread, i, client, data_folders, TOOLBOX_IDS, TOOL_IDS
            )
            futures.append(future)
        
        # Monitor progress
        start_time = time.time()
        report_interval = 5  # Report every 5 seconds
        last_report = start_time
        
        try:
            while self.running and (time.time() - start_time) < duration:
                current_time = time.time()
                
                if current_time - last_report >= report_interval:
                    self.print_metrics()
                    last_report = current_time
                
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            logging.info("Stress test interrupted by user")
        
        finally:
            # Shutdown
            logging.info("Shutting down stress test...")
            self.running = False
            
            # Wait for workers to finish
            if self.executor:
                self.executor.shutdown(wait=True)
            
            # Disconnect clients
            for client in self.clients:
                client.loop_stop()
                client.disconnect()
            
            # Final report
            self.print_metrics()
            
            # Summary
            elapsed = time.time() - start_time
            total_signals = self.metrics.messages_sent / 3
            avg_rate = total_signals / elapsed if elapsed > 0 else 0
            
            print(f"\n{'='*60}")
            print(f"STRESS TEST COMPLETE")
            print(f"{'='*60}")
            print(f"Duration:          {elapsed:.1f} seconds")
            print(f"Total Signals:     {int(total_signals):,}")
            print(f"Average Rate:      {avg_rate:.1f} signals/second")
            print(f"Target Rate:       {self.target_rate} signals/second")
            print(f"Achievement:       {(avg_rate/self.target_rate*100):.1f}%")
            print(f"{'='*60}")


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up and return the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="High-performance MQTT JSON publisher for stress testing"
    )
    
    parser.add_argument(
        "path", 
        type=str, 
        help="Path to the data folder containing JSON files"
    )
    parser.add_argument(
        "-c", "--conf", 
        type=str, 
        help="YAML configuration file", 
        default="config/mqtt_conf_docker.yaml"
    )
    
    # Stress testing options
    parser.add_argument(
        "--stress-test", 
        action="store_true",
        help="Enable stress testing mode"
    )
    parser.add_argument(
        "--rate", 
        type=int, 
        default=1000, 
        help="Target signals per second (default: 1000)"
    )
    parser.add_argument(
        "--duration", 
        type=int, 
        default=60, 
        help="Test duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--concurrent-publishers", 
        type=int, 
        default=10, 
        help="Number of concurrent publisher threads (default: 10)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1, 
        help="Number of signals to batch before publishing (default: 1)"
    )
    parser.add_argument(
        "--no-sleep", 
        action="store_true",
        help="Disable all sleep intervals for maximum throughput"
    )
    
    # Original options (for non-stress mode)
    parser.add_argument(
        "--sleep-min", 
        type=float, 
        default=0.1, 
        help="Minimum sleep interval between publishes (seconds)"
    )
    parser.add_argument(
        "--sleep-max", 
        type=float, 
        default=0.3, 
        help="Maximum sleep interval between publishes (seconds)"
    )
    parser.add_argument(
        "-r", "--repetitions", 
        type=int, 
        default=10, 
        help="Number of repetitions for publishing data (non-stress mode)"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
        help="Set the logging level"
    )
    
    return parser


def validate_and_get_data_folders(path_str: str) -> list[Path]:
    """Validate the input path and return a list of data folders containing JSON files."""
    path = Path(path_str)
    
    if path.is_file():
        raise ValueError(f"Path '{path_str}' is a file, but a directory is required")
    
    if not path.is_dir():
        raise ValueError(f"Path '{path_str}' does not exist")
    
    # Check for subdirectories first
    subdirs = [p for p in path.iterdir() if p.is_dir()]
    
    if subdirs:
        # Filter subdirectories that contain JSON files
        data_folders = []
        for subdir in subdirs:
            json_files = [f for f in subdir.iterdir() if f.is_file() and f.suffix.lower() == '.json']
            if json_files:
                data_folders.append(subdir)
        
        if not data_folders:
            raise ValueError(f"No subdirectories in '{path_str}' contain JSON files")
        
        return data_folders
    else:
        # No subdirectories, check if the directory itself contains JSON files
        json_files = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() == '.json']
        if json_files:
            return [path]
        else:
            raise ValueError(f"Directory '{path_str}' contains no JSON files")


def main():
    """Main function to run either stress test or normal publishing mode."""
    # Parse command line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
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
    
    # Validate path and get data folders
    try:
        data_folders = validate_and_get_data_folders(args.path)
        logging.info(f"Found {len(data_folders)} data folder(s) to process")
    except ValueError as e:
        logging.error(e)
        sys.exit(1)
    
    # Run stress test or normal mode
    if args.stress_test:
        publisher = StressTestPublisher(config, args)
        publisher.run_stress_test(data_folders, args.duration)
    else:
        # Import and run original publisher for backward compatibility
        from uos_publish_json import main as original_main
        original_main()


if __name__ == "__main__":
    main()