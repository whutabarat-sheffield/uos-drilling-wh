#!/usr/bin/env python3
"""
Asyncio-based MQTT JSON Data Publisher for Ultra-High Performance

This implementation uses asyncio and aiomqtt to achieve 1000+ signals/second
through asynchronous publishing and connection pooling.
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import deque
import random

try:
    import aiomqtt
except ImportError:
    print("Error: aiomqtt not installed. Install with: pip install aiomqtt")
    sys.exit(1)

import yaml
from abyss.uos_depth_est_utils import setup_logging


@dataclass
class AsyncPublishMetrics:
    """Track publishing performance metrics"""
    messages_sent: int = 0
    messages_failed: int = 0
    bytes_sent: int = 0
    start_time: float = field(default_factory=time.time)
    last_report_time: float = field(default_factory=time.time)
    latencies: deque = field(default_factory=lambda: deque(maxlen=1000))


class AsyncStressTestPublisher:
    """Asyncio-based high-performance MQTT publisher"""
    
    def __init__(self, config: Dict[str, Any], args: argparse.Namespace):
        self.config = config
        self.args = args
        self.metrics = AsyncPublishMetrics()
        self.running = True
        self.data_cache = {}
        self.semaphore = asyncio.Semaphore(args.max_concurrent_messages)
        
        # Performance settings
        self.target_rate = args.rate
        self.target_interval = 1.0 / self.target_rate if self.target_rate > 0 else 0
        
        # Connection pool
        self.clients = []
        self.client_index = 0
        self.client_lock = asyncio.Lock()
    
    async def create_client_pool(self, pool_size: int):
        """Create a pool of MQTT clients for load distribution"""
        host = self.config['mqtt']['broker']['host']
        port = self.config['mqtt']['broker']['port']
        
        for i in range(pool_size):
            client_id = f"async_stress_publisher_{i}"
            client = aiomqtt.Client(
                hostname=host,
                port=port,
                client_id=client_id,
                protocol=aiomqtt.ProtocolVersion.V5,  # Use MQTT v5 for better performance
                max_inflight_messages=200,  # Much higher than paho default
                max_queued_messages=5000,
            )
            self.clients.append(client)
            logging.debug(f"Created async MQTT client {i}")
        
        # Start all clients
        for client in self.clients:
            await client.__aenter__()
            logging.debug(f"Connected client {client._client_id}")
    
    async def get_next_client(self):
        """Get next client from pool (round-robin)"""
        async with self.client_lock:
            client = self.clients[self.client_index]
            self.client_index = (self.client_index + 1) % len(self.clients)
            return client
    
    def load_and_cache_data(self, data_folders: List[Path]) -> Dict[str, Dict[str, str]]:
        """Pre-load and cache all JSON data"""
        cache = {}
        
        for folder in data_folders:
            try:
                folder_data = {}
                
                # Read JSON files
                with open(folder / "ResultManagement.json") as f:
                    result_content = f.read()
                    result_json = json.loads(result_content)
                    folder_data['result'] = result_content
                    folder_data['result_timestamp'] = self._extract_timestamp(result_json)
                
                with open(folder / "Trace.json") as f:
                    folder_data['trace'] = f.read()
                
                with open(folder / "Heads.json") as f:
                    folder_data['heads'] = f.read()
                
                # Pre-compile topic templates
                mqtt_root = self.config['mqtt']['listener']['root']
                folder_data['topic_templates'] = {
                    'result': f"{mqtt_root}/{{toolbox_id}}/{{tool_id}}/{self.config['mqtt']['listener']['result']}",
                    'trace': f"{mqtt_root}/{{toolbox_id}}/{{tool_id}}/{self.config['mqtt']['listener']['trace']}",
                    'heads': f"{mqtt_root}/{{toolbox_id}}/{{tool_id}}/{self.config['mqtt']['listener']['heads']}"
                }
                
                cache[str(folder)] = folder_data
                logging.debug(f"Cached data from {folder}")
                
            except Exception as e:
                logging.error(f"Failed to cache data from {folder}: {e}")
        
        return cache
    
    def _extract_timestamp(self, data: dict) -> str:
        """Extract timestamp from JSON data"""
        def find_in_dict(d, target_key):
            if target_key in d:
                return d[target_key]
            for v in d.values():
                if isinstance(v, dict):
                    result = find_in_dict(v, target_key)
                    if result:
                        return result
            return None
        
        return find_in_dict(data, 'SourceTimestamp') or ""
    
    async def publish_signal_async(self, toolbox_id: str, tool_id: str, data: Dict[str, str]):
        """Publish a single signal asynchronously"""
        async with self.semaphore:  # Limit concurrent messages
            try:
                start_time = time.perf_counter()
                
                # Get client from pool
                client = await self.get_next_client()
                
                # Generate timestamp once
                new_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                
                # Build topics using pre-compiled templates
                topics = {
                    'result': data['topic_templates']['result'].format(toolbox_id=toolbox_id, tool_id=tool_id),
                    'trace': data['topic_templates']['trace'].format(toolbox_id=toolbox_id, tool_id=tool_id),
                    'heads': data['topic_templates']['heads'].format(toolbox_id=toolbox_id, tool_id=tool_id)
                }
                
                # Update payloads with new timestamp
                payloads = {
                    'result': data['result'].replace(data['result_timestamp'], new_timestamp),
                    'trace': data['trace'].replace(data['result_timestamp'], new_timestamp),
                    'heads': data['heads'].replace(data['result_timestamp'], new_timestamp)
                }
                
                # Publish all three messages concurrently
                publish_tasks = [
                    client.publish(topics['result'], payloads['result'], qos=0),
                    client.publish(topics['trace'], payloads['trace'], qos=0),
                    client.publish(topics['heads'], payloads['heads'], qos=0)
                ]
                
                await asyncio.gather(*publish_tasks)
                
                # Update metrics
                self.metrics.messages_sent += 3
                self.metrics.bytes_sent += sum(len(p) for p in payloads.values())
                
                # Track latency
                latency = time.perf_counter() - start_time
                self.metrics.latencies.append(latency)
                
            except Exception as e:
                logging.error(f"Error publishing signal: {e}")
                self.metrics.messages_failed += 3
    
    async def worker_coroutine(self, worker_id: int, data_folders: List[Path], 
                              toolbox_ids: List[str], tool_ids: List[str]):
        """Worker coroutine for publishing signals"""
        logging.debug(f"Worker {worker_id} started")
        
        # Pre-select data to reduce random selection overhead
        folder_keys = [str(f) for f in data_folders if str(f) in self.data_cache]
        if not folder_keys:
            logging.error(f"Worker {worker_id}: No valid data folders")
            return
        
        # Calculate rate per worker
        signals_per_worker = self.target_rate / self.args.concurrent_publishers
        interval = 1.0 / signals_per_worker if signals_per_worker > 0 else 0
        
        # Use asyncio sleep for precise timing
        next_publish_time = time.time()
        
        while self.running:
            try:
                # Select random data
                folder_key = random.choice(folder_keys)
                data = self.data_cache[folder_key]
                toolbox_id = random.choice(toolbox_ids)
                tool_id = random.choice(tool_ids)
                
                # Publish signal
                await self.publish_signal_async(toolbox_id, tool_id, data)
                
                # Rate limiting with precise timing
                if not self.args.no_sleep and interval > 0:
                    next_publish_time += interval
                    sleep_time = next_publish_time - time.time()
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                    else:
                        # We're behind schedule, reset
                        next_publish_time = time.time()
                
            except Exception as e:
                logging.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(0.001)  # Brief pause on error
    
    def print_metrics(self):
        """Print current performance metrics"""
        current_time = time.time()
        elapsed = current_time - self.metrics.start_time
        
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
            throughput_mbps = (mb_sent * 8) / elapsed
            
            print(f"\n{'='*60}")
            print(f"Async Stress Test Metrics - Elapsed: {elapsed:.1f}s")
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
    
    async def report_metrics_periodically(self, interval: int = 5):
        """Report metrics every N seconds"""
        while self.running:
            await asyncio.sleep(interval)
            self.print_metrics()
    
    async def run_stress_test_async(self, data_folders: List[Path], duration: int):
        """Run the async stress test"""
        logging.info(f"Starting async stress test: {self.target_rate} signals/sec for {duration}s")
        
        # Constants
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
        
        # Create client pool
        pool_size = min(self.args.concurrent_publishers, 10)  # Limit pool size
        logging.info(f"Creating connection pool with {pool_size} clients...")
        await self.create_client_pool(pool_size)
        
        # Start workers
        worker_tasks = []
        for i in range(self.args.concurrent_publishers):
            task = asyncio.create_task(
                self.worker_coroutine(i, data_folders, TOOLBOX_IDS, TOOL_IDS)
            )
            worker_tasks.append(task)
        
        # Start metrics reporter
        reporter_task = asyncio.create_task(self.report_metrics_periodically())
        
        # Run for specified duration
        try:
            await asyncio.sleep(duration)
        except asyncio.CancelledError:
            logging.info("Stress test cancelled")
        
        # Shutdown
        logging.info("Shutting down async stress test...")
        self.running = False
        
        # Cancel all tasks
        reporter_task.cancel()
        for task in worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(reporter_task, *worker_tasks, return_exceptions=True)
        
        # Close all clients
        for client in self.clients:
            await client.__aexit__(None, None, None)
        
        # Final report
        self.print_metrics()
        
        # Summary
        elapsed = time.time() - self.metrics.start_time
        total_signals = self.metrics.messages_sent / 3
        avg_rate = total_signals / elapsed if elapsed > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"ASYNC STRESS TEST COMPLETE")
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
        description="Async high-performance MQTT JSON publisher for stress testing"
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
        default=50, 
        help="Number of concurrent coroutines (default: 50)"
    )
    parser.add_argument(
        "--max-concurrent-messages", 
        type=int, 
        default=1000, 
        help="Max concurrent messages in flight (default: 1000)"
    )
    parser.add_argument(
        "--no-sleep", 
        action="store_true",
        help="Disable all sleep intervals for maximum throughput"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
        help="Set the logging level"
    )
    
    return parser


def validate_and_get_data_folders(path_str: str) -> List[Path]:
    """Validate the input path and return a list of data folders."""
    path = Path(path_str)
    
    if path.is_file():
        raise ValueError(f"Path '{path_str}' is a file, but a directory is required")
    
    if not path.is_dir():
        raise ValueError(f"Path '{path_str}' does not exist")
    
    # Check for subdirectories first
    subdirs = [p for p in path.iterdir() if p.is_dir()]
    
    if subdirs:
        data_folders = []
        for subdir in subdirs:
            json_files = [f for f in subdir.iterdir() if f.is_file() and f.suffix.lower() == '.json']
            if json_files:
                data_folders.append(subdir)
        
        if not data_folders:
            raise ValueError(f"No subdirectories in '{path_str}' contain JSON files")
        
        return data_folders
    else:
        json_files = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() == '.json']
        if json_files:
            return [path]
        else:
            raise ValueError(f"Directory '{path_str}' contains no JSON files")


async def main_async():
    """Async main function"""
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
    
    # Create and run publisher
    publisher = AsyncStressTestPublisher(config, args)
    
    # Handle signals
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: publisher.running == False)
    
    try:
        await publisher.run_stress_test_async(data_folders, args.duration)
    except KeyboardInterrupt:
        logging.info("Interrupted by user")


def main():
    """Main entry point"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()