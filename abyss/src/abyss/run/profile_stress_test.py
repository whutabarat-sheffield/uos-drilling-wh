#!/usr/bin/env python3
"""
Profile the stress test publisher to identify performance bottlenecks
"""

import cProfile
import pstats
import io
import argparse
import yaml
import sys
from pathlib import Path
from contextlib import contextmanager
import time
import threading

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from uos_publish_json_stress import StressTestPublisher, validate_and_get_data_folders
from uos_depth_est_utils import setup_logging
import logging


class PerformanceAnalyzer:
    """Analyze performance bottlenecks in the stress test"""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.timings = {
            'mqtt_publish': [],
            'data_preparation': [],
            'json_parsing': [],
            'timestamp_update': [],
            'topic_building': [],
            'total_publish': []
        }
    
    @contextmanager
    def timer(self, operation: str):
        """Context manager to time operations"""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        with self.lock:
            if operation in self.timings:
                self.timings[operation].append(elapsed)
    
    def get_stats(self):
        """Get performance statistics"""
        stats = {}
        with self.lock:
            for op, times in self.timings.items():
                if times:
                    stats[op] = {
                        'count': len(times),
                        'total': sum(times),
                        'avg': sum(times) / len(times),
                        'min': min(times),
                        'max': max(times)
                    }
        return stats
    
    def print_report(self):
        """Print performance report"""
        stats = self.get_stats()
        print("\n" + "="*60)
        print("Performance Analysis Report")
        print("="*60)
        
        for op, data in sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True):
            print(f"\n{op}:")
            print(f"  Total time: {data['total']:.3f}s")
            print(f"  Calls: {data['count']}")
            print(f"  Avg: {data['avg']*1000:.2f}ms")
            print(f"  Min: {data['min']*1000:.2f}ms")
            print(f"  Max: {data['max']*1000:.2f}ms")


class InstrumentedStressTestPublisher(StressTestPublisher):
    """Instrumented version of stress test publisher for profiling"""
    
    def __init__(self, config, args):
        super().__init__(config, args)
        self.analyzer = PerformanceAnalyzer()
    
    def publish_signal_batch(self, client, signals):
        """Instrumented version with timing"""
        messages_published = 0
        
        with self.analyzer.timer('total_publish'):
            for toolbox_id, tool_id, data in signals:
                try:
                    # Build topics
                    with self.analyzer.timer('topic_building'):
                        mqtt_root = self.config['mqtt']['listener']['root']
                        topic_result = f"{mqtt_root}/{toolbox_id}/{tool_id}/{self.config['mqtt']['listener']['result']}"
                        topic_trace = f"{mqtt_root}/{toolbox_id}/{tool_id}/{self.config['mqtt']['listener']['trace']}"
                        topic_heads = f"{mqtt_root}/{toolbox_id}/{tool_id}/{self.config['mqtt']['listener']['heads']}"
                    
                    # Generate new timestamp
                    with self.analyzer.timer('timestamp_update'):
                        new_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.localtime())
                        result_payload = data['result'].replace(data['result_timestamp'], new_timestamp)
                        trace_payload = data['trace'].replace(data['result_timestamp'], new_timestamp)
                        heads_payload = data['heads'].replace(data['result_timestamp'], new_timestamp)
                    
                    # Publish messages
                    qos = 0 if self.args.stress_test else 1
                    
                    with self.analyzer.timer('mqtt_publish'):
                        info = client.publish(topic_result, result_payload, qos=qos)
                        if info.rc != mqtt.MQTT_ERR_SUCCESS:
                            with self.lock:
                                self.metrics.messages_failed += 1
                        else:
                            with self.lock:
                                self.metrics.bytes_sent += len(result_payload)
                    
                    with self.analyzer.timer('mqtt_publish'):
                        info = client.publish(topic_trace, trace_payload, qos=qos)
                        if info.rc != mqtt.MQTT_ERR_SUCCESS:
                            with self.lock:
                                self.metrics.messages_failed += 1
                        else:
                            with self.lock:
                                self.metrics.bytes_sent += len(trace_payload)
                    
                    with self.analyzer.timer('mqtt_publish'):
                        info = client.publish(topic_heads, heads_payload, qos=qos)
                        if info.rc != mqtt.MQTT_ERR_SUCCESS:
                            with self.lock:
                                self.metrics.messages_failed += 1
                        else:
                            with self.lock:
                                self.metrics.bytes_sent += len(heads_payload)
                    
                    # Track latency
                    latency = time.time() - time.time()
                    with self.lock:
                        self.metrics.latencies.append(latency)
                    
                    messages_published += 3
                    
                except Exception as e:
                    logging.error(f"Error publishing signal: {e}")
                    with self.lock:
                        self.metrics.messages_failed += 3
        
        return messages_published
    
    def load_and_cache_data(self, data_folders):
        """Instrumented data loading"""
        cache = {}
        
        for folder in data_folders:
            try:
                folder_data = {}
                
                with self.analyzer.timer('json_parsing'):
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


def profile_stress_test(args):
    """Run profiled stress test"""
    # Load configuration
    with open(args.conf) as f:
        config = yaml.safe_load(f)
    
    # Validate path and get data folders
    data_folders = validate_and_get_data_folders(args.path)
    logging.info(f"Found {len(data_folders)} data folder(s) to process")
    
    # Create instrumented publisher
    publisher = InstrumentedStressTestPublisher(config, args)
    
    # Run with cProfile
    profiler = cProfile.Profile()
    
    logging.info("Starting profiled stress test...")
    profiler.enable()
    
    # Run for shorter duration for profiling
    publisher.run_stress_test(data_folders, min(args.duration, 10))
    
    profiler.disable()
    
    # Print profiling results
    print("\n" + "="*60)
    print("cProfile Results")
    print("="*60)
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions
    print(s.getvalue())
    
    # Print custom timing analysis
    publisher.analyzer.print_report()
    
    # Analyze thread contention
    print("\n" + "="*60)
    print("Thread Contention Analysis")
    print("="*60)
    
    # Check lock acquisition patterns
    ps = pstats.Stats(profiler)
    ps.sort_stats('time')
    
    # Look for lock-related functions
    lock_functions = []
    for func in ps.stats:
        if 'lock' in str(func).lower() or 'acquire' in str(func).lower():
            lock_functions.append((func, ps.stats[func]))
    
    if lock_functions:
        print("\nLock-related functions:")
        for func, stats in sorted(lock_functions, key=lambda x: x[1][2], reverse=True)[:10]:
            print(f"  {func}: {stats[2]:.3f}s total time")
    
    # Calculate theoretical vs actual performance
    print("\n" + "="*60)
    print("Performance Analysis")
    print("="*60)
    
    total_messages = publisher.metrics.messages_sent
    total_time = time.time() - publisher.metrics.start_time
    actual_rate = total_messages / total_time if total_time > 0 else 0
    signals_per_second = actual_rate / 3
    
    print(f"Actual signals/second: {signals_per_second:.1f}")
    print(f"Target signals/second: {args.rate}")
    print(f"Achievement: {(signals_per_second/args.rate*100):.1f}%")
    
    # Analyze bottlenecks
    analyzer_stats = publisher.analyzer.get_stats()
    if 'mqtt_publish' in analyzer_stats:
        mqtt_stats = analyzer_stats['mqtt_publish']
        mqtt_throughput = 1.0 / mqtt_stats['avg'] if mqtt_stats['avg'] > 0 else 0
        print(f"\nMQTT publish throughput: {mqtt_throughput:.1f} messages/second per thread")
        print(f"With {args.concurrent_publishers} threads: {mqtt_throughput * args.concurrent_publishers:.1f} messages/second")
        print(f"Theoretical signals/second: {(mqtt_throughput * args.concurrent_publishers) / 3:.1f}")


def main():
    """Main function for profiling"""
    parser = argparse.ArgumentParser(description="Profile MQTT stress test performance")
    
    parser.add_argument("path", type=str, help="Path to test data")
    parser.add_argument("-c", "--conf", type=str, default="config/mqtt_conf_docker.yaml")
    parser.add_argument("--rate", type=int, default=1000)
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--concurrent-publishers", type=int, default=10)
    parser.add_argument("--no-sleep", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    
    args = parser.parse_args()
    args.stress_test = True  # Always in stress test mode
    
    # Configure logging
    setup_logging(getattr(logging, args.log_level.upper()))
    
    # Add missing imports
    import json
    import paho.mqtt.client as mqtt
    
    # Make them available globally
    globals()['json'] = json
    globals()['mqtt'] = mqtt
    
    profile_stress_test(args)


if __name__ == "__main__":
    main()