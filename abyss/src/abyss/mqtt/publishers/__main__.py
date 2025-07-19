"""
MQTT Publisher Module Entry Point

Usage:
    python -m abyss.mqtt.publishers [options] path
    
Examples:
    # Standard mode
    python -m abyss.mqtt.publishers test_data
    
    # Stress test mode
    python -m abyss.mqtt.publishers test_data --stress-test --rate 1000 --duration 60
    
    # With custom config
    python -m abyss.mqtt.publishers test_data -c config/mqtt_conf.yaml
    
    # With signal tracking
    python -m abyss.mqtt.publishers test_data --track-signals
"""

import argparse
import logging
import sys
from pathlib import Path

from abyss.uos_depth_est_utils import setup_logging

from .base import PublisherConfig
from .standard import StandardPublisher
from .stress import StressTestConfig, StressTestPublisher


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up and return the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="MQTT Publisher for drilling data with multiple operation modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard publishing with patterns
  %(prog)s test_data
  
  # Stress test at 1000 signals/sec for 60 seconds
  %(prog)s test_data --stress-test --rate 1000 --duration 60
  
  # Standard mode without patterns
  %(prog)s test_data --no-patterns
  
  # With signal tracking
  %(prog)s test_data --track-signals --signal-log tracking.csv
        """
    )
    
    # Positional arguments
    parser.add_argument(
        "path",
        type=str,
        help="Path to the test data folder containing JSON files"
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--stress-test",
        action="store_true",
        help="Run in high-performance stress test mode"
    )
    mode_group.add_argument(
        "--async-mode",
        action="store_true",
        help="Run in ultra-high-performance async mode (requires aiomqtt)"
    )
    
    # Configuration
    parser.add_argument(
        "-c", "--conf",
        type=str,
        help="YAML configuration file",
        default="config/mqtt_conf_docker.yaml"
    )
    
    # Standard mode options
    standard_group = parser.add_argument_group("standard mode options")
    standard_group.add_argument(
        "--sleep-min",
        type=float,
        default=0.1,
        help="Minimum sleep interval between publishes (seconds)"
    )
    standard_group.add_argument(
        "--sleep-max",
        type=float,
        default=0.3,
        help="Maximum sleep interval between publishes (seconds)"
    )
    standard_group.add_argument(
        "-r", "--repetitions",
        type=int,
        default=10,
        help="Number of repetitions (0 for infinite)"
    )
    standard_group.add_argument(
        "--no-patterns",
        action="store_true",
        help="Disable realistic drilling patterns"
    )
    
    # Stress test options
    stress_group = parser.add_argument_group("stress test options")
    stress_group.add_argument(
        "--rate",
        type=int,
        default=1000,
        help="Target rate in signals per second"
    )
    stress_group.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Test duration in seconds"
    )
    stress_group.add_argument(
        "--threads",
        type=int,
        default=10,
        help="Number of publisher threads"
    )
    stress_group.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Messages per batch"
    )
    
    # Signal tracking
    tracking_group = parser.add_argument_group("signal tracking")
    tracking_group.add_argument(
        "--track-signals",
        action="store_true",
        help="Enable signal tracking with unique IDs"
    )
    tracking_group.add_argument(
        "--signal-log",
        type=str,
        default="sent_signals.csv",
        help="CSV file for signal tracking log"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level"
    )
    
    return parser


def main():
    """Main entry point for the MQTT publisher module."""
    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(getattr(logging, args.log_level.upper()))
    
    # Validate path
    test_data_path = Path(args.path)
    if not test_data_path.exists():
        logging.error(f"Test data path does not exist: {test_data_path}")
        sys.exit(1)
    
    # Handle async mode
    if args.async_mode:
        try:
            from .async_publisher import AsyncPublisher, AsyncConfig
            
            # Create async config
            config = AsyncConfig.from_yaml(args.conf)
            config.test_data_path = test_data_path
            config.track_signals = args.track_signals
            config.signal_log = args.signal_log
            config.target_rate = args.rate
            config.duration = args.duration
            
            # Run async publisher
            publisher = AsyncPublisher(config)
            
            # Async requires special handling
            import asyncio
            asyncio.run(publisher.run())
            
        except ImportError as e:
            logging.error(
                "Cannot run async mode: aiomqtt not installed.\n"
                "Install with: pip install aiomqtt"
            )
            sys.exit(1)
    
    # Handle stress test mode
    elif args.stress_test:
        # Create stress test config
        config = StressTestConfig.from_yaml(args.conf)
        config.test_data_path = test_data_path
        config.track_signals = args.track_signals
        config.signal_log = args.signal_log
        config.target_rate = args.rate
        config.duration = args.duration
        config.num_threads = args.threads
        config.batch_size = args.batch_size
        
        # Run stress test
        publisher = StressTestPublisher(config)
        publisher.run()
    
    # Standard mode
    else:
        # Create standard config
        config = PublisherConfig.from_yaml(args.conf)
        config.test_data_path = test_data_path
        config.sleep_min = args.sleep_min
        config.sleep_max = args.sleep_max
        config.repetitions = args.repetitions
        config.track_signals = args.track_signals
        config.signal_log = args.signal_log
        
        # Run standard publisher
        use_patterns = not args.no_patterns
        publisher = StandardPublisher(config, use_patterns=use_patterns)
        publisher.run()


if __name__ == "__main__":
    main()