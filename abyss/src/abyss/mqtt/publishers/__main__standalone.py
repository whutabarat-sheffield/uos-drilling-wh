"""
Standalone MQTT Publisher Module Entry Point

This version has no dependencies on the main abyss package.
It can run independently with just paho-mqtt and pyyaml installed.

Usage:
    python -m abyss.mqtt.publishers [options] path
    
Examples:
    # Standard mode
    python -m abyss.mqtt.publishers test_data
    
    # With custom config
    python -m abyss.mqtt.publishers test_data -c config/mqtt_conf.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

from .base import PublisherConfig
from .standard import StandardPublisher


def setup_logging(level=logging.INFO):
    """Simple logging setup without external dependencies."""
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)-8s %(name)s | %(filename)s:%(lineno)d:%(funcName)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up and return the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Lightweight MQTT Publisher for drilling data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard publishing
  %(prog)s test_data
  
  # With custom configuration
  %(prog)s test_data -c /path/to/config.yaml
  
  # With different intervals
  %(prog)s test_data --sleep-min 0.5 --sleep-max 1.0
        """
    )
    
    # Positional arguments
    parser.add_argument(
        "path",
        type=str,
        help="Path to the test data folder containing JSON files"
    )
    
    # Configuration
    parser.add_argument(
        "-c", "--conf",
        type=str,
        help="YAML configuration file",
        default="config/mqtt_conf_docker.yaml"
    )
    
    # Publishing options
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
        help="Number of repetitions (0 for infinite)"
    )
    
    # Signal tracking
    parser.add_argument(
        "--track-signals",
        action="store_true",
        help="Enable signal tracking with unique IDs"
    )
    parser.add_argument(
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
    
    # Validation
    parser.add_argument(
        "--validate-data",
        action="store_true",
        help="Validate JSON files on startup"
    )
    
    return parser


def main():
    """Main entry point for the standalone MQTT publisher."""
    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(getattr(logging, args.log_level.upper()))
    
    # Log startup
    logging.info("Starting lightweight MQTT publisher")
    
    # Validate path
    test_data_path = Path(args.path)
    if not test_data_path.exists():
        logging.error(f"Test data path does not exist: {test_data_path}")
        sys.exit(1)
    
    try:
        # Create config
        config = PublisherConfig.from_yaml(args.conf)
        config.test_data_path = test_data_path
        config.sleep_min = args.sleep_min
        config.sleep_max = args.sleep_max
        config.repetitions = args.repetitions
        config.track_signals = args.track_signals
        config.signal_log = args.signal_log
        
        # Log configuration
        logging.info(f"Configuration loaded from: {args.conf}")
        logging.info(f"Publishing from: {test_data_path}")
        logging.info(f"Target broker: {config.broker_host}:{config.broker_port}")
        logging.info(f"Sleep interval: {config.sleep_min}s - {config.sleep_max}s")
        
        # Create and run publisher
        publisher = StandardPublisher(config, use_patterns=False, validate_data=args.validate_data)
        publisher.run()
        
    except KeyboardInterrupt:
        logging.info("Publisher interrupted by user")
    except Exception as e:
        logging.error(f"Publisher failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()