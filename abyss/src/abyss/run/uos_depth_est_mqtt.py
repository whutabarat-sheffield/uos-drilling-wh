#!/usr/bin/env python3
"""
MQTT Drilling Data Analyser - Refactored Components Version

This is the main entry point for the refactored drilling data analyzer
using the new simplified orchestrator that replaces the original 
monolithic MQTTDrillingDataAnalyser class.
"""

import logging
import sys
import os
import yaml
import argparse

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from abyss.mqtt.components import DrillingDataAnalyser
from abyss.uos_depth_est_utils import setup_logging


def main():
    """
    Main entry point for the refactored drilling data analyzer.
    
    Command line arguments:
        --config: Path to YAML configuration file (default: mqtt_conf_local.yaml)
        --log-level: Logging level (default: INFO)
    
    Example usage:
        python uos_depth_est_mqtt.py
        python uos_depth_est_mqtt.py --config=mqtt_conf_local.yaml --log-level=DEBUG
        python uos_depth_est_mqtt.py --config=mqtt_conf_docker.yaml --log-level=INFO
    """
    parser = argparse.ArgumentParser(description='MQTT Drilling Data Analyzer - Refactored Components')
    parser.add_argument(
        '--config', 
        type=str,
        default='mqtt_conf_local.yaml',
        help='Path to YAML configuration file (default: mqtt_conf_local.yaml)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    setup_logging(getattr(logging, args.log_level))
    
    # Resolve config path - check multiple locations
    config_path = args.config
    if not os.path.isabs(config_path):
        # Try relative to script location first
        script_dir = os.path.dirname(__file__)
        possible_paths = [
            os.path.join(script_dir, '..', 'run', 'config', config_path),
            os.path.join(script_dir, config_path),
            config_path
        ]
        
        config_found = False
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                config_found = True
                break
        
        if not config_found:
            logging.critical("Configuration file '%s' not found in any of these locations: %s", 
                           args.config, possible_paths)
            sys.exit(1)
    
    try:
        logging.info("Starting MQTT Drilling Data Analyzer with refactored components")
        logging.info("Configuration file: %s", config_path)
        
        analyzer = DrillingDataAnalyser(config_path=config_path)
        analyzer.run()
        
    except FileNotFoundError:
        logging.critical("Configuration file '%s' not found in %s", config_path, os.getcwd())
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.critical("Invalid YAML configuration in '%s': %s", config_path, str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt, shutting down...")
        sys.exit(0)
    except Exception as e:
        logging.critical("Error: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()