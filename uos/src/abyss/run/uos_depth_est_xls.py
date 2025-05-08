### XLS Drilling Data Analyzer


import logging
import sys
import os
import yaml
import argparse


from abyss.uos_inference import (
    DepthInference
)

from abyss.uos_depth_est_utils import (
    setup_logging,
)


def main():
    """
    Main entry point for the application.
    
    Command line arguments:
        --config: Path to YAML configuration file (default: mqtt_conf.yaml)
        --log-level: Logging level (default: INFO)
    
    Example usage:
        python -m uos_depth_est file_path
        python -m uos_depth_est file_path --log-level=DEBUG
    """
    parser = argparse.ArgumentParser(description='XLS Drilling Data Analyzer')
    parser.add_argument("file_path",
        # type=str,
        # default='mqtt_conf.yaml',
        help='Path to XLS Setitec file (default: none)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the logging level (default: INFO)'
    )
    parser.add_argument(
        '--config',
        type=str,
        # choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='mqtt_conf.yaml',
        help='Set the configuration file (default: ./mqtt_conf.yaml)'
    )
    
    args = parser.parse_args()
    
    setup_logging(getattr(logging, args.log_level))
    
    ## -----------------------------------------------------------------------------
    ##
    ## Run the depth estimator here
    ##
    ## -----------------------------------------------------------------------------
    try:
        analyzer = DepthInference()
        kp = analyzer.infer3_xls(args.file_path)
        # print(f"Key points [1] {kp[0]:.2f} mm, [2] {kp[1]:.2f} mm, result: {kp[1]-kp[0]:.2f} mm depth")
        logging.info(f"Key points are: {kp}")
    except FileNotFoundError:
        logging.critical("Configuration file '%s' not found in %s", args.config, os.getcwd())
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.critical("Invalid YAML configuration in '%s': %s", args.config, str(e))
        sys.exit(1)
    except Exception as e:
        logging.critical("Error: %s", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
