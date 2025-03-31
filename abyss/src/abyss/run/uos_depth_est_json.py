import logging
import sys
import os
import json
import yaml
import argparse


from abyss.uos_inference import (
    DepthInference
)

from abyss.uos_depth_est_utils import (
    convert_mqtt_to_df,
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
    parser = argparse.ArgumentParser(description='JSON Drilling Data Analyzer')

    parser.add_argument("result_msg",
        # type=str,
        # default='mqtt_conf.yaml',
        help='Path to Result json file (default: none)'
    )

    parser.add_argument("trace_msg",
        # type=str,
        # default='mqtt_conf.yaml',
        help='Path to Trace json  file (default: none)'
    )

    parser.add_argument("config",
        # type=str,
        # default='mqtt_conf.yaml',
        help='Path to Trace json  file (default: none)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='DEBUG',
        help='Set the logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    setup_logging(getattr(logging, args.log_level))
    
    ## -----------------------------------------------------------------------------
    ##
    ## Run the depth estimator here
    ##
    ## -----------------------------------------------------------------------------
    try:
        # Load the configuration file
        with open(args.config, 'r') as file:
            conf = yaml.safe_load(file)
            # logging.debug(f"Configuration: {conf}\n\n")
        # load the data from the MQTT messages
        with open(args.result_msg, 'r') as file:
            result_msg = json.dumps(json.load(file))
            # logging.debug(f"Result message: {result_msg}\n\n")
        with open(args.trace_msg, 'r') as file:
            trace_msg = json.dumps(json.load(file))
            # logging.debug(f"Trace message: {trace_msg}\n\n")
        # Convert the MQTT messages to a DataFrame
        df = convert_mqtt_to_df(result_msg=result_msg, trace_msg=trace_msg, conf=conf)
        assert df is not None, "DataFrame conversion failed."
        logging.info(f"DataFrame successfully converted:\n {df.tail()}")
        analyzer = DepthInference()
        kp = analyzer.infer_common(df)
        print(f"Key points: {kp[0]:.2f} mm, {kp[1]:.2f} mm, depth: {kp[1]-kp[0]:.2f} mm")
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
