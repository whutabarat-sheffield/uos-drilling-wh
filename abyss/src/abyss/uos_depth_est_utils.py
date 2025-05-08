import logging
import json
import pandas as pd
from functools import reduce


# Set up logging configuration
def setup_logging(level):
    """
    Configure logging with the specified level.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def convert_mqtt_to_df(result_msg=None, trace_msg=None, conf=None):
    """
    Convert `RESULT` and/or `TRACE` MQTT messages to Pandas DataFrames and combine them.

    Parameters:
    result_msg (str, optional): The MQTT message for the `RESULT` structure as a JSON string.
    trace_msg (str, optional): The MQTT message for the `TRACE` structure as a JSON string.
    conf (dict, optional): Configuration dictionary with key paths to extract the required data. If not provided, default paths will be used.

    Returns:
    pd.DataFrame: A combined DataFrame with data from `RESULT` and `TRACE` messages, merged on step values.

    Notes:
    - If both `result_msg` and `trace_msg` are provided, the data is merged using step values as a key.
    - The paths in the configuration should be provided as tuples representing the hierarchical keys.
    """
    


    def reduce_dict(data_dict, search_key):
        # Using reduce function to filter dictionary values based on search_key
        # from https://www.geeksforgeeks.org/python-substring-key-match-in-dictionary/
        logging.info("Reducing dictionary")
        logging.debug(f"Dict: {data_dict}\n\nSearch_key: {search_key}")
    
        values = reduce(
            # lambda function that takes in two arguments, an accumulator list and a key
            lambda acc, key: acc + [data_dict[key]] if search_key in key else acc,
            # list of keys from the test_dict
            data_dict.keys(),
            # initial value for the accumulator (an empty list)
            []
        )
        return values[0]['Value']
    

    def parse_result(data, conf):
        # Default key paths for RESULT structure
        fore = ['Messages', 'Payload']
        
        data = data[fore[0]][fore[1]]   
        logging.info(conf['mqtt']['data_ids']['torque_empty_vals'])

        torque_empty_vals = reduce_dict(data, conf['mqtt']['data_ids']['torque_empty_vals'])
        logging.info(f"Torque empty values: {torque_empty_vals}")
        thrust_empty_vals = reduce_dict(data, conf['mqtt']['data_ids']['thrust_empty_vals'])
        logging.info(f"Thrust empty values: {thrust_empty_vals}")
        step_vals = reduce_dict(data, conf['mqtt']['data_ids']['step_vals'])
        logging.info(f"Step values: {step_vals}")
        # hole_id = reduce_dict(data, conf['mqtt']['data_ids']['machine_id']) + '_' + reduce_dict(data, conf['mqtt']['data_ids']['result_id'])
        hole_id = reduce_dict(data, conf['mqtt']['data_ids']['machine_id'])# TODO rethink whether this is correct
        hole_id = [str(hole_id)] * len(step_vals)
        logging.info(f"Hole ID: {hole_id}")
        local = reduce_dict(data, conf['mqtt']['data_ids']['result_id'])# TODO rethink whether this is correct
        local = [str(local)] * len(step_vals)
        logging.info(f"Local count: {local}")

        # Create a DataFrame
        logging.info("Creating DataFrame")
        try:
            df = pd.DataFrame({
                'Step (nb)': step_vals,
                'I Torque Empty (A)': torque_empty_vals,
                'I Thrust Empty (A)': thrust_empty_vals,
                # 'HOLE_ID': [str(hole_id)] * len(step_vals),
                # 'local': [str(local)] * len(step_vals),
                'HOLE_ID': hole_id,
                'local': local,
                'PREDRILLED': [1] * len(step_vals),
            })
        except:
            df = pd.DataFrame({
                'Step (nb)': step_vals,
                'I Torque Empty (A)': torque_empty_vals,
                'I Thrust Empty (A)': thrust_empty_vals,
                'HOLE_ID': [str(hole_id)], 
                'local': [str(local)],  
                'PREDRILLED': [1]
            })
        df['local'] = df['local'].astype('int32')
        logging.info(f"DataFrame: {df}")
        return df

    def parse_trace(data, conf):
        # Default key paths for RESULT structure
        fore = ['Messages', 'Payload']

        data = data[fore[0]][fore[1]]   
        
        # def get_nested_value(data, key):
        #     key_path = fore + [key]
        #     return reduce(dict.get, key_path, data)

        position = reduce_dict(data, conf['mqtt']['data_ids']['position'])
        logging.debug(f"Position: {position}")
        torque = reduce_dict(data, conf['mqtt']['data_ids']['torque'])
        logging.debug(f"Torque: {torque}")
        thrust = reduce_dict(data, conf['mqtt']['data_ids']['thrust'])
        logging.debug(f"Thrust: {thrust}")
        step = reduce_dict(data, conf['mqtt']['data_ids']['step'])
        logging.debug(f"Step: {step}")
        # hole_id = reduce_dict(data, conf['mqtt']['data_ids']['trace_result_id'])

        # Create a DataFrame
        df = pd.DataFrame({
            'Step (nb)': step,
            'Position (mm)': position,
            'I Torque (A)': torque,
            'I Thrust (A)': thrust,
            # 'HOLE_ID': [str(hole_id)],
        })

        # Process columns
        df['Step (nb)'] = df['Step (nb)'].astype('int32')
        df['Position (mm)'] = -df['Position (mm)']  # Invert position

        logging.info(f"Trace DataFrame: {df.tail()}")
        return df

    result_df = None
    trace_df = None

    logging.debug("Starting conversion of MQTT messages to DataFrame")
    logging.debug(f"Result message: {result_msg}\n\n")
    logging.debug(f"Trace message: {trace_msg}\n\n")
    logging.debug(f"Configuration: {conf}\n\n")

    if result_msg:
        logging.info("Processing RESULT message")
        logging.debug(f"Result message: {result_msg}")
        # Decode the JSON message
        try:
            result_data = json.loads(result_msg)
        except json.JSONDecodeError as e:
            logging.critical("Error decoding JSON: %s", str(e))
            return None
        logging.info(f"Decoded RESULT data: {result_data}")
        # Parse the RESULT data
        try:
            result_df = parse_result(result_data, conf)
        except Exception as e:
            logging.critical("Error parsing RESULT data: %s", str(e))

    if trace_msg:
        trace_data = json.loads(trace_msg)
        try:
            trace_df = parse_trace(trace_data, conf)
        except Exception as e:
            logging.critical("Error parsing TRACE data: %s", str(e))

    # Merge the DataFrames on the step values if both are present
    if result_df is not None and trace_df is not None:
        try:
            combined_df = pd.merge(result_df, trace_df, on='Step (nb)', how='outer')
            logging.info(str(combined_df.dtypes))
            logging.info(f"Combined DataFrame: {combined_df.head()}")
            return combined_df
        except Exception as e:
            logging.critical("Error merging RESULT and TRACE DataFrames: %s", str(e))

    return result_df or trace_df