import logging
import json
import pandas as pd
from functools import reduce


def find_in_dict(data, target_key: str) -> list:
    """
    This function performs a depth-first search through nested dictionaries and lists
    to find all occurrences of a specified key and returns their corresponding values.
    Args:
        data: The nested dictionary/list structure to search through. Can be a dict,
              list, or any combination of nested dicts and lists.
        target_key (str): The key to search for in the nested structure.
    Returns:
        list: A list containing all values found for the target key. Returns an
              empty list if the key is not found anywhere in the structure.
    Examples:
        >>> data = {"a": 1, "b": {"a": 2, "c": {"a": 3}}}
        >>> find_in_dict(data, "a")
        [1, 2, 3]
        >>> data = [{"name": "John"}, {"name": "Jane", "details": {"name": "Detail"}}]
        >>> find_in_dict(data, "name")
        ["John", "Jane", "Detail"]
        >>> find_in_dict({}, "nonexistent")
        []

    Recursively search for a key in a nested dictionary/list structure.
    Returns list of values found for that key.
    """
    results = []
    
    def _search(current_data):
        if isinstance(current_data, dict):
            for key, value in current_data.items():
                if key == target_key:
                    results.append(value)
                if isinstance(value, (dict, list)):
                    _search(value)
        elif isinstance(current_data, list):
            for item in current_data:
                if isinstance(item, (dict, list)):
                    _search(item)
    
    _search(data)
    return results


# Set up logging configuration
def setup_logging(level):
    """
    Configure logging with the specified level.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    

def reduce_dict(data_dict, search_key):
    """
    Filter dictionary values based on a substring match in the keys.
    This function uses the reduce function to find all dictionary entries where
    the key contains the specified search string, then returns the 'Value' field
    from the first matching entry.
    Args:
        data_dict (dict): The dictionary to search through
        search_key (str): The substring to search for in dictionary keys
    Returns:
        Any: The 'Value' field from the first matching dictionary entry,
             or an empty list if no matches found or invalid input
    Raises:
        IndexError: If no matching entries are found (returns empty list instead)
        KeyError: If the first matching entry doesn't contain a 'Value' key
    Example:
        >>> data = {'sensor_temp_1': {'Value': 25.5}, 'sensor_pressure_1': {'Value': 1013}}
        >>> reduce_dict(data, 'temp')
        25.5
    """
    # Using reduce function to filter dictionary values based on search_key
    # from https://www.geeksforgeeks.org/python-substring-key-match-in-dictionary/
    logging.info(f"Reducing dictionary for {search_key}")
    logging.debug(f"Dict: {data_dict}\n\nSearch_key: {search_key}")

    if not isinstance(data_dict, dict):
        logging.error("Provided data_dict is not a dictionary")
        return []
    if not isinstance(search_key, str):
        logging.error("Provided search_key is not a string")
        return []
    
    values = reduce(
        # lambda function that takes in two arguments, an accumulator list and a key
        lambda acc, key: acc + [data_dict[key]] if search_key in key else acc,
        # list of keys from the test_dict
        data_dict.keys(),
        # initial value for the accumulator (an empty list)
        []
    )
    logging.info(f"Found {len(values)} matching entries for '{search_key}'")
    logging.debug(f"Reduced values: {values}")
    return values[0]['Value']


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
        if not isinstance(step_vals, list):
            step_vals = [step_vals]
        logging.info(f"Step values: {step_vals}")
        # hole_id = reduce_dict(data, conf['mqtt']['data_ids']['machine_id']) + '_' + reduce_dict(data, conf['mqtt']['data_ids']['result_id'])
        hole_id = reduce_dict(data, conf['mqtt']['data_ids']['machine_id'])# TODO rethink whether this is correct
        # logging.debug(f"Hole ID: {hole_id}")
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
        except Exception as e:
            logging.warning("DataFrame creation failed, using fallback format", extra={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'step_vals_length': len(step_vals),
                'torque_vals_length': len(torque_empty_vals),
                'thrust_vals_length': len(thrust_empty_vals)
            })
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

        logging.debug(f"Trace DataFrame: {df.tail()}")
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
        logging.debug(f"Decoded RESULT data: {result_data}")
        # Parse the RESULT data
        try:
            result_df = parse_result(result_data, conf)
        except Exception as e:
            logging.critical("Error parsing RESULT data: %s", str(e))

    if trace_msg:
        trace_data = json.loads(trace_msg)
        logging.info("Processing TRACE message")
        try:
            trace_df = parse_trace(trace_data, conf)
        except Exception as e:
            logging.critical("Error parsing TRACE data: %s", str(e))

    # Merge the DataFrames on the step values if both are present
    if result_df is not None and trace_df is not None:
        try:
            combined_df = pd.merge(result_df, trace_df, on='Step (nb)', how='outer')
            logging.debug(str(combined_df.dtypes))
            logging.debug(f"Combined DataFrame: {combined_df.head()}")
            return combined_df
        except Exception as e:
            logging.critical("Error merging RESULT and TRACE DataFrames: %s", str(e))

    return result_df if result_df is not None else trace_df