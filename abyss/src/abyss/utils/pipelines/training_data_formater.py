import pandas as pd
import logging
# import numpy as np
from abyss.utils.functions.data_labeller import *
from abyss.utils.functions.data_visualisation import *
from abyss.utils.functions.data_organisation import *


def inference_data_pipeline(df_for_inference, ref_data_path):
    """
    Zee
    :param df_for_inference: pandas dataframe
    :param HOLE_ID: Unique id for one hole
    :param tool_age: how many holes drilled previously
    :param Predrilling_flag: whether the hole to be estimated has predrilling hole
    :return: the pandas dataframe that is ready for estimation.
    """
    logging.info("Starting inference data pipeline...")
    logging.info("Checking raw data...")
    data = raw_data_checker(df_for_inference)
    logging.info("Adding entry point...")
    data, enter_pos = add_entry_point(data)
    logging.info("Organising data...")
    data = data_orgnization(data)
    logging.info("Calling data scaling function...")
    data = data_scaler(data, ref_data_path=ref_data_path)
    logging.info("Inference pipeline finished!")
    return data, enter_pos


if __name__ == "__main__":
    '''read data to pandas data frame'''
    raw_data_path = '../../test_data/MQTT_Curve_Info_20240304_14H27.json'
    ref_data_path = '../../test_data/ref_15T.csv'

    df = pd.read_csv(
        raw_data_path,
        delimiter=';')
    df['HOLE_ID'] = 'test'  # Unique ID for each different hole
    df['local'] = 50  # Tool age: how many holes drilled before.
    df['PREDRILLED'] = 0  # Predrilling flag, 0: False 1: True

    data, enter_pos = inference_data_pipeline(df_for_inference=df, ref_data_path=ref_data_path)
    print(data.keys())
    print(enter_pos)

