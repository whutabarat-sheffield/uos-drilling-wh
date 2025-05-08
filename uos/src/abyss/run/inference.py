import torch
import numpy as np
import pandas as pd
from utils.pipelines.inference_pipeline import *
from utils.pipelines.training_data_formater import *
from utils.functions.load_model import load_model

"""
This py is for inference
Args:

"""


if __name__ == "__main__":

    '''read data to pandas data frame'''
    raw_data_path = '../../test_data/MQTT_Curve_Info_20240304_14H27.json'
    ref_data_path = '../../test_data/ref_15T.csv'

    df = pd.read_csv(
        raw_data_path,
        delimiter=';')

    '''provide additional information'''
    hole_id = 'test'
    df['HOLE_ID'] = hole_id  # Unique ID for each different hole
    df['local'] = 0  # Tool age: how many holes drilled before.
    df['PREDRILLED'] = 1  # Predrilling flag, 0: False 1: True

    '''prepare data, model, and estimator'''
    data, enter_pos = inference_data_pipeline(df_for_inference=df, ref_data_path=ref_data_path)
    model = load_model(idx_cv=4)
    hole_depth = depth_estimation_pipeline(model, data, start_xpos=enter_pos[hole_id])
    print(hole_depth)

