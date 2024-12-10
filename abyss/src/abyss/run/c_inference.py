import numpy as np
import pandas as pd
from pprint import pprint

from abyss.dataparser import loadSetitecXls
from abyss.utils.pipelines.inference_pipeline import *
from abyss.utils.pipelines.training_data_formater import *
from abyss.utils.functions.load_model import load_model


"""
This py is for inference
Args:

"""

class DepthInference:
    def __init__(self, n_model = 4, ref_data_path=None, ):
        self.model = load_model(n_model, ref_data_path)
        self.ref_data_path = ref_data_path

    def inference(self, raw_data_path, hole_id = 'test', local = 0, PREDRILLED = 1):
        self.raw_data_path = raw_data_path
        '''read data to pandas data frame'''
        df = pd.read_csv(
            self.raw_data_path,
            delimiter=';')

        '''provide additional information'''
        df['HOLE_ID'] = hole_id  # Unique ID for each different hole
        df['local'] = local  # Tool age: how many holes drilled before.
        df['PREDRILLED'] = PREDRILLED  # Predrilling flag, 0: False 1: True

        '''prepare data, model, and estimator'''
        data, enter_pos = inference_data_pipeline(df_for_inference=df, ref_data_path=self.ref_data_path)
        # pprint(enter_pos)

        model = load_model(idx_cv=4)
        # hole_depth = depth_estimation_pipeline(model, data, start_xpos=enter_pos[hole_id])
        exit_depth = exit_estimation_pipeline(model, data)
        return enter_pos[hole_id], exit_depth
   
    
    def infer_json(self, json_path, hole_id = 'test', local = 0, PREDRILLED = 1):
        self.raw_data_path = raw_data_path
        '''read data to pandas data frame'''
        df = pd.read_csv(
            self.raw_data_path,
            delimiter=';')
        

        '''provide additional information'''
        df['HOLE_ID'] = hole_id  # Unique ID for each different hole
        df['local'] = local  # Tool age: how many holes drilled before.
        df['PREDRILLED'] = PREDRILLED  # Predrilling flag, 0: False 1: True

        return self.infer_common(df, hole_id, local, PREDRILLED)
        
    def infer_common(self, data, hole_id = 'test', local = 0, PREDRILLED = 1):
        '''prepare data, model, and estimator'''
        self._df = data
        data, enter_pos = inference_data_pipeline(df_for_inference=self._df, ref_data_path=self.ref_data_path)
        # pprint(enter_pos)

        model = load_model(idx_cv=4)
        # hole_depth = depth_estimation_pipeline(model, data, start_xpos=enter_pos[hole_id])
        exit_depth = exit_estimation_pipeline(model, data)
        return enter_pos[hole_id], exit_depth
    
    def infer_xls(self, xls_path, hole_id = 'test', local = 0, PREDRILLED = 1):
        df = loadSetitecXls(xls_path, version='auto_data')

        '''provide additional information'''
        df['HOLE_ID'] = hole_id  # Unique ID for each different hole
        df['local'] = local  # Tool age: how many holes drilled before.
        df['PREDRILLED'] = PREDRILLED  # Predrilling flag, 0: False 1: True

        return self.infer_common(df, hole_id, local, PREDRILLED)


if __name__ == "__main__":
    import os
    '''read data to pandas data frame'''
    raw_data_path = '../test_data/MQTT_Curve_Info_20240304_14H27.json'
    ref_data_path = '../test_data/ref_15T.csv'
    # pprint(os.getcwd())
    # pprint(os.listdir('../'))

    # df = pd.read_csv(
    #     raw_data_path,
    #     delimiter=';')

    # '''provide additional information'''
    # hole_id = 'test'
    # df['HOLE_ID'] = hole_id  # Unique ID for each different hole
    # df['local'] = 0  # Tool age: how many holes drilled before.
    # df['PREDRILLED'] = 1  # Predrilling flag, 0: False 1: True

    # '''prepare data, model, and estimator'''
    # data, enter_pos = inference_data_pipeline(df_for_inference=df, ref_data_path=ref_data_path)
    # model = load_model(idx_cv=4)
    # hole_depth = depth_estimation_pipeline(model, data, start_xpos=enter_pos[hole_id])
    # print(hole_depth)

    d_est = DepthInference(4, ref_data_path)
    result = d_est.inference(raw_data_path, hole_id = 'test', local = 0, PREDRILLED = 1)
    print(f'HAMBURG EXAMPLE .json: {result}, depth: {result[1]-result[0]}')
    result1 = d_est.infer_json(raw_data_path, hole_id = 'test', local = 0, PREDRILLED = 1)
    print(f'HAMBURG EXAMPLE .json split function: {result1}, depth: {result1[1]-result1[0]}')
    #E00401006B6B2977_113-V3-022_ST_2193_63
    print((result, result1))
    result2 = d_est.infer_xls('../test_data/E00401006B6B2977_113-V3-022_ST_2193_63.xls', hole_id = 'test', local = 0, PREDRILLED = 1)
    print(f'MDB EXAMPLE .xls: {result2}, depth: {result2[1]-result2[0]}')