from dataparser import loadSetitecXls
import numpy as np
import pandas as pd
from utils.pipelines.inference_pipeline import *
from utils.pipelines.training_data_formater import *
from utils.functions.load_model import load_model
from pprint import pprint

"""
This py is for inference
Args:

"""

class DepthInference:
    """
    Class for performing depth inference on drilling data.

    Args:
        n_model (int): Number of the model to load.
        ref_data_path (str): Path to the reference data.

    Attributes:
        model: The loaded model.
        ref_data_path (str): Path to the reference data.

    Methods:
        inference: Perform depth inference on raw data.
        infer_json: Perform depth inference on data from a JSON file.
        infer_common: Common method for performing depth inference.
        infer_xls: Perform depth inference on data from an XLS file.
    """
    def __init__(self, n_model=4, ref_data_path=None):
        assert ref_data_path is not None, "Please provide a reference data path."
        self.ref_data_path = ref_data_path
        self.model = load_model(n_model, ref_data_path)


    def inference(self, raw_data_path, hole_id='test', local=0, PREDRILLED=1):
        """
        Perform depth inference on raw data.

        Args:
            raw_data_path (str): Path to the raw data file.
            hole_id (str): Unique ID for each different hole.
            local (int): Tool age: how many holes drilled before.
            PREDRILLED (int): Predrilling flag, 0: False 1: True

        Returns:
            tuple: A tuple containing the entry position and exit depth.
        """
        self.raw_data_path = raw_data_path
        # read data to pandas data frame
        df = pd.read_csv(self.raw_data_path, delimiter=';')

        # provide additional information
        df['HOLE_ID'] = hole_id
        df['local'] = local
        df['PREDRILLED'] = PREDRILLED

        # prepare data, model, and estimator
        data, enter_pos = inference_data_pipeline(df_for_inference=df, ref_data_path=self.ref_data_path)
        model = load_model(idx_cv=4)
        exit_depth = exit_estimation_pipeline(model, data)
        return enter_pos[hole_id], exit_depth

    def infer_json(self, json_path, hole_id='test', local=0, PREDRILLED=1):
        """
        Perform depth inference on data from a JSON file.

        Args:
            json_path (str): Path to the JSON file.
            hole_id (str): Unique ID for each different hole.
            local (int): Tool age: how many holes drilled before.
            PREDRILLED (int): Predrilling flag, 0: False 1: True

        Returns:
            tuple: A tuple containing the enter position and exit depth.
        """
        self.raw_data_path = raw_data_path
        # read data to pandas data frame
        df = pd.read_csv(self.raw_data_path, delimiter=';')

        # provide additional information
        df['HOLE_ID'] = hole_id
        df['local'] = local
        df['PREDRILLED'] = PREDRILLED

        return self.infer_common(df, hole_id, local, PREDRILLED)

    def infer_common(self, data, hole_id='test', local=0, PREDRILLED=1):
        """
        Common method for performing depth inference.

        Args:
            data: The data for inference.
            hole_id (str): Unique ID for each different hole.
            local (int): Tool age: how many holes drilled before.
            PREDRILLED (int): Predrilling flag, 0: False 1: True

        Returns:
            tuple: A tuple containing the enter position and exit depth.
        """
        self._df = data
        data, enter_pos = inference_data_pipeline(df_for_inference=self._df, ref_data_path=self.ref_data_path)
        model = load_model(idx_cv=4)
        exit_depth = exit_estimation_pipeline(model, data)
        return enter_pos[hole_id], exit_depth

    def infer_xls(self, xls_path, hole_id='test', local=0, PREDRILLED=1):
        """
        Perform depth inference on data from an XLS file.

        Args:
            xls_path (str): Path to the XLS file.
            hole_id (str): Unique ID for each different hole.
            local (int): Tool age: how many holes drilled before.
            PREDRILLED (int): Predrilling flag, 0: False 1: True

        Returns:
            tuple: A tuple containing the enter position and exit depth.
        """
        df = loadSetitecXls(xls_path, version='auto_data')

        # provide additional information
        df['HOLE_ID'] = hole_id
        df['local'] = local
        df['PREDRILLED'] = PREDRILLED

        return self.infer_common(df, hole_id, local, PREDRILLED)


if __name__ == "__main__":

    '''read data to pandas data frame'''
    raw_data_path = '../../test_data/MQTT_Curve_Info_20240304_14H27.json'
    ref_data_path = '../../test_data/ref_15T.csv'

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

    fn='E00401006B6B2977_113-V3-022_ST_2193_63'
    result2 = d_est.infer_xls(f'../../test_data/{fn}.xls', hole_id = 'test', local = 0, PREDRILLED = 1)
    print(f'{fn}.xls: {result2}, depth: {result2[1]-result2[0]}')

    #17070143_17070143_ST_885_99
    fn='17070143_17070143_ST_885_99'
    result3 = d_est.infer_xls(f'../../test_data/{fn}.xls', hole_id = 'test', local = 0, PREDRILLED = 1)
    print(f'{fn}.xls: {result3}, depth: {result3[1]-result3[0]}')

    #17070141_17070141_ST_753_55
    fn='17070141_17070141_ST_753_55'
    result4 = d_est.infer_xls(f'../../test_data/{fn}.xls', hole_id = 'test', local = 0, PREDRILLED = 1)
    print(f'{fn}.xls: {result4}, depth: {result4[1]-result4[0]}')

    #17070141_17070141_ST_798_100
    fn='17070141_17070141_ST_798_100'
    result4 = d_est.infer_xls(f'../../test_data/{fn}.xls', hole_id = 'test', local = 0, PREDRILLED = 1)
    print(f'{fn}.xls: {result4}, depth: {result4[1]-result4[0]}')
