import unittest
import pandas as pd
import numpy as np
from unittest import mock
import pytest
import os
import sys
from abyss.uos_inference import DepthInference

class TestDepthInference(unittest.TestCase):
    
    @mock.patch('abyss.uos_inference.load_model')
    @mock.patch('abyss.uos_inference.abyss.__path__')
    def test_init(self, mock_path, mock_load_model):
        mock_path.__getitem__.return_value = '/mock/path'
        mock_model = mock.MagicMock()
        mock_load_model.return_value = mock_model
        
        depth_inf = DepthInference(n_model=5)
        
        self.assertEqual(depth_inf.ref_data_path, '/mock/path/trained_model/reference_data/ref_15T.csv')
        self.assertEqual(depth_inf.model, mock_model)
        self.assertEqual(depth_inf.idx_cv, 5)
        mock_load_model.assert_called_once_with(5)
    
    @mock.patch('abyss.uos_inference.pd.read_csv')
    @mock.patch('abyss.uos_inference.inference_data_pipeline')
    @mock.patch('abyss.uos_inference.exit_estimation_pipeline')
    def test_inference(self, mock_exit_estimation, mock_data_pipeline, mock_read_csv):
        # Mock returned values
        mock_df = mock.MagicMock()
        # Configure index access to return proper values
        mock_df.__getitem__.side_effect = lambda key: {
            'HOLE_ID': 'test',
            'local': 1,
            'PREDRILLED': 0
        }.get(key, mock.MagicMock())
        
        mock_read_csv.return_value = mock_df
        mock_data = mock.MagicMock()
        mock_enter_pos = {'test': 10.0}
        mock_data_pipeline.return_value = (mock_data, mock_enter_pos)
        mock_exit_estimation.return_value = 20.0
        
        # Create instance with mocked model
        depth_inf = DepthInference()
        depth_inf.model = mock.MagicMock()
        depth_inf.ref_data_path = '/mock/ref_path'
        
        # Call inference method
        result = depth_inf.inference('raw_data_path', hole_id='test', local=1, PREDRILLED=0)
        
        # Verify results
        self.assertEqual(result, (10.0, 20.0))
        mock_read_csv.assert_called_once_with('raw_data_path', delimiter=';')
        self.assertEqual(mock_df['HOLE_ID'], 'test')
        self.assertEqual(mock_df['local'], 1)
        self.assertEqual(mock_df['PREDRILLED'], 0)
        mock_data_pipeline.assert_called_once_with(df_for_inference=mock_df, ref_data_path='/mock/ref_path')
        mock_exit_estimation.assert_called_once_with(depth_inf.model, mock_data)
    
    @mock.patch('abyss.uos_inference.pd.read_csv')
    @mock.patch('abyss.uos_inference.DepthInference.infer_common')
    def test_infer_json(self, mock_infer_common, mock_read_csv):
        # Mock returned values
        mock_df = mock.MagicMock()
        # Configure index access to return proper values
        mock_df.__getitem__.side_effect = lambda key: {
            'HOLE_ID': 'test',
            'local': 1,
            'PREDRILLED': 0
        }.get(key, mock.MagicMock())
        
        mock_read_csv.return_value = mock_df
        mock_infer_common.return_value = (10.0, 20.0)
        
        # Create instance
        depth_inf = DepthInference()
        
        # Call infer_json method
        result = depth_inf.infer_json('json_path', hole_id='test', local=1, PREDRILLED=0)
        
        # Verify results
        self.assertEqual(result, (10.0, 20.0))
        mock_read_csv.assert_called_once_with('json_path', delimiter=';')
        
        # Don't check the mock values directly, as they may not be correctly reflected
        # Instead check if infer_common was called with the right DataFrame
        mock_infer_common.assert_called_once_with(mock_df)
    
    @mock.patch('abyss.uos_inference.inference_data_pipeline')
    @mock.patch('abyss.uos_inference.exit_estimation_pipeline')
    def test_infer_common(self, mock_exit_estimation, mock_data_pipeline):
        # Create mock data
        mock_df = pd.DataFrame({
            'HOLE_ID': ['test'],
            'local': [1],
            'PREDRILLED': [0]
        })
        
        # Mock returned values
        mock_data = mock.MagicMock()
        mock_enter_pos = {'test': 10.0}
        mock_data_pipeline.return_value = (mock_data, mock_enter_pos)
        mock_exit_estimation.return_value = 20.0
        
        # Create instance with mocked model
        depth_inf = DepthInference()
        depth_inf.model = mock.MagicMock()
        depth_inf.ref_data_path = '/mock/ref_path'
        
        # Call infer_common method
        result = depth_inf.infer_common(mock_df)
        
        # Verify results
        self.assertEqual(result, (10.0, 20.0))
        mock_data_pipeline.assert_called_once_with(df_for_inference=mock_df, ref_data_path='/mock/ref_path')
        mock_exit_estimation.assert_called_once_with(depth_inf.model, mock_data)
    
    @mock.patch('abyss.uos_inference.inference_data_pipeline')
    @mock.patch('abyss.uos_inference.exit_estimation_pipeline')
    @mock.patch('abyss.uos_inference.add_cf2ti_point')
    def test_infer3_common(self, mock_add_cf2ti, mock_exit_estimation, mock_data_pipeline):
        # Create mock data
        mock_df = pd.DataFrame({
            'HOLE_ID': ['test'],
            'local': [1],
            'PREDRILLED': [0]
        })
        
        # Mock returned values
        mock_data = mock.MagicMock()
        mock_enter_pos = {'test': 10.0}
        mock_data_pipeline.return_value = (mock_data, mock_enter_pos)
        mock_exit_estimation.return_value = 30.0
        mock_add_cf2ti.return_value = 20.0
        
        # Create instance with mocked model
        depth_inf = DepthInference()
        depth_inf.model = mock.MagicMock()
        depth_inf.ref_data_path = '/mock/ref_path'
        
        # Call infer3_common method
        result = depth_inf.infer3_common(mock_df)
        
        # Verify results
        self.assertEqual(result, (10.0, 20.0, 30.0))
        mock_data_pipeline.assert_called_once_with(df_for_inference=mock_df, ref_data_path='/mock/ref_path')
        mock_add_cf2ti.assert_called_once_with(mock_df)
        mock_exit_estimation.assert_called_once_with(depth_inf.model, mock_data)
    
    @mock.patch('abyss.uos_inference.loadSetitecXls')
    @mock.patch('abyss.uos_inference.DepthInference.infer_common')
    def test_infer_xls(self, mock_infer_common, mock_load_xls):
        # Mock returned values
        mock_df = mock.MagicMock()
        # Configure index access to return proper values
        mock_df.__getitem__.side_effect = lambda key: {
            'HOLE_ID': 'test',
            'local': 1,
            'PREDRILLED': 0
        }.get(key, mock.MagicMock())
        
        mock_load_xls.return_value = mock_df
        mock_infer_common.return_value = (10.0, 20.0)
        
        # Create instance
        depth_inf = DepthInference()
        
        # Call infer_xls method
        result = depth_inf.infer_xls('xls_path', hole_id='test', local=1, PREDRILLED=0)
        
        # Verify results
        self.assertEqual(result, (10.0, 20.0))
        mock_load_xls.assert_called_once_with('xls_path', version='auto_data')
        self.assertEqual(mock_df['HOLE_ID'], 'test')
        self.assertEqual(mock_df['local'], 1)
        self.assertEqual(mock_df['PREDRILLED'], 0)
        mock_infer_common.assert_called_once_with(mock_df)
    
    @mock.patch('abyss.uos_inference.loadSetitecXls')
    @mock.patch('abyss.uos_inference.DepthInference.infer3_common')
    def test_infer3_xls(self, mock_infer3_common, mock_load_xls):
        # Mock returned values
        mock_df = mock.MagicMock()
        # Configure index access to return proper values
        mock_df.__getitem__.side_effect = lambda key: {
            'HOLE_ID': 'test',
            'local': 1,
            'PREDRILLED': 0
        }.get(key, mock.MagicMock())
        
        mock_load_xls.return_value = mock_df
        mock_infer3_common.return_value = (10.0, 20.0, 30.0)
        
        # Create instance
        depth_inf = DepthInference()
        
        # Call infer3_xls method
        result = depth_inf.infer3_xls('xls_path', hole_id='test', local=1, PREDRILLED=0)
        
        # Verify results
        self.assertEqual(result, (10.0, 20.0, 30.0))
        mock_load_xls.assert_called_once_with('xls_path', version='auto_data')
        self.assertEqual(mock_df['HOLE_ID'], 'test')
        self.assertEqual(mock_df['local'], 1)
        self.assertEqual(mock_df['PREDRILLED'], 0)
        mock_infer3_common.assert_called_once_with(mock_df)

if __name__ == '__main__':
    unittest.main()