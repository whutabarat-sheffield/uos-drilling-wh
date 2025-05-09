import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, Mock, MagicMock
import os
from pathlib import Path
from functools import reduce
import json


from abyss.uos_depth_est_core import (
    get_setitec_signals,
    calculate_quotient_remainder,
    calculate_fastener_grip_length,
    depth_est_xls_persegment_stats,
    depth_est_persegment_stats,
    keypoint_recognition_gradient,
    standard_xls_loading,
    kp_conversion,
    kp_recognition_gradient
)


class TestCalculateQuotientRemainder:
    def test_standard_case(self):
        quotient, remainder = calculate_quotient_remainder(5.0, 1.5875)
        assert quotient == 3
        assert remainder == pytest.approx(5.0 - 3 * 1.5875)

    def test_exact_division(self):
        quotient, remainder = calculate_quotient_remainder(4.7625, 1.5875)
        assert quotient == 3
        assert remainder == pytest.approx(0.0)

    def test_type_error(self):
        with pytest.raises(TypeError):
            calculate_quotient_remainder("5.0", 1.5875)
        with pytest.raises(TypeError):
            calculate_quotient_remainder(5.0, "1.5875")


class TestCalculateFastenerGripLength:
    def test_standard_case(self):
        d1, d2, d3 = calculate_fastener_grip_length(5.0, 1.5875, 0.396875, 1.42875)
        assert d1 == 4
        # Expected values will depend on where the remainder falls relative to thresholds
        remainder = 5.0 % 1.5875
        rule2 = remainder > 1.42875
        expected_d2 = 3 + rule2
        assert d2 == expected_d2

        if remainder < 0.396875:
            expected_d3 = 3
        elif remainder > 1.42875:
            expected_d3 = 5
        else:
            expected_d3 = 4
        assert d3 == expected_d3

    def test_lower_threshold(self):
        d1, d2, d3 = calculate_fastener_grip_length(3.2, 1.5875, 0.396875, 1.42875)
        quotient = 3.2 // 1.5875
        assert d1 == quotient + 1
        remainder = 3.2 % 1.5875
        assert d3 == quotient + (0 if remainder < 0.396875 else (2 if remainder > 1.42875 else 1))

    def test_upper_threshold(self):
        d1, d2, d3 = calculate_fastener_grip_length(4.8, 1.5875, 0.396875, 1.42875)
        quotient = 4.8 // 1.5875
        assert d1 == quotient + 1
        remainder = 4.8 % 1.5875
        assert d3 == quotient + (0 if remainder < 0.396875 else (2 if remainder > 1.42875 else 1))


@pytest.fixture
def mock_signals():
    pos = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    torque = np.array([1.0, 1.2, 1.5, 1.8, 2.0])
    return pos, torque


class TestGetSetitecSignals:
    @patch('abyss.uos_depth_est_core.loadSetitecXls')
    def test_get_setitec_signals(self, mock_load):
        # Setup mock
        mock_df = pd.DataFrame({
            'Position (mm)': [-0.1, -0.2, -0.3, -0.4, -0.5],
            'I Torque (A)': [1.0, 1.2, 1.5, 1.8, 2.0],
            'I Torque Empty (A)': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        mock_load.return_value = mock_df
        
        # Call function
        position, torque_full = get_setitec_signals("dummy_file.xls")
        
        # Assertions
        assert mock_load.called_once_with("dummy_file.xls", version="auto_data")
        np.testing.assert_array_equal(position, np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
        np.testing.assert_array_equal(torque_full, np.array([1.1, 1.4, 1.8, 2.2, 2.5]))


class TestDepthEstimation:
    @patch('abyss.uos_depth_est_core.get_setitec_signals')
    @patch('abyss.uos_depth_est_core.depth_est_persegment_stats')
    def test_depth_est_xls_persegment_stats(self, mock_depth_est, mock_get_signals):
        # Setup mocks
        mock_pos = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        mock_torque = np.array([1.0, 1.2, 1.5, 1.8, 2.0])
        mock_get_signals.return_value = (mock_pos, mock_torque)
        mock_depth_est.return_value = [0.1, 0.3, 0.5]
        
        # Call function
        result = depth_est_xls_persegment_stats("dummy_file.xls", bit_depth=0.7, stat='median')
        
        # Assertions
        mock_get_signals.assert_called_once_with("dummy_file.xls")
        mock_depth_est.assert_called_once_with(mock_torque, mock_pos, bit_depth=0.7, stat='median', simple_results=True, debug=False)
        assert result == [0.1, 0.3, 0.5]

    @patch('pandas.DataFrame')
    def test_depth_est_persegment_stats(self, mock_df_class, mock_signals):
        pos, torque = mock_signals
        
        # This is a complex function to test fully without integration testing
        # Here's a partial test focusing on the function signature and basic input validation
        with pytest.raises(ValueError):
            depth_est_persegment_stats(torque, pos, bit_depth=0.7, stat='invalid_stat')


class TestKeypointRecognition:
    def test_keypoint_recognition_gradient(self, mock_signals):
        pos, signal = mock_signals
        
        # Create a simple dataframe for testing
        df = pd.DataFrame({'pos': pos, 'signal': signal})
        
        # Test the function with small window sizes to accommodate our small test data
        result_df, l_pos = keypoint_recognition_gradient(pos, signal, smo_w=2, wsize=2, bit_depth=0.7)
        
        # Verify the result structure
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(l_pos, list)
        assert len(l_pos) == 1
        assert all(key in l_pos[0] for key in ['meanmin', 'meanmax', 'medianmin', 'medianmax'])