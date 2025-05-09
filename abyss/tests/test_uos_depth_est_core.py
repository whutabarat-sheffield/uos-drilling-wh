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
    convert_raw_mqtt_to_df,

    # standard_xls_loading,
    # kp_conversion,
    # kp_recognition_gradient
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
        
class TestConvertRawMqttToDF:
    def test_convert_raw_mqtt_with_default_conf(self):
        # Create sample MQTT message in JSON format with the expected structure
        sample_msg = json.dumps({
            "Messages": {
                "Payload": {
                    "nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecxls.ResultManagement.Results.0.ResultMetaData.SerialNumber": {
                        "Value": "12345"
                    },
                    "nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecxls.ResultManagement.Results.0.ResultContent.StepResults.0.StepResultValues.Position": {
                        "Value": [-0.1, -0.2, -0.3]
                    },
                    "nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecxls.ResultManagement.Results.0.ResultContent.StepResults.0.StepResultValues.Torque": {
                        "Value": [1.0, 1.2, 1.5]
                    },
                    "nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecxls.ResultManagement.Results.0.ResultContent.StepResults.0.StepResultValues.TorqueEmpty": {
                        "Value": [0.1, 0.2, 0.3]
                    },
                    "nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecxls.ResultManagement.Results.0.ResultContent.StepResults.0.StepResultValues.StepNb": {
                        "Value": [1, 1, 1]
                    },
                    "nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecxls.ResultManagement.Results.0.ResultMetaData.ToolAge": {
                        "Value": 10
                    },
                    "nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecxls.ResultManagement.Results.0.ResultMetaData.Predrilled": {
                        "Value": 1
                    }
                }
            }
        })
        
        # Call the function
        df = convert_raw_mqtt_to_df(sample_msg)
        
        # Assertions
        assert isinstance(df, pd.DataFrame)
        assert 'Position (mm)' in df.columns
        assert 'I Torque (A)' in df.columns
        assert 'I Torque Empty (A)' in df.columns
        assert 'Step (nb)' in df.columns
        assert 'HOLE_ID' in df.columns
        assert 'local' in df.columns
        assert 'PREDRILLED' in df.columns
        assert df.shape[0] == 3
        assert df['HOLE_ID'].iloc[0] == "12345"
        assert df['Position (mm)'].iloc[0] == 0.1  # Negated from -0.1
        assert df['I Torque (A)'].iloc[0] == 1.0
        assert df['I Torque Empty (A)'].iloc[0] == 0.1
        assert df['Step (nb)'].iloc[0] == 1
        assert df['local'].iloc[0] == 10
        assert df['PREDRILLED'].iloc[0] == 1

    def test_convert_raw_mqtt_with_custom_conf(self):
        # Create sample MQTT message with different structure
        sample_msg = json.dumps({
            "data": {
                "drilling": {
                    "hole_id": {
                        "value": "67890"
                    },
                    "position": {
                        "value": [-0.5, -0.6, -0.7]
                    },
                    "torque": {
                        "value": [2.0, 2.2, 2.5]
                    },
                    "torque_empty": {
                        "value": [0.5, 0.6, 0.7]
                    },
                    "step": {
                        "value": [2, 2, 2]
                    },
                    "tool_age": {
                        "value": 20
                    },
                    "predrilled_status": {
                        "value": 0
                    }
                }
            }
        })
        
        # Custom configuration
        custom_conf = {
            'hole_id': ('data', 'drilling', 'hole_id', 'value'),
            'position_keys': ('data', 'drilling', 'position', 'value'),
            'torque_keys': ('data', 'drilling', 'torque', 'value'),
            'torque_empty_keys': ('data', 'drilling', 'torque_empty', 'value'),
            'step_keys': ('data', 'drilling', 'step', 'value'),
            'local_keys': ('data', 'drilling', 'tool_age', 'value'),
            'predrilled_keys': ('data', 'drilling', 'predrilled_status', 'value')
        }
        
        # Call the function
        df = convert_raw_mqtt_to_df(sample_msg, conf=custom_conf)
        
        # Assertions
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 3
        assert df['HOLE_ID'].iloc[0] == "67890"
        assert df['Position (mm)'].iloc[0] == 0.5  # Negated from -0.5
        assert df['I Torque (A)'].iloc[0] == 2.0
        assert df['I Torque Empty (A)'].iloc[0] == 0.5
        assert df['Step (nb)'].iloc[0] == 2
        assert df['local'].iloc[0] == 20
        assert df['PREDRILLED'].iloc[0] == 0

    def test_convert_raw_mqtt_with_missing_optional_keys(self):
        # Create sample MQTT message with missing optional fields
        sample_msg = json.dumps({
            "Messages": {
                "Payload": {
                    "nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecxls.ResultManagement.Results.0.ResultMetaData.SerialNumber": {
                        "Value": "12345"
                    },
                    "nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecxls.ResultManagement.Results.0.ResultContent.StepResults.0.StepResultValues.Position": {
                        "Value": [-0.1, -0.2, -0.3]
                    },
                    "nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecxls.ResultManagement.Results.0.ResultContent.StepResults.0.StepResultValues.Torque": {
                        "Value": [1.0, 1.2, 1.5]
                    },
                    "nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecxls.ResultManagement.Results.0.ResultContent.StepResults.0.StepResultValues.TorqueEmpty": {
                        "Value": [0.1, 0.2, 0.3]
                    },
                    "nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecxls.ResultManagement.Results.0.ResultContent.StepResults.0.StepResultValues.StepNb": {
                        "Value": [1, 1, 1]
                    }
                    # Missing ToolAge and Predrilled
                }
            }
        })
        
        # Custom conf with nonexistent paths for optional fields
        custom_conf = {
            'hole_id': ('Messages', 'Payload', 'nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecxls.ResultManagement.Results.0.ResultMetaData.SerialNumber', 'Value'),
            'position_keys': ('Messages', 'Payload', 'nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecxls.ResultManagement.Results.0.ResultContent.StepResults.0.StepResultValues.Position', 'Value'),
            'torque_keys': ('Messages', 'Payload', 'nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecxls.ResultManagement.Results.0.ResultContent.StepResults.0.StepResultValues.Torque', 'Value'),
            'torque_empty_keys': ('Messages', 'Payload', 'nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecxls.ResultManagement.Results.0.ResultContent.StepResults.0.StepResultValues.TorqueEmpty', 'Value'),
            'step_keys': ('Messages', 'Payload', 'nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecxls.ResultManagement.Results.0.ResultContent.StepResults.0.StepResultValues.StepNb', 'Value'),
            'local_keys': ('Messages', 'Payload', 'missing_local_key', 'Value'),  # Key that doesn't exist
            'predrilled_keys': ('Messages', 'Payload', 'missing_predrilled_key', 'Value')  # Key that doesn't exist
        }
        
        # Call the function
        df = convert_raw_mqtt_to_df(sample_msg, conf=custom_conf)
        
        # Assertions
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 3
        assert df['HOLE_ID'].iloc[0] == "12345"
        assert df['local'].iloc[0] == 0  # Default value when local_keys path doesn't resolve
        assert df['PREDRILLED'].iloc[0] == 1  # Default value when predrilled_keys path doesn't resolve

    def test_convert_raw_mqtt_null_hole_id(self):
        # Create sample MQTT message with null hole_id
        sample_msg = json.dumps({
            "Messages": {
                "Payload": {
                    "nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecxls.ResultManagement.Results.0.ResultMetaData.SerialNumber": {
                        "Value": None
                    },
                    "nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecxls.ResultManagement.Results.0.ResultContent.StepResults.0.StepResultValues.Position": {
                        "Value": [-0.1, -0.2, -0.3]
                    },
                    "nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecxls.ResultManagement.Results.0.ResultContent.StepResults.0.StepResultValues.Torque": {
                        "Value": [1.0, 1.2, 1.5]
                    },
                    "nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecxls.ResultManagement.Results.0.ResultContent.StepResults.0.StepResultValues.TorqueEmpty": {
                        "Value": [0.1, 0.2, 0.3]
                    },
                    "nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecxls.ResultManagement.Results.0.ResultContent.StepResults.0.StepResultValues.StepNb": {
                        "Value": [1, 1, 1]
                    }
                }
            }
        })
        
        # Call the function
        df = convert_raw_mqtt_to_df(sample_msg)
        
        # Assertions
        assert isinstance(df, pd.DataFrame)
        assert df['HOLE_ID'].iloc[0] == "UNKNOWN"  # Default value when hole_id is None
