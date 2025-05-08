import pytest
import json
import pandas as pd
from unittest import mock
import logging
from abyss.uos_depth_est_utils import convert_mqtt_to_df

@pytest.fixture
def sample_config():
    return {
        'mqtt': {
            'data_ids': {
                'torque_empty_vals': 'TORQUE_EMPTY',
                'thrust_empty_vals': 'THRUST_EMPTY',
                'step_vals': 'STEP',
                'machine_id': 'MACHINE_ID',
                'result_id': 'RESULT_ID',
                'position': 'POSITION',
                'torque': 'TORQUE',
                'thrust': 'THRUST',
                'step': 'STEP'
            }
        }
    }

@pytest.fixture
def sample_result_msg():
    result_data = {
        'Messages': {
            'Payload': {
                'TORQUE_EMPTY': {'Value': [10.1, 10.2, 10.3]},
                'THRUST_EMPTY': {'Value': [5.1, 5.2, 5.3]},
                'STEP': {'Value': [1, 2, 3]},
                'MACHINE_ID': {'Value': 42},
                'RESULT_ID': {'Value': 123}
            }
        }
    }
    return json.dumps(result_data)

@pytest.fixture
def sample_trace_msg():
    trace_data = {
        'Messages': {
            'Payload': {
                'POSITION': {'Value': [100.1, 200.2, 300.3]},
                'TORQUE': {'Value': [15.1, 15.2, 15.3]},
                'THRUST': {'Value': [8.1, 8.2, 8.3]},
                'STEP': {'Value': [1, 2, 3]}
            }
        }
    }
    return json.dumps(trace_data)

@pytest.fixture
def invalid_json():
    return "{ this is not valid json }"

def test_convert_mqtt_to_df_with_result_only(sample_result_msg, sample_config):
    with mock.patch('logging.info'), mock.patch('logging.debug'), mock.patch('logging.critical'):
        df = convert_mqtt_to_df(result_msg=sample_result_msg, conf=sample_config)
    
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df['Step (nb)']) == [1, 2, 3]
    assert list(df['I Torque Empty (A)']) == [10.1, 10.2, 10.3]
    assert list(df['I Thrust Empty (A)']) == [5.1, 5.2, 5.3]
    assert all(df['HOLE_ID'] == '42')
    assert all(df['local'] == 123)
    assert all(df['PREDRILLED'] == 1)

def test_convert_mqtt_to_df_with_trace_only(sample_trace_msg, sample_config):
    with mock.patch('logging.info'), mock.patch('logging.debug'), mock.patch('logging.critical'):
        df = convert_mqtt_to_df(trace_msg=sample_trace_msg, conf=sample_config)
    
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df['Step (nb)']) == [1, 2, 3]
    assert list(df['Position (mm)']) == [-100.1, -200.2, -300.3]  # Note the negation
    assert list(df['I Torque (A)']) == [15.1, 15.2, 15.3]
    assert list(df['I Thrust (A)']) == [8.1, 8.2, 8.3]

def test_convert_mqtt_to_df_with_both_messages(sample_result_msg, sample_trace_msg, sample_config):
    with mock.patch('logging.info'), mock.patch('logging.debug'), mock.patch('logging.critical'):
        df = convert_mqtt_to_df(result_msg=sample_result_msg, trace_msg=sample_trace_msg, conf=sample_config)
    
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df['Step (nb)']) == [1, 2, 3]
    assert list(df['I Torque Empty (A)']) == [10.1, 10.2, 10.3]
    assert list(df['I Thrust Empty (A)']) == [5.1, 5.2, 5.3]
    assert list(df['Position (mm)']) == [-100.1, -200.2, -300.3]
    assert list(df['I Torque (A)']) == [15.1, 15.2, 15.3]
    assert list(df['I Thrust (A)']) == [8.1, 8.2, 8.3]
    assert all(df['HOLE_ID'] == '42')
    assert all(df['local'] == 123)
    assert all(df['PREDRILLED'] == 1)

def test_convert_mqtt_to_df_with_invalid_json(invalid_json, sample_config):
    with mock.patch('logging.info'), mock.patch('logging.debug'), mock.patch('logging.critical'):
        df = convert_mqtt_to_df(result_msg=invalid_json, conf=sample_config)
    
    assert df is None

def test_convert_mqtt_to_df_with_no_messages(sample_config):
    with mock.patch('logging.info'), mock.patch('logging.debug'), mock.patch('logging.critical'):
        df = convert_mqtt_to_df(conf=sample_config)
    
    assert df is None

def test_convert_mqtt_to_df_with_missing_fields_in_result(sample_config):
    result_data = {
        'Messages': {
            'Payload': {
                'TORQUE_EMPTY': {'Value': [10.1, 10.2, 10.3]},
                # Missing THRUST_EMPTY
                'STEP': {'Value': [1, 2, 3]},
                'MACHINE_ID': {'Value': 42},
                'RESULT_ID': {'Value': 123}
            }
        }
    }
    result_msg = json.dumps(result_data)
    
    with mock.patch('logging.info'), mock.patch('logging.debug'), mock.patch('logging.critical'):
        df = convert_mqtt_to_df(result_msg=result_msg, conf=sample_config)
    
    assert df is None  # Should fail because of missing field