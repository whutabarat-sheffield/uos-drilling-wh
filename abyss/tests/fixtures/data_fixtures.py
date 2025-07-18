"""
Data generation fixtures for testing.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any


@pytest.fixture
def sample_drilling_data() -> pd.DataFrame:
    """Create sample drilling data for testing."""
    num_points = 100
    
    # Generate synthetic drilling data
    time_stamps = pd.date_range(start='2024-01-01', periods=num_points, freq='100ms')
    position = np.linspace(0, 50, num_points)  # mm
    torque = np.sin(np.linspace(0, 4*np.pi, num_points)) * 10 + 50  # Nm
    torque_empty = np.random.normal(5, 0.5, num_points)  # Nm
    step_nb = np.repeat(range(1, 6), 20)[:num_points]
    
    df = pd.DataFrame({
        'timestamp': time_stamps,
        'Position (mm)': position,
        'I Torque (A)': torque,
        'I Torque Empty (A)': torque_empty,
        'Step (nb)': step_nb,
        'HOLE_ID': 'toolbox1/tool1',
        'local': 'test_run_001'
    })
    
    return df


@pytest.fixture
def sample_keypoints() -> List[float]:
    """Create sample keypoint data."""
    return [0.0, 5.5, 12.3, 18.7, 25.0, 31.2, 37.8, 44.1, 50.0]


@pytest.fixture
def sample_depth_estimation() -> List[float]:
    """Create sample depth estimation results."""
    return [5.5, 6.8, 6.6, 6.3, 6.2, 6.5, 6.7, 6.1]


@pytest.fixture
def correlation_test_data():
    """Create data for testing message correlation."""
    base_time = datetime.now()
    
    # Create matching messages with slight time offsets
    messages = []
    
    for i in range(5):
        tool_id = f"tool{i+1}"
        timestamp = base_time + timedelta(seconds=i*10)
        
        # Result message
        messages.append({
            'type': 'result',
            'tool_id': tool_id,
            'timestamp': timestamp.timestamp(),
            'data': {'value': 100 + i}
        })
        
        # Trace message (slightly after result)
        messages.append({
            'type': 'trace',
            'tool_id': tool_id,
            'timestamp': (timestamp + timedelta(milliseconds=500)).timestamp(),
            'data': {'steps': [1, 2, 3]}
        })
        
        # Heads message (slightly before result)
        messages.append({
            'type': 'heads',
            'tool_id': tool_id,
            'timestamp': (timestamp - timedelta(milliseconds=200)).timestamp(),
            'data': {'serial': f'HEAD_{i+1:03d}'}
        })
    
    return messages


@pytest.fixture
def large_message_data():
    """Create large messages for performance testing."""
    def _create_large_message(size_mb: float = 1.0) -> Dict[str, Any]:
        # Calculate number of floats needed for desired size
        num_floats = int(size_mb * 1024 * 1024 / 8)  # 8 bytes per float
        
        return {
            'timestamp': datetime.now().isoformat(),
            'data': {
                'array': np.random.random(num_floats).tolist(),
                'metadata': {
                    'size_mb': size_mb,
                    'num_points': num_floats
                }
            }
        }
    
    return _create_large_message


@pytest.fixture
def edge_case_data():
    """Create edge case data for robustness testing."""
    return {
        'empty_data': {},
        'null_values': {'value': None, 'list': [None, None]},
        'special_chars': {'text': 'Test\n\r\t\x00Special'},
        'unicode': {'text': '测试数据 🚀'},
        'nested_deep': {
            'level1': {
                'level2': {
                    'level3': {
                        'level4': {
                            'level5': 'deep_value'
                        }
                    }
                }
            }
        },
        'large_list': list(range(10000)),
        'mixed_types': {
            'int': 42,
            'float': 3.14159,
            'bool': True,
            'str': 'text',
            'list': [1, 'two', 3.0],
            'dict': {'nested': True}
        }
    }