"""
Central pytest configuration for the Abyss test suite.

This file provides shared fixtures and configuration for all tests.
"""

import pytest
import logging
import tempfile
import yaml
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress noisy loggers during tests
logging.getLogger('paho.mqtt').setLevel(logging.WARNING)


# Configuration Fixtures
@pytest.fixture
def sample_mqtt_config() -> Dict[str, Any]:
    """Provide a standard MQTT configuration for testing."""
    return {
        'mqtt': {
            'broker': {
                'host': 'localhost',
                'port': 1883,
                'username': 'test_user',
                'password': 'test_pass',
                'keepalive': 60
            },
            'listener': {
                'root': 'test/drilling',
                'result': 'Result',
                'trace': 'Trace',
                'heads': 'Heads',
                'time_window': 30.0,
                'cleanup_interval': 60,
                'max_buffer_size': 10000,
                'duplicate_handling': 'ignore',
                'duplicate_time_window': 1.0
            },
            'estimation': {
                'keypoints': 'Keypoints',
                'depth_estimation': 'DepthEstimation'
            },
            'data_ids': {
                'head_id': 'AssetManagement.Assets.Heads.0.Identification.SerialNumber'
            }
        }
    }


@pytest.fixture
def temp_config_file(sample_mqtt_config):
    """Create a temporary config file with test configuration."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_mqtt_config, f)
        config_path = f.name
    
    yield config_path
    
    # Cleanup
    Path(config_path).unlink(missing_ok=True)


@pytest.fixture
def minimal_config() -> Dict[str, Any]:
    """Provide minimal valid configuration for quick tests."""
    return {
        'mqtt': {
            'broker': {'host': 'localhost', 'port': 1883},
            'listener': {
                'root': 'test',
                'result': 'R',
                'trace': 'T',
                'heads': 'H'
            }
        }
    }


# Data Creation Fixtures
@pytest.fixture
def create_timestamped_data():
    """Factory fixture for creating TimestampedData objects."""
    from abyss.uos_depth_est import TimestampedData
    
    def _create(data: Dict[str, Any] = None, source: str = None, timestamp: float = None):
        if data is None:
            data = {'test_id': 1, 'value': 42.0}
        if source is None:
            source = 'test/drilling/toolbox1/tool1/Result'
        if timestamp is None:
            timestamp = time.time()
        
        return TimestampedData(
            _timestamp=timestamp,
            _data=data,
            _source=source
        )
    
    return _create


@pytest.fixture
def create_mqtt_message():
    """Factory fixture for creating realistic MQTT messages."""
    def _create(
        msg_type: str = 'result',
        tool_id: str = 'tool1',
        toolbox_id: str = 'toolbox1',
        hole_id: str = None,
        timestamp: float = None
    ) -> Dict[str, Any]:
        if timestamp is None:
            timestamp = time.time()
        if hole_id is None:
            hole_id = f"{toolbox_id}/{tool_id}"
        
        base_message = {
            "Messages": {
                "Payload": {
                    "HOLE_ID": hole_id,
                    "local": f"result_{int(timestamp)}",
                    "SourceTimestamp": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%dT%H:%M:%SZ")
                }
            }
        }
        
        if msg_type == 'trace':
            base_message["Messages"]["Payload"]["Step (nb)"] = [1, 2, 3, 4]
            base_message["Messages"]["Payload"]["Trace_Data"] = [0.1, 0.2, 0.3, 0.4]
        elif msg_type == 'result':
            base_message["Messages"]["Payload"]["Result_Data"] = {
                "value": 123.45,
                "status": "OK"
            }
        elif msg_type == 'heads':
            base_message["Messages"]["Payload"]["HeadSerial"] = f"HEAD_{tool_id}"
        
        return base_message
    
    return _create


# Component Fixtures
@pytest.fixture
def config_manager(temp_config_file):
    """Create a ConfigurationManager instance for testing."""
    from abyss.mqtt.components.config_manager import ConfigurationManager
    return ConfigurationManager(temp_config_file)


@pytest.fixture
def message_buffer(sample_mqtt_config):
    """Create a MessageBuffer instance for testing."""
    from abyss.mqtt.components.message_buffer import MessageBuffer
    return MessageBuffer(
        config=sample_mqtt_config,
        cleanup_interval=60,
        max_buffer_size=1000,
        max_age_seconds=300
    )


@pytest.fixture
def simple_correlator(sample_mqtt_config):
    """Create a SimpleMessageCorrelator instance for testing."""
    from abyss.mqtt.components.simple_correlator import SimpleMessageCorrelator
    return SimpleMessageCorrelator(
        config=sample_mqtt_config,
        time_window=30.0
    )


# Performance Monitoring Fixtures
@pytest.fixture
def performance_monitor(request):
    """Monitor test performance metrics."""
    import psutil
    import gc
    
    process = psutil.Process()
    gc.collect()
    
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    start_cpu_percent = process.cpu_percent(interval=0.1)
    
    metrics = {}
    yield metrics
    
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    end_cpu_percent = process.cpu_percent(interval=0.1)
    
    duration = end_time - start_time
    memory_delta = end_memory - start_memory
    
    # Log performance metrics
    logger = logging.getLogger('performance')
    logger.info(f"Test Duration: {duration:.3f}s")
    logger.info(f"Memory Delta: {memory_delta:+.1f}MB (Start: {start_memory:.1f}MB, End: {end_memory:.1f}MB)")
    logger.info(f"CPU Usage: Start: {start_cpu_percent:.1f}%, End: {end_cpu_percent:.1f}%")
    
    # Store metrics in the yielded dictionary
    metrics.update({
        'duration': duration,
        'memory_delta': memory_delta,
        'start_memory': start_memory,
        'end_memory': end_memory,
        'cpu_start': start_cpu_percent,
        'cpu_end': end_cpu_percent
    })


# Test Markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "memory: marks tests that check memory usage"
    )
    config.addinivalue_line(
        "markers", "threading: marks tests that verify thread safety"
    )


# Cleanup Fixtures
@pytest.fixture(autouse=True)
def cleanup_test_files(request):
    """Automatically cleanup test-generated files."""
    test_files: List[Path] = []
    
    def register_file(filepath: Path):
        test_files.append(filepath)
    
    request.addfinalizer(lambda: [f.unlink(missing_ok=True) for f in test_files])
    return register_file


# Mock Fixtures for Common Dependencies
@pytest.fixture
def mock_mqtt_client():
    """Provide a mock MQTT client."""
    from unittest.mock import Mock, MagicMock
    
    client = Mock()
    client.publish = MagicMock(return_value=Mock(rc=0))
    client.subscribe = MagicMock()
    client.connect = MagicMock()
    client.disconnect = MagicMock()
    client.loop_start = MagicMock()
    client.loop_stop = MagicMock()
    
    return client


@pytest.fixture
def mock_depth_inference():
    """Provide a mock DepthInference instance."""
    from unittest.mock import Mock
    
    mock = Mock()
    mock.infer3_common.return_value = [0.0, 1.5, 3.0, 4.5, 6.0]
    return mock


# Import fixtures from fixture modules
from .fixtures.config_fixtures import *
from .fixtures.mqtt_fixtures import *
from .fixtures.data_fixtures import *


# Utility Functions
def wait_for_condition(condition_func, timeout=5.0, interval=0.1):
    """Wait for a condition to become true."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if condition_func():
            return True
        time.sleep(interval)
    return False