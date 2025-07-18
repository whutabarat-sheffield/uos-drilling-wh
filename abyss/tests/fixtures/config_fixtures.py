"""
Configuration-related test fixtures.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any


@pytest.fixture
def invalid_yaml_config():
    """Create an invalid YAML configuration file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
mqtt:
  broker:
    host: localhost
    port: 1883
    [invalid yaml syntax here
        """)
        yield f.name
    
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def missing_required_config():
    """Create a config file missing required fields."""
    config = {
        'mqtt': {
            'broker': {
                'host': 'localhost'
                # Missing 'port'
            },
            'listener': {
                # Missing 'root', 'result', 'trace'
                'time_window': 30.0
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        yield f.name
    
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def config_with_env_vars():
    """Create a config that references environment variables."""
    config = {
        'mqtt': {
            'broker': {
                'host': '${MQTT_HOST:localhost}',
                'port': '${MQTT_PORT:1883}',
                'username': '${MQTT_USER:}',
                'password': '${MQTT_PASS:}'
            },
            'listener': {
                'root': '${MQTT_ROOT:drilling/data}',
                'result': 'Result',
                'trace': 'Trace',
                'heads': 'Heads'
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        yield f.name
    
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def multi_environment_config():
    """Create a config with multiple environment settings."""
    config = {
        'environments': {
            'development': {
                'mqtt': {
                    'broker': {'host': 'localhost', 'port': 1883},
                    'listener': {'root': 'dev/drilling'}
                }
            },
            'testing': {
                'mqtt': {
                    'broker': {'host': 'test-broker', 'port': 1883},
                    'listener': {'root': 'test/drilling'}
                }
            },
            'production': {
                'mqtt': {
                    'broker': {'host': 'prod-broker', 'port': 8883},
                    'listener': {'root': 'prod/drilling'}
                }
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        yield f.name
    
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def performance_tuned_config():
    """Create a config optimized for performance testing."""
    return {
        'mqtt': {
            'broker': {
                'host': 'localhost',
                'port': 1883,
                'keepalive': 60,
                'max_inflight_messages': 1000,
                'max_queued_messages': 10000
            },
            'listener': {
                'root': 'perf/test',
                'result': 'R',
                'trace': 'T',
                'heads': 'H',
                'time_window': 5.0,
                'cleanup_interval': 300,
                'max_buffer_size': 100000,
                'duplicate_handling': 'ignore',
                'batch_size': 1000
            },
            'performance': {
                'thread_pool_size': 10,
                'enable_profiling': True,
                'metrics_interval': 10
            }
        }
    }