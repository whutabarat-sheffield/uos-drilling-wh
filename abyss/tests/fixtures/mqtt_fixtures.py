"""
MQTT-related test fixtures.
"""

import pytest
import json
from datetime import datetime
from typing import Dict, Any


@pytest.fixture
def mqtt_result_payload() -> bytes:
    """Create a standard MQTT result message payload."""
    message = {
        "Messages": {
            "Payload": {
                "HOLE_ID": "toolbox1/tool1",
                "local": "result_123",
                "SourceTimestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "Result_Data": {
                    "value": 123.45,
                    "status": "OK"
                }
            }
        }
    }
    return json.dumps(message).encode('utf-8')


@pytest.fixture
def mqtt_trace_payload() -> bytes:
    """Create a standard MQTT trace message payload."""
    message = {
        "Messages": {
            "Payload": {
                "HOLE_ID": "toolbox1/tool1",
                "local": "trace_123",
                "SourceTimestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "Step (nb)": [1, 2, 3, 4, 5],
                "Trace_Data": [0.1, 0.2, 0.3, 0.4, 0.5]
            }
        }
    }
    return json.dumps(message).encode('utf-8')


@pytest.fixture
def mqtt_heads_payload() -> bytes:
    """Create a standard MQTT heads message payload."""
    message = {
        "Messages": {
            "Payload": {
                "HOLE_ID": "toolbox1/tool1",
                "local": "heads_123",
                "SourceTimestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "HeadSerial": "HEAD_001",
                "HeadType": "TypeA",
                "HeadStatus": "Active"
            }
        }
    }
    return json.dumps(message).encode('utf-8')


@pytest.fixture
def mqtt_topic_factory():
    """Factory for creating MQTT topics."""
    def _create_topic(
        root: str = "test/drilling",
        toolbox_id: str = "toolbox1",
        tool_id: str = "tool1",
        message_type: str = "Result"
    ) -> str:
        return f"{root}/{toolbox_id}/{tool_id}/{message_type}"
    
    return _create_topic


@pytest.fixture
def mqtt_message_factory():
    """Factory for creating complete MQTT messages with topic and payload."""
    def _create_message(
        topic: str,
        payload: Dict[str, Any],
        qos: int = 0,
        retain: bool = False
    ) -> Dict[str, Any]:
        return {
            "topic": topic,
            "payload": json.dumps(payload).encode('utf-8'),
            "qos": qos,
            "retain": retain,
            "timestamp": datetime.now().isoformat()
        }
    
    return _create_message