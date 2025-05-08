import sys
print(f"Python path: {sys.path}")
from pathlib import Path

import pytest
# from src.models import Message, MatchedMessageSet

# Direct path manipulation
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root / 'src'))
# print(f"Added {project_root / 'src'} to path")
# print(f"Python path: {sys.path}")

# Direct module import
import importlib.util
module_path = Path(__file__).parent.parent / 'src' / 'abyss' / 'mqtt_correlation' / 'models.py'
spec = importlib.util.spec_from_file_location("models", module_path)
if spec is None:
    raise ImportError(f"Could not load spec for module at {module_path}")
if spec.loader is None:
    raise ImportError(f"Could not load loader for module at {module_path}")
models = importlib.util.module_from_spec(spec)
sys.modules["models"] = models
spec.loader.exec_module(models)

Message = models.Message
MatchedMessageSet = models.MatchedMessageSet

# from abyss.mqtt_correlation.models import Message, MatchedMessageSet

def test_message_creation():
    # Create a basic message
    msg = Message(type_id="sensor", key="device123", data={"temperature": 25.5})
    
    # Check fields
    assert msg.type_id == "sensor"
    assert msg.key == "device123"
    assert msg.data == {"temperature": 25.5}
    assert isinstance(msg.metadata, dict)
    assert isinstance(msg.timestamp, float)
    assert isinstance(msg.id, str)

def test_matched_message_set():
    # Create messages
    msg1 = Message(type_id="sensor", key="device123", data={"temperature": 25.5})
    msg2 = Message(type_id="control", key="device123", data={"command": "on"})
    
    # Create matched set
    matched_set = MatchedMessageSet(
        key="device123",
        messages={"sensor": msg1, "control": msg2}
    )
    
    # Check fields
    assert matched_set.key == "device123"
    assert matched_set.messages["sensor"] == msg1
    assert matched_set.messages["control"] == msg2
    
    # Test completeness check
    assert matched_set.is_complete({"sensor", "control"})
    assert not matched_set.is_complete({"sensor", "control", "status"})