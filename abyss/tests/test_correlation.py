import pytest
import time
from abyss.mqtt.async_components.models import Message
from abyss.mqtt.async_components.correlation import CorrelationEngine

def test_correlation_basic_matching():
    # Create correlation engine requiring sensor and control messages
    engine = CorrelationEngine(required_types={"sensor", "control"})
    
    # Create messages with same key
    msg1 = Message(type_id="sensor", key="device123", data={"temperature": 25.5})
    msg2 = Message(type_id="control", key="device123", data={"command": "on"})
    
    # Add first message - should not complete a match
    result1 = engine.add_message(msg1)
    assert result1 is None
    
    # Add second message - should complete a match
    result2 = engine.add_message(msg2)
    assert result2 is not None
    assert result2.key == "device123"
    assert set(result2.messages.keys()) == {"sensor", "control"}
    
    # Check that match can be retrieved
    assert len(engine.complete_matches) == 1
    match = engine.get_next_match()
    assert match is not None
    assert match.key == "device123"
    
    # Should be empty now
    assert engine.get_next_match() is None

def test_correlation_different_keys():
    # Create correlation engine
    engine = CorrelationEngine(required_types={"sensor", "control"})
    
    # Create messages with different keys
    msg1 = Message(type_id="sensor", key="device123", data={"temperature": 25.5})
    msg2 = Message(type_id="control", key="device456", data={"command": "on"})
    
    # Add messages - neither should complete a match
    assert engine.add_message(msg1) is None
    assert engine.add_message(msg2) is None
    assert engine.get_next_match() is None

def test_correlation_timeout():
    # Create correlation engine with short timeout
    engine = CorrelationEngine(required_types={"sensor", "control"}, match_timeout=0.1)
    
    # Create message
    msg1 = Message(type_id="sensor", key="device123", data={"temperature": 25.5})
    
    # Add message and verify it's in partial matches
    engine.add_message(msg1)
    assert "device123" in engine.partial_matches
    
    # Wait for timeout and force cleanup
    time.sleep(0.2)
    engine.last_cleanup = 0  # Force cleanup on next add
    
    # Add unrelated message to trigger cleanup
    msg2 = Message(type_id="sensor", key="device456", data={"temperature": 26.5})
    engine.add_message(msg2)
    
    # Check that expired match was removed
    assert "device123" not in engine.partial_matches