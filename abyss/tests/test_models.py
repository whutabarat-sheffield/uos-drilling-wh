from abyss.mqtt.models import Message, MatchedMessageSet

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