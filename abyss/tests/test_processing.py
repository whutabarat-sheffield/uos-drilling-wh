import pytest
import json
from abyss.mqtt_correlation.processing import JSONMessageProcessor

@pytest.mark.asyncio
async def test_json_processor_basic():
    # Create processor
    processor = JSONMessageProcessor(
        type_id="sensor",
        key_field="device_id"
    )
    
    # Create test payload
    payload = json.dumps({
        "device_id": "device123",
        "temperature": 25.5,
        "humidity": 60
    }).encode("utf-8")
    
    # Process message
    type_id, key, data = await processor.process("sensors/temperature", payload)
    
    # Check results
    assert type_id == "sensor"
    assert key == "device123"
    assert data["temperature"] == 25.5
    assert data["humidity"] == 60
    assert data["_topic"] == "sensors/temperature"

@pytest.mark.asyncio
async def test_json_processor_with_prefix():
    # Create processor with topic prefix
    processor = JSONMessageProcessor(
        type_id="control",
        key_field="device_id",
        topic_prefix="control/"
    )
    
    # Create test payload
    payload = json.dumps({
        "device_id": "device123",
        "command": "on",
        "value": 100
    }).encode("utf-8")
    
    # Process message
    type_id, key, data = await processor.process("control/commands", payload)
    
    # Check results
    assert type_id == "control"
    assert key == "device123"
    assert data["command"] == "on"
    assert data["_topic"] == "commands"  # Prefix removed

@pytest.mark.asyncio
async def test_json_processor_invalid_json():
    # Create processor
    processor = JSONMessageProcessor(
        type_id="sensor",
        key_field="device_id"
    )
    
    # Create invalid payload
    payload = b"This is not JSON"
    
    # Process message
    type_id, key, data = await processor.process("sensors/temperature", payload)
    
    # Check results
    assert type_id == "sensor"
    assert key == "unknown"  # Default key for missing field
    assert "raw" in data
    assert data["raw"] == "This is not JSON"