import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from abyss.mqtt_correlation.service import MQTTCorrelationService
from abyss.mqtt_correlation.processing import JSONMessageProcessor

@pytest.mark.asyncio
async def test_basic_correlation_flow():
    # Mock the MQTT client to avoid actual connections
    with patch('aiomqtt.Client') as mock_client_class:
        # Setup mock behavior
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # Create messages context manager
        messages_context = AsyncMock()
        mock_client.messages.return_value = messages_context
        
        # Create the service
        service = MQTTCorrelationService(
            broker="test.mosquitto.org",
            required_types={"sensor", "control"}
        )
        
        # Add processors
        service.add_processor(
            "sensor/+/data",
            JSONMessageProcessor(
                type_id="sensor",
                key_field="device_id"
            )
        )
        
        service.add_processor(
            "control/+/command",
            JSONMessageProcessor(
                type_id="control",
                key_field="device_id"
            )
        )
        
        # Add a callback for matched sets
        callback_mock = AsyncMock()
        service.add_callback(callback_mock)
        
        # Prepare test messages
        sensor_message = MagicMock()
        sensor_message.topic.value = "sensor/temp/data"
        sensor_message.payload = json.dumps({
            "device_id": "device123",
            "temperature": 25.5
        }).encode()
        
        control_message = MagicMock()
        control_message.topic.value = "control/switch/command"
        control_message.payload = json.dumps({
            "device_id": "device123",
            "command": "on"
        }).encode()
        
        # Configure the messages context to yield our test messages
        async def async_iter():
            # This is called when an async for loop iterates through messages
            # Yield one message, then simulate service stopping
            yield sensor_message
            yield control_message
            service.subscriber.running = False
        
        messages_context.__aiter__.return_value = async_iter()
        
        # Start the service (this starts the message loop)
        await service.start()
        
        # Verify the callback was called with a matched set
        callback_mock.assert_called_once()
        
        # Check the matched set contains the expected messages
        called_with = callback_mock.call_args[0][0]
        assert called_with.key == "device123"
        assert "sensor" in called_with.messages
        assert "control" in called_with.messages
        assert called_with.messages["sensor"].data["temperature"] == 25.5
        assert called_with.messages["control"].data["command"] == "on"