import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch, call
from abyss.mqtt.service import MQTTCorrelationService
from abyss.mqtt.processing import JSONMessageProcessor
from abyss.mqtt.models import MatchedMessageSet

@pytest.mark.asyncio
async def test_basic_correlation_flow():
    """Test the end-to-end correlation flow with mocked MQTT messages."""

    callback_completed = asyncio.Future()

    async def test_callback(matched_set: MatchedMessageSet):
        if not callback_completed.done():
            callback_completed.set_result(matched_set)

    with patch('aiomqtt.Client') as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        messages_context = AsyncMock()
        mock_client.messages.return_value = messages_context

        service = MQTTCorrelationService(
            broker="test.mosquitto.org",
            required_types={"sensor", "control"},
            match_timeout=5.0
        )

        service.add_processor("sensor/+/data", JSONMessageProcessor(type_id="sensor", key_field="device_id"))
        service.add_processor("control/+/command", JSONMessageProcessor(type_id="control", key_field="device_id"))
        service.add_callback(test_callback)

        sensor_message = MagicMock(topic="sensor/temp/data", payload=json.dumps({"device_id": "device123", "temperature": 25.5}).encode())
        control_message = MagicMock(topic="control/switch/command", payload=json.dumps({"device_id": "device123", "command": "on"}).encode())

        async def async_iter():
            for message in [sensor_message, control_message]:
                yield message
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.1)

        messages_context.__aiter__.return_value = async_iter()

        service_task = asyncio.create_task(service.start())

        matched_set = await asyncio.wait_for(callback_completed, timeout=2.0)

        mock_client.connect.assert_called_once()
        mock_client.subscribe.assert_has_calls([
            call("sensor/+/data"),
            call("control/+/command")
        ], any_order=True)

        assert matched_set.key == "device123"
        assert matched_set.messages["sensor"].data["temperature"] == 25.5
        assert matched_set.messages["control"].data["command"] == "on"

        await service.stop()
        service_task.cancel()
        try:
            await service_task
        except asyncio.CancelledError:
            pass

# Add additional test cases for error conditions and edge cases
@pytest.mark.asyncio
async def test_correlation_timeout():
    """Test the behavior when correlation times out."""
    # Create timeout tracking
    timeout_occurred = asyncio.Future()
    service_task = None
    
    async def test_callback(matched_set: MatchedMessageSet):
        # This shouldn't be called in this test
        assert False, "Callback should not be called for timeout test"
    
    # We'll mock logs to capture the timeout warning
    with patch('aiomqtt.Client') as mock_client_class, \
         patch('logging.warning') as mock_warning:
        
        # Setup mock client
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # Create messages context manager
        messages_context = AsyncMock()
        mock_client.messages.return_value = messages_context
        
        # Create the service with a very short timeout
        service = MQTTCorrelationService(
            broker="test.mosquitto.org",
            required_types={"sensor", "control"},
            match_timeout=0.2  # Very short timeout for test
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
        
        # Add callback
        service.add_callback(test_callback)
        
        # Only provide one message to trigger a timeout
        sensor_message = MagicMock()
        sensor_message.topic.value = "sensor/temp/data"
        sensor_message.payload = json.dumps({
            "device_id": "device123",
            "temperature": 25.5
        }).encode()
        
        # Configure async iterator with only one message
        async def async_iter():
            yield sensor_message
            # Wait long enough for timeout
            await asyncio.sleep(0.3)
            # Set the future to indicate timeout has occurred
            if not timeout_occurred.done():
                timeout_occurred.set_result(True)
            # Wait a bit more and then stop
            await asyncio.sleep(0.3)
            # Then stop
            await service.stop()
            
        # Configure messages context to use our iterator
        messages_context.__aiter__.return_value = async_iter()
        
        # Start the service before waiting for timeout
        service_task = asyncio.create_task(service.start())
        
        # Wait for timeout to occur
        try:
            await asyncio.wait_for(timeout_occurred, timeout=1.0)
        except asyncio.TimeoutError:
            pass  # If we time out here, the test will fail in the next assertion
            
        # Verify timeout was logged (with more flexible matching)
        assert any("timeout" in str(args).lower() and "correlation" in str(args).lower()
                  for args, _ in mock_warning.call_args_list), f"No timeout warning logged: {mock_warning.call_args_list}"
        
        await asyncio.sleep(0.5)  # Wait for timeout and cleanup
        
        # Verify timeout was logged
        assert any("Correlation timeout" in str(args) 
                  for args, _ in mock_warning.call_args_list)
        
        # Cleanup
        await service.stop()
        if service_task and not service_task.done():
            service_task.cancel()
            try:
                await service_task
            except asyncio.CancelledError:
                pass