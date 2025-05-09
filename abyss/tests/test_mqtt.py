import pytest
import asyncio
import ssl
from unittest.mock import AsyncMock, MagicMock, patch, call
from pathlib import Path

import aiomqtt
from abyss.mqtt.mqtt import MQTTSubscriber
from abyss.mqtt.confighandler import ConfigHandler

# Path to test configuration
TEST_CONFIG_PATH = Path(__file__).parent.parent / "src/abyss/run/mqtt_conf.yaml"

#-----------------------------------------
# Fixtures
#-----------------------------------------

@pytest.fixture
def mqtt_config():
    """Fixture providing MQTT configuration from yaml file."""
    return ConfigHandler.from_yaml(TEST_CONFIG_PATH)

@pytest.fixture
def mock_mqtt_client():
    """Fixture providing a mocked MQTT client."""
    mock_client = AsyncMock()
    
    # Setup context manager behavior
    messages_context = AsyncMock()
    mock_client.messages = messages_context
    messages_context.__aenter__.return_value = messages_context
    messages_context.__aexit__.return_value = None
    
    # Default empty iterator
    messages_context.__aiter__.return_value = (msg for msg in [])
    
    return mock_client

#-----------------------------------------
# Basic Initialization Tests
#-----------------------------------------

def test_init_with_default_values():
    """Test initialization with only the required broker parameter"""
    subscriber = MQTTSubscriber(broker="test.mosquitto.org")
    
    assert subscriber.broker == "test.mosquitto.org"
    assert subscriber.port == 1883  # Default port
    assert subscriber.client_id is None
    assert subscriber.username is None
    assert subscriber.password is None
    assert subscriber.use_tls is False
    assert subscriber.subscribed_topics == set()
    assert subscriber.message_callbacks == {}
    assert subscriber.running is False

def test_init_with_custom_values():
    """Test initialization with custom values for all parameters"""
    subscriber = MQTTSubscriber(
        broker="broker.hivemq.com",
        port=8883,
        client_id="test-client",
        username="user",
        password="pass",
        use_tls=True
    )
    
    assert subscriber.broker == "broker.hivemq.com"
    assert subscriber.port == 8883
    assert subscriber.client_id == "test-client"
    assert subscriber.username == "user"
    assert subscriber.password == "pass"
    assert subscriber.use_tls is True
    assert subscriber.subscribed_topics == set()
    assert subscriber.message_callbacks == {}
    assert subscriber.running is False

#-----------------------------------------
# ConfigHandler Integration Tests
#-----------------------------------------

def test_init_with_config(mqtt_config):
    """Test initializing MQTTSubscriber with configuration from ConfigHandler."""
    subscriber = MQTTSubscriber(
        broker=mqtt_config.mqtt.broker.host,
        port=mqtt_config.mqtt.broker.port,
        username=mqtt_config.mqtt.broker.username,
        password=mqtt_config.mqtt.broker.password
    )
    
    # Verify subscriber is initialized with config values
    assert subscriber.broker == mqtt_config.mqtt.broker.host
    assert subscriber.port == mqtt_config.mqtt.broker.port
    assert subscriber.username == mqtt_config.mqtt.broker.username
    assert subscriber.password == mqtt_config.mqtt.broker.password

def test_add_topic_handlers_from_config(mqtt_config):
    """Test adding topic handlers using topics from ConfigHandler."""
    subscriber = MQTTSubscriber(broker=mqtt_config.mqtt.broker.host)
    
    # Mock callback
    async def mock_callback(topic, payload):
        pass
    
    # Add handlers for topics defined in config
    result_topic = mqtt_config.mqtt.listener.get_result_topic()
    trace_topic = mqtt_config.mqtt.listener.get_trace_topic()
    
    subscriber.add_topic_handler(result_topic, mock_callback)
    subscriber.add_topic_handler(trace_topic, mock_callback)
    
    # Verify topics were added
    assert result_topic in subscriber.subscribed_topics
    assert trace_topic in subscriber.subscribed_topics
    assert subscriber.message_callbacks[result_topic] == mock_callback
    assert subscriber.message_callbacks[trace_topic] == mock_callback

#-----------------------------------------
# Topic Matching Tests
#-----------------------------------------

def test_topic_matching():
    """Test the _topic_matches method with various patterns."""
    subscriber = MQTTSubscriber(broker="test.mosquitto.org")
    
    # Exact match
    assert subscriber._topic_matches("a/b/c", "a/b/c") is True
    
    # Single-level wildcard
    assert subscriber._topic_matches("a/b/c", "a/+/c") is True
    assert subscriber._topic_matches("a/b/c", "+/+/+") is True
    assert subscriber._topic_matches("a/b/c", "+/b/+") is True
    assert subscriber._topic_matches("a/b/c", "a/+/+") is True
    
    # Multi-level wildcard
    assert subscriber._topic_matches("a/b/c", "a/#") is True
    assert subscriber._topic_matches("a/b/c/d", "a/b/#") is True
    
    # Non-matches
    assert subscriber._topic_matches("a/b/c", "a/d/c") is False
    assert subscriber._topic_matches("a/b/c", "a/b") is False
    assert subscriber._topic_matches("a/b", "a/b/c") is False
    
    # Edge cases
    assert subscriber._topic_matches("", "") is True
    assert subscriber._topic_matches("a", "+") is True
    assert subscriber._topic_matches("a", "#") is True

#-----------------------------------------
# Async Core Functionality Tests
#-----------------------------------------

@pytest.mark.asyncio
async def test_subscription_at_startup(mock_mqtt_client):
    """Test that topics are subscribed to on startup."""
    # Create subscriber instance with mocked client
    with patch('aiomqtt.Client', return_value=mock_mqtt_client):
        subscriber = MQTTSubscriber(broker="test.mosquitto.org")
        
        # Add topic handlers
        callback = AsyncMock()
        subscriber.add_topic_handler("test/topic1", callback)
        subscriber.add_topic_handler("test/topic2", callback)
        
        # Set up a future to control when to stop the subscriber
        stop_future = asyncio.Future()
        
        # Start subscriber in background task
        async def run_subscriber():
            subscriber.running = True
            try:
                await subscriber.start()
            except asyncio.CancelledError:
                pass
        
        task = asyncio.create_task(run_subscriber())
        
        try:
            # Wait for client to be accessed and subscriptions to be made
            await asyncio.sleep(0.5)
            
            # Check that we subscribed to both topics
            assert mock_mqtt_client.subscribe.await_count == 2
            mock_mqtt_client.subscribe.assert_has_awaits([
                call("test/topic1"),
                call("test/topic2")
            ], any_order=True)
        finally:
            # Clean up
            subscriber.running = False
            task.cancel()
            await asyncio.wait([task], timeout=0.5)

@pytest.mark.asyncio
async def test_message_handling(mqtt_config):
    """Test message handling with a real callback."""
    # Create subscriber
    subscriber = MQTTSubscriber(
        broker=mqtt_config.mqtt.broker.host,
        port=mqtt_config.mqtt.broker.port
    )
    
    # Create a mock client with messages
    mock_client = AsyncMock()
    
    # Setup context manager behavior
    messages_context = AsyncMock()
    mock_client.messages = messages_context
    messages_context.__aenter__.return_value = messages_context
    messages_context.__aexit__.return_value = None
    
    # Create mock messages
    result_topic = mqtt_config.mqtt.listener.get_result_topic()
    test_payload = b'{"test": "data"}'
    
    mock_message = MagicMock()
    mock_message.topic.value = result_topic
    mock_message.payload = test_payload
    
    # Setup async message iterator
    async def mock_message_gen():
        yield mock_message
        # Stop after first message
        subscriber.running = False
        
    messages_context.__aiter__.return_value = mock_message_gen()
    
    # Setup callback to verify message reception
    received_messages = []
    
    async def test_callback(topic, payload):
        received_messages.append((topic, payload))
    
    # Add the callback for our test topic
    subscriber.add_topic_handler(result_topic, test_callback)
    
    # Run the subscriber with our mock client
    with patch('aiomqtt.Client', return_value=mock_client):
        subscriber.running = True
        await subscriber.start()
    
    # Verify the message was received and callback was called
    assert len(received_messages) == 1
    assert received_messages[0][0] == result_topic
    assert received_messages[0][1] == test_payload

@pytest.mark.asyncio
async def test_with_config_topics_and_messages(mqtt_config):
    """Test subscribing to topics from config and receiving messages."""
    # Create the subscriber
    subscriber = MQTTSubscriber(
        broker=mqtt_config.mqtt.broker.host,
        port=mqtt_config.mqtt.broker.port,
        username=mqtt_config.mqtt.broker.username or None,
        password=mqtt_config.mqtt.broker.password or None
    )
    
    # Get topics from config
    result_topic = mqtt_config.mqtt.listener.get_result_topic()
    trace_topic = mqtt_config.mqtt.listener.get_trace_topic()
    
    # Track received messages
    result_received = asyncio.Future()
    trace_received = asyncio.Future()
    
    # Define callbacks
    async def result_callback(topic, payload):
        if not result_received.done():
            result_received.set_result((topic, payload))
    
    async def trace_callback(topic, payload):
        if not trace_received.done():
            trace_received.set_result((topic, payload))
    
    # Add topic handlers
    subscriber.add_topic_handler(result_topic, result_callback)
    subscriber.add_topic_handler(trace_topic, trace_callback)
    
    # Create mock client
    mock_client = AsyncMock()
    
    # Setup context manager
    messages_context = AsyncMock()
    mock_client.messages = messages_context
    messages_context.__aenter__.return_value = messages_context
    messages_context.__aexit__.return_value = None
    
    # Create mock messages
    result_message = MagicMock()
    result_message.topic.value = result_topic
    result_message.payload = b'{"result": "data"}'
    
    trace_message = MagicMock()
    trace_message.topic.value = trace_topic
    trace_message.payload = b'{"trace": "data"}'
    
    # Setup message sequence
    async def message_generator():
        yield result_message
        # Give time for the callback to execute
        await asyncio.sleep(0.1)
        yield trace_message
        # Give time for the callback to execute
        await asyncio.sleep(0.1)
        # All messages delivered, stop
        subscriber.running = False
        
    messages_context.__aiter__.return_value = message_generator()
    
    # Run subscriber with mocked client
    client_mock = MagicMock(return_value=mock_client)
    with patch('aiomqtt.Client', client_mock):
        subscriber.running = True
        task = asyncio.create_task(subscriber.start())
        
        try:
            # Wait for both messages or timeout
            await asyncio.wait_for(asyncio.gather(
                result_received, trace_received
            ), timeout=1.0)
            
            # Verify messages were received correctly
            result_data = result_received.result()
            trace_data = trace_received.result()
            
            assert result_data[0] == result_topic
            assert result_data[1] == b'{"result": "data"}'
            
            assert trace_data[0] == trace_topic
            assert trace_data[1] == b'{"trace": "data"}'
            
            # Verify client was configured correctly
            client_mock.assert_called_once_with(
                hostname=mqtt_config.mqtt.broker.host,
                port=mqtt_config.mqtt.broker.port,
                identifier=None,
                username=mqtt_config.mqtt.broker.username or None,
                password=mqtt_config.mqtt.broker.password or None
            )
            
            # Verify subscriptions
            assert mock_client.subscribe.await_count >= 2
            mock_client.subscribe.assert_has_awaits([
                call(result_topic),
                call(trace_topic)
            ], any_order=True)
            
        finally:
            # Clean up
            subscriber.running = False
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling during message processing."""
    subscriber = MQTTSubscriber(broker="test.mosquitto.org")
    
    # Add a callback that raises an exception
    async def failing_callback(topic, payload):
        raise ValueError("Test exception")
    
    subscriber.add_topic_handler("test/topic", failing_callback)
    
    # Mock client with a message that will trigger the failing callback
    mock_client = AsyncMock()
    
    # Setup context manager
    messages_context = AsyncMock()
    mock_client.messages = messages_context
    messages_context.__aenter__.return_value = messages_context
    messages_context.__aexit__.return_value = None
    
    # Create mock message
    mock_message = MagicMock()
    mock_message.topic.value = "test/topic"
    mock_message.payload = b"test"
    
    # Make the iterator yield our message then raise exception
    async def message_generator():
        yield mock_message
        # Message delivered, continue running to see if error is handled
        await asyncio.sleep(0.1)
        # Stop after checking error handling
        subscriber.running = False
        
    messages_context.__aiter__.return_value = message_generator()
    
    # Capture print output to verify error handling
    with patch('builtins.print') as mock_print, \
         patch('aiomqtt.Client', return_value=mock_client):
        
        subscriber.running = True
        await subscriber.start()
        
        # Verify the error was properly handled (printed)
        mock_print.assert_called_with("MQTT error: Test exception")
        
        # Running should be set to False when an exception occurs
        assert subscriber.running is False

@pytest.mark.asyncio
async def test_stop_signal():
    """Test that the stop signal properly terminates message processing."""
    subscriber = MQTTSubscriber(broker="test.mosquitto.org")
    
    # Track number of messages processed
    processed_count = 0
    
    async def counting_callback(topic, payload):
        nonlocal processed_count
        processed_count += 1
    
    subscriber.add_topic_handler("test/topic", counting_callback)
    
    # Mock client with infinite messages until stopped
    mock_client = AsyncMock()
    
    # Setup context manager
    messages_context = AsyncMock()
    mock_client.messages = messages_context
    messages_context.__aenter__.return_value = messages_context
    messages_context.__aexit__.return_value = None
    
    # Create mock message that keeps yielding until stopped
    mock_message = MagicMock()
    mock_message.topic.value = "test/topic"
    mock_message.payload = b"test"
    
    # A generator that yields messages until stopped
    async def infinite_message_generator():
        while subscriber.running:
            yield mock_message
            await asyncio.sleep(0.01)  # Small delay to prevent tight loop
        
    messages_context.__aiter__.return_value = infinite_message_generator()
    
    # Run subscriber and stop it after processing some messages
    with patch('aiomqtt.Client', return_value=mock_client):
        subscriber.running = True
        task = asyncio.create_task(subscriber.start())
        
        # Let it process a few messages
        await asyncio.sleep(0.05)
        
        # Stop the subscriber
        subscriber.stop()
        
        # Wait for task to complete
        try:
            await asyncio.wait_for(task, timeout=0.5)
        except asyncio.TimeoutError:
            # If it doesn't stop properly, cancel it
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            pytest.fail("Subscriber didn't stop when requested")
        
        # Verify that messages were processed before stopping
        assert processed_count > 0
        # Verify that the subscriber is stopped
        assert subscriber.running is False

# Note: This method implementation should be in the MQTTSubscriber class, not in the test file.
# Remove this code from here as it's likely a copy that was accidentally added to the test file.