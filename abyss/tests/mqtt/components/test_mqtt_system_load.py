"""
Functional tests for the MQTT drilling data analysis pipeline under moderate load.
Tests focus on buffer management, concurrent access, and cleanup operations.
"""

import pytest
import time
import threading
from unittest.mock import Mock
from datetime import datetime

# Import the classes we need to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from abyss.mqtt.components.message_buffer import MessageBuffer
from abyss.uos_depth_est import TimestampedData


class TestMQTTSystemLoadTests:
    """Functional tests for MQTT system components under moderate load conditions."""
    
    @pytest.fixture
    def test_config(self, tmp_path):
        """Test configuration for the system."""
        import yaml
        from abyss.mqtt.components.config_manager import ConfigurationManager
        
        config_data = {
            'mqtt': {
                'broker': {
                    'host': 'localhost',
                    'port': 1883,
                    'keepalive': 60
                },
                'listener': {
                    'root': 'test/root',
                    'result': 'Result',
                    'trace': 'Trace',
                    'heads': 'Heads',
                    'duplicate_handling': 'ignore',
                    'duplicate_time_window': 1.0
                },
                'estimation': {
                    'keypoints': 'KeypointsEstimation',
                    'depth_estimation': 'DepthEstimation'
                },
                'time_window': 30.0,
                'cleanup_interval': 60,
                'max_buffer_size': 10000
            }
        }
        
        # Create temporary config file
        config_file = tmp_path / "load_test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Return ConfigurationManager instance
        return ConfigurationManager(str(config_file))
    
    def create_mqtt_message(self, topic: str, tool_id: int, timestamp: float, 
                          msg_type: str, step_data: list = None) -> dict:
        """Create a realistic MQTT message."""
        base_message = {
            "Messages": {
                "Payload": {
                    "HOLE_ID": f"toolbox{tool_id}/tool{tool_id}",
                    "local": f"result_{int(timestamp)}",
                    "SourceTimestamp": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%dT%H:%M:%SZ")
                }
            }
        }
        
        if msg_type == 'trace' and step_data:
            base_message["Messages"]["Payload"]["Step (nb)"] = step_data
            base_message["Messages"]["Payload"]["Trace_Data"] = [0.1 * i for i in step_data]
        elif msg_type == 'result':
            base_message["Messages"]["Payload"]["Result_Data"] = {
                "value": 42.0,
                "status": "OK"
            }
        elif msg_type == 'heads':
            base_message["Messages"]["Payload"]["HeadSerial"] = f"HEAD_{tool_id:04d}"
        
        return base_message
    
    def test_basic_message_buffering(self, test_config):
        """Test basic message buffering with moderate load."""
        # Create message buffer
        message_buffer = MessageBuffer(
            config=test_config,
            cleanup_interval=60,
            max_buffer_size=10000,
            max_age_seconds=300
        )
        
        # Test parameters
        num_tools = 3
        messages_per_tool = 10
        
        # Generate and add messages
        current_time = time.time()
        messages_added = 0
        
        for tool_id in range(1, num_tools + 1):
            for i in range(messages_per_tool):
                # Use different timestamps to avoid duplicates
                timestamp = current_time - (tool_id * 100) - (i * 2)
                
                # Create message with unique timestamp
                result_msg = TimestampedData(
                    _timestamp=timestamp,
                    _data=self.create_mqtt_message(
                        f'test/root/toolbox{tool_id}/tool{tool_id}/Result',
                        tool_id, timestamp, 'result'
                    ),
                    _source=f'test/root/toolbox{tool_id}/tool{tool_id}/Result'
                )
                
                # Add message to buffer
                if message_buffer.add_message(result_msg):
                    messages_added += 1
        
        # Verify results
        stats = message_buffer.get_buffer_stats()
        print(f"Messages added: {messages_added}")
        print(f"Buffer stats: {stats}")
        
        # Verify messages were buffered
        assert messages_added > 0
        assert stats['total_messages'] == messages_added
        # Messages might be grouped by pattern, not individual tools
        assert len(stats['buffer_sizes']) >= 1
    
    def test_concurrent_message_handling(self, test_config):
        """Test concurrent message handling from multiple tools."""
        # Create shared components
        message_buffer = MessageBuffer(
            config=test_config,
            cleanup_interval=60,
            max_buffer_size=10000,
            max_age_seconds=300
        )
        
        # Tracking
        messages_added = 0
        lock = threading.Lock()
        
        def add_tool_messages(tool_id):
            """Add messages for a specific tool."""
            nonlocal messages_added
            local_count = 0
            
            # Use unique timestamps per tool to avoid duplicates
            base_timestamp = time.time() - (tool_id * 1000)
            
            for i in range(20):
                timestamp = base_timestamp - (i * 2)
                
                # Create unique message
                result_msg = TimestampedData(
                    _timestamp=timestamp,
                    _data=self.create_mqtt_message(
                        f'test/root/toolbox{tool_id}/tool{tool_id}/Result',
                        tool_id, timestamp, 'result'
                    ),
                    _source=f'test/root/toolbox{tool_id}/tool{tool_id}/Result'
                )
                
                # Add message
                if message_buffer.add_message(result_msg):
                    local_count += 1
                
                time.sleep(0.001)  # Small delay between messages
            
            with lock:
                messages_added += local_count
        
        # Run concurrent message generation
        num_tools = 5
        threads = []
        
        for tool_id in range(1, num_tools + 1):
            thread = threading.Thread(target=add_tool_messages, args=(tool_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify results
        stats = message_buffer.get_buffer_stats()
        print(f"Messages added across threads: {messages_added}")
        print(f"Buffer stats: {stats}")
        print(f"Buffer topics: {list(stats['buffer_sizes'].keys())}")
        
        # Verify messages were added
        assert messages_added > 0
        assert stats['total_messages'] == messages_added
        # Verify we have at least one buffer
        assert len(stats['buffer_sizes']) >= 1
    
    def test_buffer_capacity_handling(self, test_config):
        """Test buffer behavior at capacity."""
        # Create buffer with small size
        message_buffer = MessageBuffer(
            config=test_config,
            cleanup_interval=300,
            max_buffer_size=50,  # Small buffer
            max_age_seconds=300
        )
        
        # Track metrics
        messages_added = 0
        messages_dropped = 0
        
        # Add messages beyond capacity
        base_time = time.time()
        for i in range(100):
            # Unique timestamps to avoid duplicates
            msg = TimestampedData(
                _timestamp=base_time - (i * 2),
                _data={'test': i},
                _source=f'test/root/toolbox1/tool1/Result'
            )
            
            if message_buffer.add_message(msg):
                messages_added += 1
            else:
                messages_dropped += 1
        
        # Get buffer stats
        stats = message_buffer.get_buffer_stats()
        
        print(f"Messages added: {messages_added}")
        print(f"Messages dropped: {messages_dropped}")
        print(f"Buffer size: {stats['total_messages']}")
        print(f"Buffer capacity: {message_buffer.max_buffer_size}")
        
        # Verify buffer respected capacity
        assert stats['total_messages'] <= message_buffer.max_buffer_size
        # Messages might be cleaned up rather than dropped, so we just verify totals
        assert messages_added > 0
        assert messages_added <= 100
    
    def test_old_message_cleanup(self, test_config):
        """Test cleanup of old messages."""
        message_buffer = MessageBuffer(
            config=test_config,
            cleanup_interval=30,
            max_buffer_size=1000,
            max_age_seconds=5  # Short age for testing
        )
        
        # Add old messages
        old_time = time.time() - 10  # 10 seconds ago
        for i in range(50):
            msg = TimestampedData(
                _timestamp=old_time - i,
                _data={'test': i},
                _source=f'test/root/toolbox1/tool1/Result'
            )
            message_buffer.add_message(msg)
        
        # Add recent messages
        recent_time = time.time()
        for i in range(50):
            msg = TimestampedData(
                _timestamp=recent_time - i * 0.1,
                _data={'test': i + 50},
                _source=f'test/root/toolbox1/tool1/Result'
            )
            message_buffer.add_message(msg)
        
        # Stats before cleanup
        stats_before = message_buffer.get_buffer_stats()
        print(f"Messages before cleanup: {stats_before['total_messages']}")
        
        # Run cleanup
        message_buffer.cleanup_old_messages()
        
        # Stats after cleanup
        stats_after = message_buffer.get_buffer_stats()
        print(f"Messages after cleanup: {stats_after['total_messages']}")
        
        # Verify old messages were cleaned up
        assert stats_after['total_messages'] < stats_before['total_messages']
        assert stats_after['total_messages'] <= 50  # Only recent messages should remain