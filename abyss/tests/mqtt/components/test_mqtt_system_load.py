"""
System-wide load tests for the complete MQTT drilling data analysis pipeline.
"""

import pytest
import time
import threading
import random
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import logging

# Import the classes we need to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from abyss.mqtt.components.drilling_analyser import DrillingDataAnalyser
from abyss.mqtt.components.message_buffer import MessageBuffer
from abyss.mqtt.components.simple_correlator import SimpleMessageCorrelator
from abyss.mqtt.components.message_processor import MessageProcessor
from abyss.mqtt.components.client_manager import MQTTClientManager
from abyss.mqtt.components.result_publisher import ResultPublisher
from abyss.uos_depth_est import TimestampedData


class TestMQTTSystemLoadTests:
    """Load tests for the complete MQTT system under various conditions."""
    
    @pytest.fixture
    def test_config(self):
        """Test configuration for the system."""
        return {
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
            base_message["Messages"]["Payload"]["Trace_Data"] = [random.random() for _ in step_data]
        elif msg_type == 'result':
            base_message["Messages"]["Payload"]["Result_Data"] = {
                "value": random.random() * 100,
                "status": "OK"
            }
        elif msg_type == 'heads':
            base_message["Messages"]["Payload"]["HeadSerial"] = f"HEAD_{tool_id:04d}"
        
        return base_message
    
    def test_high_throughput_message_processing(self, test_config):
        """Test system handling high throughput of correlated messages."""
        # Create system components
        message_buffer = MessageBuffer(
            config=test_config,
            cleanup_interval=60,
            max_buffer_size=10000,
            max_age_seconds=300
        )
        
        correlator = SimpleMessageCorrelator(
            config=test_config,
            time_window=30.0
        )
        
        # Mock components that would require external dependencies
        mock_depth_inference = Mock()
        mock_depth_inference.infer3_common.return_value = [0.0, 1.5, 3.0, 4.5]
        
        mock_data_converter = Mock()
        mock_data_converter.convert_messages_to_df.return_value = Mock(
            iloc=[{'HOLE_ID': 'toolbox1/tool1', 'local': 'result_123'}],
            columns=['HOLE_ID', 'local', 'Step (nb)'],
            __getitem__=lambda self, key: Mock(unique=lambda: [1, 2, 3, 4])
        )
        
        processor = MessageProcessor(
            depth_inference=mock_depth_inference,
            data_converter=mock_data_converter,
            config=test_config,
            algo_version="1.0"
        )
        
        # Test parameters
        num_tools = 5
        messages_per_tool = 100
        correlation_success_count = 0
        processing_success_count = 0
        
        # Generate and process messages
        start_time = time.time()
        
        for tool_id in range(1, num_tools + 1):
            for i in range(messages_per_tool):
                timestamp = time.time() + i * 0.1
                
                # Create correlated message pairs
                result_msg = TimestampedData(
                    _timestamp=timestamp,
                    _data=self.create_mqtt_message(
                        f'test/root/toolbox{tool_id}/tool{tool_id}/Result',
                        tool_id, timestamp, 'result'
                    ),
                    _source=f'test/root/toolbox{tool_id}/tool{tool_id}/Result'
                )
                
                trace_msg = TimestampedData(
                    _timestamp=timestamp + random.uniform(-0.5, 0.5),
                    _data=self.create_mqtt_message(
                        f'test/root/toolbox{tool_id}/tool{tool_id}/Trace',
                        tool_id, timestamp, 'trace', [1, 2, 3, 4]
                    ),
                    _source=f'test/root/toolbox{tool_id}/tool{tool_id}/Trace'
                )
                
                # Add messages to buffer
                message_buffer.add_message(result_msg)
                message_buffer.add_message(trace_msg)
                
                # Occasionally add heads message
                if i % 5 == 0:
                    heads_msg = TimestampedData(
                        _timestamp=timestamp + random.uniform(-0.2, 0.2),
                        _data=self.create_mqtt_message(
                            f'test/root/toolbox{tool_id}/tool{tool_id}/Heads',
                            tool_id, timestamp, 'heads'
                        ),
                        _source=f'test/root/toolbox{tool_id}/tool{tool_id}/Heads'
                    )
                    message_buffer.add_message(heads_msg)
        
        # Process correlations
        def process_callback(matches):
            nonlocal processing_success_count
            result = processor.process_matching_messages(matches)
            if result.success:
                processing_success_count += 1
        
        # Run correlation multiple times to process all messages
        for _ in range(10):
            buffers = message_buffer.get_all_buffers()
            if correlator.find_and_process_matches(buffers, process_callback):
                correlation_success_count += 1
            time.sleep(0.1)
        
        elapsed_time = time.time() - start_time
        
        # Calculate metrics
        total_messages = num_tools * messages_per_tool * 2
        messages_per_second = total_messages / elapsed_time
        
        print(f"Total messages processed: {total_messages}")
        print(f"Messages per second: {messages_per_second:.0f}")
        print(f"Correlation cycles with matches: {correlation_success_count}")
        print(f"Successful processing results: {processing_success_count}")
        
        # Verify system performance
        assert processing_success_count > 0
        assert messages_per_second > 100  # Should handle at least 100 msg/sec
    
    def test_concurrent_multi_tool_processing(self, test_config):
        """Test concurrent processing of messages from multiple tools."""
        # Create shared components
        message_buffer = MessageBuffer(
            config=test_config,
            cleanup_interval=60,
            max_buffer_size=10000,
            max_age_seconds=300
        )
        
        correlator = SimpleMessageCorrelator(
            config=test_config,
            time_window=2.0  # Tighter window for this test
        )
        
        # Tracking
        tool_message_counts = {}
        processed_tools = set()
        lock = threading.Lock()
        
        def generate_tool_messages(tool_id):
            """Generate messages for a specific tool."""
            message_count = 0
            for i in range(50):
                timestamp = time.time()
                
                # Create message pair
                result_msg = TimestampedData(
                    _timestamp=timestamp,
                    _data=self.create_mqtt_message(
                        f'test/root/toolbox{tool_id}/tool{tool_id}/Result',
                        tool_id, timestamp, 'result'
                    ),
                    _source=f'test/root/toolbox{tool_id}/tool{tool_id}/Result'
                )
                
                trace_msg = TimestampedData(
                    _timestamp=timestamp + random.uniform(-0.1, 0.1),
                    _data=self.create_mqtt_message(
                        f'test/root/toolbox{tool_id}/tool{tool_id}/Trace',
                        tool_id, timestamp, 'trace', list(range(i % 5 + 1))
                    ),
                    _source=f'test/root/toolbox{tool_id}/tool{tool_id}/Trace'
                )
                
                # Add with small random delay
                if message_buffer.add_message(result_msg):
                    message_count += 1
                time.sleep(random.uniform(0.001, 0.005))
                
                if message_buffer.add_message(trace_msg):
                    message_count += 1
                time.sleep(random.uniform(0.01, 0.02))
            
            with lock:
                tool_message_counts[tool_id] = message_count
        
        def process_callback(matches):
            """Track which tools get processed."""
            if matches:
                tool_key = matches[0].source.split('/')[1:3]
                tool_id = int(tool_key[0].replace('toolbox', ''))
                with lock:
                    processed_tools.add(tool_id)
        
        # Run concurrent message generation
        num_tools = 10
        with ThreadPoolExecutor(max_workers=num_tools) as executor:
            futures = [
                executor.submit(generate_tool_messages, tool_id)
                for tool_id in range(1, num_tools + 1)
            ]
            
            # Run correlation in parallel
            correlation_thread = threading.Thread(
                target=lambda: [
                    correlator.find_and_process_matches(
                        message_buffer.get_all_buffers(),
                        process_callback
                    ) or time.sleep(0.05)
                    for _ in range(100)
                ]
            )
            correlation_thread.start()
            
            # Wait for message generation
            for future in as_completed(futures):
                future.result()
            
            # Wait for correlation to finish
            correlation_thread.join(timeout=5)
        
        # Verify results
        total_messages = sum(tool_message_counts.values())
        print(f"Total messages generated: {total_messages}")
        print(f"Tools processed: {len(processed_tools)}/{num_tools}")
        print(f"Processed tool IDs: {sorted(processed_tools)}")
        
        # All tools should have been processed
        assert len(processed_tools) >= num_tools * 0.8  # Allow 20% margin
    
    @patch('paho.mqtt.client.Client')
    def test_mqtt_client_stress_test(self, mock_mqtt_client, test_config):
        """Test MQTT client handling under stress conditions."""
        # Setup mock MQTT client
        mock_client_instance = MagicMock()
        mock_mqtt_client.return_value = mock_client_instance
        mock_client_instance.publish.return_value = Mock(rc=0)  # Success
        
        # Create client manager
        message_queue = []
        
        def message_handler(msg):
            message_queue.append(msg)
        
        client_manager = MQTTClientManager(
            config=test_config,
            message_handler=message_handler
        )
        
        # Simulate high-rate message reception
        num_messages = 1000
        start_time = time.time()
        
        # Create mock on_message callback
        result_client = client_manager.create_result_listener()
        on_message = result_client.on_message
        
        # Simulate rapid message arrival
        for i in range(num_messages):
            mock_msg = Mock()
            mock_msg.topic = f'test/root/toolbox{i % 5 + 1}/tool{i % 5 + 1}/Result'
            mock_msg.payload = json.dumps({
                "Messages": {
                    "Payload": {
                        "SourceTimestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "data": f"test_data_{i}"
                    }
                }
            }).encode()
            
            # Call the message handler
            on_message(mock_client_instance, None, mock_msg)
            
            # Simulate realistic message rate
            if i % 100 == 0:
                time.sleep(0.01)
        
        elapsed_time = time.time() - start_time
        
        # Verify performance
        print(f"Messages received: {len(message_queue)}")
        print(f"Messages per second: {len(message_queue) / elapsed_time:.0f}")
        print(f"Processing time: {elapsed_time:.2f}s")
        
        assert len(message_queue) == num_messages
        assert elapsed_time < 10  # Should complete within 10 seconds
    
    def test_warning_system_under_load(self, test_config):
        """Test that warning systems trigger appropriately under load."""
        with patch('logging.warning') as mock_warning:
            # Create buffer with small size to trigger warnings
            message_buffer = MessageBuffer(
                config=test_config,
                cleanup_interval=300,
                max_buffer_size=100,
                max_age_seconds=300
            )
            
            # Generate messages to trigger various warnings
            for i in range(200):
                msg = TimestampedData(
                    _timestamp=time.time() - (i * 2),  # Old messages
                    _data={'test': i},
                    _source=f'test/root/toolbox1/tool1/Result'
                )
                message_buffer.add_message(msg)
                
                # Force metric checks
                if i % 50 == 0:
                    message_buffer._check_message_age_warnings()
                    message_buffer._check_drop_rate_warning()
            
            # Verify warnings were triggered
            warning_messages = [
                call[0][0] for call in mock_warning.call_args_list
            ]
            
            print(f"Total warnings: {len(warning_messages)}")
            print("Warning types:")
            for msg in set(warning_messages):
                if 'capacity' in msg:
                    print("  - Buffer capacity warning")
                elif 'Old messages' in msg:
                    print("  - Old messages warning")
                elif 'drop rate' in msg:
                    print("  - High drop rate warning")
            
            # Should have triggered multiple warning types
            assert len(warning_messages) > 0
            assert any('capacity' in msg for msg in warning_messages)
    
    def test_recovery_from_overload(self, test_config):
        """Test system recovery after overload conditions."""
        message_buffer = MessageBuffer(
            config=test_config,
            cleanup_interval=30,
            max_buffer_size=1000,
            max_age_seconds=60
        )
        
        # Phase 1: Overload the system
        print("Phase 1: Creating overload condition...")
        overload_start = time.time()
        
        for i in range(2000):
            msg = TimestampedData(
                _timestamp=time.time(),
                _data={'test': i, 'large_data': 'x' * 1000},
                _source=f'test/root/toolbox{i % 10}/tool{i % 10}/Result'
            )
            message_buffer.add_message(msg)
        
        overload_metrics = message_buffer.get_metrics()
        print(f"Overload metrics - Dropped: {overload_metrics['messages_dropped']}, "
              f"Drop rate: {overload_metrics['drop_rate']:.2%}")
        
        # Phase 2: Allow system to recover
        print("\nPhase 2: Recovery phase...")
        time.sleep(2)
        message_buffer.cleanup_old_messages()
        
        # Phase 3: Normal operation
        print("\nPhase 3: Testing normal operation after recovery...")
        recovery_success = 0
        
        for i in range(100):
            msg = TimestampedData(
                _timestamp=time.time(),
                _data={'test': i},
                _source=f'test/root/toolbox1/tool1/Result'
            )
            if message_buffer.add_message(msg):
                recovery_success += 1
            time.sleep(0.01)
        
        # Verify recovery
        final_stats = message_buffer.get_buffer_stats()
        final_metrics = message_buffer.get_metrics()
        
        print(f"\nRecovery results:")
        print(f"Success rate after recovery: {recovery_success}/100")
        print(f"Final buffer size: {final_stats['total_messages']}")
        print(f"Total cleanup cycles: {final_metrics['cleanup_cycles']}")
        
        # System should recover and handle new messages
        assert recovery_success >= 90  # 90% success rate after recovery
        assert final_stats['total_messages'] < message_buffer.max_buffer_size