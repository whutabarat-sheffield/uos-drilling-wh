"""Tests specifically for HeadId field inclusion in ResultPublisher messages.

This module focuses exclusively on verifying that the HeadId field from ProcessingResult
is correctly included in all published MQTT messages across different result types
(success, error, insufficient data). For general ResultPublisher functionality tests,
see test_result_publisher.py.
"""

import pytest
import json
import tempfile
import yaml
import os
from unittest.mock import MagicMock, patch
from abyss.mqtt.components.result_publisher import ResultPublisher
from abyss.mqtt.components.message_processor import ProcessingResult
from abyss.mqtt.components.config_manager import ConfigurationManager


class TestResultPublisherHeadId:
    """Tests for head_id inclusion in ResultPublisher
    
    Test coverage:
    - test_successful_result_includes_head_id: Verifies HeadId in successful results
    - test_successful_result_handles_null_head_id: Verifies null HeadId handling
    - test_topic_construction_with_head_id: Verifies topic construction (not HeadId specific)
    - test_insufficient_data_result_includes_head_id: Verifies HeadId with insufficient data
    - test_error_result_includes_head_id: Verifies HeadId in error results
    - test_error_result_with_null_head_id: Verifies null HeadId in error results
    """
    
    @pytest.fixture
    def mock_mqtt_client(self):
        """Mock MQTT client"""
        client = MagicMock()
        client.publish.return_value.rc = 0  # MQTT_ERR_SUCCESS
        return client
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration"""
        config_data = {
            'mqtt': {
                'broker': {
                    'host': 'localhost',
                    'port': 1883,
                    'username': '',
                    'password': ''
                },
                'listener': {
                    'root': 'OPCPUBSUB',
                    'result': 'ResultManagement',
                    'trace': 'ResultManagement/Trace',
                    'heads': 'AssetManagement'
                },
                'estimation': {
                    'keypoints': 'DepthEstimation/KeyPoints',
                    'depth_estimation': 'DepthEstimation/DepthEstimation'
                }
            }
        }
        
        # Create temporary config file and ConfigurationManager
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            config_manager = ConfigurationManager(config_file)
            yield config_manager
        finally:
            os.unlink(config_file)
    
    @pytest.fixture
    def result_publisher(self, mock_mqtt_client, mock_config):
        """Create ResultPublisher with mocked dependencies"""
        return ResultPublisher(mock_mqtt_client, mock_config)
    
    @pytest.fixture
    def processing_result_with_head_id(self):
        """Create ProcessingResult with head_id"""
        return ProcessingResult(
            success=True,
            keypoints=[10.0, 20.0, 30.0],
            depth_estimation=[10.0, 10.0],
            machine_id='TEST_MACHINE_123',
            result_id='RESULT_456',
            head_id='HEAD_789'
        )
    
    @pytest.fixture
    def processing_result_without_head_id(self):
        """Create ProcessingResult without head_id"""
        return ProcessingResult(
            success=True,
            keypoints=[10.0, 20.0, 30.0],
            depth_estimation=[10.0, 10.0],
            machine_id='TEST_MACHINE_123',
            result_id='RESULT_456',
            head_id=None
        )
    
    def test_successful_result_includes_head_id(self, result_publisher, processing_result_with_head_id):
        """Test that successful results include HeadId in published data"""
        toolbox_id = "TOOLBOX123"
        tool_id = "TOOL456"
        timestamp = 1234567890.0
        algo_version = "1.0"
        
        # Publish the result
        result_publisher.publish_processing_result(
            processing_result_with_head_id, toolbox_id, tool_id, timestamp, algo_version
        )
        
        # Method returns None, not True
        
        # Verify MQTT client was called twice (keypoints + depth estimation)
        assert result_publisher.mqtt_client.publish.call_count == 2
        
        # Get the published messages
        calls = result_publisher.mqtt_client.publish.call_args_list
        
        # Check keypoints message
        keypoints_topic, keypoints_payload = calls[0][0]
        keypoints_data = json.loads(keypoints_payload)
        
        assert 'HeadId' in keypoints_data
        assert keypoints_data['HeadId'] == 'HEAD_789'
        assert keypoints_data['Value'] == [10.0, 20.0, 30.0]
        assert keypoints_data['MachineId'] == 'TEST_MACHINE_123'
        assert keypoints_data['ResultId'] == 'RESULT_456'
        assert keypoints_data['AlgoVersion'] == '1.0'
        
        # Check depth estimation message
        depth_topic, depth_payload = calls[1][0]
        depth_data = json.loads(depth_payload)
        
        assert 'HeadId' in depth_data
        assert depth_data['HeadId'] == 'HEAD_789'
        assert depth_data['Value'] == [10.0, 10.0]
        assert depth_data['MachineId'] == 'TEST_MACHINE_123'
        assert depth_data['ResultId'] == 'RESULT_456'
        assert depth_data['AlgoVersion'] == '1.0'
    
    def test_successful_result_handles_null_head_id(self, result_publisher, processing_result_without_head_id):
        """Test that results with null head_id still include the HeadId field"""
        toolbox_id = "TOOLBOX123"
        tool_id = "TOOL456"
        timestamp = 1234567890.0
        algo_version = "1.0"
        
        # Publish the result
        result_publisher.publish_processing_result(
            processing_result_without_head_id, toolbox_id, tool_id, timestamp, algo_version
        )
        
        # Method returns None, not True
        
        # Get the published messages
        calls = result_publisher.mqtt_client.publish.call_args_list
        
        # Check keypoints message
        keypoints_topic, keypoints_payload = calls[0][0]
        keypoints_data = json.loads(keypoints_payload)
        
        assert 'HeadId' in keypoints_data
        assert keypoints_data['HeadId'] is None
        
        # Check depth estimation message
        depth_topic, depth_payload = calls[1][0]
        depth_data = json.loads(depth_payload)
        
        assert 'HeadId' in depth_data
        assert depth_data['HeadId'] is None
    
    
    def test_topic_construction_with_head_id(self, result_publisher, processing_result_with_head_id):
        """Test that topics are constructed correctly"""
        toolbox_id = "TOOLBOX123"
        tool_id = "TOOL456"
        timestamp = 1234567890.0
        algo_version = "1.0"
        
        # Publish the result
        result_publisher.publish_processing_result(
            processing_result_with_head_id, toolbox_id, tool_id, timestamp, algo_version
        )
        
        # Get the published topics
        calls = result_publisher.mqtt_client.publish.call_args_list
        
        keypoints_topic = calls[0][0][0]
        depth_topic = calls[1][0][0]
        
        # Verify topic structure
        expected_keypoints_topic = "OPCPUBSUB/TOOLBOX123/TOOL456/DepthEstimation/KeyPoints"
        expected_depth_topic = "OPCPUBSUB/TOOLBOX123/TOOL456/DepthEstimation/DepthEstimation"
        
        assert keypoints_topic == expected_keypoints_topic
        assert depth_topic == expected_depth_topic
    
    def test_insufficient_data_result_includes_head_id(self, result_publisher, mock_config):
        """Test that insufficient data results include HeadId"""
        processing_result = ProcessingResult(
            success=True,
            keypoints=None,  # Missing keypoints triggers insufficient data path
            depth_estimation=None,
            machine_id='TEST_MACHINE_123',
            result_id='RESULT_456',
            head_id='HEAD_789'
        )
        
        toolbox_id = "TOOLBOX123"
        tool_id = "TOOL456"
        timestamp = 1234567890.0
        algo_version = "1.0"
        
        # Publish the result
        result_publisher.publish_processing_result(
            processing_result, toolbox_id, tool_id, timestamp, algo_version
        )
        
        # Method returns None, not True
        
        # Verify MQTT client was called twice (keypoints + depth estimation)
        assert result_publisher.mqtt_client.publish.call_count == 2
        
        # Get the published messages
        calls = result_publisher.mqtt_client.publish.call_args_list
        
        # Check keypoints message
        keypoints_topic, keypoints_payload = calls[0][0]
        keypoints_data = json.loads(keypoints_payload)
        
        assert 'HeadId' in keypoints_data
        assert keypoints_data['HeadId'] == 'HEAD_789'
        assert keypoints_data['Value'] == 'Not enough steps to estimate keypoints'
        assert keypoints_data['MachineId'] == 'TEST_MACHINE_123'
        assert keypoints_data['ResultId'] == 'RESULT_456'
        
        # Check depth estimation message
        depth_topic, depth_payload = calls[1][0]
        depth_data = json.loads(depth_payload)
        
        assert 'HeadId' in depth_data
        assert depth_data['HeadId'] == 'HEAD_789'
        assert depth_data['Value'] == 'Not enough steps to estimate depth'
        assert depth_data['MachineId'] == 'TEST_MACHINE_123'
        assert depth_data['ResultId'] == 'RESULT_456'
    
    def test_error_result_includes_head_id(self, result_publisher):
        """Test that error results include HeadId"""
        processing_result = ProcessingResult(
            success=True,  # Changed to True to pass validation
            keypoints=None,
            depth_estimation=None,
            machine_id='TEST_MACHINE_123',
            result_id='RESULT_456',
            head_id='HEAD_789',
            error_message='Test processing error'  # This will trigger error type
        )
        
        toolbox_id = "TOOLBOX123"
        tool_id = "TOOL456"
        timestamp = 1234567890.0
        algo_version = "1.0"
        
        # Publish the result
        result_publisher.publish_processing_result(
            processing_result, toolbox_id, tool_id, timestamp, algo_version
        )
        
        # Method returns None, not True
        
        # Verify MQTT client was called twice (keypoints + depth estimation)
        assert result_publisher.mqtt_client.publish.call_count == 2
        
        # Get the published messages
        calls = result_publisher.mqtt_client.publish.call_args_list
        
        # Check keypoints message
        keypoints_topic, keypoints_payload = calls[0][0]
        keypoints_data = json.loads(keypoints_payload)
        
        assert 'HeadId' in keypoints_data
        assert keypoints_data['HeadId'] == 'HEAD_789'
        assert keypoints_data['Value'] == 'Error in keypoint estimation: Test processing error'
        assert keypoints_data['MachineId'] == 'TEST_MACHINE_123'
        assert keypoints_data['ResultId'] == 'RESULT_456'
        assert keypoints_data['Error'] is True
        
        # Check depth estimation message
        depth_topic, depth_payload = calls[1][0]
        depth_data = json.loads(depth_payload)
        
        assert 'HeadId' in depth_data
        assert depth_data['HeadId'] == 'HEAD_789'
        assert depth_data['Value'] == 'Error in depth estimation: Test processing error'
        assert depth_data['MachineId'] == 'TEST_MACHINE_123'
        assert depth_data['ResultId'] == 'RESULT_456'
        assert depth_data['Error'] is True
    
    def test_error_result_with_null_head_id(self, result_publisher):
        """Test that error results handle null head_id"""
        processing_result = ProcessingResult(
            success=True,  # Changed to True to pass validation
            keypoints=None,
            depth_estimation=None,
            machine_id='TEST_MACHINE_123',
            result_id='RESULT_456',
            head_id=None,
            error_message='Test processing error'  # This will trigger error type
        )
        
        toolbox_id = "TOOLBOX123"
        tool_id = "TOOL456"
        timestamp = 1234567890.0
        algo_version = "1.0"
        
        # Publish the result
        result_publisher.publish_processing_result(
            processing_result, toolbox_id, tool_id, timestamp, algo_version
        )
        
        # Method returns None, not True
        
        # Get the published messages
        calls = result_publisher.mqtt_client.publish.call_args_list
        
        # Check keypoints message
        keypoints_topic, keypoints_payload = calls[0][0]
        keypoints_data = json.loads(keypoints_payload)
        
        assert 'HeadId' in keypoints_data
        assert keypoints_data['HeadId'] is None
        
        # Check depth estimation message
        depth_topic, depth_payload = calls[1][0]
        depth_data = json.loads(depth_payload)
        
        assert 'HeadId' in depth_data
        assert depth_data['HeadId'] is None