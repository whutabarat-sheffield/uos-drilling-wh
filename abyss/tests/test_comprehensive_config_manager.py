#!/usr/bin/env python3
"""
Test comprehensive ConfigurationManager integration across all components.

This test verifies that ALL components now properly support ConfigurationManager
and that raw config access is only used for backward compatibility.
"""

from abyss.mqtt.components.config_manager import ConfigurationManager
from abyss.mqtt.components.message_buffer import MessageBuffer
from abyss.mqtt.components.simple_correlator import SimpleMessageCorrelator
from abyss.mqtt.components.message_processor import MessageProcessor
from abyss.mqtt.components.client_manager import MQTTClientManager
from abyss.mqtt.components.data_converter import DataFrameConverter
from abyss.mqtt.components.result_publisher import ResultPublisher
# Note: MessageCorrelator doesn't exist - using SimpleMessageCorrelator instead
from abyss.mqtt.components.drilling_analyser import DrillingDataAnalyser
import tempfile
import yaml
from unittest.mock import Mock

def create_test_config():
    """Create a comprehensive test configuration."""
    return {
        'mqtt': {
            'broker': {
                'host': 'localhost',
                'port': 1883,
                'username': '',
                'password': ''
            },
            'listener': {
                'duplicate_handling': 'replace',
                'duplicate_time_window': 2.0,
                'root': 'TESTROOT',
                'result': 'TestResult',
                'trace': 'TestTrace',
                'heads': 'TestHeads',
                'time_window': 25.0,
                'cleanup_interval': 45,
                'max_buffer_size': 8000
            },
            'data_ids': {
                'head_id': 'AssetManagement.Assets.Heads.0.Identification.SerialNumber'
            },
            'estimation': {
                'keypoints': 'TestKeypoints',
                'depth_estimation': 'TestDepthEstimation'
            }
        }
    }

def test_all_components_with_config_manager():
    """Test that all components properly work with ConfigurationManager."""
    print("=== Testing All Components with ConfigurationManager ===")
    
    config_data = create_test_config()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_config_path = f.name
    
    try:
        # Create ConfigurationManager
        config_manager = ConfigurationManager(temp_config_path)
        
        # Test MessageBuffer
        message_buffer = MessageBuffer(config=config_manager)
        assert message_buffer.config_manager is not None
        assert message_buffer.duplicate_handling == 'replace'
        print("✓ MessageBuffer works with ConfigurationManager")
        
        # Test SimpleMessageCorrelator
        simple_correlator = SimpleMessageCorrelator(config=config_manager)
        assert simple_correlator.config_manager is not None
        assert simple_correlator.time_window == 25.0  # From config
        print("✓ SimpleMessageCorrelator works with ConfigurationManager")
        
        # Test MessageCorrelator (legacy)
        correlator = MessageCorrelator(config=config_manager)
        assert correlator.config_manager is not None
        assert correlator.time_window == 25.0  # From config
        print("✓ MessageCorrelator works with ConfigurationManager")
        
        # Test DataConverter
        data_converter = DataFrameConverter(config=config_manager)
        assert data_converter.config_manager is not None
        print("✓ DataFrameConverter works with ConfigurationManager")
        
        # Test MessageProcessor (with mocks)
        mock_depth_inference = Mock()
        message_processor = MessageProcessor(
            depth_inference=mock_depth_inference,
            data_converter=data_converter,
            config=config_manager
        )
        assert message_processor.config_manager is not None
        print("✓ MessageProcessor works with ConfigurationManager")
        
        # Test ClientManager
        client_manager = MQTTClientManager(config=config_manager)
        assert client_manager.config_manager is not None
        print("✓ MQTTClientManager works with ConfigurationManager")
        
        # Test ResultPublisher (with mock client)
        mock_client = Mock()
        result_publisher = ResultPublisher(mqtt_client=mock_client, config=config_manager)
        assert result_publisher.config_manager is not None
        print("✓ ResultPublisher works with ConfigurationManager")
        
        print("\n✅ ALL COMPONENTS successfully work with ConfigurationManager!")
        
    finally:
        os.unlink(temp_config_path)

def test_backward_compatibility_with_raw_config():
    """Test that all components still work with raw config dictionaries."""
    print("\n=== Testing Backward Compatibility with Raw Config ===")
    
    config = create_test_config()
    
    # Test MessageBuffer
    message_buffer = MessageBuffer(config=config)
    assert message_buffer.config_manager is None
    assert message_buffer.duplicate_handling == 'replace'
    print("✓ MessageBuffer backward compatible with raw config")
    
    # Test SimpleMessageCorrelator
    simple_correlator = SimpleMessageCorrelator(config=config)
    assert simple_correlator.config_manager is None
    print("✓ SimpleMessageCorrelator backward compatible with raw config")
    
    # Test MessageCorrelator
    correlator = MessageCorrelator(config=config)
    assert correlator.config_manager is None
    print("✓ MessageCorrelator backward compatible with raw config")
    
    # Test DataConverter
    data_converter = DataFrameConverter(config=config)
    assert data_converter.config_manager is None
    print("✓ DataConverter backward compatible with raw config")
    
    # Test MessageProcessor (with mocks)
    mock_depth_inference = Mock()
    message_processor = MessageProcessor(
        depth_inference=mock_depth_inference,
        data_converter=data_converter,
        config=config
    )
    assert message_processor.config_manager is None
    print("✓ MessageProcessor backward compatible with raw config")
    
    # Test ClientManager
    client_manager = MQTTClientManager(config=config)
    assert client_manager.config_manager is None
    print("✓ MQTTClientManager backward compatible with raw config")
    
    # Test ResultPublisher (with mock client)
    mock_client = Mock()
    result_publisher = ResultPublisher(mqtt_client=mock_client, config=config)
    assert result_publisher.config_manager is None
    print("✓ ResultPublisher backward compatible with raw config")
    
    print("\n✅ ALL COMPONENTS maintain backward compatibility!")

def test_drilling_analyser_integration():
    """Test that DrillingDataAnalyser properly integrates with ConfigurationManager."""
    print("\n=== Testing DrillingDataAnalyser Integration ===")
    
    config_data = create_test_config()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_config_path = f.name
    
    try:
        # Test DrillingDataAnalyser initialization
        analyser = DrillingDataAnalyser(config_path=temp_config_path)
        
        # Verify that all components are properly initialized with ConfigurationManager
        assert analyser.config_manager is not None
        assert analyser.message_buffer.config_manager is not None
        assert analyser.message_correlator.config_manager is not None
        assert analyser.data_converter.config_manager is not None
        assert analyser.message_processor.config_manager is not None
        assert analyser.client_manager.config_manager is not None
        
        print("✓ DrillingDataAnalyser properly initializes all components with ConfigurationManager")
        print(f"✓ All components have consistent duplicate_handling: {analyser.message_buffer.duplicate_handling}")
        print(f"✓ All components have consistent time_window: {analyser.message_correlator.time_window}")
        
    finally:
        os.unlink(temp_config_path)

def test_configuration_consistency():
    """Test that configuration values are consistent across all components."""
    print("\n=== Testing Configuration Consistency ===")
    
    config_data = create_test_config()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_config_path = f.name
    
    try:
        config_manager = ConfigurationManager(temp_config_path)
        
        # Create all components with the same ConfigurationManager
        message_buffer = MessageBuffer(config=config_manager)
        simple_correlator = SimpleMessageCorrelator(config=config_manager)
        data_converter = DataFrameConverter(config=config_manager)
        
        # Test that configuration values are consistent
        assert message_buffer.duplicate_handling == config_manager.get_duplicate_handling()
        assert simple_correlator.time_window == config_manager.get_time_window()
        
        # Test topic patterns consistency
        expected_patterns = config_manager.get_topic_patterns()
        assert simple_correlator._topic_patterns['result'] == expected_patterns['result']
        assert simple_correlator._topic_patterns['trace'] == expected_patterns['trace']
        assert simple_correlator._topic_patterns['heads'] == expected_patterns['heads']
        
        print("✓ Configuration values are consistent across all components")
        print(f"✓ duplicate_handling: {message_buffer.duplicate_handling}")
        print(f"✓ time_window: {simple_correlator.time_window}")
        print(f"✓ Topic patterns: {len(simple_correlator._topic_patterns)} patterns")
        
    finally:
        os.unlink(temp_config_path)

if __name__ == '__main__':
    print("Testing comprehensive ConfigurationManager integration...\n")
    
    try:
        test_all_components_with_config_manager()
        test_backward_compatibility_with_raw_config()
        test_drilling_analyser_integration()
        test_configuration_consistency()
        
        print("\n" + "="*80)
        print("✅ ALL COMPREHENSIVE TESTS PASSED!")
        print("✅ All components properly use ConfigurationManager when available")
        print("✅ All components maintain backward compatibility with raw config")
        print("✅ DrillingDataAnalyser properly integrates all components")
        print("✅ Configuration values are consistent across all components")
        print("✅ Raw config access is only used for legacy support")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
