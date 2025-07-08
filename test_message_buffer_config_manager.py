#!/usr/bin/env python3
"""
Test the improved MessageBuffer with ConfigurationManager integration.

This test verifies that:
1. MessageBuffer can accept both ConfigurationManager and raw config
2. Configuration access is consistent between both methods
3. Backward compatibility is maintained
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'abyss', 'src'))

from abyss.mqtt.components.config_manager import ConfigurationManager
from abyss.mqtt.components.message_buffer import MessageBuffer
import tempfile
import yaml

def test_message_buffer_with_config_manager():
    """Test MessageBuffer with ConfigurationManager instance."""
    print("=== Testing MessageBuffer with ConfigurationManager ===")
    
    # Create test config
    config_data = {
        'mqtt': {
            'broker': {
                'host': 'localhost',
                'port': 1883
            },
            'listener': {
                'duplicate_handling': 'error',
                'duplicate_time_window': 2.5,
                'root': 'TEST_ROOT',
                'result': 'TestResult',
                'trace': 'TestTrace',
                'heads': 'TestHeads'
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_config_path = f.name
    
    try:
        # Create ConfigurationManager
        config_manager = ConfigurationManager(temp_config_path)
        
        # Create MessageBuffer with ConfigurationManager
        message_buffer = MessageBuffer(config=config_manager)
        
        # Test that configuration is properly accessed
        assert message_buffer.duplicate_handling == 'error'
        assert message_buffer.config_manager is not None
        
        # Test topic patterns
        expected_patterns = {
            'result': 'TEST_ROOT/+/+/TestResult',
            'trace': 'TEST_ROOT/+/+/TestTrace',
            'heads': 'TEST_ROOT/+/+/TestHeads'
        }
        
        for key, expected in expected_patterns.items():
            assert message_buffer._topic_patterns[key] == expected, \
                f"Topic pattern mismatch for {key}: expected {expected}, got {message_buffer._topic_patterns[key]}"
        
        print("✓ MessageBuffer correctly uses ConfigurationManager")
        print(f"✓ duplicate_handling: {message_buffer.duplicate_handling}")
        print(f"✓ Topic patterns: {message_buffer._topic_patterns}")
        
    finally:
        os.unlink(temp_config_path)

def test_message_buffer_with_raw_config():
    """Test MessageBuffer with raw config dictionary (backward compatibility)."""
    print("\n=== Testing MessageBuffer with Raw Config (Backward Compatibility) ===")
    
    # Create raw config dictionary
    config = {
        'mqtt': {
            'listener': {
                'duplicate_handling': 'replace',
                'duplicate_time_window': 1.5,
                'root': 'RAW_ROOT',
                'result': 'RawResult',
                'trace': 'RawTrace'
                # Note: no 'heads' configured
            }
        }
    }
    
    # Create MessageBuffer with raw config
    message_buffer = MessageBuffer(config=config)
    
    # Test that configuration is properly accessed
    assert message_buffer.duplicate_handling == 'replace'
    assert message_buffer.config_manager is None  # Should be None for raw config
    
    # Test topic patterns
    expected_patterns = {
        'result': 'RAW_ROOT/+/+/RawResult',
        'trace': 'RAW_ROOT/+/+/RawTrace'
    }
    
    for key, expected in expected_patterns.items():
        assert message_buffer._topic_patterns[key] == expected, \
            f"Topic pattern mismatch for {key}: expected {expected}, got {message_buffer._topic_patterns[key]}"
    
    # Test that heads pattern is not created when not configured
    assert 'heads' not in message_buffer._topic_patterns
    
    print("✓ MessageBuffer correctly handles raw config")
    print(f"✓ duplicate_handling: {message_buffer.duplicate_handling}")
    print(f"✓ Topic patterns: {message_buffer._topic_patterns}")
    print("✓ Heads pattern correctly omitted when not configured")

def test_both_approaches_equivalent():
    """Test that both approaches produce equivalent results."""
    print("\n=== Testing Equivalence Between Approaches ===")
    
    # Create test config
    config_data = {
        'mqtt': {
            'broker': {
                'host': 'localhost',
                'port': 1883
            },
            'listener': {
                'duplicate_handling': 'ignore',
                'duplicate_time_window': 3.0,
                'root': 'EQUIV_ROOT',
                'result': 'EquivResult',
                'trace': 'EquivTrace',
                'heads': 'EquivHeads'
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_config_path = f.name
    
    try:
        # Create both versions
        config_manager = ConfigurationManager(temp_config_path)
        raw_config = config_manager.get_raw_config()
        
        buffer_with_manager = MessageBuffer(config=config_manager)
        buffer_with_raw = MessageBuffer(config=raw_config)
        
        # Test equivalence
        assert buffer_with_manager.duplicate_handling == buffer_with_raw.duplicate_handling
        assert buffer_with_manager._topic_patterns == buffer_with_raw._topic_patterns
        
        print("✓ Both approaches produce equivalent configurations")
        print(f"✓ Both have duplicate_handling: {buffer_with_manager.duplicate_handling}")
        print(f"✓ Both have same topic patterns: {len(buffer_with_manager._topic_patterns)} patterns")
        
    finally:
        os.unlink(temp_config_path)

if __name__ == '__main__':
    print("Testing improved MessageBuffer with ConfigurationManager integration...\n")
    
    try:
        test_message_buffer_with_config_manager()
        test_message_buffer_with_raw_config()
        test_both_approaches_equivalent()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("✅ MessageBuffer properly uses ConfigurationManager when available")
        print("✅ Backward compatibility maintained for raw config")
        print("✅ Configuration access is consistent and type-safe")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
