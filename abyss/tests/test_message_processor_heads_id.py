import pytest
import json
from unittest.mock import MagicMock
from abyss.mqtt.components.message_processor import MessageProcessor, ProcessingResult
from abyss.uos_depth_est import TimestampedData


class TestMessageProcessorHeadsId:
    """Tests for head_id extraction in MessageProcessor"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration with head_id path"""
        return {
            'mqtt': {
                'listener': {
                    'result': 'ResultManagement',
                    'trace': 'ResultManagement/Trace',
                    'heads': 'AssetManagement/Head'
                },
                'data_ids': {
                    'head_id': 'AssetManagement.Assets.Heads.0.Identification.SerialNumber'
                }
            }
        }
    
    @pytest.fixture
    def mock_depth_inference(self):
        """Mock depth inference instance"""
        mock = MagicMock()
        mock.infer3_common.return_value = [10.0, 20.0, 30.0]
        return mock
    
    @pytest.fixture
    def mock_data_converter(self):
        """Mock data converter instance"""
        import pandas as pd
        
        mock = MagicMock()
        
        # Create a real DataFrame for testing
        test_df = pd.DataFrame({
            'Step (nb)': [1, 2, 3],
            'HOLE_ID': ['TEST123', 'TEST123', 'TEST123'],
            'local': [456, 456, 456],
            'Position (mm)': [10.0, 20.0, 30.0],
            'I Torque (A)': [1.0, 2.0, 3.0],
            'I Thrust (A)': [0.5, 1.0, 1.5]
        })
        
        mock.convert_messages_to_df.return_value = test_df
        
        # Mock the heads data extraction
        mock._extract_heads_data.return_value = {'head_id': 'HEAD987654'}
        
        return mock
    
    @pytest.fixture
    def message_processor(self, mock_depth_inference, mock_data_converter, mock_config):
        """Create MessageProcessor with mocked dependencies"""
        return MessageProcessor(
            depth_inference=mock_depth_inference,
            data_converter=mock_data_converter,
            config=mock_config
        )
    
    def test_extract_heads_id_success(self, message_processor):
        """Test successful head_id extraction from heads message"""
        heads_data = {
            'AssetManagement': {
                'Assets': {
                    'Heads': [
                        {
                            'Identification': {
                                'SerialNumber': 'HEAD12345'
                            }
                        }
                    ]
                }
            }
        }
        
        heads_msg = TimestampedData(
            _timestamp=1234567890.0,
            _data=json.dumps(heads_data),
            _source='test/AssetManagement/Head'
        )
        
        result = message_processor._extract_heads_id(heads_msg)
        
        assert result == 'HEAD987654'  # From mock data converter
    
    def test_extract_heads_id_no_message(self, message_processor):
        """Test head_id extraction with no heads message"""
        result = message_processor._extract_heads_id(None)
        assert result is None
    
    def test_extract_heads_id_invalid_json(self, message_processor):
        """Test head_id extraction with invalid JSON"""
        heads_msg = TimestampedData(
            _timestamp=1234567890.0,
            _data='invalid json',
            _source='test/AssetManagement/Head'
        )
        
        result = message_processor._extract_heads_id(heads_msg)
        assert result is None
    
    def test_heads_id_extraction_integration(self, message_processor):
        """Test that head_id extraction works in isolation"""
        # Test the head_id extraction method directly
        heads_data = {
            'AssetManagement': {
                'Assets': {
                    'Heads': [
                        {
                            'Identification': {
                                'SerialNumber': 'INTEGRATION_TEST_HEAD'
                            }
                        }
                    ]
                }
            }
        }
        
        heads_msg = TimestampedData(
            _timestamp=1234567890.0,
            _data=json.dumps(heads_data),
            _source='test/toolbox/tool/AssetManagement/Head'
        )
        
        # Extract head_id
        head_id = message_processor._extract_heads_id(heads_msg)
        
        # Should get the mocked value from data converter
        assert head_id == 'HEAD987654'
    
    def test_process_matching_messages_no_heads_message(self, message_processor):
        """Test processing without heads message"""
        # Create test messages (no heads message)
        result_msg = TimestampedData(
            _timestamp=1234567890.0,
            _data='{"test": "result_data"}',
            _source='test/toolbox/tool/ResultManagement'
        )
        
        trace_msg = TimestampedData(
            _timestamp=1234567890.0,
            _data='{"test": "trace_data"}',
            _source='test/toolbox/tool/ResultManagement/Trace'
        )
        
        matches = [result_msg, trace_msg]  # No heads message
        
        # Process the messages
        result = message_processor.process_matching_messages(matches)
        
        # Verify the result has None for head_id
        assert isinstance(result, ProcessingResult)
        assert result.success is True
        assert result.heads_id is None  # Should be None when no heads message
        assert result.machine_id == 'TEST123'
        assert result.result_id == '456'
    
    def test_processing_result_includes_heads_id_in_error_cases(self, message_processor):
        """Test that error cases also include heads_id"""
        # Test with missing trace message
        result_msg = TimestampedData(
            _timestamp=1234567890.0,
            _data='{"test": "result_data"}',
            _source='test/toolbox/tool/ResultManagement'
        )
        
        matches = [result_msg]  # Missing trace message
        
        result = message_processor.process_matching_messages(matches)
        
        assert isinstance(result, ProcessingResult)
        assert result.success is False
        assert result.heads_id is None  # Should be None initially
        assert result.error_message is not None
        assert "Missing required result or trace message" in str(result.error_message)
    
    def test_heads_id_preserved_across_processing_steps(self, message_processor):
        """Test that heads_id is preserved across all processing steps"""
        # Set up a heads_id value manually
        message_processor.heads_id = "MANUAL_HEAD_ID"
        
        # Mock insufficient data scenario
        message_processor.data_converter.convert_messages_to_df.return_value = None
        
        result_msg = TimestampedData(
            _timestamp=1234567890.0,
            _data='{"test": "result_data"}',
            _source='test/toolbox/tool/ResultManagement'
        )
        
        trace_msg = TimestampedData(
            _timestamp=1234567890.0,
            _data='{"test": "trace_data"}',
            _source='test/toolbox/tool/ResultManagement/Trace'
        )
        
        matches = [result_msg, trace_msg]
        
        result = message_processor.process_matching_messages(matches)
        
        # Even in error case, heads_id should be preserved
        assert result.heads_id == "MANUAL_HEAD_ID"
        assert result.success is False