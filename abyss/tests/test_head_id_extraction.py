import pytest
from unittest.mock import MagicMock
from abyss.mqtt.components.data_converter import DataFrameConverter


class TestHeadIdExtraction:
    """Tests for head_id extraction functionality in DataFrameConverter"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration with head_id path"""
        return {
            'mqtt': {
                'data_ids': {
                    'head_id': 'AssetManagement.Assets.Heads.0.Identification.SerialNumber'
                }
            }
        }
    
    @pytest.fixture
    def data_converter(self, mock_config):
        """Create DataFrameConverter with mock config"""
        return DataFrameConverter(mock_config)
    
    def test_extract_value_by_path_simple(self, data_converter):
        """Test extracting value with simple path"""
        data = {
            'AssetManagement': {
                'Assets': {
                    'Heads': [
                        {
                            'Identification': {
                                'SerialNumber': 'HEAD123456'
                            }
                        }
                    ]
                }
            }
        }
        
        result = data_converter._extract_value_by_path(
            data, 'AssetManagement.Assets.Heads.0.Identification.SerialNumber'
        )
        
        assert result == 'HEAD123456'
    
    def test_extract_value_by_path_missing_key(self, data_converter):
        """Test extracting value with missing key"""
        data = {
            'AssetManagement': {
                'Assets': {
                    'Heads': []  # Empty array
                }
            }
        }
        
        result = data_converter._extract_value_by_path(
            data, 'AssetManagement.Assets.Heads.0.Identification.SerialNumber'
        )
        
        assert result is None
    
    def test_extract_value_by_path_invalid_array_index(self, data_converter):
        """Test extracting value with invalid array index"""
        data = {
            'AssetManagement': {
                'Assets': {
                    'Heads': [
                        {'id': 'head1'}
                    ]
                }
            }
        }
        
        result = data_converter._extract_value_by_path(
            data, 'AssetManagement.Assets.Heads.5.Identification.SerialNumber'
        )
        
        assert result is None
    
    def test_extract_head_id_from_config_success(self, data_converter):
        """Test successful head_id extraction using configuration"""
        heads_data = {
            'AssetManagement': {
                'Assets': {
                    'Heads': [
                        {
                            'Identification': {
                                'SerialNumber': 'HEAD987654'
                            }
                        }
                    ]
                }
            }
        }
        
        result = data_converter._extract_head_id_from_config(heads_data)
        
        assert result == 'HEAD987654'
    
    def test_extract_head_id_from_config_missing_path(self):
        """Test head_id extraction with missing config path"""
        config_without_head_id = {
            'mqtt': {
                'data_ids': {
                    # head_id path is missing
                }
            }
        }
        
        converter = DataFrameConverter(config_without_head_id)
        heads_data = {'some': 'data'}
        
        result = converter._extract_head_id_from_config(heads_data)
        
        assert result is None
    
    def test_extract_heads_data_includes_head_id(self, data_converter):
        """Test that _extract_heads_data includes head_id in results"""
        heads_data = {
            'AssetManagement': {
                'Assets': {
                    'Heads': [
                        {
                            'Identification': {
                                'SerialNumber': 'HEAD555'
                            }
                        }
                    ]
                }
            },
            'SourceTimestamp': '2023-01-01T00:00:00Z',
            'Value': 'some_value'
        }
        
        result = data_converter._extract_heads_data(heads_data)
        
        # Should include head_id plus other extracted fields
        assert 'head_id' in result
        assert result['head_id'] == 'HEAD555'
        assert 'sourcetimestamp' in result
        assert result['sourcetimestamp'] == '2023-01-01T00:00:00Z'
    
    def test_extract_heads_data_missing_head_id_path(self, data_converter):
        """Test that _extract_heads_data handles missing head_id gracefully"""
        heads_data = {
            'SomeOther': {
                'Structure': 'without AssetManagement'
            },
            'SourceTimestamp': '2023-01-01T00:00:00Z'
        }
        
        result = data_converter._extract_heads_data(heads_data)
        
        # Should not include head_id but should include other fields
        assert 'head_id' not in result
        assert 'sourcetimestamp' in result
    
    def test_extract_value_by_path_numeric_conversion(self, data_converter):
        """Test that numeric values are converted to strings"""
        data = {
            'path': {
                'to': {
                    'numeric': 12345
                }
            }
        }
        
        result = data_converter._extract_value_by_path(data, 'path.to.numeric')
        
        assert result == '12345'
        assert isinstance(result, str)
    
    def test_extract_value_by_path_empty_path(self, data_converter):
        """Test extracting with empty path"""
        data = {'some': 'data'}
        
        result = data_converter._extract_value_by_path(data, '')
        
        assert result is None
    
    def test_extract_value_by_path_non_dict_data(self, data_converter):
        """Test extracting from non-dictionary data"""
        data = "not a dictionary"
        
        result = data_converter._extract_value_by_path(data, 'some.path')
        
        assert result is None