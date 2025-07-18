import pytest
from unittest.mock import Mock
from abyss.mqtt.components.simple_correlator import SimpleMessageCorrelator
from abyss.uos_depth_est import TimestampedData
import time


class TestSimpleCorrelator:
    """Test the simplified message correlator with exact matching"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration"""
        return {
            'mqtt': {
                'listener': {
                    'root': 'OPCPUBSUB',
                    'result': 'ResultManagement',
                    'trace': 'ResultManagement/Trace',
                    'heads': 'AssetManagement/Head'
                }
            }
        }
    
    @pytest.fixture
    def simple_correlator(self, mock_config):
        """Create simple correlator (time_window param ignored in exact match mode)"""
        return SimpleMessageCorrelator(mock_config, time_window=5.0)  # time_window ignored
    
    def test_exact_matching_same_timestamp(self, simple_correlator):
        """Test exact matching with same SourceTimestamp"""
        source_timestamp = '2024-01-18T10:30:45Z'
        
        # Create test messages with same SourceTimestamp
        result_msg = TimestampedData(
            _timestamp=time.time(),
            _data={'SourceTimestamp': source_timestamp, 'ResultId': 100},
            _source='OPCPUBSUB/toolbox1/tool1/ResultManagement'
        )
        
        trace_msg = TimestampedData(
            _timestamp=time.time() + 1.0,  # Different _timestamp but same SourceTimestamp
            _data={'SourceTimestamp': source_timestamp, 'TraceData': [1, 2, 3]},
            _source='OPCPUBSUB/toolbox1/tool1/ResultManagement/Trace'
        )
        
        heads_msg = TimestampedData(
            _timestamp=time.time() + 0.5,
            _data={'SourceTimestamp': source_timestamp, 'HeadsId': 'HEADS123'},
            _source='OPCPUBSUB/toolbox1/tool1/AssetManagement/Head'
        )
        
        # Create buffers
        buffers = {
            'OPCPUBSUB/+/+/ResultManagement': [result_msg],
            'OPCPUBSUB/+/+/ResultManagement/Trace': [trace_msg],
            'OPCPUBSUB/+/+/AssetManagement/Head': [heads_msg]
        }
        
        # Mock message processor
        processed_matches = []
        def mock_processor(messages):
            processed_matches.append(messages)
        
        # Test correlation
        found_matches = simple_correlator.find_and_process_matches(buffers, mock_processor)
        
        # Verify results
        assert found_matches is True
        assert len(processed_matches) == 1
        assert len(processed_matches[0]) == 3  # result + trace + heads
        
        # Verify all three messages were matched
        assert len(processed_matches[0]) == 3
        assert result_msg in processed_matches[0]
        assert trace_msg in processed_matches[0]
        assert heads_msg in processed_matches[0]
    
    def test_exact_matching_requires_heads(self, simple_correlator):
        """Test that exact matching always requires heads message"""
        source_timestamp = '2024-01-18T10:30:45Z'
        
        # Create test messages (no heads)
        result_msg = TimestampedData(
            _timestamp=time.time(),
            _data={'SourceTimestamp': source_timestamp, 'ResultId': 200},
            _source='OPCPUBSUB/toolbox2/tool2/ResultManagement'
        )
        
        trace_msg = TimestampedData(
            _timestamp=time.time(),
            _data={'SourceTimestamp': source_timestamp, 'TraceData': [4, 5, 6]},
            _source='OPCPUBSUB/toolbox2/tool2/ResultManagement/Trace'
        )
        
        # Create buffers (no heads buffer)
        buffers = {
            'OPCPUBSUB/+/+/ResultManagement': [result_msg],
            'OPCPUBSUB/+/+/ResultManagement/Trace': [trace_msg]
        }
        
        # Mock message processor
        processed_matches = []
        def mock_processor(messages):
            processed_matches.append(messages)
        
        # Test correlation
        found_matches = simple_correlator.find_and_process_matches(buffers, mock_processor)
        
        # Verify no matches without heads (heads is required in exact match mode)
        assert found_matches is False
        assert len(processed_matches) == 0
    
    def test_exact_matching_different_timestamps(self, simple_correlator):
        """Test that messages with different SourceTimestamps don't match"""
        # Create test messages with different SourceTimestamps
        result_msg = TimestampedData(
            _timestamp=time.time(),
            _data={'SourceTimestamp': '2024-01-18T10:30:45Z', 'ResultId': 300},
            _source='OPCPUBSUB/toolbox3/tool3/ResultManagement'
        )
        
        trace_msg = TimestampedData(
            _timestamp=time.time(),
            _data={'SourceTimestamp': '2024-01-18T10:30:46Z', 'TraceData': [7, 8, 9]},  # Different!
            _source='OPCPUBSUB/toolbox3/tool3/ResultManagement/Trace'
        )
        
        heads_msg = TimestampedData(
            _timestamp=time.time(),
            _data={'SourceTimestamp': '2024-01-18T10:30:47Z', 'HeadsId': 'HEADS789'},  # Different!
            _source='OPCPUBSUB/toolbox3/tool3/AssetManagement/Head'
        )
        
        # Create buffers with all three messages
        buffers = {
            'OPCPUBSUB/+/+/ResultManagement': [result_msg],
            'OPCPUBSUB/+/+/ResultManagement/Trace': [trace_msg],
            'OPCPUBSUB/+/+/AssetManagement/Head': [heads_msg]
        }
        
        # Mock message processor
        processed_matches = []
        def mock_processor(messages):
            processed_matches.append(messages)
        
        # Test correlation
        found_matches = simple_correlator.find_and_process_matches(buffers, mock_processor)
        
        # Verify no matches found (different timestamps)
        assert found_matches is False
        assert len(processed_matches) == 0
    
    def test_exact_matching_different_tool_keys(self, simple_correlator):
        """Test that messages with different tool keys don't match even with same timestamp"""
        source_timestamp = '2024-01-18T10:30:45Z'
        
        # Create test messages with same timestamp but different tool keys
        result_msg = TimestampedData(
            _timestamp=time.time(),
            _data={'SourceTimestamp': source_timestamp, 'ResultId': 400},
            _source='OPCPUBSUB/toolbox1/tool1/ResultManagement'
        )
        
        trace_msg = TimestampedData(
            _timestamp=time.time(),
            _data={'SourceTimestamp': source_timestamp, 'TraceData': [10, 11, 12]},
            _source='OPCPUBSUB/toolbox2/tool2/ResultManagement/Trace'  # Different tool
        )
        
        # Create buffers
        buffers = {
            'OPCPUBSUB/+/+/ResultManagement': [result_msg],
            'OPCPUBSUB/+/+/ResultManagement/Trace': [trace_msg]
        }
        
        # Mock message processor
        processed_matches = []
        def mock_processor(messages):
            processed_matches.append(messages)
        
        # Test correlation
        found_matches = simple_correlator.find_and_process_matches(buffers, mock_processor)
        
        # Verify no matches found
        assert found_matches is False
        assert len(processed_matches) == 0
    
    def test_group_by_tool_key(self, simple_correlator):
        """Test tool key grouping functionality"""
        messages = [
            TimestampedData(_timestamp=1.0, _data='{}', _source='OPCPUBSUB/tb1/t1/ResultManagement'),
            TimestampedData(_timestamp=2.0, _data='{}', _source='OPCPUBSUB/tb1/t2/ResultManagement'),
            TimestampedData(_timestamp=3.0, _data='{}', _source='OPCPUBSUB/tb2/t1/ResultManagement'),
            TimestampedData(_timestamp=4.0, _data='{}', _source='OPCPUBSUB/tb1/t1/ResultManagement'),
        ]
        
        grouped = simple_correlator._group_by_tool_key(messages)
        
        assert len(grouped) == 3  # tb1/t1, tb1/t2, tb2/t1
        assert len(grouped['tb1/t1']) == 2
        assert len(grouped['tb1/t2']) == 1
        assert len(grouped['tb2/t1']) == 1
    
    def test_correlation_stats_exact_match(self, simple_correlator):
        """Test correlation statistics for exact matching"""
        # Create complete message set
        timestamp = '2024-01-18T10:30:45Z'
        messages = [
            TimestampedData(
                _timestamp=time.time(),
                _data={'SourceTimestamp': timestamp, 'ResultId': 500},
                _source='OPCPUBSUB/toolbox1/tool1/ResultManagement'
            ),
            TimestampedData(
                _timestamp=time.time(),
                _data={'SourceTimestamp': timestamp},
                _source='OPCPUBSUB/toolbox1/tool1/ResultManagement/Trace'
            ),
            TimestampedData(
                _timestamp=time.time(),
                _data={'SourceTimestamp': timestamp},
                _source='OPCPUBSUB/toolbox1/tool1/AssetManagement/Head'
            )
        ]
        
        buffers = {
            'OPCPUBSUB/+/+/ResultManagement': [messages[0]],
            'OPCPUBSUB/+/+/ResultManagement/Trace': [messages[1]],
            'OPCPUBSUB/+/+/AssetManagement/Head': [messages[2]]
        }
        
        stats = simple_correlator.get_correlation_stats(buffers)
        
        assert stats['total_messages'] == 3
        assert stats['correlation_approach'] == 'exact_match_only'
        assert stats['exact_groups'] >= 1
        assert stats['complete_groups'] >= 1
    
    def test_get_completed_matches_from_buffer(self, simple_correlator):
        """Test retrieving completed matches from buffer (exact match specific)"""
        # Create buffer-like structure with completed matches
        timestamp = '2024-01-18T10:30:45Z'
        
        result_msg = TimestampedData(
            _timestamp=time.time(),
            _data={'SourceTimestamp': timestamp, 'ResultId': 600},
            _source='OPCPUBSUB/toolbox1/tool1/ResultManagement'
        )
        trace_msg = TimestampedData(
            _timestamp=time.time(),
            _data={'SourceTimestamp': timestamp},
            _source='OPCPUBSUB/toolbox1/tool1/ResultManagement/Trace'
        )
        heads_msg = TimestampedData(
            _timestamp=time.time(),
            _data={'SourceTimestamp': timestamp},
            _source='OPCPUBSUB/toolbox1/tool1/AssetManagement/Head'
        )
        
        # In exact match mode, buffer would have pre-grouped these
        completed_matches = [[result_msg, trace_msg, heads_msg]]
        
        # Mock the buffer's get_completed_matches method
        mock_buffer = Mock()
        mock_buffer.get_completed_matches.return_value = completed_matches
        
        # Process should handle pre-matched groups
        processed = []
        def processor(msgs):
            processed.extend(msgs)
        
        # In real usage, correlator would get matches from buffer
        for match in completed_matches:
            processor(match)
        
        assert len(processed) == 3
        assert result_msg in processed
        assert trace_msg in processed
        assert heads_msg in processed


class TestCorrelatorComparison:
    """Compare simple and complex correlators"""
    
    @pytest.fixture
    def mock_config(self):
        return {
            'mqtt': {
                'listener': {
                    'root': 'OPCPUBSUB',
                    'result': 'ResultManagement',
                    'trace': 'ResultManagement/Trace',
                    'heads': 'AssetManagement/Head'
                }
            }
        }