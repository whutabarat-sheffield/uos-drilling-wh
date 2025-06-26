import pytest
from unittest.mock import Mock
from abyss.mqtt.components.simple_correlator import SimpleMessageCorrelator
from abyss.mqtt.components.correlator import MessageCorrelator
from abyss.uos_depth_est import TimestampedData
import time


class TestSimpleCorrelator:
    """Test the simplified message correlator"""
    
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
        """Create simple correlator"""
        return SimpleMessageCorrelator(mock_config, time_window=5.0)
    
    @pytest.fixture
    def complex_correlator(self, mock_config):
        """Create complex correlator for comparison"""
        return MessageCorrelator(mock_config, time_window=5.0)
    
    def test_simple_correlator_basic_matching(self, simple_correlator):
        """Test basic message matching with simple correlator"""
        timestamp = time.time()
        
        # Create test messages
        result_msg = TimestampedData(
            _timestamp=timestamp,
            _data='{"test": "result"}',
            _source='OPCPUBSUB/toolbox1/tool1/ResultManagement'
        )
        
        trace_msg = TimestampedData(
            _timestamp=timestamp + 1.0,  # 1 second later
            _data='{"test": "trace"}',
            _source='OPCPUBSUB/toolbox1/tool1/ResultManagement/Trace'
        )
        
        heads_msg = TimestampedData(
            _timestamp=timestamp + 0.5,  # 0.5 seconds later
            _data='{"test": "heads"}',
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
        
        # Verify messages are marked as processed
        assert getattr(result_msg, 'processed', False) is True
        assert getattr(trace_msg, 'processed', False) is True
        assert getattr(heads_msg, 'processed', False) is True
    
    def test_simple_correlator_no_heads_message(self, simple_correlator):
        """Test correlation without heads message"""
        timestamp = time.time()
        
        # Create test messages (no heads)
        result_msg = TimestampedData(
            _timestamp=timestamp,
            _data='{"test": "result"}',
            _source='OPCPUBSUB/toolbox2/tool2/ResultManagement'
        )
        
        trace_msg = TimestampedData(
            _timestamp=timestamp + 2.0,  # 2 seconds later
            _data='{"test": "trace"}',
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
        
        # Verify results
        assert found_matches is True
        assert len(processed_matches) == 1
        assert len(processed_matches[0]) == 2  # result + trace only
    
    def test_simple_correlator_time_window_rejection(self, simple_correlator):
        """Test that messages outside time window are rejected"""
        timestamp = time.time()
        
        # Create test messages with large time difference
        result_msg = TimestampedData(
            _timestamp=timestamp,
            _data='{"test": "result"}',
            _source='OPCPUBSUB/toolbox3/tool3/ResultManagement'
        )
        
        trace_msg = TimestampedData(
            _timestamp=timestamp + 10.0,  # 10 seconds later (outside 5s window)
            _data='{"test": "trace"}',
            _source='OPCPUBSUB/toolbox3/tool3/ResultManagement/Trace'
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
    
    def test_simple_correlator_different_tool_keys(self, simple_correlator):
        """Test that messages with different tool keys don't match"""
        timestamp = time.time()
        
        # Create test messages with different tool keys
        result_msg = TimestampedData(
            _timestamp=timestamp,
            _data='{"test": "result"}',
            _source='OPCPUBSUB/toolbox1/tool1/ResultManagement'
        )
        
        trace_msg = TimestampedData(
            _timestamp=timestamp + 1.0,
            _data='{"test": "trace"}',
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
    
    def test_correlation_stats(self, simple_correlator):
        """Test correlation statistics"""
        # Create some test messages
        messages = [
            TimestampedData(_timestamp=1.0, _data='{}', _source='test'),
            TimestampedData(_timestamp=2.0, _data='{}', _source='test')
        ]
        
        buffers = {
            'OPCPUBSUB/+/+/ResultManagement': messages,
            'OPCPUBSUB/+/+/ResultManagement/Trace': []
        }
        
        stats = simple_correlator.get_correlation_stats(buffers)
        
        assert stats['total_messages'] == 2
        assert stats['time_window'] == 5.0
        assert stats['correlation_approach'] == 'simple_key_based'
        assert 'unprocessed_messages' in stats


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
    
    def test_both_correlators_same_result(self, mock_config):
        """Test that both correlators produce the same results for basic cases"""
        simple_correlator = SimpleMessageCorrelator(mock_config, time_window=5.0)
        complex_correlator = MessageCorrelator(mock_config, time_window=5.0)
        
        timestamp = time.time()
        
        # Create identical test messages
        result_msg1 = TimestampedData(
            _timestamp=timestamp,
            _data='{"test": "result"}',
            _source='OPCPUBSUB/toolbox1/tool1/ResultManagement'
        )
        result_msg2 = TimestampedData(
            _timestamp=timestamp,
            _data='{"test": "result"}',
            _source='OPCPUBSUB/toolbox1/tool1/ResultManagement'
        )
        
        trace_msg1 = TimestampedData(
            _timestamp=timestamp + 1.0,
            _data='{"test": "trace"}',
            _source='OPCPUBSUB/toolbox1/tool1/ResultManagement/Trace'
        )
        trace_msg2 = TimestampedData(
            _timestamp=timestamp + 1.0,
            _data='{"test": "trace"}',
            _source='OPCPUBSUB/toolbox1/tool1/ResultManagement/Trace'
        )
        
        # Create identical buffers
        buffers_simple = {
            'OPCPUBSUB/+/+/ResultManagement': [result_msg1],
            'OPCPUBSUB/+/+/ResultManagement/Trace': [trace_msg1]
        }
        
        buffers_complex = {
            'OPCPUBSUB/+/+/ResultManagement': [result_msg2],
            'OPCPUBSUB/+/+/ResultManagement/Trace': [trace_msg2]
        }
        
        # Mock processors
        simple_matches = []
        complex_matches = []
        
        def simple_processor(messages):
            simple_matches.append(len(messages))
            
        def complex_processor(messages):
            complex_matches.append(len(messages))
        
        # Test both correlators
        simple_found = simple_correlator.find_and_process_matches(buffers_simple, simple_processor)
        complex_found = complex_correlator.find_and_process_matches(buffers_complex, complex_processor)
        
        # Both should find matches
        assert simple_found == complex_found == True
        assert len(simple_matches) == len(complex_matches) == 1
        assert simple_matches[0] == complex_matches[0] == 2  # Both found result+trace