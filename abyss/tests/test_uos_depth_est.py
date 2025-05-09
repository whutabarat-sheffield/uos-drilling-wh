import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from collections import defaultdict
import logging
import json
from abyss.uos_depth_est import MQTTDrillingDataAnalyser
from abyss.uos_depth_est import MQTTDrillingDataAnalyser
from abyss.uos_depth_est import MQTTDrillingDataAnalyser
from abyss.uos_depth_est import MQTTDrillingDataAnalyser
from abyss.uos_depth_est import MQTTDrillingDataAnalyser
from abyss.uos_depth_est import MQTTDrillingDataAnalyser

class TimestampedData:
    """Mock TimestampedData class for testing"""
    def __init__(self, _timestamp, _data, _source):
        self.timestamp = _timestamp
        self.data = _data
        self.source = _source
        self.processed = False

class TestMQTTDrillingDataAnalyser:
    @pytest.fixture
    def analyser(self):
        """Create a mock instance of MQTTDrillingDataAnalyser for testing"""
        with patch('abyss.uos_depth_est.MQTTDrillingDataAnalyser') as MockAnalyser:
            instance = MockAnalyser.return_value
            
            # Configure the mock
            instance.config = {
                'mqtt': {
                    'listener': {
                        'root': 'test/root',
                        'result': 'result', 
                        'trace': 'trace'
                    }
                }
            }
            instance.time_window = 1.0  # 1 second time window
            instance.buffers = defaultdict(list)
            instance.process_matching_messages = Mock()
            
            yield instance

    def test_matching_messages_are_processed(self, analyser):
        """Test that matching messages are properly identified and processed"""
        # Setup test data
        now = datetime.now().timestamp()
        
        # Create result and trace messages with matching tool IDs and timestamps
        result_msg = TimestampedData(
            _timestamp=now,
            _data={"test": "result_data"},
            _source="test/root/toolbox1/tool1/result"
        )
        
        trace_msg = TimestampedData(
            _timestamp=now + 0.5,  # Within time window
            _data={"test": "trace_data"},
            _source="test/root/toolbox1/tool1/trace"
        )
        
        # Setup the buffers
        result_topic = "test/root/+/+/result"
        trace_topic = "test/root/+/+/trace"
        analyser.buffers[result_topic] = [result_msg]
        analyser.buffers[trace_topic] = [trace_msg]
        
        # Call the method under test
        MQTTDrillingDataAnalyser.find_and_process_matches(analyser)
        
        # Verify the messages were processed
        analyser.process_matching_messages.assert_called_once_with([result_msg, trace_msg])
        assert result_msg.processed
        assert trace_msg.processed

    def test_non_matching_tool_ids_not_processed(self, analyser):
        """Test that messages with different tool IDs are not processed together"""
        now = datetime.now().timestamp()
        
        # Create messages with different tool IDs
        result_msg = TimestampedData(
            _timestamp=now,
            _data={"test": "result_data"},
            _source="test/root/toolbox1/tool1/result"
        )
        
        trace_msg = TimestampedData(
            _timestamp=now + 0.5,
            _data={"test": "trace_data"},
            _source="test/root/toolbox2/tool2/trace"  # Different toolbox/tool
        )
        
        # Setup the buffers
        result_topic = "test/root/+/+/result"
        trace_topic = "test/root/+/+/trace"
        analyser.buffers[result_topic] = [result_msg]
        analyser.buffers[trace_topic] = [trace_msg]
        
        # Call the method
        MQTTDrillingDataAnalyser.find_and_process_matches(analyser)
        
        # Verify processing was not called
        analyser.process_matching_messages.assert_not_called()
        assert not result_msg.processed
        assert not trace_msg.processed

    def test_timestamps_outside_window_not_processed(self, analyser):
        """Test that messages with timestamps outside the window are not processed"""
        now = datetime.now().timestamp()
        
        # Create messages with timestamps outside the matching window
        result_msg = TimestampedData(
            _timestamp=now,
            _data={"test": "result_data"},
            _source="test/root/toolbox1/tool1/result"
        )
        
        trace_msg = TimestampedData(
            _timestamp=now + 2.0,  # Outside time window (1.0 second)
            _data={"test": "trace_data"},
            _source="test/root/toolbox1/tool1/trace"
        )
        
        # Setup the buffers
        result_topic = "test/root/+/+/result"
        trace_topic = "test/root/+/+/trace"
        analyser.buffers[result_topic] = [result_msg]
        analyser.buffers[trace_topic] = [trace_msg]
        
        # Call the method
        MQTTDrillingDataAnalyser.find_and_process_matches(analyser)
        
        # Verify processing was not called
        analyser.process_matching_messages.assert_not_called()
        assert not result_msg.processed
        assert not trace_msg.processed

    def test_buffer_cleanup(self, analyser):
        """Test that processed messages are correctly managed in the buffer"""
        now = datetime.now().timestamp()
        
        # Create matching messages
        result_msg = TimestampedData(
            _timestamp=now,
            _data={"test": "result_data"},
            _source="test/root/toolbox1/tool1/result"
        )
        
        trace_msg = TimestampedData(
            _timestamp=now + 0.5,
            _data={"test": "trace_data"},
            _source="test/root/toolbox1/tool1/trace"
        )
        
        # Setup the buffers
        result_topic = "test/root/+/+/result"
        trace_topic = "test/root/+/+/trace"
        analyser.buffers[result_topic] = [result_msg]
        analyser.buffers[trace_topic] = [trace_msg]
        
        # Mock datetime.now to return a specific time
        with patch('abyss.uos_depth_est.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.fromtimestamp(now + 5)  # Future time
            mock_datetime.fromtimestamp = datetime.fromtimestamp
            
            # Call the method
            MQTTDrillingDataAnalyser.find_and_process_matches(analyser)
        
        # Verify buffers are updated correctly
        assert len(analyser.buffers[result_topic]) == 0
        assert len(analyser.buffers[trace_topic]) == 0

    def test_multiple_matching_pairs(self, analyser):
        """Test handling multiple matching message pairs"""
        now = datetime.now().timestamp()
        
        # Create first pair of messages
        result_msg1 = TimestampedData(
            _timestamp=now,
            _data={"test": "result_data1"},
            _source="test/root/toolbox1/tool1/result"
        )
        trace_msg1 = TimestampedData(
            _timestamp=now + 0.5,
            _data={"test": "trace_data1"},
            _source="test/root/toolbox1/tool1/trace"
        )
        
        # Create second pair of messages
        result_msg2 = TimestampedData(
            _timestamp=now + 10,
            _data={"test": "result_data2"},
            _source="test/root/toolbox2/tool2/result"
        )
        trace_msg2 = TimestampedData(
            _timestamp=now + 10.3,
            _data={"test": "trace_data2"},
            _source="test/root/toolbox2/tool2/trace"
        )
        
        # Setup the buffers
        result_topic = "test/root/+/+/result"
        trace_topic = "test/root/+/+/trace"
        analyser.buffers[result_topic] = [result_msg1, result_msg2]
        analyser.buffers[trace_topic] = [trace_msg1, trace_msg2]
        
        # Call the method
        MQTTDrillingDataAnalyser.find_and_process_matches(analyser)
        
        # Verify both pairs were processed
        assert analyser.process_matching_messages.call_count == 2
        assert result_msg1.processed
        assert trace_msg1.processed
        assert result_msg2.processed
        assert trace_msg2.processed

    def test_exception_handling(self, analyser):
        """Test that exceptions are properly handled"""
        now = datetime.now().timestamp()
        
        # Create matching messages
        result_msg = TimestampedData(
            _timestamp=now,
            _data={"test": "result_data"},
            _source="test/root/toolbox1/tool1/result"
        )
        trace_msg = TimestampedData(
            _timestamp=now + 0.5,
            _data={"test": "trace_data"},
            _source="test/root/toolbox1/tool1/trace"
        )
        
        # Setup the buffers
        result_topic = "test/root/+/+/result"
        trace_topic = "test/root/+/+/trace"
        analyser.buffers[result_topic] = [result_msg]
        analyser.buffers[trace_topic] = [trace_msg]
        
        # Make process_matching_messages throw an exception
        analyser.process_matching_messages.side_effect = Exception("Test exception")
        
        # Mock logging
        with patch('abyss.uos_depth_est.logging.error') as mock_log:
            # Call the method
            MQTTDrillingDataAnalyser.find_and_process_matches(analyser)
            
            # Verify exception was logged
            mock_log.assert_called_with("Error in find_and_process_matches: %s", "Test exception")

    def test_mixed_matching_and_nonmatching_pairs(self, analyser):
        """Test handling mixed matching and non-matching message pairs"""
        now = datetime.now().timestamp()
        
        # Create matching pair
        result_msg1 = TimestampedData(
            _timestamp=now,
            _data={"test": "result_data1"},
            _source="test/root/toolbox1/tool1/result"
        )
        trace_msg1 = TimestampedData(
            _timestamp=now + 0.5,  # Within time window
            _data={"test": "trace_data1"},
            _source="test/root/toolbox1/tool1/trace"
        )
        
        # Create non-matching pair (different tool ID)
        result_msg2 = TimestampedData(
            _timestamp=now + 10,
            _data={"test": "result_data2"},
            _source="test/root/toolbox2/tool2/result"
        )
        trace_msg2 = TimestampedData(
            _timestamp=now + 10.3,
            _data={"test": "trace_data2"},
            _source="test/root/toolbox3/tool3/trace"  # Different toolbox/tool
        )
        
        # Create non-matching pair (outside time window)
        result_msg3 = TimestampedData(
            _timestamp=now + 20,
            _data={"test": "result_data3"},
            _source="test/root/toolbox4/tool4/result"
        )
        trace_msg3 = TimestampedData(
            _timestamp=now + 22,  # Outside time window (> 1.0 second)
            _data={"test": "trace_data3"},
            _source="test/root/toolbox4/tool4/trace"
        )
        
        # Setup the buffers
        result_topic = "test/root/+/+/result"
        trace_topic = "test/root/+/+/trace"
        analyser.buffers[result_topic] = [result_msg1, result_msg2, result_msg3]
        analyser.buffers[trace_topic] = [trace_msg1, trace_msg2, trace_msg3]
        
        # Call the method under test
        MQTTDrillingDataAnalyser.find_and_process_matches(analyser)
        
        # Verify only the matching pair was processed
        analyser.process_matching_messages.assert_called_once_with([result_msg1, trace_msg1])
        
        # Verify processed state of all messages
        assert result_msg1.processed
        assert trace_msg1.processed
        assert not result_msg2.processed
        assert not trace_msg2.processed
        assert not result_msg3.processed
        assert not trace_msg3.processed