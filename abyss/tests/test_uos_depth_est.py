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


    def test_out_of_order_mixed_messages(self, analyser):
        """Test handling of out-of-order messages with mixed paired and unpaired messages"""
        now = datetime.now().timestamp()
        
        # Create 10 pairs of messages (5 matching, 5 non-matching)
        # and add them in a non-sequential order
        
        # Pair 1 - Matching
        result_msg1 = TimestampedData(
            _timestamp=now,
            _data={"test": "result_data1"},
            _source="test/root/toolbox1/tool1/result"
        )
        trace_msg1 = TimestampedData(
            _timestamp=now + 0.3,
            _data={"test": "trace_data1"},
            _source="test/root/toolbox1/tool1/trace"
        )
        
        # Pair 2 - Matching
        result_msg2 = TimestampedData(
            _timestamp=now + 5.0,
            _data={"test": "result_data2"},
            _source="test/root/toolbox2/tool2/result"
        )
        trace_msg2 = TimestampedData(
            _timestamp=now + 5.2,
            _data={"test": "trace_data2"},
            _source="test/root/toolbox2/tool2/trace"
        )
        
        # Pair 3 - Matching
        result_msg3 = TimestampedData(
            _timestamp=now + 10.0,
            _data={"test": "result_data3"},
            _source="test/root/toolbox3/tool3/result"
        )
        trace_msg3 = TimestampedData(
            _timestamp=now + 10.4,
            _data={"test": "trace_data3"},
            _source="test/root/toolbox3/tool3/trace"
        )
        
        # Pair 4 - Matching
        result_msg4 = TimestampedData(
            _timestamp=now + 15.0,
            _data={"test": "result_data4"},
            _source="test/root/toolbox4/tool4/result"
        )
        trace_msg4 = TimestampedData(
            _timestamp=now + 15.3,
            _data={"test": "trace_data4"},
            _source="test/root/toolbox4/tool4/trace"
        )
        
        # Pair 5 - Matching
        result_msg5 = TimestampedData(
            _timestamp=now + 20.0,
            _data={"test": "result_data5"},
            _source="test/root/toolbox5/tool5/result"
        )
        trace_msg5 = TimestampedData(
            _timestamp=now + 20.4,
            _data={"test": "trace_data5"},
            _source="test/root/toolbox5/tool5/trace"
        )
        
        # Pair 6 - Non-matching (different tool ID)
        result_msg6 = TimestampedData(
            _timestamp=now + 25.0,
            _data={"test": "result_data6"},
            _source="test/root/toolbox6/tool6/result"
        )
        trace_msg6 = TimestampedData(
            _timestamp=now + 25.3,
            _data={"test": "trace_data6"},
            _source="test/root/toolbox6A/tool6A/trace"  # Different toolbox/tool
        )
        
        # Pair 7 - Non-matching (outside time window)
        result_msg7 = TimestampedData(
            _timestamp=now + 30.0,
            _data={"test": "result_data7"},
            _source="test/root/toolbox7/tool7/result"
        )
        trace_msg7 = TimestampedData(
            _timestamp=now + 31.5,  # Outside time window (> 1.0 second)
            _data={"test": "trace_data7"},
            _source="test/root/toolbox7/tool7/trace"
        )
        
        # Pair 8 - Non-matching (different tool ID)
        result_msg8 = TimestampedData(
            _timestamp=now + 35.0,
            _data={"test": "result_data8"},
            _source="test/root/toolbox8/tool8/result"
        )
        trace_msg8 = TimestampedData(
            _timestamp=now + 35.3,
            _data={"test": "trace_data8"},
            _source="test/root/toolbox8B/tool8B/trace"  # Different toolbox/tool
        )
        
        # Pair 9 - Non-matching (outside time window)
        result_msg9 = TimestampedData(
            _timestamp=now + 40.0,
            _data={"test": "result_data9"},
            _source="test/root/toolbox9/tool9/result"
        )
        trace_msg9 = TimestampedData(
            _timestamp=now + 41.2,  # Outside time window (> 1.0 second)
            _data={"test": "trace_data9"},
            _source="test/root/toolbox9/tool9/trace"
        )
        
        # Pair 10 - Non-matching (different tool ID)
        result_msg10 = TimestampedData(
            _timestamp=now + 45.0,
            _data={"test": "result_data10"},
            _source="test/root/toolbox10/tool10/result"
        )
        trace_msg10 = TimestampedData(
            _timestamp=now + 45.3,
            _data={"test": "trace_data10"},
            _source="test/root/toolbox10C/tool10C/trace"  # Different toolbox/tool
        )
        
        # Add all messages to the buffer in a non-sequential order
        result_topic = "test/root/+/+/result"
        trace_topic = "test/root/+/+/trace"
        
        # Adding messages in a mixed, non-sequential order 
        # to test the analyzer's ability to match related messages
        # regardless of the order they arrive
        analyser.buffers[result_topic] = [
            result_msg3,  # Matching pair
            result_msg8,  # Non-matching (different tool)
            result_msg1,  # Matching pair
            result_msg6,  # Non-matching (different tool)
            result_msg5,  # Matching pair
            result_msg9,  # Non-matching (outside window)
            result_msg2,  # Matching pair
            result_msg10, # Non-matching (different tool)
            result_msg4,  # Matching pair
            result_msg7,  # Non-matching (outside window)
        ]
        
        analyser.buffers[trace_topic] = [
            trace_msg10,  # Non-matching (different tool)
            trace_msg4,   # Matching pair
            trace_msg7,   # Non-matching (outside window)
            trace_msg2,   # Matching pair
            trace_msg9,   # Non-matching (outside window)
            trace_msg1,   # Matching pair
            trace_msg6,   # Non-matching (different tool)
            trace_msg3,   # Matching pair
            trace_msg8,   # Non-matching (different tool)
            trace_msg5,   # Matching pair
        ]
        
        # Call the method under test
        MQTTDrillingDataAnalyser.find_and_process_matches(analyser)
        
        # Verify that only the matching pairs were processed
        # process_matching_messages should be called 5 times - once for each valid pair
        assert analyser.process_matching_messages.call_count == 5
        
        # Verify only matching pairs were processed
        assert result_msg1.processed and trace_msg1.processed  # Pair 1
        assert result_msg2.processed and trace_msg2.processed  # Pair 2
        assert result_msg3.processed and trace_msg3.processed  # Pair 3
        assert result_msg4.processed and trace_msg4.processed  # Pair 4
        assert result_msg5.processed and trace_msg5.processed  # Pair 5
        
        # Verify that non-matching pairs were not processed
        assert not result_msg6.processed and not trace_msg6.processed  # Pair 6
        assert not result_msg7.processed and not trace_msg7.processed  # Pair 7
        assert not result_msg8.processed and not trace_msg8.processed  # Pair 8
        assert not result_msg9.processed and not trace_msg9.processed  # Pair 9
        assert not result_msg10.processed and not trace_msg10.processed  # Pair 10

    
    def test_out_of_order_mixed_messages_with_cleanup_checks(self, analyser):
        """Test handling of out-of-order messages with mixed paired and unpaired messages"""
        now = datetime.now().timestamp()
        
        # Create 10 pairs of messages (5 matching, 5 non-matching)
        # and add them in a non-sequential order
        
        # Pair 1 - Matching
        result_msg1 = TimestampedData(
            _timestamp=now,
            _data={"test": "result_data1"},
            _source="test/root/toolbox1/tool1/result"
        )
        trace_msg1 = TimestampedData(
            _timestamp=now + 0.3,
            _data={"test": "trace_data1"},
            _source="test/root/toolbox1/tool1/trace"
        )
        
        # Pair 2 - Matching
        result_msg2 = TimestampedData(
            _timestamp=now + 5.0,
            _data={"test": "result_data2"},
            _source="test/root/toolbox2/tool2/result"
        )
        trace_msg2 = TimestampedData(
            _timestamp=now + 5.2,
            _data={"test": "trace_data2"},
            _source="test/root/toolbox2/tool2/trace"
        )
        
        # Pair 3 - Matching
        result_msg3 = TimestampedData(
            _timestamp=now + 10.0,
            _data={"test": "result_data3"},
            _source="test/root/toolbox3/tool3/result"
        )
        trace_msg3 = TimestampedData(
            _timestamp=now + 10.4,
            _data={"test": "trace_data3"},
            _source="test/root/toolbox3/tool3/trace"
        )
        
        # Pair 4 - Matching
        result_msg4 = TimestampedData(
            _timestamp=now + 15.0,
            _data={"test": "result_data4"},
            _source="test/root/toolbox4/tool4/result"
        )
        trace_msg4 = TimestampedData(
            _timestamp=now + 15.3,
            _data={"test": "trace_data4"},
            _source="test/root/toolbox4/tool4/trace"
        )
        
        # Pair 5 - Matching
        result_msg5 = TimestampedData(
            _timestamp=now + 20.0,
            _data={"test": "result_data5"},
            _source="test/root/toolbox5/tool5/result"
        )
        trace_msg5 = TimestampedData(
            _timestamp=now + 20.4,
            _data={"test": "trace_data5"},
            _source="test/root/toolbox5/tool5/trace"
        )
        
        # Pair 6 - Non-matching (different tool ID)
        result_msg6 = TimestampedData(
            _timestamp=now + 25.0,
            _data={"test": "result_data6"},
            _source="test/root/toolbox6/tool6/result"
        )
        trace_msg6 = TimestampedData(
            _timestamp=now + 25.3,
            _data={"test": "trace_data6"},
            _source="test/root/toolbox6A/tool6A/trace"  # Different toolbox/tool
        )
        
        # Pair 7 - Non-matching (outside time window)
        result_msg7 = TimestampedData(
            _timestamp=now + 30.0,
            _data={"test": "result_data7"},
            _source="test/root/toolbox7/tool7/result"
        )
        trace_msg7 = TimestampedData(
            _timestamp=now + 31.5,  # Outside time window (> 1.0 second)
            _data={"test": "trace_data7"},
            _source="test/root/toolbox7/tool7/trace"
        )
        
        # Pair 8 - Non-matching (different tool ID)
        result_msg8 = TimestampedData(
            _timestamp=now + 35.0,
            _data={"test": "result_data8"},
            _source="test/root/toolbox8/tool8/result"
        )
        trace_msg8 = TimestampedData(
            _timestamp=now + 35.3,
            _data={"test": "trace_data8"},
            _source="test/root/toolbox8B/tool8B/trace"  # Different toolbox/tool
        )
        
        # Pair 9 - Non-matching (outside time window)
        result_msg9 = TimestampedData(
            _timestamp=now + 40.0,
            _data={"test": "result_data9"},
            _source="test/root/toolbox9/tool9/result"
        )
        trace_msg9 = TimestampedData(
            _timestamp=now + 41.2,  # Outside time window (> 1.0 second)
            _data={"test": "trace_data9"},
            _source="test/root/toolbox9/tool9/trace"
        )
        
        # Pair 10 - Non-matching (different tool ID)
        result_msg10 = TimestampedData(
            _timestamp=now + 45.0,
            _data={"test": "result_data10"},
            _source="test/root/toolbox10/tool10/result"
        )
        trace_msg10 = TimestampedData(
            _timestamp=now + 45.3,
            _data={"test": "trace_data10"},
            _source="test/root/toolbox10C/tool10C/trace"  # Different toolbox/tool
        )
        
        # Add all messages to the buffer in a non-sequential order
        result_topic = "test/root/+/+/result"
        trace_topic = "test/root/+/+/trace"
        
        # Store original messages for comparison
        result_messages = [
            result_msg3,  # Matching pair
            result_msg8,  # Non-matching (different tool)
            result_msg1,  # Matching pair
            result_msg6,  # Non-matching (different tool)
            result_msg5,  # Matching pair
            result_msg9,  # Non-matching (outside window)
            result_msg2,  # Matching pair
            result_msg10, # Non-matching (different tool)
            result_msg4,  # Matching pair
            result_msg7,  # Non-matching (outside window)
        ]
        
        trace_messages = [
            trace_msg10,  # Non-matching (different tool)
            trace_msg4,   # Matching pair
            trace_msg7,   # Non-matching (outside window)
            trace_msg2,   # Matching pair
            trace_msg9,   # Non-matching (outside window)
            trace_msg1,   # Matching pair
            trace_msg6,   # Non-matching (different tool)
            trace_msg3,   # Matching pair
            trace_msg8,   # Non-matching (different tool)
            trace_msg5,   # Matching pair
        ]
        
        # Adding messages to the buffer
        analyser.buffers[result_topic] = result_messages.copy()
        analyser.buffers[trace_topic] = trace_messages.copy()
        
        # Call the method under test
        MQTTDrillingDataAnalyser.find_and_process_matches(analyser)
        
        # Verify that only the matching pairs were processed
        # process_matching_messages should be called 5 times - once for each valid pair
        assert analyser.process_matching_messages.call_count == 5
        
        # Verify only matching pairs were processed
        assert result_msg1.processed and trace_msg1.processed  # Pair 1
        assert result_msg2.processed and trace_msg2.processed  # Pair 2
        assert result_msg3.processed and trace_msg3.processed  # Pair 3
        assert result_msg4.processed and trace_msg4.processed  # Pair 4
        assert result_msg5.processed and trace_msg5.processed  # Pair 5
        
        # Verify that non-matching pairs were not processed
        assert not result_msg6.processed and not trace_msg6.processed  # Pair 6
        assert not result_msg7.processed and not trace_msg7.processed  # Pair 7
        assert not result_msg8.processed and not trace_msg8.processed  # Pair 8
        assert not result_msg9.processed and not trace_msg9.processed  # Pair 9
        assert not result_msg10.processed and not trace_msg10.processed  # Pair 10
        
        # Verify that processed messages have been removed from the buffers
        # Only unprocessed messages should remain
        remaining_result_messages = analyser.buffers[result_topic]
        remaining_trace_messages = analyser.buffers[trace_topic]
        
        # Check that processed messages were removed
        assert result_msg1 not in remaining_result_messages
        assert result_msg2 not in remaining_result_messages
        assert result_msg3 not in remaining_result_messages
        assert result_msg4 not in remaining_result_messages
        assert result_msg5 not in remaining_result_messages
        
        assert trace_msg1 not in remaining_trace_messages
        assert trace_msg2 not in remaining_trace_messages
        assert trace_msg3 not in remaining_trace_messages
        assert trace_msg4 not in remaining_trace_messages
        assert trace_msg5 not in remaining_trace_messages
        
        # Check that unprocessed messages are still in the buffers
        assert result_msg6 in remaining_result_messages
        assert result_msg7 in remaining_result_messages
        assert result_msg8 in remaining_result_messages
        assert result_msg9 in remaining_result_messages
        assert result_msg10 in remaining_result_messages
        
        assert trace_msg6 in remaining_trace_messages
        assert trace_msg7 in remaining_trace_messages
        assert trace_msg8 in remaining_trace_messages
        assert trace_msg9 in remaining_trace_messages
        assert trace_msg10 in remaining_trace_messages
        
        # Now let's test cleanup of old messages
        # Mock datetime.now to return a time far in the future
        with patch('abyss.uos_depth_est.datetime') as mock_datetime:
            # Set time to be beyond the cleanup interval
            future_time = datetime.fromtimestamp(now + analyser.cleanup_interval + 5)
            mock_datetime.now.return_value = future_time
            mock_datetime.fromtimestamp = datetime.fromtimestamp
            
            # Call cleanup method
            MQTTDrillingDataAnalyser.cleanup_old_messages(analyser)
            
            # Verify all processed messages were removed
            assert len(analyser.buffers[result_topic]) == 5
            assert len(analyser.buffers[trace_topic]) == 5