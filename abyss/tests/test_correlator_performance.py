import pytest
import time
from unittest.mock import Mock
from abyss.mqtt.components.correlator import MessageCorrelator
from abyss.mqtt.components.simple_correlator import SimpleMessageCorrelator
from abyss.uos_depth_est import TimestampedData


class TestCorrelatorPerformance:
    """Performance comparison between complex and simple correlators"""
    
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
    
    def create_test_messages(self, num_tools=10, messages_per_tool=5):
        """Create test messages for performance testing"""
        result_messages = []
        trace_messages = []
        heads_messages = []
        
        base_time = time.time()
        
        for tool_idx in range(num_tools):
            toolbox_id = f"toolbox{tool_idx}"
            tool_id = f"tool{tool_idx}"
            
            for msg_idx in range(messages_per_tool):
                timestamp = base_time + msg_idx * 2.0  # 2 second intervals
                
                result_msg = TimestampedData(
                    _timestamp=timestamp,
                    _data=f'{{"tool": "{tool_id}", "msg": {msg_idx}}}',
                    _source=f'OPCPUBSUB/{toolbox_id}/{tool_id}/ResultManagement'
                )
                
                trace_msg = TimestampedData(
                    _timestamp=timestamp + 0.5,  # Slightly offset
                    _data=f'{{"tool": "{tool_id}", "trace": {msg_idx}}}',
                    _source=f'OPCPUBSUB/{toolbox_id}/{tool_id}/ResultManagement/Trace'
                )
                
                heads_msg = TimestampedData(
                    _timestamp=timestamp + 0.2,  # Small offset
                    _data=f'{{"tool": "{tool_id}", "heads": {msg_idx}}}',
                    _source=f'OPCPUBSUB/{toolbox_id}/{tool_id}/AssetManagement/Head'
                )
                
                result_messages.append(result_msg)
                trace_messages.append(trace_msg)
                heads_messages.append(heads_msg)
        
        return result_messages, trace_messages, heads_messages
    
    def test_performance_comparison_small_dataset(self, mock_config):
        """Compare performance with small dataset"""
        # Create test data
        result_messages, trace_messages, heads_messages = self.create_test_messages(
            num_tools=5, messages_per_tool=3
        )
        
        buffers = {
            'OPCPUBSUB/+/+/ResultManagement': result_messages,
            'OPCPUBSUB/+/+/ResultManagement/Trace': trace_messages,
            'OPCPUBSUB/+/+/AssetManagement/Head': heads_messages
        }
        
        # Mock processor
        simple_matches = []
        complex_matches = []
        
        def simple_processor(messages):
            simple_matches.append(len(messages))
            
        def complex_processor(messages):
            complex_matches.append(len(messages))
        
        # Test simple correlator
        simple_correlator = SimpleMessageCorrelator(mock_config, time_window=5.0)
        start_time = time.perf_counter()
        simple_found = simple_correlator.find_and_process_matches(buffers, simple_processor)
        simple_duration = time.perf_counter() - start_time
        
        # Reset processed flags for complex correlator test
        for msg_list in buffers.values():
            for msg in msg_list:
                if hasattr(msg, 'processed'):
                    delattr(msg, 'processed')
        
        # Test complex correlator
        complex_correlator = MessageCorrelator(mock_config, time_window=5.0)
        start_time = time.perf_counter()
        complex_found = complex_correlator.find_and_process_matches(buffers, complex_processor)
        complex_duration = time.perf_counter() - start_time
        
        # Both should find matches
        assert simple_found == complex_found == True
        assert len(simple_matches) == len(complex_matches)
        
        # Log performance results
        print(f"\nPerformance Comparison (Small Dataset):")
        print(f"Simple Correlator:  {simple_duration:.4f}s")
        print(f"Complex Correlator: {complex_duration:.4f}s")
        print(f"Speedup: {complex_duration/simple_duration:.2f}x")
        print(f"Matches found: {len(simple_matches)}")
    
    def test_performance_comparison_large_dataset(self, mock_config):
        """Compare performance with larger dataset"""
        # Create larger test data
        result_messages, trace_messages, heads_messages = self.create_test_messages(
            num_tools=20, messages_per_tool=10
        )
        
        buffers = {
            'OPCPUBSUB/+/+/ResultManagement': result_messages,
            'OPCPUBSUB/+/+/ResultManagement/Trace': trace_messages,
            'OPCPUBSUB/+/+/AssetManagement/Head': heads_messages
        }
        
        # Mock processor (lightweight)
        simple_count = 0
        complex_count = 0
        
        def simple_processor(messages):
            nonlocal simple_count
            simple_count += 1
            
        def complex_processor(messages):
            nonlocal complex_count
            complex_count += 1
        
        # Test simple correlator
        simple_correlator = SimpleMessageCorrelator(mock_config, time_window=5.0)
        start_time = time.perf_counter()
        simple_found = simple_correlator.find_and_process_matches(buffers, simple_processor)
        simple_duration = time.perf_counter() - start_time
        
        # Reset processed flags for complex correlator test
        for msg_list in buffers.values():
            for msg in msg_list:
                if hasattr(msg, 'processed'):
                    delattr(msg, 'processed')
        
        # Test complex correlator
        complex_correlator = MessageCorrelator(mock_config, time_window=5.0)
        start_time = time.perf_counter()
        complex_found = complex_correlator.find_and_process_matches(buffers, complex_processor)
        complex_duration = time.perf_counter() - start_time
        
        # Both should find matches
        assert simple_found == complex_found == True
        assert simple_count == complex_count
        
        # Log performance results
        print(f"\nPerformance Comparison (Large Dataset):")
        print(f"Simple Correlator:  {simple_duration:.4f}s")
        print(f"Complex Correlator: {complex_duration:.4f}s")
        print(f"Speedup: {complex_duration/simple_duration:.2f}x")
        print(f"Matches found: {simple_count}")
        print(f"Total messages: {len(result_messages) + len(trace_messages) + len(heads_messages)}")
    
    def test_code_complexity_metrics(self):
        """Compare code complexity metrics"""
        import inspect
        
        # Get source code for both correlators
        simple_source = inspect.getsource(SimpleMessageCorrelator)
        complex_source = inspect.getsource(MessageCorrelator)
        
        # Count lines of code (rough metric)
        simple_lines = len([line for line in simple_source.split('\n') if line.strip() and not line.strip().startswith('#')])
        complex_lines = len([line for line in complex_source.split('\n') if line.strip() and not line.strip().startswith('#')])
        
        # Count methods
        simple_methods = len([name for name, method in inspect.getmembers(SimpleMessageCorrelator, predicate=inspect.isfunction)])
        complex_methods = len([name for name, method in inspect.getmembers(MessageCorrelator, predicate=inspect.isfunction)])
        
        print(f"\nCode Complexity Comparison:")
        print(f"Simple Correlator:  {simple_lines} lines, {simple_methods} methods")
        print(f"Complex Correlator: {complex_lines} lines, {complex_methods} methods")
        print(f"Line reduction: {((complex_lines - simple_lines) / complex_lines * 100):.1f}%")
        print(f"Method reduction: {((complex_methods - simple_methods) / complex_methods * 100):.1f}%")
        
        # Simple correlator should be significantly smaller
        assert simple_lines < complex_lines
        assert simple_methods <= complex_methods