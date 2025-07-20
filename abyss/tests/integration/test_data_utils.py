"""
Shared utilities for test data transformation.

This module provides functions to load and transform test data
from various formats, particularly OPC UA JSON format.
"""

import json
import os
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from src.abyss.uos_depth_est import TimestampedData


def load_opcua_test_data(test_data_path: str) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """
    Load OPC UA format test data from JSON files.
    
    Args:
        test_data_path: Directory containing test JSON files
        
    Returns:
        Tuple of (result_data, trace_data, heads_data) or None values if files not found
    """
    result_data = None
    trace_data = None
    heads_data = None
    
    try:
        # Load ResultManagement data
        with open(os.path.join(test_data_path, "ResultManagement.json"), 'r') as f:
            result_json = json.load(f)
            result_data = result_json.get('Messages', {}).get('Payload', {})
            
        # Load Trace data
        with open(os.path.join(test_data_path, "Trace.json"), 'r') as f:
            trace_json = json.load(f)
            trace_data = trace_json.get('Messages', {}).get('Payload', {})
            
        # Load Heads data
        with open(os.path.join(test_data_path, "Heads.json"), 'r') as f:
            heads_json = json.load(f)
            heads_data = heads_json.get('Messages', {}).get('Payload', {})
            
    except Exception as e:
        print(f"Warning: Could not load test data from JSON files: {e}")
        
    return result_data, trace_data, heads_data


def transform_opcua_result_data(opcua_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform OPC UA ResultManagement data to expected format.
    
    Args:
        opcua_payload: Raw OPC UA payload data
        
    Returns:
        Transformed data in expected format
    """
    return {
        'ResultManagement': {
            'Results': [{
                'ResultMetaData': {
                    'SerialNumber': opcua_payload.get(
                        'nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecgabriele.ResultManagement.Results.0.ResultMetaData.SerialNumber', 
                        {}
                    ).get('Value', '0'),
                    'ResultId': opcua_payload.get(
                        'nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecgabriele.ResultManagement.Results.0.ResultMetaData.ResultId', 
                        {}
                    ).get('Value', '432')
                },
                'ResultContent': {
                    'StepResults': [{
                        'StepResultValues': {
                            'IntensityTorqueEmpty': opcua_payload.get(
                                'nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecgabriele.ResultManagement.Results.0.ResultContent.StepResults.0.StepResultValues.IntensityTorqueEmpty.MeasuredValue', 
                                {}
                            ).get('Value', []),
                            'IntensityThrustEmpty': opcua_payload.get(
                                'nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecgabriele.ResultManagement.Results.0.ResultContent.StepResults.0.StepResultValues.IntensityThrustEmpty.MeasuredValue', 
                                {}
                            ).get('Value', []),
                            'StepNumber': opcua_payload.get(
                                'nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecgabriele.ResultManagement.Results.0.ResultContent.StepResults.0.StepResultValues.StepNumber.MeasuredValue', 
                                {}
                            ).get('Value', [])
                        }
                    }]
                }
            }]
        }
    }


def transform_opcua_trace_data(opcua_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform OPC UA Trace data to expected format.
    
    Args:
        opcua_payload: Raw OPC UA payload data
        
    Returns:
        Transformed data in expected format
    """
    return {
        'ResultManagement': {
            'Results': [{
                'ResultContent': {
                    'Trace': {
                        'StepTraces': {
                            'PositionTrace': {
                                'StepResultId': opcua_payload.get(
                                    'nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecgabriele.ResultManagement.Results.0.ResultContent.Trace.StepTraces.PositionTrace.StepResultId', 
                                    {}
                                ).get('Value', '432'),
                                'StepTraceContent': [{
                                    'Values': opcua_payload.get(
                                        'nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecgabriele.ResultManagement.Results.0.ResultContent.Trace.StepTraces.PositionTrace.StepTraceContent[0].Values', 
                                        {}
                                    ).get('Value', [])[:100]  # Limit to first 100 values
                                }]
                            },
                            'IntensityThrustTrace': {
                                'StepTraceContent': [{
                                    'Values': opcua_payload.get(
                                        'nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecgabriele.ResultManagement.Results.0.ResultContent.Trace.StepTraces.IntensityThrustTrace.StepTraceContent[0].Values', 
                                        {}
                                    ).get('Value', [])[:100] if 'IntensityThrustTrace' in str(opcua_payload) else []
                                }]
                            },
                            'IntensityTorqueTrace': {
                                'StepTraceContent': [{
                                    'Values': opcua_payload.get(
                                        'nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecgabriele.ResultManagement.Results.0.ResultContent.Trace.StepTraces.IntensityTorqueTrace.StepTraceContent[0].Values', 
                                        {}
                                    ).get('Value', [])[:100] if 'IntensityTorqueTrace' in str(opcua_payload) else []
                                }]
                            }
                        }
                    }
                }
            }]
        }
    }


def transform_opcua_heads_data(opcua_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform OPC UA Heads data to expected format.
    
    Args:
        opcua_payload: Raw OPC UA payload data
        
    Returns:
        Transformed data in expected format
    """
    return {
        'AssetManagement': {
            'Assets': {
                'Heads': [{
                    'Identification': {
                        'SerialNumber': opcua_payload.get(
                            'nsu=http://airbus.com/IJT/ADrilling;s=Objects.DeviceSet.setitecgabriele.AssetManagement.Assets.Heads.0.Identification.SerialNumber', 
                            {}
                        ).get('Value', 'HEAD001')
                    }
                }]
            }
        }
    }


def create_synthetic_test_message(message_type: str, index: int, base_time: float) -> TimestampedData:
    """
    Create a synthetic test message when real data is not available.
    
    Args:
        message_type: Type of message ('result', 'trace', or 'heads')
        index: Message index for variation
        base_time: Base timestamp
        
    Returns:
        TimestampedData message
    """
    if message_type == 'result':
        source = f"OPCPUBSUB/TEST001/tool{index%3}/ResultManagement"
        data = {
            'ResultManagement': {
                'Results': [{
                    'ResultMetaData': {
                        'SerialNumber': f'SN{index:05d}',
                        'ResultId': f'RID{index:05d}'
                    }
                }]
            }
        }
    elif message_type == 'trace':
        source = f"OPCPUBSUB/TEST001/tool{index%3}/ResultManagement/Trace" 
        data = {
            'ResultManagement': {
                'Results': [{
                    'ResultContent': {
                        'Trace': {
                            'StepTraces': {
                                'PositionTrace': {'StepTraceContent': [{'Values': [index * 0.1]}]},
                                'IntensityThrustTrace': {'StepTraceContent': [{'Values': [index * 0.2]}]},
                                'IntensityTorqueTrace': {'StepTraceContent': [{'Values': [index * 0.3]}]}
                            }
                        }
                    }
                }]
            }
        }
    else:  # heads
        source = f"OPCPUBSUB/TEST001/tool{index%3}/AssetManagement/Heads"
        data = {
            'AssetManagement': {
                'Assets': {
                    'Heads': [{
                        'Identification': {
                            'SerialNumber': f'HEAD{index:03d}'
                        }
                    }]
                }
            }
        }
        
    return TimestampedData(
        _data=data,
        _timestamp=base_time + index * 0.01,
        _source=source
    )