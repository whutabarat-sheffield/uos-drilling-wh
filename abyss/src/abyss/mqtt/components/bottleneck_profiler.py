"""
Bottleneck Profiler Module

Profiles the message processing pipeline to identify performance bottlenecks.
"""

import time
import logging
import json
from contextlib import contextmanager
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading


@dataclass
class ProfileSection:
    """Represents a profiled section of code."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BottleneckReport:
    """Report identifying system bottlenecks."""
    primary_bottleneck: str
    bottleneck_percentage: float
    stage_timings: Dict[str, float]
    recommendations: List[str]
    confidence: float  # 0-1 confidence in the analysis


class BottleneckProfiler:
    """
    Profiles message processing pipeline to identify bottlenecks.
    
    Designed to run on-demand rather than continuously to minimize overhead.
    """
    
    def __init__(self, sample_rate: float = 0.1):
        """
        Initialize BottleneckProfiler.
        
        Args:
            sample_rate: Fraction of messages to profile (0.1 = 10%)
        """
        self.sample_rate = sample_rate
        self.sample_counter = 0
        
        # Timing storage per stage
        self.stage_timings: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Current profile stack for nested timing
        self._profile_stack: List[ProfileSection] = []
        self._thread_local = threading.local()
        
        # Lock for thread-safe operations
        self._lock = threading.Lock()
        
        # Profiling enabled flag
        self.profiling_enabled = False
        
    def should_profile(self) -> bool:
        """Determine if current operation should be profiled."""
        if not self.profiling_enabled:
            return False
            
        with self._lock:
            self.sample_counter += 1
            return (self.sample_counter % int(1 / self.sample_rate)) == 0
    
    @contextmanager
    def profile_section(self, section_name: str, **metadata):
        """
        Context manager to profile a code section.
        
        Args:
            section_name: Name of the section being profiled
            **metadata: Additional metadata to store with timing
            
        Usage:
            with profiler.profile_section('mqtt_receive'):
                # Code to profile
        """
        if not self.should_profile():
            yield
            return
        
        # Use thread-local storage for the profile stack
        if not hasattr(self._thread_local, 'profile_stack'):
            self._thread_local.profile_stack = []
        
        section = ProfileSection(
            name=section_name,
            start_time=time.time(),
            metadata=metadata
        )
        
        self._thread_local.profile_stack.append(section)
        
        try:
            yield
        finally:
            if self._thread_local.profile_stack and self._thread_local.profile_stack[-1] == section:
                section.end_time = time.time()
                section.duration = section.end_time - section.start_time
                
                # Store timing
                with self._lock:
                    self.stage_timings[section_name].append(section.duration)
                
                self._thread_local.profile_stack.pop()
    
    def profile_processing_pipeline(self, messages: List[Any], 
                                  correlator: Any, 
                                  processor: Any, 
                                  publisher: Any) -> Dict[str, Any]:
        """
        Profile the complete message processing pipeline.
        
        Args:
            messages: Messages to process
            correlator: Message correlator instance
            processor: Message processor instance
            publisher: Result publisher instance
            
        Returns:
            Profiling results with timings
        """
        self.profiling_enabled = True
        profile_results = {}
        
        try:
            # Profile message parsing
            with self.profile_section('json_parsing'):
                parsed_messages = []
                for msg in messages:
                    if isinstance(msg, str):
                        parsed = json.loads(msg)
                    else:
                        parsed = msg
                    parsed_messages.append(parsed)
            
            # Profile correlation
            with self.profile_section('correlation'):
                matched_sets = correlator.find_matches(parsed_messages)
            
            # Profile depth estimation
            results = []
            with self.profile_section('depth_estimation'):
                for match_set in matched_sets:
                    result = processor.process_matching_messages(match_set)
                    results.append(result)
            
            # Profile publishing
            with self.profile_section('mqtt_publish'):
                for result in results:
                    publisher.publish_result(result)
            
            # Generate report
            profile_results = self.generate_profile_report()
            
        finally:
            self.profiling_enabled = False
        
        return profile_results
    
    def profile_individual_stage(self, stage_name: str, func: Callable, *args, **kwargs) -> Any:
        """
        Profile an individual processing stage.
        
        Args:
            stage_name: Name of the stage
            func: Function to profile
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Result of the function call
        """
        with self.profile_section(stage_name):
            return func(*args, **kwargs)
    
    def generate_profile_report(self) -> Dict[str, Any]:
        """
        Generate a profiling report.
        
        Returns:
            Dictionary containing profiling analysis
        """
        with self._lock:
            if not self.stage_timings:
                return {
                    'error': 'No profiling data collected',
                    'recommendation': 'Enable profiling and process some messages'
                }
            
            # Calculate average times per stage
            avg_timings = {}
            total_time = 0
            
            for stage, timings in self.stage_timings.items():
                if timings:
                    avg_time = sum(timings) / len(timings)
                    avg_timings[stage] = avg_time
                    total_time += avg_time
            
            if total_time == 0:
                return {'error': 'No timing data available'}
            
            # Calculate percentages
            stage_percentages = {
                stage: (time / total_time * 100)
                for stage, time in avg_timings.items()
            }
            
            # Identify bottleneck
            bottleneck_stage = max(stage_percentages.items(), key=lambda x: x[1])
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                bottleneck_stage[0], 
                bottleneck_stage[1],
                avg_timings
            )
            
            # Calculate confidence based on sample count
            min_samples = min(len(timings) for timings in self.stage_timings.values() if timings)
            confidence = min(1.0, min_samples / 100)  # Full confidence at 100+ samples
            
            return {
                'primary_bottleneck': bottleneck_stage[0],
                'bottleneck_percentage': bottleneck_stage[1],
                'stage_timings_ms': {k: v * 1000 for k, v in avg_timings.items()},
                'stage_percentages': stage_percentages,
                'total_pipeline_time_ms': total_time * 1000,
                'recommendations': recommendations,
                'confidence': confidence,
                'samples_per_stage': {k: len(v) for k, v in self.stage_timings.items()}
            }
    
    def _generate_recommendations(self, bottleneck: str, percentage: float, 
                                timings: Dict[str, float]) -> List[str]:
        """Generate recommendations based on bottleneck analysis."""
        recommendations = []
        
        if bottleneck == 'depth_estimation':
            recommendations.append("Depth estimation is the bottleneck")
            if timings.get('depth_estimation', 0) > 0.1:  # >100ms
                recommendations.append("Consider optimizing the ML model or using GPU acceleration")
                recommendations.append("Evaluate if batch processing can improve throughput")
            else:
                recommendations.append("Processing time is reasonable but dominates pipeline")
                recommendations.append("Scale horizontally by adding more processing instances")
        
        elif bottleneck == 'mqtt_publish':
            recommendations.append("MQTT publishing is the bottleneck")
            recommendations.append("Check network latency to MQTT broker")
            recommendations.append("Consider batching publish operations")
            recommendations.append("Verify broker can handle the publish rate")
        
        elif bottleneck == 'correlation':
            recommendations.append("Message correlation is the bottleneck")
            recommendations.append("Review correlation time window settings")
            recommendations.append("Consider optimizing correlation algorithm")
            if len(self.stage_timings.get('correlation', [])) > 0:
                avg_correlation = sum(self.stage_timings['correlation']) / len(self.stage_timings['correlation'])
                if avg_correlation > 0.05:  # >50ms
                    recommendations.append("Correlation taking too long - check buffer sizes")
        
        elif bottleneck == 'json_parsing':
            recommendations.append("JSON parsing is the bottleneck")
            recommendations.append("Consider using a faster JSON library (ujson, orjson)")
            recommendations.append("Validate message format to avoid parsing errors")
        
        # General recommendations
        if percentage > 80:
            recommendations.append(f"{bottleneck} consumes {percentage:.1f}% of processing time")
            recommendations.append("Focus optimization efforts on this stage")
        elif percentage > 50:
            recommendations.append("Processing is relatively balanced")
            recommendations.append("Consider overall system scaling rather than optimization")
        
        return recommendations
    
    def identify_bottleneck(self, profile_data: Dict[str, Any]) -> BottleneckReport:
        """
        Analyze profile data to identify bottlenecks.
        
        Args:
            profile_data: Raw profiling data
            
        Returns:
            BottleneckReport with analysis
        """
        if 'error' in profile_data:
            return BottleneckReport(
                primary_bottleneck='unknown',
                bottleneck_percentage=0.0,
                stage_timings={},
                recommendations=[profile_data['error']],
                confidence=0.0
            )
        
        return BottleneckReport(
            primary_bottleneck=profile_data['primary_bottleneck'],
            bottleneck_percentage=profile_data['bottleneck_percentage'],
            stage_timings=profile_data['stage_timings_ms'],
            recommendations=profile_data['recommendations'],
            confidence=profile_data['confidence']
        )
    
    def reset_profiling_data(self):
        """Reset all collected profiling data."""
        with self._lock:
            self.stage_timings.clear()
            self.sample_counter = 0
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics without full analysis."""
        with self._lock:
            return {
                'stages_profiled': list(self.stage_timings.keys()),
                'samples_collected': {k: len(v) for k, v in self.stage_timings.items()},
                'profiling_enabled': self.profiling_enabled,
                'sample_rate': self.sample_rate
            }