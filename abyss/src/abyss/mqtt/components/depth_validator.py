"""
Depth Validator Module

Provides configurable validation for depth estimation results.
Allows users to control how negative depth values are handled.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum

from .config_manager import ConfigurationManager
from .message_processor import ProcessingResult


class DepthBehavior(Enum):
    """Configurable behaviors for handling negative depth values."""
    PUBLISH = "publish"      # Always publish, even with negative values
    SKIP = "skip"           # Don't publish when negative values detected
    WARNING = "warning"     # Publish but log warnings


@dataclass
class ValidationResult:
    """Result of depth validation."""
    is_valid: bool
    action: str  # 'publish', 'skip', 'publish_with_warning'
    reason: Optional[str] = None
    warnings: Optional[List[str]] = None


class DepthValidator:
    """
    Configurable validator for depth estimation results.
    
    Allows users to control how the system handles edge cases like
    negative depth values through configuration rather than code changes.
    """
    
    def __init__(self, config: ConfigurationManager):
        """
        Initialize depth validator with configuration.
        
        Args:
            config: Configuration manager for accessing settings
        """
        self.config = config
        
        # Load validation settings
        self.negative_depth_behavior = self._get_behavior_setting()
        self.track_sequential = config.get('mqtt.depth_validation.track_sequential_negatives', True)
        self.sequential_threshold = config.get('mqtt.depth_validation.sequential_threshold', 5)
        self.sequential_window = config.get('mqtt.depth_validation.sequential_window_minutes', 5) * 60
        
        # Tracking for sequential negative depths
        self._negative_occurrences = deque(maxlen=100)
        self._last_sequential_warning = 0
        
        logging.info("Depth validator initialized", extra={
            'negative_depth_behavior': self.negative_depth_behavior.value,
            'track_sequential': self.track_sequential,
            'sequential_threshold': self.sequential_threshold
        })
    
    def _get_behavior_setting(self) -> DepthBehavior:
        """Get and validate the negative depth behavior setting."""
        behavior_str = self.config.get('mqtt.depth_validation.negative_depth_behavior', 'publish')
        
        try:
            return DepthBehavior(behavior_str.lower())
        except ValueError:
            logging.warning(f"Invalid negative_depth_behavior '{behavior_str}', defaulting to 'publish'")
            return DepthBehavior.PUBLISH
    
    def validate(self, result: ProcessingResult) -> ValidationResult:
        """
        Validate a processing result.
        
        Args:
            result: Processing result to validate
            
        Returns:
            ValidationResult indicating what action to take
        """
        # Check if result has required data
        if not result.success or result.depth_estimation is None:
            return ValidationResult(
                is_valid=False,
                action='skip',
                reason='No depth estimation available'
            )
        
        # Check for negative depth values
        negative_values = [d for d in result.depth_estimation if d < 0]
        
        if negative_values:
            return self._handle_negative_depths(result, negative_values)
        
        # All validations passed
        return ValidationResult(
            is_valid=True,
            action='publish'
        )
    
    def _handle_negative_depths(self, 
                               result: ProcessingResult,
                               negative_values: List[float]) -> ValidationResult:
        """Handle the case where negative depth values are detected."""
        # Track occurrence if enabled
        if self.track_sequential:
            self._track_negative_occurrence()
        
        # Determine action based on configuration
        if self.negative_depth_behavior == DepthBehavior.SKIP:
            return ValidationResult(
                is_valid=False,
                action='skip',
                reason=f'Negative depth values detected: {negative_values}',
                warnings=[f"Skipping publish due to negative depths: {negative_values}"]
            )
        
        elif self.negative_depth_behavior == DepthBehavior.WARNING:
            warnings = [
                f"Negative depth values detected: {negative_values}",
                f"Publishing with warning as configured"
            ]
            
            # Add sequential warning if applicable
            sequential_warning = self._check_sequential_pattern()
            if sequential_warning:
                warnings.append(sequential_warning)
            
            return ValidationResult(
                is_valid=True,
                action='publish_with_warning',
                reason='Negative depths with warning behavior',
                warnings=warnings
            )
        
        else:  # DepthBehavior.PUBLISH
            # Log for awareness but publish normally
            logging.debug("Negative depths detected, publishing as configured", extra={
                'negative_values': negative_values,
                'behavior': 'publish'
            })
            
            return ValidationResult(
                is_valid=True,
                action='publish',
                warnings=[f"Negative depths present but publishing as configured"]
            )
    
    def _track_negative_occurrence(self):
        """Track occurrence of negative depth for pattern detection."""
        current_time = time.time()
        self._negative_occurrences.append(current_time)
    
    def _check_sequential_pattern(self) -> Optional[str]:
        """Check if we have sequential negative depths that warrant a warning."""
        if not self.track_sequential:
            return None
        
        current_time = time.time()
        
        # Remove old occurrences
        cutoff_time = current_time - self.sequential_window
        valid_occurrences = [t for t in self._negative_occurrences if t > cutoff_time]
        self._negative_occurrences.clear()
        self._negative_occurrences.extend(valid_occurrences)
        
        # Check if we've hit the threshold
        if len(valid_occurrences) >= self.sequential_threshold:
            # Rate limit the warning
            if current_time - self._last_sequential_warning > 300:  # 5 minutes
                self._last_sequential_warning = current_time
                return (f"Multiple negative depths detected: {len(valid_occurrences)} "
                       f"in last {self.sequential_window/60:.1f} minutes")
        
        return None
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get current validation statistics."""
        current_time = time.time()
        cutoff_time = current_time - self.sequential_window
        recent_negatives = sum(1 for t in self._negative_occurrences if t > cutoff_time)
        
        return {
            'behavior': self.negative_depth_behavior.value,
            'recent_negative_count': recent_negatives,
            'sequential_threshold': self.sequential_threshold,
            'tracking_enabled': self.track_sequential
        }