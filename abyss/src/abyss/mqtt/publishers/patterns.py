"""
Drilling operation patterns for realistic MQTT message simulation.
"""

import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Generator, Tuple


class DrillPattern(Enum):
    """Types of drilling operation patterns."""
    NORMAL = "normal"
    ACTIVE_DRILLING = "active_drilling"
    BIT_CHANGE = "bit_change"
    CALIBRATION = "calibration"
    IDLE = "idle"
    MAINTENANCE = "maintenance"


@dataclass
class PatternConfig:
    """Configuration for a drilling pattern."""
    name: str
    min_interval: float  # seconds
    max_interval: float  # seconds
    burst_probability: float  # 0.0 to 1.0
    burst_count: Tuple[int, int]  # (min, max) messages in burst
    duration_range: Tuple[float, float]  # (min, max) duration in seconds


class PatternGenerator:
    """Generate realistic drilling operation patterns."""
    
    # Pattern configurations based on real drilling scenarios
    PATTERNS = {
        DrillPattern.NORMAL: PatternConfig(
            name="Normal Operation",
            min_interval=0.8,
            max_interval=1.5,
            burst_probability=0.1,
            burst_count=(3, 5),
            duration_range=(30.0, 120.0)
        ),
        DrillPattern.ACTIVE_DRILLING: PatternConfig(
            name="Active Drilling",
            min_interval=0.1,
            max_interval=0.3,
            burst_probability=0.3,
            burst_count=(5, 10),
            duration_range=(60.0, 300.0)
        ),
        DrillPattern.BIT_CHANGE: PatternConfig(
            name="Bit Change",
            min_interval=5.0,
            max_interval=30.0,
            burst_probability=0.05,
            burst_count=(2, 3),
            duration_range=(180.0, 600.0)
        ),
        DrillPattern.CALIBRATION: PatternConfig(
            name="Calibration",
            min_interval=2.0,
            max_interval=5.0,
            burst_probability=0.2,
            burst_count=(3, 6),
            duration_range=(60.0, 180.0)
        ),
        DrillPattern.IDLE: PatternConfig(
            name="Idle",
            min_interval=10.0,
            max_interval=60.0,
            burst_probability=0.01,
            burst_count=(1, 2),
            duration_range=(300.0, 1800.0)
        ),
        DrillPattern.MAINTENANCE: PatternConfig(
            name="Maintenance",
            min_interval=30.0,
            max_interval=120.0,
            burst_probability=0.0,
            burst_count=(1, 1),
            duration_range=(600.0, 3600.0)
        )
    }
    
    # Transition probabilities between patterns
    TRANSITIONS = {
        DrillPattern.NORMAL: {
            DrillPattern.NORMAL: 0.7,
            DrillPattern.ACTIVE_DRILLING: 0.15,
            DrillPattern.CALIBRATION: 0.05,
            DrillPattern.IDLE: 0.05,
            DrillPattern.BIT_CHANGE: 0.03,
            DrillPattern.MAINTENANCE: 0.02
        },
        DrillPattern.ACTIVE_DRILLING: {
            DrillPattern.ACTIVE_DRILLING: 0.8,
            DrillPattern.NORMAL: 0.1,
            DrillPattern.BIT_CHANGE: 0.05,
            DrillPattern.CALIBRATION: 0.03,
            DrillPattern.IDLE: 0.01,
            DrillPattern.MAINTENANCE: 0.01
        },
        DrillPattern.BIT_CHANGE: {
            DrillPattern.CALIBRATION: 0.4,
            DrillPattern.NORMAL: 0.3,
            DrillPattern.ACTIVE_DRILLING: 0.2,
            DrillPattern.IDLE: 0.1,
            DrillPattern.BIT_CHANGE: 0.0,
            DrillPattern.MAINTENANCE: 0.0
        },
        DrillPattern.CALIBRATION: {
            DrillPattern.NORMAL: 0.5,
            DrillPattern.ACTIVE_DRILLING: 0.3,
            DrillPattern.IDLE: 0.1,
            DrillPattern.CALIBRATION: 0.1,
            DrillPattern.BIT_CHANGE: 0.0,
            DrillPattern.MAINTENANCE: 0.0
        },
        DrillPattern.IDLE: {
            DrillPattern.IDLE: 0.6,
            DrillPattern.NORMAL: 0.2,
            DrillPattern.CALIBRATION: 0.1,
            DrillPattern.MAINTENANCE: 0.05,
            DrillPattern.ACTIVE_DRILLING: 0.05,
            DrillPattern.BIT_CHANGE: 0.0
        },
        DrillPattern.MAINTENANCE: {
            DrillPattern.CALIBRATION: 0.5,
            DrillPattern.IDLE: 0.3,
            DrillPattern.NORMAL: 0.2,
            DrillPattern.MAINTENANCE: 0.0,
            DrillPattern.ACTIVE_DRILLING: 0.0,
            DrillPattern.BIT_CHANGE: 0.0
        }
    }
    
    def __init__(self, initial_pattern: DrillPattern = DrillPattern.NORMAL):
        self.current_pattern = initial_pattern
        self.pattern_start_time = time.time()
        self.pattern_duration = self._get_pattern_duration()
        self._in_burst = False
        self._burst_remaining = 0
    
    def _get_pattern_duration(self) -> float:
        """Get a random duration for the current pattern."""
        config = self.PATTERNS[self.current_pattern]
        return random.uniform(*config.duration_range)
    
    def _transition_pattern(self):
        """Transition to a new pattern based on probabilities."""
        transitions = self.TRANSITIONS[self.current_pattern]
        patterns = list(transitions.keys())
        probabilities = list(transitions.values())
        
        self.current_pattern = random.choices(patterns, weights=probabilities)[0]
        self.pattern_start_time = time.time()
        self.pattern_duration = self._get_pattern_duration()
    
    def get_next_interval(self) -> float:
        """Get the next interval before publishing."""
        # Check if we should transition patterns
        if time.time() - self.pattern_start_time > self.pattern_duration:
            self._transition_pattern()
        
        config = self.PATTERNS[self.current_pattern]
        
        # Handle burst mode
        if self._in_burst and self._burst_remaining > 0:
            self._burst_remaining -= 1
            if self._burst_remaining == 0:
                self._in_burst = False
            return random.uniform(0.05, 0.1)  # Very short interval during burst
        
        # Check if we should start a burst
        if random.random() < config.burst_probability:
            self._in_burst = True
            self._burst_remaining = random.randint(*config.burst_count)
            return random.uniform(0.05, 0.1)
        
        # Normal interval
        return random.uniform(config.min_interval, config.max_interval)
    
    def get_current_pattern_info(self) -> dict:
        """Get information about the current pattern."""
        config = self.PATTERNS[self.current_pattern]
        elapsed = time.time() - self.pattern_start_time
        remaining = max(0, self.pattern_duration - elapsed)
        
        return {
            'pattern': self.current_pattern.value,
            'name': config.name,
            'elapsed': elapsed,
            'remaining': remaining,
            'in_burst': self._in_burst,
            'burst_remaining': self._burst_remaining
        }
    
    def generate_intervals(self, count: int = None) -> Generator[float, None, None]:
        """Generate intervals indefinitely or for a specific count."""
        generated = 0
        while count is None or generated < count:
            yield self.get_next_interval()
            generated += 1