from typing import Dict, List, Optional, Set
import time
import logging
from abyss.mqtt.models import Message, MatchedMessageSet

class CorrelationEngine:
    """Engine for correlating messages based on matching keys."""
    
    def __init__(self, required_types: Set[str], match_timeout: float = 60.0):
        """Initialize correlation engine."""
        self.required_types = required_types
        self.match_timeout = match_timeout
        self.partial_matches: Dict[str, Dict[str, Message]] = {}
        self.complete_matches: List[MatchedMessageSet] = []
        self.last_cleanup = time.time()
    
    def add_message(self, message: Message) -> Optional[MatchedMessageSet]:
        """Add a message and check if it completes a match."""
        key = message.key
        type_id = message.type_id
        
        # Create entry for this key if it doesn't exist
        if key not in self.partial_matches:
            self.partial_matches[key] = {}
            
        # Add this message to the partial match
        self.partial_matches[key][type_id] = message
        
        # Always clean up expired matches when adding a new message
        self._cleanup_expired_matches()
        
        # Check if we have a complete match
        if set(self.partial_matches[key].keys()) == self.required_types:
            # Create matched set
            matched_set = MatchedMessageSet(
                key=key,
                messages=self.partial_matches[key].copy()
            )
            
            # Remove the matched messages from partial matches
            del self.partial_matches[key]
            
            # Store in complete matches
            self.complete_matches.append(matched_set)
            
            return matched_set
            
        # No complete match yet
        return None
    
    def get_next_match(self) -> Optional[MatchedMessageSet]:
        """Get the next available complete match."""
        if self.complete_matches:
            return self.complete_matches.pop(0)
        return None
        
    def get_all_matches(self) -> List[MatchedMessageSet]:
        """Get all available complete matches."""
        matches = self.complete_matches.copy()
        self.complete_matches = []
        return matches
    
    def _cleanup_expired_matches(self) -> None:
        """Remove partial matches that have expired."""
        current_time = time.time()
        # Always run cleanup, don't throttle in tests
        
        expired_keys = []
        
        for key, matches in self.partial_matches.items():
            # Check if any message is older than the timeout
            for _, message in matches.items():
                if current_time - message.timestamp > self.match_timeout:
                    expired_keys.append(key)
                    logging.warning(f"Correlation timeout for key {key}, type {message.type_id}")
                    break
                    
        # Remove expired matches
        for key in expired_keys:
            del self.partial_matches[key]
            
        # Update cleanup timestamp
        self.last_cleanup = current_time