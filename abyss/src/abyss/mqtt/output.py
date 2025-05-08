from abc import ABC, abstractmethod
from typing import List, Callable, Awaitable, Optional
from abyss.mqtt.models import MatchedMessageSet

class OutputHandler(ABC):
    """Base class for handling matched message sets."""
    
    @abstractmethod
    async def handle(self, matched_set: MatchedMessageSet) -> None:
        """Handle a matched message set.
        
        Args:
            matched_set: Set of matched messages
        """
        pass
        
class CallbackOutputHandler(OutputHandler):
    """Calls a callback function with matched message sets."""
    
    def __init__(self, callback: Callable[[MatchedMessageSet], Awaitable[None]]):
        """Initialize callback output handler.
        
        Args:
            callback: Async function to call with matched message sets
        """
        self.callback = callback
        
    async def handle(self, matched_set: MatchedMessageSet) -> None:
        """Call the callback with the matched message set.
        
        Args:
            matched_set: Set of matched messages
        """
        await self.callback(matched_set)
        
class MultiOutputHandler(OutputHandler):
    """Delegates to multiple output handlers."""
    
    def __init__(self, handlers: Optional[List[OutputHandler]] = None):
        """Initialize multi-output handler.
        
        Args:
            handlers: List of output handlers
        """
        self.handlers = handlers if handlers is not None else []
        
    def add_handler(self, handler: OutputHandler):
        """Add an output handler.
        
        Args:
            handler: Output handler to add
        """
        self.handlers.append(handler)
        
    async def handle(self, matched_set: MatchedMessageSet) -> None:
        """Call all handlers with the matched message set.
        
        Args:
            matched_set: Set of matched messages
        """
        for handler in self.handlers:
            await handler.handle(matched_set)