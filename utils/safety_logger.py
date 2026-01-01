"""
Minimal safety logging for MCP system.
Console-only logging - no file I/O.
"""

import datetime
from typing import Optional


class SafetyLogger:
    """
    Minimal logger for safety features.
    Prints to console only.
    """
    
    def __init__(self):
        """Initialize logger."""
        pass
    
    def _format_timestamp(self) -> str:
        """Get formatted timestamp."""
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def log_tool_call(self, tool: str, arguments: dict, success: bool, result: Optional[dict] = None):
        """
        Log a tool call attempt.
        
        Args:
            tool: Tool name
            arguments: Tool arguments
            success: Whether the call succeeded
            result: Optional result dict
        """
        timestamp = self._format_timestamp()
        status = "SUCCESS" if success else "FAILED"
        
        print(f"[{timestamp}] TOOL_CALL: {tool} | Status: {status}")
        print(f"  Arguments: {arguments}")
        
        if result and not success:
            error = result.get("error", "Unknown error")
            print(f"  Error: {error}")
    
    def log_rejected_action(self, reason: str, tool: str = None, arguments: dict = None):
        """
        Log a rejected action.
        
        Args:
            reason: Why the action was rejected
            tool: Optional tool name
            arguments: Optional tool arguments
        """
        timestamp = self._format_timestamp()
        print(f"[{timestamp}] REJECTED: {reason}")
        if tool:
            print(f"  Tool: {tool}")
        if arguments:
            print(f"  Arguments: {arguments}")
    
    def log_reminder_prevented(self, item: str):
        """
        Log when a reminder is prevented due to do_not_remind list.
        
        Args:
            item: Item that was prevented from being reminded
        """
        timestamp = self._format_timestamp()
        print(f"[{timestamp}] REMINDER_PREVENTED: '{item}' is in do_not_remind list")


# Global logger instance
_logger = SafetyLogger()


def log_tool_call(tool: str, arguments: dict, success: bool, result: Optional[dict] = None):
    """Log a tool call."""
    _logger.log_tool_call(tool, arguments, success, result)


def log_rejected_action(reason: str, tool: str = None, arguments: dict = None):
    """Log a rejected action."""
    _logger.log_rejected_action(reason, tool, arguments)


def log_reminder_prevented(item: str):
    """Log when a reminder is prevented."""
    _logger.log_reminder_prevented(item)

