"""
Notification service for proactive house checks.
Runs house checks when user location changes to detect situations that need user attention.
"""

from typing import Callable, Optional
from mcp.server import MCPServer
from mcp.router import MCPRouter


class NotificationService:
    """
    Service that runs house checks to detect and notify about potential issues.
    
    The service:
    - Runs house checks when user location changes
    - Detects devices ON in rooms other than user location
    - Automatically decides if notification is needed (no LLM required)
    - Respects user preferences (devices marked as "don't notify")
    """
    
    def __init__(self, mcp_server: MCPServer, mcp_router: MCPRouter):
        """
        Initialize notification service.
        
        Args:
            mcp_server: MCPServer instance
            mcp_router: MCPRouter instance
        """
        self.mcp_server = mcp_server
        self.mcp_router = mcp_router
        self._notification_callback: Optional[Callable[[str], None]] = None
    
    def set_notification_callback(self, callback: Callable[[str], None]) -> None:
        """
        Set callback function to be called when a notification is generated.
        
        Args:
            callback: Function that takes a message string and displays it to user
        """
        self._notification_callback = callback
    
    def run_house_check(self) -> Optional[dict]:
        """
        Run a single house check to detect potential issues.
        
        Returns:
            dict with notification result if notification was sent, None otherwise
            Format: {
                "notified": bool,
                "message": str (if notified),
                "tool_result": dict (if tool was called)
            }
        """
        # Get current state
        current_state = self.mcp_server.get_current_state()
        
        # Detect potential issues (devices ON in other rooms)
        potential_issues = self.mcp_server.detect_potential_issues()
        
        # If no potential issues, no need to notify
        if not potential_issues:
            return None
        
        # Filter out devices that are in notification_preferences (user said "keep it on")
        notification_prefs = current_state.get("notification_preferences", [])
        devices_to_notify = []
        
        for issue in potential_issues:
            room = issue["room"]
            device = issue["device"]
            device_key = f"{room} {device}"
            
            # Check if this device is in notification preferences
            if device_key not in notification_prefs:
                devices_to_notify.append(issue)
        
        # If no devices need notification (all are in preferences), return None
        if not devices_to_notify:
            print(f"[HOUSE CHECK] Devices detected but all are in notification_preferences - skipping notification")
            return None
        
        # Generate notification message
        message = self._build_notification_message(devices_to_notify, current_state)
        
        # Execute notification via chat_message tool
        tool_result = self.mcp_router.execute({
            "tool": "chat_message",
            "arguments": {
                "message": message
            }
        })
        
        print(f"[HOUSE CHECK] Notification sent: {message}")
        print(f"[HOUSE CHECK] Tool execution result: {tool_result}")
        
        # If notification was successfully sent, call callback
        if tool_result.get("success") and self._notification_callback:
            notification_message = tool_result.get("message", "")
            if notification_message:
                print(f"[HOUSE CHECK] Calling notification callback with message: {notification_message}")
                self._notification_callback(notification_message)
        
        return {
            "notified": True,
            "message": tool_result.get("message", ""),
            "tool_result": tool_result
        }
    
    def _build_notification_message(self, devices_to_notify: list, current_state: dict) -> str:
        """
        Build notification message for devices that need attention.
        
        Args:
            devices_to_notify: List of devices that should trigger notifications
            current_state: Current system state
            
        Returns:
            Notification message string
        """
        if not devices_to_notify:
            return ""
        
        # Build list of device descriptions
        device_descriptions = []
        for issue in devices_to_notify:
            room = issue["room"]
            device = issue["device"]
            device_descriptions.append(f"{room} {device}")
        
        # Format message based on number of devices
        if len(device_descriptions) == 1:
            device_desc = device_descriptions[0]
            message = f"I noticed the {device_desc} is still ON. Would you like me to turn it off?"
        else:
            devices_list = ", ".join(device_descriptions[:-1]) + f", and {device_descriptions[-1]}"
            message = f"I noticed these devices are still ON: {devices_list}. Would you like me to turn them off?"
        
        return message

