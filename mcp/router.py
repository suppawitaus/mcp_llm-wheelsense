"""
MCP Execution Router - dispatches LLM tool calls to MCP server tools.
"""

from mcp.server import MCPServer
from core.state import StateManager
from utils.safety_logger import log_tool_call, log_rejected_action, log_reminder_prevented


class MCPRouter:
    """
    Routes LLM tool calls to the appropriate MCP server tools.
    
    Accepts LLM response format:
    {
        "tool": "tool_name",
        "arguments": {...}
    }
    """
    
    def __init__(self, mcp_server: MCPServer):
        """
        Initialize router with an MCP server instance.
        
        Args:
            mcp_server: MCPServer instance to route calls to
        """
        self.mcp_server = mcp_server
        
        # Tool name mapping
        self._tool_map = {
            "chat_message": self._route_chat_message,
            "e_device_control": self._route_e_device_control,
            "schedule_modifier": self._route_schedule_modifier,
            "rag_query": self._route_rag_query,
        }
    
    def execute(self, llm_response: dict, user_message: str = None) -> dict:
        """
        Execute a tool call from LLM response.
        
        Args:
            llm_response: Dict with format:
                {
                    "tool": str,  # Tool name
                    "arguments": dict  # Tool arguments
                }
        
        Returns:
            dict: Tool execution result from MCP server
        """
        # Validate input format
        if not isinstance(llm_response, dict):
            result = {
                "success": False,
                "error": "LLM response must be a dictionary",
                "tool": None
            }
            log_rejected_action("Invalid LLM response format (not a dictionary)")
            return result
        
        tool_name = llm_response.get("tool")
        arguments = llm_response.get("arguments", {})
        
        if not tool_name:
            result = {
                "success": False,
                "error": "Missing 'tool' field in LLM response",
                "tool": None
            }
            log_rejected_action("Missing 'tool' field in LLM response")
            return result
        
        if not isinstance(arguments, dict):
            result = {
                "success": False,
                "error": "Arguments must be a dictionary",
                "tool": tool_name
            }
            log_rejected_action("Invalid arguments format (not a dictionary)", tool_name, arguments)
            return result
        
        # Route to appropriate tool handler
        if tool_name not in self._tool_map:
            result = {
                "success": False,
                "error": f"Unknown tool: '{tool_name}'. Available tools: {list(self._tool_map.keys())}",
                "tool": tool_name
            }
            log_rejected_action(f"Unknown tool: '{tool_name}'", tool_name, arguments)
            return result
        
        try:
            handler = self._tool_map[tool_name]
            print(f"[ROUTER DEBUG] Routing tool '{tool_name}' with arguments: {arguments}")
            
            # Pass user_message to schedule_modifier handler
            if tool_name == "schedule_modifier":
                result = handler(arguments, user_message=user_message)
            else:
                result = handler(arguments)
            
            print(f"[ROUTER DEBUG] Tool '{tool_name}' result: success={result.get('success')}, error={result.get('error')}")
            if result.get("success"):
                print(f"[ROUTER DEBUG] Tool '{tool_name}' result details: {result}")
            
            # Log every tool call
            log_tool_call(tool_name, arguments, result.get("success", False), result)
            
            return result
        except Exception as e:
            result = {
                "success": False,
                "error": f"Tool execution error: {str(e)}",
                "tool": tool_name
            }
            log_rejected_action(f"Tool execution error: {str(e)}", tool_name, arguments)
            return result
    
    def _route_chat_message(self, arguments: dict) -> dict:
        """Route chat_message tool call with reminder prevention."""
        message = arguments.get("message")
        if message is None:
            return {
                "success": False,
                "tool": "chat_message",
                "message": "",
                "error": "Missing required argument: 'message'"
            }
        
        # Check if this is a reminder that should be prevented
        state_manager = self.mcp_server.state_manager
        message_lower = message.lower().strip()
        
        # Check do_not_remind list - check both exact match and if message contains any item
        do_not_remind_list = state_manager.get_do_not_remind()
        
        # Check exact match
        if message_lower in do_not_remind_list:
            log_reminder_prevented(message_lower)
            return {
                "success": False,
                "tool": "chat_message",
                "message": "",
                "error": f"Reminder prevented: '{message_lower}' is in do_not_remind list"
            }
        
        # Check if message contains any item from do_not_remind list
        for item in do_not_remind_list:
            if item.lower() in message_lower or message_lower in item.lower():
                log_reminder_prevented(item)
                return {
                    "success": False,
                    "tool": "chat_message",
                    "message": "",
                    "error": f"Reminder prevented: '{item}' is in do_not_remind list"
                }
        
        return self.mcp_server.chat_message(message)
    
    def _route_e_device_control(self, arguments: dict) -> dict:
        """Route e_device_control tool call."""
        room = arguments.get("room")
        device = arguments.get("device")
        action = arguments.get("action")
        
        missing = []
        if room is None:
            missing.append("room")
        if device is None:
            missing.append("device")
        if action is None:
            missing.append("action")
        
        if missing:
            return {
                "success": False,
                "tool": "e_device_control",
                "room": room or "",
                "device": device or "",
                "action": action or "",
                "previous_state": None,
                "new_state": None,
                "message": "",
                "error": f"Missing required arguments: {', '.join(missing)}"
            }
        
        return self.mcp_server.e_device_control(room, device, action)
    
    def _route_schedule_modifier(self, arguments: dict, user_message: str = None) -> dict:
        """Route schedule_modifier tool call."""
        modify_type = arguments.get("modify_type")  # Required - no default
        time = arguments.get("time")  # Optional
        activity = arguments.get("activity")  # Optional
        old_time = arguments.get("old_time")  # Optional, for change operation
        old_activity = arguments.get("old_activity")  # Optional, for change operation
        
        # Validate modify_type is provided
        if not modify_type:
            return {
                "success": False,
                "tool": "schedule_modifier",
                "message": "",
                "error": "modify_type is required. Must be 'add', 'delete', or 'change'"
            }
        
        return self.mcp_server.schedule_modifier(
            modify_type=modify_type,
            time=time,
            activity=activity,
            old_time=old_time,
            old_activity=old_activity,
            user_message=user_message
        )
    
    def _route_rag_query(self, arguments: dict) -> dict:
        """Route rag_query tool call."""
        query = arguments.get("query")
        user_condition = arguments.get("user_condition")  # Optional
        
        if query is None:
            return {
                "success": False,
                "tool": "rag_query",
                "found": False,
                "chunks": None,
                "error": "Missing required argument: 'query'"
            }
        
        return self.mcp_server.rag_query(query, user_condition)
    
    def process_user_response_for_preferences(self, user_message: str, recent_notification: dict = None) -> dict:
        """
        Process user message to detect "leave it on" intent and update notification preferences.
        
        This should be called after a notification to check if user wants to disable
        future notifications for the mentioned device.
        
        Args:
            user_message: User's response message
            recent_notification: Optional dict with recent notification info:
                {
                    "room": str,
                    "device": str,
                    "message": str
                }
        
        Returns:
            dict with format:
            {
                "preference_updated": bool,
                "room": str or None,
                "device": str or None,
                "message": str
            }
        """
        if not user_message:
            return {
                "preference_updated": False,
                "room": None,
                "device": None,
                "message": ""
            }
        
        message_lower = user_message.lower().strip()
        
        # Keywords that indicate "leave it on" / "don't notify"
        leave_it_on_keywords = [
            "leave it on",
            "leave it",
            "that's fine",
            "thats fine",
            "it's fine",
            "its fine",
            "that's okay",
            "thats okay",
            "it's okay",
            "its okay",
            "don't worry",
            "dont worry",
            "no problem",
            "it's intentional",
            "its intentional",
            "keep it on",
            "keep on"
        ]
        
        # Check if message contains any "leave it on" keywords
        contains_leave_it_on = any(keyword in message_lower for keyword in leave_it_on_keywords)
        
        if not contains_leave_it_on:
            return {
                "preference_updated": False,
                "room": None,
                "device": None,
                "message": ""
            }
        
        # If we have recent notification info, update preferences for that device
        if recent_notification:
            room = recent_notification.get("room")
            device = recent_notification.get("device")
            
            if room and device:
                state_manager = self.mcp_server.state_manager
                success = state_manager.set_notification_preference(room, device, do_not_notify=True)
                
                if success:
                    return {
                        "preference_updated": True,
                        "room": room,
                        "device": device,
                        "message": f"Got it! I won't notify you about {room} {device} anymore."
                    }
        
        # If no specific device context, but user said "leave it on",
        # we could try to extract device info from the message
        # For now, return that we detected the intent but couldn't update
        return {
            "preference_updated": False,
            "room": None,
            "device": None,
            "message": "I understand you want to leave something on, but I need more context about which device."
        }

