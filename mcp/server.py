"""
MCP Server implementation with tools for chat and device control.
No UI dependencies - pure state management and tool execution.
"""

from datetime import datetime
from core.state import StateManager, _validate_schedule_item
from config import ROOMS
from core.activity_derivation import ActivityDerivationService


def _normalize_time_format(time_str: str) -> str:
    """
    Normalize time string to HH:MM format.
    Handles formats like: "14.00", "14:00", "2.30", "14:30"
    
    Args:
        time_str: Time string in various formats
        
    Returns:
        Normalized time string in HH:MM format, or original if normalization fails
    """
    if not time_str:
        return time_str
    
    # Replace dot with colon
    time_str_normalized = time_str.replace(".", ":")
    
    # Parse and normalize
    try:
        parts = time_str_normalized.split(":")
        if len(parts) == 2:
            hours = int(parts[0])
            minutes = int(parts[1]) if parts[1] else 0
            # Validate range
            if 0 <= hours <= 23 and 0 <= minutes <= 59:
                return f"{hours:02d}:{minutes:02d}"
    except (ValueError, IndexError):
        pass
    
    return time_str  # Return original if normalization fails


def _extract_location_from_message(message: str) -> str:
    """
    Extract location/room name from user message.
    
    Looks for patterns like "in bedroom", "in living room", "in kitchen", etc.
    
    Args:
        message: User message string
        
    Returns:
        Room name if found (normalized), None otherwise
    """
    if not message:
        return None
    
    message_lower = message.lower()
    from config import ROOMS
    
    # Look for "in [room]" pattern
    for room in ROOMS.keys():
        room_lower = room.lower()
        # Check for "in bedroom", "in living room", etc.
        if f"in {room_lower}" in message_lower or f"in the {room_lower}" in message_lower:
            return room
    
    return None

def _normalize_room_name(room_str: str) -> str:
    """
    Normalize room name to match ROOMS config.
    Examples: "livingroom" → "Living Room", "bedroom" → "Bedroom"
    
    Args:
        room_str: Room name string (may be in various formats)
        
    Returns:
        Normalized room name matching ROOMS config, or original if no mapping found
    """
    if not room_str:
        return room_str
    
    room_lower = room_str.lower().strip()
    
    # Map variations to exact room names from ROOMS config
    room_mapping = {
        "bedroom": "Bedroom",
        "bathroom": "Bathroom",
        "kitchen": "Kitchen",
        "livingroom": "Living Room",
        "living room": "Living Room",
        "living": "Living Room"
    }
    
    normalized = room_mapping.get(room_lower)
    if normalized:
        return normalized
    
    # If not in mapping, check if it matches any room name exactly (case-insensitive)
    for room_name in ROOMS.keys():
        if room_name.lower() == room_lower:
            return room_name
    
    return room_str  # Return original if no mapping found


def _normalize_device_name(device_str: str, room: str = None) -> str:
    """
    Normalize device name to match ROOMS config.
    Examples: "light" → "Light", "ac" → "AC", "tv" → "TV"
    
    Args:
        device_str: Device name string (may be in various formats)
        room: Optional room name to validate device exists in that room
        
    Returns:
        Normalized device name matching ROOMS config, or original if no mapping found
    """
    if not device_str:
        return device_str
    
    device_lower = device_str.lower().strip()
    
    # Map common variations to exact device names
    device_mapping = {
        "light": "Light",
        "lights": "Light",
        "lamp": "Light",
        "lamps": "Light",
        "ac": "AC",
        "air conditioner": "AC",
        "airconditioner": "AC",
        "air conditioning": "AC",
        "tv": "TV",
        "television": "TV",
        "fan": "Fan",
        "fans": "Fan",
        "alarm": "Alarm",
        "alarms": "Alarm"
    }
    
    normalized = device_mapping.get(device_lower)
    if normalized:
        # If room is provided, validate device exists in that room
        if room:
            room_devices = ROOMS.get(room, [])
            if normalized in room_devices:
                return normalized
            else:
                # Device doesn't exist in this room, try to find it in any room
                print(f"[MCP WARNING] Device '{normalized}' not found in room '{room}'. Available devices: {room_devices}")
        else:
            return normalized
    
    # If not in mapping, check if it matches any device name exactly (case-insensitive)
    # Check all rooms if room not specified, or just the specified room
    rooms_to_check = [room] if room and room in ROOMS else ROOMS.keys()
    
    for room_name in rooms_to_check:
        for device_name in ROOMS.get(room_name, []):
            if device_name.lower() == device_lower:
                return device_name
    
    return device_str  # Return original if no mapping found


def _extract_date_from_message(message: str) -> str:
    """
    Extract date from user message and convert to YYYY-MM-DD format.
    
    Handles:
    - Relative dates: "tomorrow", "next Monday", "next week"
    - Absolute dates: "March 15th", "15th March", "2024-03-15"
    - Implicit dates: no date mentioned = today
    
    Args:
        message: User message string
        
    Returns:
        Date string in YYYY-MM-DD format, or None if no date found (defaults to today)
    """
    if not message:
        return None
    
    from datetime import datetime, timedelta
    
    message_lower = message.lower().strip()
    today = datetime.now()
    
    # Relative dates
    if "tomorrow" in message_lower:
        tomorrow = today + timedelta(days=1)
        return tomorrow.strftime("%Y-%m-%d")
    
    if "next week" in message_lower:
        next_week = today + timedelta(days=7)
        return next_week.strftime("%Y-%m-%d")
    
    # Day of week patterns: "next Monday", "next monday", "next tuesday", etc.
    days_of_week = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6
    }
    
    for day_name, day_num in days_of_week.items():
        if f"next {day_name}" in message_lower:
            days_ahead = (day_num - today.weekday()) % 7
            if days_ahead == 0:  # If today is that day, go to next week
                days_ahead = 7
            target_date = today + timedelta(days=days_ahead)
            return target_date.strftime("%Y-%m-%d")
    
    # Absolute date patterns: "March 15th", "15th March", "March 15", "15 March"
    # Try to parse common date formats
    import re
    
    # Pattern: "March 15th", "March 15", "15th March", "15 March"
    month_names = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12
    }
    
    # Pattern 1: "Month Day" or "Month Dayth"
    for month_name, month_num in month_names.items():
        pattern = rf"{month_name}\s+(\d{{1,2}})(?:st|nd|rd|th)?"
        match = re.search(pattern, message_lower)
        if match:
            day = int(match.group(1))
            try:
                # Use current year, or next year if date has passed
                target_date = datetime(today.year, month_num, day)
                if target_date < today:
                    target_date = datetime(today.year + 1, month_num, day)
                return target_date.strftime("%Y-%m-%d")
            except ValueError:
                pass  # Invalid date, continue
    
    # Pattern 2: "Day Month" or "Dayth Month"
    for month_name, month_num in month_names.items():
        pattern = rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+{month_name}"
        match = re.search(pattern, message_lower)
        if match:
            day = int(match.group(1))
            try:
                target_date = datetime(today.year, month_num, day)
                if target_date < today:
                    target_date = datetime(today.year + 1, month_num, day)
                return target_date.strftime("%Y-%m-%d")
            except ValueError:
                pass  # Invalid date, continue
    
    # Pattern 3: YYYY-MM-DD format
    date_pattern = r"\d{4}-\d{2}-\d{2}"
    match = re.search(date_pattern, message)
    if match:
        date_str = match.group(0)
        try:
            parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
            return parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            pass
    
    # No date found - return None (will default to today)
    return None


class MCPServer:
    """
    MCP Server that provides tools for the LLM to interact with the system.
    
    Tools:
    - chat_message: Send a message to the user
    - e_device_control: Control electrical devices (ON/OFF)
    - schedule_modifier: Modify schedule (add/delete/change items)
    - rag_query: Query RAG system for health knowledge
    """
    
    def __init__(self, state_manager: StateManager):
        """
        Initialize MCP server with a state manager.
        
        Args:
            state_manager: StateManager instance to use for state operations
        """
        self.state_manager = state_manager
        self._rag_retriever = None  # Lazy-loaded RAG retriever
        self._activity_derivation = ActivityDerivationService()  # Activity derivation service
    
    def chat_message(self, message: str) -> dict:
        """
        Tool: Send a chat message to the user.
        
        This tool does not modify state - it only returns a message
        that should be displayed to the user.
        
        Args:
            message: The message content to send to the user
            
        Returns:
            dict with structure:
            {
                "success": bool,
                "tool": "chat_message",
                "message": str,
                "error": str (only if success=False)
            }
        """
        if not message or not isinstance(message, str):
            return {
                "success": False,
                "tool": "chat_message",
                "message": "",
                "error": "Message must be a non-empty string"
            }
        
        return {
            "success": True,
            "tool": "chat_message",
            "message": message.strip(),
            "error": None
        }
    
    def e_device_control(self, room: str, device: str, action: str) -> dict:
        """
        Tool: Control an electrical device (turn ON or OFF).
        
        Updates the device state in the MCP state manager.
        
        Args:
            room: Room name (e.g., "Bedroom", "Living Room")
            device: Device name (e.g., "light1", "light2")
            action: Action to perform ("ON" or "OFF")
            
        Returns:
            dict with structure:
            {
                "success": bool,
                "tool": "e_device_control",
                "room": str,
                "device": str,
                "action": str ("ON" or "OFF"),
                "previous_state": bool,
                "new_state": bool,
                "message": str,
                "error": str (only if success=False)
            }
        """
        # Validate inputs
        if not room or not isinstance(room, str):
            return {
                "success": False,
                "tool": "e_device_control",
                "room": room or "",
                "device": device or "",
                "action": action or "",
                "previous_state": None,
                "new_state": None,
                "message": "",
                "error": "Room must be a non-empty string"
            }
        
        if not device or not isinstance(device, str):
            return {
                "success": False,
                "tool": "e_device_control",
                "room": room,
                "device": device or "",
                "action": action or "",
                "previous_state": None,
                "new_state": None,
                "message": "",
                "error": "Device must be a non-empty string"
            }
        
        # Validate and normalize action
        action_upper = action.upper() if action else ""
        if action_upper not in ["ON", "OFF"]:
            return {
                "success": False,
                "tool": "e_device_control",
                "room": room,
                "device": device,
                "action": action or "",
                "previous_state": None,
                "new_state": None,
                "message": "",
                "error": f"Invalid action: '{action}'. Must be 'ON' or 'OFF'."
            }
        
        # Handle case where device name might include room name (e.g., "Kitchen Light")
        # Check if device string contains a room name
        device_room = None
        device_only = device
        for room_name in ROOMS.keys():
            room_lower = room_name.lower()
            device_lower = device.lower()
            # Check if device string starts with room name (e.g., "Kitchen Light")
            if device_lower.startswith(room_lower):
                # Extract device part after room name
                remaining = device[len(room_name):].strip()
                if remaining:
                    device_room = room_name
                    device_only = remaining
                    print(f"[MCP DEBUG] Extracted room from device name: '{device}' -> room='{device_room}', device='{device_only}'")
                    break
        
        # Use extracted room if found, otherwise use provided room
        room_to_use = device_room if device_room else room
        device_to_use = device_only
        
        # Normalize room name to match config
        room_normalized = _normalize_room_name(room_to_use)
        
        # Normalize device name to match config
        device_normalized = _normalize_device_name(device_to_use, room_normalized)
        print(f"[MCP DEBUG] e_device_control: room='{room}' -> normalized='{room_normalized}', device='{device}' -> normalized='{device_normalized}', action='{action_upper}'")
        
        # Get current state before change
        previous_state = self.state_manager.get_device_state(room_normalized, device_normalized)
        print(f"[MCP DEBUG] Previous state: {previous_state}")
        
        # Convert action to boolean
        new_state = action_upper == "ON"
        print(f"[MCP DEBUG] New state: {new_state}")
        
        # Update device state
        success = self.state_manager.set_device_state(room_normalized, device_normalized, new_state)
        print(f"[MCP DEBUG] set_device_state result: success={success}")
        
        if success:
            # Verify the state was actually updated
            verify_state = self.state_manager.get_device_state(room_normalized, device_normalized)
            print(f"[MCP DEBUG] Verified state after update: {verify_state} (expected: {new_state})")
            if verify_state != new_state:
                print(f"[MCP ERROR] State mismatch! Expected {new_state}, got {verify_state}")
        
        if success:
            state_text = "ON" if new_state else "OFF"
            result = {
                "success": True,
                "tool": "e_device_control",
                "room": room_normalized,  # Return normalized room name
                "device": device_normalized,  # Return normalized device name
                "action": action_upper,
                "previous_state": previous_state,
                "new_state": new_state,
                "message": f"Set {room_normalized} {device_normalized} to {state_text}",
                "error": None
            }
            print(f"[MCP DEBUG] Returning success result: {result}")
            return result
        else:
            result = {
                "success": False,
                "tool": "e_device_control",
                "room": room_normalized,  # Return normalized room name
                "device": device_normalized,  # Return normalized device name
                "action": action_upper,
                "previous_state": previous_state,
                "new_state": None,
                "message": "",
                "error": f"Device '{device_normalized}' not found in room '{room_normalized}'. Available devices: {ROOMS.get(room_normalized, [])}"
            }
            print(f"[MCP DEBUG] Returning failure result: {result}")
            return result
    
    def get_current_state(self, custom_date: str = None, current_activity: dict = None) -> dict:
        """
        Get the current system state for the LLM.
        
        Args:
            custom_date: Optional custom date string in YYYY-MM-DD format (for custom clock)
            current_activity: Optional dict with current activity context: {"activity": str, "time": str, "location": str or None}
        
        Returns:
            dict with structure:
            {
                "current_location": str,
                "devices": {room: {device: bool}},
                "do_not_remind": list[str],
                "notification_preferences": list[str],
                "current_activity": dict or None
            }
        """
        # Phase 5: Cleanup old one-time events periodically (every state fetch)
        # This prevents accumulation of old events
        self.state_manager.cleanup_old_one_time_events()
        
        state_summary = self.state_manager.get_state_summary(custom_date=custom_date)
        state_summary["current_activity"] = current_activity  # Add current activity to state
        return state_summary
    
    def detect_potential_issues(self) -> list:
        """
        Detect situations where something might be "off" in the house.
        
        This identifies devices that are ON in rooms other than where the user is located.
        These may be unintended and worth notifying the user about.
        
        Returns:
            List of dictionaries with format:
            [
                {
                    "room": str,
                    "device": str,
                    "state": bool,
                    "user_location": str
                },
                ...
            ]
        """
        current_location = self.state_manager.current_location
        all_devices = self.state_manager.get_all_devices()
        issues = []
        
        for room, room_devices in all_devices.items():
            # Skip the user's current room
            if room == current_location:
                continue
            
            # Check each device in this room
            for device, state in room_devices.items():
                # If device is ON and we should notify about it
                if state and self.state_manager.should_notify_about_device(room, device):
                    issues.append({
                        "room": room,
                        "device": device,
                        "state": state,
                        "user_location": current_location
                    })
        
        return issues
    
    def _is_one_time_activity(self, activity: str, user_message: str = None) -> bool:
        """
        Determine if an activity is a one-time event or recurring routine.
        
        Args:
            activity: Activity name
            user_message: Optional user message for context
            
        Returns:
            True if one-time event, False if recurring routine
        """
        if not activity:
            return True  # Default to one-time if no activity
        
        activity_lower = activity.lower()
        message_lower = (user_message or "").lower()
        
        # Known recurring activities (daily routines)
        recurring_keywords = [
            "wake up", "wake", "breakfast", "lunch", "dinner",
            "work", "continue working", "exercise", "morning exercise",
            "relaxation", "relaxation time", "prepare for bed", "sleep", "bedtime"
        ]
        
        # Check if activity matches recurring patterns
        for keyword in recurring_keywords:
            if keyword in activity_lower:
                # Check if user message overrides with "every day" or similar
                if user_message:
                    recurring_phrases = ["every day", "daily", "always", "regularly", "every morning", "every evening"]
                    if any(phrase in message_lower for phrase in recurring_phrases):
                        return False  # Explicitly recurring
                return False  # It's recurring
        
        # One-time event keywords
        one_time_keywords = [
            "meeting", "appointment", "doctor", "dentist", "gym",
            "visit", "event", "party", "wedding", "birthday",
            "conference", "seminar", "workshop", "class", "therapy",
            "checkup", "consultation", "session"
        ]
        
        # Check if activity matches one-time patterns
        for keyword in one_time_keywords:
            if keyword in activity_lower:
                return True  # It's one-time
        
        # Check user message for context clues
        if user_message:
            # Phrases that indicate one-time events
            one_time_phrases = [
                "i have a", "i have an", "i need to", "i'm going to",
                "i'm attending", "i'm visiting", "i'm going to the",
                "this afternoon", "this evening", "this morning"
            ]
            
            # Phrases that indicate recurring
            recurring_phrases = [
                "every day", "daily", "always", "usually", "regularly",
                "every morning", "every evening", "every week"
            ]
            
            for phrase in one_time_phrases:
                if phrase in message_lower:
                    return True
            
            for phrase in recurring_phrases:
                if phrase in message_lower:
                    return False
        
        # Default: If activity is not in known recurring list, assume one-time
        # This is safer - user can always add it as recurring explicitly later
        return True
    
    def _get_base_schedule_item(self, time: str, activity: str = None) -> dict:
        """
        Check if an item exists in the base schedule and return it.
        Used to determine if we should preserve action/location from base schedule.
        
        Args:
            time: Time string to match
            activity: Optional activity string to match
            
        Returns:
            Base schedule item if found, None otherwise
        """
        base_schedule = self.state_manager.get_user_schedule()
        for item in base_schedule:
            if item.get("time") == time:
                if activity is None or item.get("activity") == activity:
                    return item
        return None
    
    def _apply_derivation(self, item: dict, preserve_from_base: bool = False, user_message: str = None) -> dict:
        """
        Apply activity derivation to a schedule item.
        
        Args:
            item: Schedule item dict with at least "time" and "activity"
            preserve_from_base: If True, check base schedule and preserve action/location if found
            user_message: Optional user message to extract location override from
            
        Returns:
            Schedule item with derived action/location fields added
        """
        activity = item.get("activity")
        if not activity:
            return item
        
        # Check if we should preserve from base schedule
        if preserve_from_base:
            base_item = self._get_base_schedule_item(item.get("time"), activity)
            if base_item:
                # Preserve action and location from base schedule
                if "action" in base_item:
                    item["action"] = base_item["action"].copy()
                    if "devices" in item["action"]:
                        item["action"]["devices"] = [d.copy() for d in item["action"]["devices"]]
                if "location" in base_item:
                    item["location"] = base_item["location"]
                return item
        
        # Derive from activity
        derived = self._activity_derivation.derive_fields(activity)
        if derived["action"]:
            item["action"] = derived["action"]
        if derived["location"]:
            item["location"] = derived["location"]
        
        # Override location if user specified it in the message
        if user_message:
            user_location = _extract_location_from_message(user_message)
            if user_location:
                # Override location and update action devices to use the new location
                item["location"] = user_location
                if "action" in item and item["action"] and "devices" in item["action"]:
                    # Update all devices in action to use the new location
                    for device_spec in item["action"]["devices"]:
                        device_spec["room"] = user_location
        
        return item
    
    def schedule_modifier(self, modify_type: str, time: str = None, 
                         activity: str = None, old_time: str = None, 
                         old_activity: str = None, user_message: str = None,
                         action: dict = None, location: str = None) -> dict:
        """
        Tool: Modify schedule using daily clone.
        
        This tool can modify schedules for today or future dates. The system automatically:
        - Extracts dates from user messages ("tomorrow", "next Monday", "March 15th", etc.)
        - Detects if an activity is one-time (meetings, appointments) or recurring (daily routines)
        - Routes one-time events to one_time_events table (date-specific)
        - Routes recurring activities to base schedule (applies to all future days)
        
        For ADD operations:
        - One-time events: Stored in one_time_events table for the target date
        - Recurring activities: Stored in base schedule for all future days
        - If date is today: Also added to today's daily clone immediately
        - If date is future: Will appear in daily clone when that day arrives
        
        Args:
            modify_type: Type of modification:
                        - "add": Add a new schedule item (requires time and activity)
                        - "delete": Delete an existing schedule item (requires time)
                        - "change": Change an existing item's time/activity (requires old_time, time, activity)
            time: Time string (e.g., "08:00") - required for add/delete, new_time for change
            activity: Activity string - required for add, new_activity for change
            old_time: Old time string - required for change operation (when changing time)
            old_activity: Old activity string - optional for change (for validation)
            user_message: Optional user message - used to extract date and detect activity type
            action: Optional action dict (e.g., {"devices": [...]}) - if provided, overrides derivation
            location: Optional location string - if provided, overrides derivation
            
        Note: action and location are automatically derived from activity by the system if not provided.
              The LLM cannot directly set these fields, but the UI can pass them explicitly.
              Date is automatically extracted from user_message if present.
            
        Returns:
            dict with structure:
            {
                "success": bool,
                "tool": "schedule_modifier",
                "message": str,
                "error": str (only if success=False)
            }
        """
        from datetime import datetime
        
        # Determine what "today" is - use real datetime
        # Note: If custom clock date support is needed, it should be passed as a parameter
        today = datetime.now().strftime("%Y-%m-%d")
        print(f"[SCHEDULE MODIFIER] Using date as 'today': {today}")
        
        # Normalize time format if provided
        if time:
            time = _normalize_time_format(time)
        if old_time:
            old_time = _normalize_time_format(old_time)
        
        # Validate modify_type is provided
        if not modify_type:
            return {
                "success": False,
                "tool": "schedule_modifier",
                "message": "",
                "error": "modify_type is required. Must be 'add', 'delete', or 'change'"
            }
        
        try:
            if modify_type == "add":
                # Add new schedule item (fails if already exists)
                if not time:
                    return {
                        "success": False,
                        "tool": "schedule_modifier",
                        "message": "",
                        "error": "time required for add operation"
                    }
                
                if not activity:
                    return {
                        "success": False,
                        "tool": "schedule_modifier",
                        "message": "",
                        "error": "activity required for add operation"
                    }
                
                # Build schedule item with time and activity
                new_item = {"time": time, "activity": activity}
                
                # Store explicitly provided fields
                explicit_action = action
                explicit_location = location
                
                # Apply derivation - check base schedule first for preservation
                new_item = self._apply_derivation(new_item, preserve_from_base=True, user_message=user_message)
                
                # Override with explicitly provided fields if they were given
                if explicit_action is not None:
                    new_item["action"] = explicit_action
                if explicit_location is not None:
                    new_item["location"] = explicit_location
                
                # Validate the schedule item
                is_valid, error_msg = _validate_schedule_item(new_item)
                if not is_valid:
                    return {
                        "success": False,
                        "tool": "schedule_modifier",
                        "message": "",
                        "error": f"Invalid schedule item: {error_msg}"
                    }
                
                # Extract date from user message (or default to today)
                target_date = _extract_date_from_message(user_message) if user_message else None
                if target_date is None:
                    target_date = today
                
                # Validate date is not in the past
                try:
                    target_datetime = datetime.strptime(target_date, "%Y-%m-%d")
                    if target_datetime.date() < datetime.now().date():
                        return {
                            "success": False,
                            "tool": "schedule_modifier",
                            "message": "",
                            "error": f"Cannot schedule items for past dates. Date '{target_date}' is in the past."
                        }
                except ValueError:
                    return {
                        "success": False,
                        "tool": "schedule_modifier",
                        "message": "",
                        "error": f"Invalid date format: '{target_date}'. Expected YYYY-MM-DD format."
                    }
                
                # Detect if this is a one-time event or recurring activity
                is_one_time = self._is_one_time_activity(activity, user_message)
                
                if is_one_time:
                    # One-time event: Add to one_time_events table
                    self.state_manager.add_schedule_addon(
                        target_date,
                        time,
                        activity,
                        action=new_item.get("action"),
                        location=new_item.get("location")
                    )
                    
                    # If date is today, also add to today's daily clone
                    if target_date == today:
                        daily_clone = self.state_manager.get_daily_clone()
                        daily_clone.append(new_item)
                        daily_clone.sort(key=lambda x: x.get("time", ""))
                        self.state_manager.set_daily_clone(daily_clone)
                        print(f"[SCHEDULE MODIFIER] Added one-time event to today's daily clone: {activity} at {time}")
                    else:
                        print(f"[SCHEDULE MODIFIER] Added one-time event for future date {target_date}: {activity} at {time}")
                    
                    # Do NOT add to base schedule (one-time events don't recur)
                    message = f"Added one-time event '{activity}' at {time}"
                    if target_date != today:
                        message += f" for {target_date}"
                else:
                    # Recurring activity: Add to base schedule and today's clone if date is today
                    if target_date == today:
                        # Add to today's daily clone
                        daily_clone = self.state_manager.get_daily_clone()
                        daily_clone.append(new_item)
                        daily_clone.sort(key=lambda x: x.get("time", ""))
                        self.state_manager.set_daily_clone(daily_clone)
                        print(f"[SCHEDULE MODIFIER] Added to today's daily clone: {activity} at {time}")
                    
                    # Add to base schedule (for all future days)
                    base_schedule = self.state_manager.get_user_schedule()
                    exists_in_base = any(item.get("time") == time for item in base_schedule)
                    if not exists_in_base:
                        self.state_manager.update_base_schedule([new_item.copy()])
                        print(f"[SCHEDULE MODIFIER] Added to base schedule for recurring: {activity} at {time}")
                    else:
                        self.state_manager.update_base_schedule([new_item.copy()])
                        print(f"[SCHEDULE MODIFIER] Updated base schedule item at {time}: {activity}")
                    
                    message = f"Added recurring activity '{activity}' at {time}"
                    if target_date != today:
                        message += f" (will appear in schedule starting from {target_date})"
                
                # Debug: Verify the add was applied with all fields (only for today's clone)
                if target_date == today:
                    verify_clone = self.state_manager.get_daily_clone()
                    verify_item = next((item for item in verify_clone if item.get("time") == time and item.get("activity") == activity), None)
                    if verify_item:
                        fields_present = []
                        if verify_item.get("time") == time:
                            fields_present.append("time")
                        if verify_item.get("activity") == activity:
                            fields_present.append("activity")
                        if "action" in verify_item:
                            fields_present.append("action")
                        if "location" in verify_item:
                            fields_present.append("location")
                        
                        print(f"[SCHEDULE MODIFIER] [OK] VERIFIED: Schedule item added successfully - {activity} at {time}")
                        print(f"[SCHEDULE MODIFIER] Fields present: {fields_present}")
                    else:
                        print(f"[SCHEDULE MODIFIER] [WARNING] Schedule add may not have been applied correctly - {activity} not found at {time}")
                
                return {
                    "success": True,
                    "tool": "schedule_modifier",
                    "modify_type": modify_type,
                    "time": time,
                    "activity": activity,
                    "message": message,
                    "error": None
                }
            
            elif modify_type == "delete":
                # Delete existing schedule item (fails if not exists)
                if not time:
                    return {
                        "success": False,
                        "tool": "schedule_modifier",
                        "message": "",
                        "error": "time required for delete operation"
                    }
                
                # Remove from today's daily clone
                daily_clone = self.state_manager.get_daily_clone()
                found = False
                for idx, item in enumerate(daily_clone):
                    if item.get("time") == time:
                        removed_activity = item.get("activity", "")
                        daily_clone.pop(idx)
                        found = True
                        print(f"[SCHEDULE MODIFIER] Removed item at {time} ('{removed_activity}') from today's schedule")
                        break
                
                if found:
                    self.state_manager.set_daily_clone(daily_clone)
                    
                    # Debug: Verify the delete was applied
                    verify_clone = self.state_manager.get_daily_clone()
                    verify_item = next((item for item in verify_clone if item.get("time") == time), None)
                    if not verify_item:
                        print(f"[SCHEDULE MODIFIER] [OK] VERIFIED: Schedule item deleted successfully - no item at {time}")
                    else:
                        print(f"[SCHEDULE MODIFIER] [WARNING] Schedule delete may not have been applied correctly - item still exists at {time}")
                    
                    return {
                        "success": True,
                        "tool": "schedule_modifier",
                        "modify_type": modify_type,
                        "time": time,
                        "message": f"Deleted schedule item at {time}",
                        "error": None
                    }
                else:
                    return {
                        "success": False,
                        "tool": "schedule_modifier",
                        "message": "",
                        "error": f"Schedule item at {time} not found in today's schedule"
                    }
            
            elif modify_type == "change":
                # Change existing item's time/activity (fails if item doesn't exist)
                # Supports partial updates: only change fields that are provided
                if not old_time:
                    return {
                        "success": False,
                        "tool": "schedule_modifier",
                        "message": "",
                        "error": "old_time required for change operation"
                    }
                
                # At least one of time or activity must be provided
                if not time and not activity:
                    return {
                        "success": False,
                        "tool": "schedule_modifier",
                        "message": "",
                        "error": "At least one of time or activity must be provided for change operation"
                    }
                
                # Action and location are not provided by LLM - they are preserved from existing items
                
                # Change item in today's daily clone
                daily_clone = self.state_manager.get_daily_clone()
                found = False
                for idx, item in enumerate(daily_clone):
                    if item.get("time") == old_time:
                        # Validate activity if old_activity provided
                        if old_activity and item.get("activity") != old_activity:
                            return {
                                "success": False,
                                "tool": "schedule_modifier",
                                "message": "",
                                "error": f"Activity mismatch: expected '{old_activity}' but found '{item.get('activity')}' at {old_time}"
                            }
                        
                        # Build new item with partial updates
                        # Only update fields that are provided, keep others unchanged
                        new_item = {}
                        # Update time if provided, otherwise keep old time
                        if time:
                            new_item["time"] = time
                        else:
                            new_item["time"] = item.get("time")
                        # Update activity if provided, otherwise keep old activity
                        new_activity = activity if activity else item.get("activity")
                        new_item["activity"] = new_activity
                        
                        # Derive action/location from new activity
                        # Check if item exists in base schedule - if so, preserve from base
                        base_item = self._get_base_schedule_item(new_item["time"], new_activity)
                        if base_item:
                            # Preserve from base schedule
                            if "action" in base_item:
                                new_item["action"] = base_item["action"].copy()
                                if "devices" in new_item["action"]:
                                    new_item["action"]["devices"] = [d.copy() for d in new_item["action"]["devices"]]
                            if "location" in base_item:
                                new_item["location"] = base_item["location"]
                        else:
                            # Derive from activity
                            derived = self._activity_derivation.derive_fields(new_activity)
                            if derived["action"]:
                                new_item["action"] = derived["action"]
                            if derived["location"]:
                                new_item["location"] = derived["location"]
                        
                        # Validate the new schedule item
                        is_valid, error_msg = _validate_schedule_item(new_item)
                        if not is_valid:
                            return {
                                "success": False,
                                "tool": "schedule_modifier",
                                "message": "",
                                "error": f"Invalid schedule item: {error_msg}"
                            }
                        
                        # Remove old item and add new item
                        old_activity_name = item.get("activity", "")
                        daily_clone.pop(idx)
                        daily_clone.append(new_item)
                        daily_clone.sort(key=lambda x: x.get("time", ""))
                        found = True
                        # Build change message based on what was updated
                        change_parts = []
                        if time and time != old_time:
                            change_parts.append(f"time from {old_time} to {time}")
                        if activity and activity != old_activity_name:
                            change_parts.append(f"activity from '{old_activity_name}' to '{activity}'")
                        change_msg = " and ".join(change_parts) if change_parts else "item"
                        print(f"[SCHEDULE MODIFIER] Changed {change_msg} in today's schedule")
                        break
                
                if found:
                    self.state_manager.set_daily_clone(daily_clone)
                    
                    # Debug: Verify the change was applied
                    verify_clone = self.state_manager.get_daily_clone()
                    new_time = new_item.get("time")
                    new_activity = new_item.get("activity")
                    verify_item = next((item for item in verify_clone if item.get("time") == new_time and item.get("activity") == new_activity), None)
                    old_item_still_exists = next((item for item in verify_clone if item.get("time") == old_time and item.get("activity") == old_activity_name), None)
                    
                    if verify_item and not old_item_still_exists:
                        print(f"[SCHEDULE MODIFIER] [OK] VERIFIED: Schedule change applied successfully - {new_activity} is now at {new_time} (old time {old_time} removed)")
                    elif verify_item:
                        print(f"[SCHEDULE MODIFIER] [OK] VERIFIED: Schedule change applied - {new_activity} is now at {new_time} (but old item may still exist)")
                    else:
                        print(f"[SCHEDULE MODIFIER] [WARNING] Schedule change may not have been applied correctly - {new_activity} not found at {new_time}")
                    
                    # Log current schedule state for debugging
                    schedule_items = [f"{item.get('time')}: {item.get('activity')}" for item in verify_clone]
                    print(f"[SCHEDULE MODIFIER] Current schedule state: {schedule_items}")
                    
                    # Build message based on what was changed
                    change_parts = []
                    if time and time != old_time:
                        change_parts.append(f"time from {old_time} to {time}")
                    if activity and activity != old_activity_name:
                        change_parts.append(f"activity to '{activity}'")
                    change_msg = " and ".join(change_parts) if change_parts else "schedule item"
                    
                    return {
                        "success": True,
                        "tool": "schedule_modifier",
                        "modify_type": modify_type,
                        "old_time": old_time,
                        "time": new_time,
                        "activity": new_activity,
                        "message": f"Changed {change_msg}",
                        "error": None
                    }
                else:
                    return {
                        "success": False,
                        "tool": "schedule_modifier",
                        "message": "",
                        "error": f"Schedule item at {old_time} not found in today's schedule"
                    }
            
            else:
                return {
                    "success": False,
                    "tool": "schedule_modifier",
                    "message": "",
                    "error": f"Invalid modify_type: '{modify_type}'. Must be 'add', 'delete', or 'change'"
                }
        
        except Exception as e:
            return {
                "success": False,
                "tool": "schedule_modifier",
                "message": "",
                "error": f"Error modifying schedule: {str(e)}"
            }
    
    def _get_rag_retriever(self):
        """
        Get or initialize RAG retriever instance (lazy loading).
        
        Returns:
            Retriever instance, or None if initialization fails
        """
        if self._rag_retriever is None:
            try:
                from rag.retrieval.retriever import Retriever
                self._rag_retriever = Retriever()
                print("[RAG] RAG retriever initialized successfully")
            except Exception as e:
                print(f"[RAG] ERROR: Failed to initialize RAG retriever: {e}")
                return None
        return self._rag_retriever
    
    def rag_query(self, query: str, user_condition: str = None) -> dict:
        """
        Tool: Query RAG system for health knowledge.
        
        Args:
            query: Health-related query string
            user_condition: Optional user condition context (e.g., "diabetes")
        
        Returns:
            dict with structure:
            {
                "success": bool,
                "tool": "rag_query",
                "found": bool,
                "chunks": list or None,
                "error": str (only if success=False)
            }
        """
        # Validate query
        if not query or not isinstance(query, str):
            return {
                "success": False,
                "tool": "rag_query",
                "found": False,
                "chunks": None,
                "error": "Query must be a non-empty string"
            }
        
        # Get RAG retriever
        retriever = self._get_rag_retriever()
        if retriever is None:
            return {
                "success": False,
                "tool": "rag_query",
                "found": False,
                "chunks": None,
                "error": "RAG system not available"
            }
        
        # Build enhanced query: combine user query + user condition if available
        enhanced_query = query.strip()
        if user_condition and user_condition.strip():
            query_lower = query.lower().strip()
            condition_lower = user_condition.lower()
            
            # For exercise/activity queries, add wheelchair-specific terms if condition mentions wheelchair
            is_exercise_query = any(word in query_lower for word in ["exercise", "activity", "workout", "physical", "fitness", "movement"])
            has_wheelchair = "wheelchair" in condition_lower or "uses a wheelchair" in condition_lower
            
            if is_exercise_query and has_wheelchair:
                # Prioritize wheelchair exercise knowledge
                enhanced_query = f"{query.strip()} wheelchair exercises wheelchair users seated exercises"
                print(f"[RAG] Enhanced exercise query for wheelchair user: {enhanced_query[:100]}...")
            else:
                # General enhancement with key terms extraction
                # Extract key terms from user condition for better matching
                key_terms = []
                
                # Extract mobility-related terms
                if "wheelchair" in condition_lower:
                    key_terms.append("wheelchair")
                if "mobility" in condition_lower:
                    key_terms.append("mobility")
                
                # Extract health condition terms
                health_conditions = ["diabetes", "hypertension", "arthritis", "copd", "dementia", "depression", "stroke", "parkinson"]
                for condition in health_conditions:
                    if condition in condition_lower:
                        key_terms.append(condition)
                        break  # Usually only one primary condition
                
                # Build enhanced query with key terms prioritized
                if key_terms:
                    enhanced_query = f"{query.strip()} {' '.join(key_terms)} {user_condition.strip()}"
                else:
                    enhanced_query = f"{query.strip()} {user_condition.strip()}"
                
                print(f"[RAG] Enhanced query with key terms: {enhanced_query[:100]}...")
        else:
            print(f"[RAG] Query (no condition): {enhanced_query[:100]}...")
        
        try:
            # Call RAG retriever with higher threshold for better precision
            # Using threshold=0.5 to ensure only highly relevant chunks are returned
            result = retriever.retrieve(enhanced_query, top_k=3, threshold=0.5)
            
            if result.get("found"):
                chunks = result.get("chunks", [])
                print(f"[RAG] Found {len(chunks)} relevant chunk(s)")
                return {
                    "success": True,
                    "tool": "rag_query",
                    "found": True,
                    "chunks": chunks,
                    "error": None
                }
            else:
                print(f"[RAG] No relevant results found (below threshold)")
                return {
                    "success": True,
                    "tool": "rag_query",
                    "found": False,
                    "chunks": None,
                    "error": None
                }
        
        except Exception as e:
            print(f"[RAG] ERROR during retrieval: {e}")
            return {
                "success": False,
                "tool": "rag_query",
                "found": False,
                "chunks": None,
                "error": f"RAG retrieval error: {str(e)}"
            }
