"""
Activity Derivation Service for automatically deriving action and location fields
from activity names in schedule items.

This service maintains a mapping of activity names to their default device actions
and required locations, allowing the system to automatically populate these fields
when the LLM creates or modifies schedule items.
"""

from config import ROOMS


# Activity-to-action/location mapping based on existing base schedule patterns
ACTIVITY_DEFAULTS = {
    "Wake up": {
        "action": {
            "devices": [
                {"room": "Bedroom", "device": "Alarm", "state": "ON"},
                {"room": "Bedroom", "device": "Light", "state": "ON"}
            ]
        },
        "location": "Bedroom"
    },
    "Morning exercise": {
        "action": None,  # No default device actions
        "location": None  # No default location requirement
    },
    "Breakfast": {
        "action": None,
        "location": "Kitchen"
    },
    "Work": {
        "action": {
            "devices": [
                {"room": "Living Room", "device": "Light", "state": "ON"},
                {"room": "Living Room", "device": "AC", "state": "ON"}
            ]
        },
        "location": "Living Room"
    },
    "Meeting": {
        "action": {
            "devices": [
                {"room": "Living Room", "device": "Light", "state": "ON"},
                {"room": "Living Room", "device": "AC", "state": "ON"}
            ]
        },
        "location": "Living Room"
    },
    "Continue Working": {
        "action": {
            "devices": [
                {"room": "Living Room", "device": "Light", "state": "ON"},
                {"room": "Living Room", "device": "AC", "state": "ON"}
            ]
        },
        "location": "Living Room"
    },
    "Lunch": {
        "action": None,
        "location": "Kitchen"
    },
    "Dinner": {
        "action": None,
        "location": "Kitchen"
    },
    "Relaxation time": {
        "action": None,
        "location": None
    },
    "Prepare for bed": {
        "action": {
            "devices": [
                {"room": "Bedroom", "device": "AC", "state": "ON"},
                {"room": "Bedroom", "device": "Light", "state": "ON"}
            ]
        },
        "location": "Bedroom"
    },
    "Sleep": {
        "action": {
            "devices": [
                {"room": "Bedroom", "device": "Light", "state": "OFF"}
            ]
        },
        "location": "Bedroom"
    }
}


class ActivityDerivationService:
    """
    Service that derives action and location fields from activity names.
    
    This allows the system to automatically populate device control actions
    and location requirements when the LLM creates or modifies schedule items,
    without requiring the LLM to reason about these technical details.
    """
    
    def __init__(self, activity_defaults: dict = None):
        """
        Initialize the derivation service.
        
        Args:
            activity_defaults: Optional custom mapping of activity names to defaults.
                              If None, uses the default ACTIVITY_DEFAULTS mapping.
        """
        self._activity_defaults = activity_defaults or ACTIVITY_DEFAULTS.copy()
    
    def derive_fields(self, activity: str) -> dict:
        """
        Derive action and location fields from an activity name.
        
        Args:
            activity: Activity name string (e.g., "Wake up", "Work", "Breakfast")
            
        Returns:
            Dictionary with structure:
            {
                "action": dict or None,  # {"devices": [...]} or None if no default action
                "location": str or None   # Room name or None if no default location
            }
        """
        if not activity or not isinstance(activity, str):
            return {"action": None, "location": None}
        
        activity_normalized = activity.strip()
        
        # Look up in mapping (case-insensitive)
        for mapped_activity, defaults in self._activity_defaults.items():
            if mapped_activity.lower() == activity_normalized.lower():
                # Return a copy to avoid modifying the original
                result = {
                    "action": defaults.get("action").copy() if defaults.get("action") else None,
                    "location": defaults.get("location")
                }
                # Deep copy action devices if present
                if result["action"] and "devices" in result["action"]:
                    result["action"] = {
                        "devices": [device.copy() for device in result["action"]["devices"]]
                    }
                return result
        
        # No mapping found - return None for both fields
        return {"action": None, "location": None}
    
    def add_activity_mapping(self, activity: str, action: dict = None, location: str = None) -> bool:
        """
        Add or update an activity mapping.
        
        Args:
            activity: Activity name
            action: Optional action dict with devices
            location: Optional location string (room name)
            
        Returns:
            True if successful, False if validation fails
        """
        if not activity or not isinstance(activity, str):
            return False
        
        # Validate location if provided
        if location and location not in ROOMS:
            return False
        
        # Validate action structure if provided
        if action:
            if not isinstance(action, dict):
                return False
            if "devices" not in action:
                return False
            if not isinstance(action["devices"], list):
                return False
            # Validate each device
            for device_spec in action["devices"]:
                if not isinstance(device_spec, dict):
                    return False
                room = device_spec.get("room")
                device = device_spec.get("device")
                state = device_spec.get("state")
                if room not in ROOMS or device not in ROOMS.get(room, []):
                    return False
                if state not in ["ON", "OFF"]:
                    return False
        
        # Add or update mapping
        self._activity_defaults[activity] = {
            "action": action.copy() if action else None,
            "location": location
        }
        
        return True
    
    def get_activity_mapping(self, activity: str) -> dict:
        """
        Get the mapping for a specific activity.
        
        Args:
            activity: Activity name
            
        Returns:
            Dictionary with action and location, or None if not found
        """
        return self.derive_fields(activity)
    
    def get_all_mappings(self) -> dict:
        """
        Get all activity mappings.
        
        Returns:
            Copy of the activity defaults dictionary
        """
        return self._activity_defaults.copy()

