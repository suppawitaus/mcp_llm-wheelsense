"""
Streamlit UI for MCP Smart Environment System.
Single page with room map (left) and chat interface (right).
"""

import streamlit as st
import streamlit.components.v1 as components
import time
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta, time as dt_time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from core.state import StateManager
from mcp.server import MCPServer
from mcp.router import MCPRouter
from llm.client import LLMClient
from services.notification import NotificationService
from database.manager import DatabaseManager
from config import ROOMS, OLLAMA_HOST, MODEL_NAME, DATABASE_PATH, PROJECT_ROOT


# ========== ENVIRONMENT VALIDATION ==========
def validate_environment():
    """
    Validate the environment and dependencies.
    Returns dict with validation results.
    """
    validation_results = {
        "all_valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check Python version
    if sys.version_info < (3, 8):
        validation_results["all_valid"] = False
        validation_results["errors"].append(
            f"Python 3.8+ is required. Current version: {sys.version_info.major}.{sys.version_info.minor}"
        )
    
    # Check database directory
    db_path = Path(DATABASE_PATH)
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Test write permissions
        test_file = db_path.parent / ".write_test"
        try:
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            validation_results["all_valid"] = False
            validation_results["errors"].append(
                f"Database directory is not writable: {db_path.parent}\nError: {str(e)}"
            )
    except Exception as e:
        validation_results["all_valid"] = False
        validation_results["errors"].append(
            f"Cannot create database directory: {db_path.parent}\nError: {str(e)}"
        )
    
    # Check RAG embeddings (warning only, not critical)
    # Use absolute path based on project root for reliability
    rag_index = PROJECT_ROOT / "rag" / "embeddings" / "faiss_index.bin"
    rag_mapping = PROJECT_ROOT / "rag" / "embeddings" / "id_to_chunk.json"
    if not rag_index.exists() or not rag_mapping.exists():
        validation_results["warnings"].append(
            "RAG embeddings not found. RAG functionality will not work.\n"
            "Expected files:\n"
            f"- {rag_index}\n"
            f"- {rag_mapping}\n\n"
            "These files should be included in the repository."
        )
    
    return validation_results


# Initialize components in session state
if 'db_manager' not in st.session_state:
    try:
        st.session_state.db_manager = DatabaseManager()
        print("[APP] Database manager initialized successfully")
    except Exception as e:
        st.error(f"❌ **Critical Error: Failed to initialize database**\n\n{str(e)}\n\nPlease check:\n- Database path permissions\n- Disk space availability\n- File system access")
        st.stop()

if 'state_manager' not in st.session_state:
    try:
        st.session_state.state_manager = StateManager(db_manager=st.session_state.db_manager)
        print("[APP] State manager initialized successfully")
    except Exception as e:
        st.error(f"❌ **Critical Error: Failed to initialize state manager**\n\n{str(e)}\n\nPlease check database connection and try again.")
        st.stop()

if 'mcp_server' not in st.session_state:
    st.session_state.mcp_server = MCPServer(st.session_state.state_manager)

if 'mcp_router' not in st.session_state:
    st.session_state.mcp_router = MCPRouter(st.session_state.mcp_server)

if 'llm_client' not in st.session_state:
    st.session_state.llm_client = LLMClient()
    
    # Validate Ollama connection and model availability
    validation = st.session_state.llm_client.validate_connection()
    if not validation["valid"]:
        error_msg = validation["message"]
        if not validation["ollama_accessible"]:
            st.error(
                f"❌ **Cannot Connect to Ollama**\n\n"
                f"{error_msg}\n\n"
                f"**Please ensure:**\n"
                f"1. Ollama service is running in Docker\n"
                f"2. The host URL is correct: `{OLLAMA_HOST}`\n"
                f"3. Check Ollama service status: `docker compose ps ollama`\n\n"
                f"**Troubleshooting:**\n"
                f"```bash\n# Check Ollama logs\ndocker compose logs ollama\n\n# Verify Ollama is healthy\ndocker compose exec ollama ollama list\n```"
            )
        elif not validation["model_available"]:
            st.error(
                f"❌ **Model Not Found**\n\n"
                f"{error_msg}\n\n"
                f"**To install the model in Docker:**\n"
                f"```bash\ndocker compose exec ollama ollama pull {MODEL_NAME}\n```\n\n"
                f"**Or check available models:**\n"
                f"```bash\ndocker compose exec ollama ollama list\n```"
            )
        else:
            st.error(
                f"❌ **Ollama Validation Error**\n\n"
                f"{error_msg}\n\n"
                f"Please check your Ollama installation and configuration."
            )
        st.stop()

# Validate environment (database, RAG files, etc.)
if 'environment_validated' not in st.session_state:
    env_validation = validate_environment()
    st.session_state.environment_validated = True
    
    # Show errors (critical issues)
    if env_validation["errors"]:
        error_text = "\n\n".join([f"❌ {err}" for err in env_validation["errors"]])
        st.error(f"**Critical Environment Issues:**\n\n{error_text}")
        st.stop()
    
    # Show warnings (non-critical issues)
    if env_validation["warnings"]:
        warning_text = "\n\n".join([f"⚠️ {warn}" for warn in env_validation["warnings"]])
        st.warning(f"**Environment Warnings:**\n\n{warning_text}")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Context management for performance optimization
if 'conversation_summary' not in st.session_state:
    st.session_state.conversation_summary = {
        "last_summarized_turn": 0,
        "summary_text": "",
        "key_events": []
    }
if 'turn_count' not in st.session_state:
    st.session_state.turn_count = 0
if 'user_profile_cache' not in st.session_state:
    st.session_state.user_profile_cache = None

if 'notification_service' not in st.session_state:
    st.session_state.notification_service = NotificationService(
        st.session_state.mcp_server,
        st.session_state.mcp_router
    )

# Helper function for summarizing long messages
def _summarize_long_message(message: str, max_length: int = 500) -> str:
    """
    Summarize a long message by keeping only the first sentence.
    
    Args:
        message: The message to summarize
        max_length: Maximum character length before summarization (default: 500)
        
    Returns:
        Original message if within limit, otherwise first sentence + "..."
    """
    if len(message) <= max_length:
        return message
    
    # Find first sentence boundary
    for punct in ['.', '!', '?']:
        idx = message.find(punct)
        if idx != -1:
            return message[:idx+1] + "..."
    
    # Fallback: truncate at max_length
    return message[:max_length-3] + "..."

# Define callback function (always create it)
def notification_callback(message: str):
    """Callback to add notification to chat history."""
    try:
        print(f"[CALLBACK] ========== NOTIFICATION CALLBACK CALLED ==========")
        print(f"[CALLBACK] Received message: '{message}'")
        print(f"[CALLBACK] Message type: {type(message)}")
        
        # Verify chat_history exists
        if 'chat_history' not in st.session_state:
            print(f"[CALLBACK] ✗ ERROR: chat_history not in session_state!")
            st.session_state.chat_history = []
        else:
            print(f"[CALLBACK] ✓ chat_history exists, current length: {len(st.session_state.chat_history)}")
        
        # Add to notification history
        if 'notification_history' not in st.session_state:
            st.session_state.notification_history = []
        st.session_state.notification_history.append({
            'message': message,
            'timestamp': time.time()
        })
        print(f"[CALLBACK] ✓ Added to notification_history")
        
        # Create notification entry for chat history
        notification_entry = {
            'role': 'assistant',
            'content': f"🔔 {message}",
            'is_notification': True,
            'activity': st.session_state.current_activity.get('activity') if st.session_state.current_activity else None
        }
        print(f"[CALLBACK] Created notification_entry: {notification_entry}")
        
        # Add to chat history
        st.session_state.chat_history.append(notification_entry)
        new_length = len(st.session_state.chat_history)
        print(f"[CALLBACK] ✓ Added to chat_history")
        print(f"[CALLBACK] New chat_history length: {new_length}")
        
        # Verify it was added correctly
        last_msg = st.session_state.chat_history[-1]
        if last_msg == notification_entry:
            print(f"[CALLBACK] ✓ Verified: Last message matches what we added")
        else:
            print(f"[CALLBACK] ⚠ WARNING: Last message doesn't match!")
            print(f"[CALLBACK] Expected: {notification_entry}")
            print(f"[CALLBACK] Got: {last_msg}")
        
        print(f"[CALLBACK] ========== CALLBACK COMPLETED ==========")
    except Exception as e:
        print(f"[CALLBACK] ✗ FATAL ERROR in notification_callback: {e}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise to see the error

# Ensure callback is always set
st.session_state.notification_service.set_notification_callback(notification_callback)
st.session_state.notification_callback = notification_callback

if 'recent_notification' not in st.session_state:
    st.session_state.recent_notification = None

if 'current_activity' not in st.session_state:
    st.session_state.current_activity = None  # {"activity": str, "time": str, "location": str or None}

if 'last_location' not in st.session_state:
    st.session_state.last_location = None

if 'custom_clock_time' not in st.session_state:
    st.session_state.custom_clock_time = None  # None means use real time, otherwise stores (hours, minutes)
if 'custom_clock_date' not in st.session_state:
    st.session_state.custom_clock_date = None  # Custom date in YYYY-MM-DD format, None means use current date
if 'custom_time_set_timestamp' not in st.session_state:
    st.session_state.custom_time_set_timestamp = None  # Timestamp when custom time was set (for calculating elapsed time)
if 'show_time_modal' not in st.session_state:
    st.session_state.show_time_modal = False  # Whether to show the time customization modal
if 'modal_hours' not in st.session_state:
    st.session_state.modal_hours = "8"  # Default hours in modal
if 'modal_minutes' not in st.session_state:
    st.session_state.modal_minutes = "00"  # Default minutes in modal

# Flag to trigger rerun from JavaScript (via hidden input)
if 'js_trigger_rerun' not in st.session_state:
    st.session_state.js_trigger_rerun = False


# Page configuration
st.set_page_config(
    page_title="Smart Room Control",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide sidebar completely
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🏠 Smart Room Control Interface")

# ========== LOCATION CHANGE HOUSE CHECK ==========
# Run house check whenever user location changes
current_location = st.session_state.state_manager.current_location

# Check if location has changed
if st.session_state.last_location is None:
    # First run - just store the location (don't check on initial load)
    st.session_state.last_location = current_location
    print(f"[LOCATION INIT] Initial location set to: {current_location}")
elif st.session_state.last_location != current_location:
    # Location changed - run house check
    print(f"[LOCATION CHANGE] Location changed from '{st.session_state.last_location}' to '{current_location}'")
    
    # Update stored location BEFORE running check to avoid duplicate checks
    old_location_for_check = st.session_state.last_location
    st.session_state.last_location = current_location
    
    # Run house check
    check_result = st.session_state.notification_service.run_house_check()
    
    # Debug output
    print(f"[HOUSE CHECK] Location change check result: {check_result}")
    
    if check_result and check_result.get("notified"):
        # Store recent notification info for preference updates
        message = check_result.get("message", "")
        # Get all potential issues (devices that are ON in other rooms)
        potential_issues = st.session_state.mcp_server.detect_potential_issues()
        if potential_issues:
            # Filter out devices that are in notification_preferences (user said "keep it on")
            current_state = st.session_state.mcp_server.get_current_state()
            notification_prefs = current_state.get("notification_preferences", [])
            devices_to_notify = []
            
            for issue in potential_issues:
                room = issue["room"]
                device = issue["device"]
                device_key = f"{room} {device}"
                
                # Only include devices that are not in notification preferences
                if device_key not in notification_prefs:
                    devices_to_notify.append({
                        "room": room,
                        "device": device
                    })
            
            # Store all devices that need notification (not just first one)
            if devices_to_notify:
                st.session_state.recent_notification = {
                    "devices": devices_to_notify,  # List of all devices
                    "message": message
                }
        # Rerun to show notification in chat
        st.rerun()
    elif check_result:
        # Check ran but no notification (all devices in preferences or no issues)
        print(f"[HOUSE CHECK] Check ran but no notification sent. Result: {check_result}")

# ========== SCHEDULE CHECK ==========
# Algorithm:
# 1. Get the current time that matches what's displayed on the UI clock
# 2. Check every minute (7:00, 7:01, 7:02, etc.) if it matches schedule
# 3. If match found and not already sent, send notification

# Initialize error tracking
if 'schedule_check_errors' not in st.session_state:
    st.session_state.schedule_check_errors = []

def log_schedule_error(message: str):
    """Log error to both console and UI display."""
    error_entry = {
        'time': datetime.now().strftime("%H:%M:%S"),
        'message': message
    }
    st.session_state.schedule_check_errors.append(error_entry)
    # Keep only last 10 errors
    if len(st.session_state.schedule_check_errors) > 10:
        st.session_state.schedule_check_errors = st.session_state.schedule_check_errors[-10:]
    print(f"[ERROR] {message}")

def _find_next_activity_time(daily_clone: list, current_time_str: str) -> str:
    """
    Find the next scheduled activity time after current_time_str.
    
    Args:
        daily_clone: List of schedule items sorted by time
        current_time_str: Current time in HH:MM format
        
    Returns:
        Next activity time in HH:MM format, or None if no next activity
    """
    # Parse current time
    try:
        current_parts = current_time_str.split(":")
        current_hours = int(current_parts[0])
        current_minutes = int(current_parts[1])
        current_total = current_hours * 60 + current_minutes
    except (ValueError, IndexError):
        return None
    
    # Find next activity
    for item in daily_clone:
        item_time = item.get("time", "")
        try:
            item_parts = item_time.split(":")
            item_hours = int(item_parts[0])
            item_minutes = int(item_parts[1])
            item_total = item_hours * 60 + item_minutes
            
            if item_total > current_total:
                return item_time
        except (ValueError, IndexError):
            continue
    
    return None

# Use Streamlit fragment to run schedule check every minute
@st.fragment(run_every=60)  # Run every 60 seconds (1 minute)
def schedule_check_fragment():
    """Fragment that runs every minute to check schedule and send notifications."""
    print(f"[SCHEDULE CHECK] ========== SCHEDULE CHECK FRAGMENT EXECUTED ==========")
    
    try:
        # Initialize tracking for sent notifications
        if 'sent_schedule_notifications' not in st.session_state:
            st.session_state.sent_schedule_notifications = set()
            print("[SCHEDULE CHECK] Initialized sent_schedule_notifications set")

        # Calculate current time (same calculation as clock display)
        gmt7 = timezone(timedelta(hours=7))
        if st.session_state.custom_clock_time:
            # Custom time: calculate from start time + elapsed seconds
            custom_hours, custom_minutes = st.session_state.custom_clock_time
            custom_time_timestamp = st.session_state.custom_time_set_timestamp if st.session_state.custom_time_set_timestamp else time.time()
            elapsed_seconds = int(time.time() - custom_time_timestamp)
            total_seconds = custom_hours * 3600 + custom_minutes * 60 + elapsed_seconds
            current_hours = (total_seconds // 3600) % 24
            current_minutes = (total_seconds % 3600) // 60
            current_time_str = f"{current_hours:02d}:{current_minutes:02d}"  # Format: "07:00", "07:01", etc.
            # Use custom date if set, otherwise use current date
            if st.session_state.custom_clock_date:
                current_date = st.session_state.custom_clock_date
            else:
                current_date = datetime.now(gmt7).strftime("%Y-%m-%d")
            print(f"[SCHEDULE CHECK] Using custom time - Current: {current_time_str}, Date: {current_date}")
        else:
            # Real GMT+7 time
            current_time = datetime.now(gmt7)
            current_time_str = current_time.strftime("%H:%M")  # Format: "07:00", "07:01", etc.
            current_date = current_time.strftime("%Y-%m-%d")
            print(f"[SCHEDULE CHECK] Using real time - Current: {current_time_str}, Date: {current_date}")

        # Track last checked minute to only check once per minute
        current_minute_key = current_time_str  # "07:00", "07:01", etc.
        if 'last_schedule_check_minute' not in st.session_state:
            st.session_state.last_schedule_check_minute = current_minute_key
            print(f"[SCHEDULE CHECK] First run - Initialized last_schedule_check_minute to: {current_minute_key}")
            # First run: don't check yet, just initialize
        else:
            # Only check if minute has changed (avoid checking multiple times in the same minute)
            last_checked = st.session_state.last_schedule_check_minute
            print(f"[SCHEDULE CHECK] Last checked minute: {last_checked}, Current minute: {current_minute_key}")
            
            if last_checked != current_minute_key:
                # Minute changed! Check schedule now
                print(f"[SCHEDULE CHECK] ✓ Minute changed from {last_checked} to {current_minute_key} - Checking schedule...")
                st.session_state.last_schedule_check_minute = current_minute_key
                
                # Check if current time matches any schedule item
                try:
                    schedule_items = st.session_state.state_manager.check_schedule_notifications(
                        current_time_str, current_date
                    )
                    print(f"[SCHEDULE CHECK] Schedule check returned {len(schedule_items)} matching item(s)")
                    
                    if schedule_items:
                        print(f"[SCHEDULE CHECK] Matching items: {schedule_items}")
                    else:
                        print(f"[SCHEDULE CHECK] No matching schedule items for {current_time_str}")
                except Exception as e:
                    print(f"[SCHEDULE CHECK] ERROR in check_schedule_notifications: {e}")
                    import traceback
                    traceback.print_exc()
                    schedule_items = []
                
                # Send notifications for matching items
                for item in schedule_items:
                    try:
                        # Create unique key to prevent duplicates
                        notification_key = f"{item['time']}_{item['activity']}_{item['type']}"
                        print(f"[SCHEDULE CHECK] Processing item: {item}")
                        print(f"[SCHEDULE CHECK] Notification key: {notification_key}")
                        print(f"[SCHEDULE CHECK] Already sent keys: {list(st.session_state.sent_schedule_notifications)}")
                        
                        # Check if already sent
                        if notification_key not in st.session_state.sent_schedule_notifications:
                            # Format notification message
                            message = f"It's time to: {item['activity']}"
                            
                            print(f"[SCHEDULE CHECK] ✓ Match found! Notification not sent yet.")
                            print(f"[SCHEDULE CHECK] Preparing to send: {message}")
                            
                            # Send notification via callback
                            if 'notification_callback' in st.session_state:
                                print(f"[SCHEDULE CHECK] ✓ notification_callback found in session_state")
                                try:
                                    callback_func = st.session_state.notification_callback
                                    print(f"[SCHEDULE CHECK] Callback function type: {type(callback_func)}")
                                    print(f"[SCHEDULE CHECK] Calling notification_callback with message: '{message}'")
                                    
                                    # Call the callback
                                    callback_func(message)
                                    
                                    print(f"[SCHEDULE CHECK] ✓ Callback executed successfully")
                                    print(f"[SCHEDULE CHECK] Chat history length after callback: {len(st.session_state.chat_history)}")
                                    
                                    # Verify notification was added
                                    last_message = st.session_state.chat_history[-1] if st.session_state.chat_history else None
                                    if last_message and last_message.get('is_notification'):
                                        print(f"[SCHEDULE CHECK] ✓ Verified: Last message in chat_history is a notification")
                                        print(f"[SCHEDULE CHECK] Last message content: {last_message.get('content')}")
                                    else:
                                        print(f"[SCHEDULE CHECK] ⚠ WARNING: Last message is NOT a notification!")
                                        print(f"[SCHEDULE CHECK] Last message: {last_message}")
                                    
                                    # Mark as sent
                                    st.session_state.sent_schedule_notifications.add(notification_key)
                                    print(f"[SCHEDULE CHECK] ✓ Notification marked as sent. Key: {notification_key}")
                                    
                                    # Find next activity time to determine when current activity ends
                                    daily_clone = st.session_state.state_manager.get_daily_clone(current_date=current_date)
                                    next_activity_time = _find_next_activity_time(daily_clone, current_time_str)
                                    
                                    # Clear previous current_activity before setting new one
                                    if st.session_state.current_activity:
                                        print(f"[SCHEDULE CHECK] Clearing previous current_activity: {st.session_state.current_activity.get('activity')}")
                                    
                                    # Store current activity context with end time
                                    st.session_state.current_activity = {
                                        "activity": item['activity'],
                                        "time": item['time'],
                                        "location": item.get("location"),
                                        "end_time": next_activity_time  # None if no next activity
                                    }
                                    print(f"[SCHEDULE CHECK] Set current_activity: {item['activity']} from {item['time']} until {next_activity_time}")
                                    
                                    # Step 2: Execute action if action field exists (pure logic, no LLM)
                                    action_field = item.get("action")
                                    print(f"[SCHEDULE CHECK] Checking for action field: {action_field is not None}")
                                    if action_field:
                                        print(f"[SCHEDULE CHECK] Action field type: {type(action_field)}, content: {action_field}")
                                        devices = action_field.get("devices")
                                        print(f"[SCHEDULE CHECK] Devices from action: {devices}")
                                        
                                        if devices and isinstance(devices, list) and len(devices) > 0:
                                            print(f"[SCHEDULE CHECK] Executing action with {len(devices)} device(s)")
                                            
                                            # Track execution results
                                            success_count = 0
                                            failure_count = 0
                                            
                                            for idx, device_spec in enumerate(devices):
                                                print(f"[SCHEDULE CHECK] Processing device {idx + 1}/{len(devices)}: {device_spec}")
                                                room = device_spec.get("room")
                                                device = device_spec.get("device")
                                                state = device_spec.get("state")  # "ON" or "OFF"
                                                
                                                print(f"[SCHEDULE CHECK] Device spec - room: {room}, device: {device}, state: {state}")
                                                
                                                # Validate device exists
                                                if not room or not device or not state:
                                                    print(f"[SCHEDULE CHECK] ✗ Invalid device spec: missing room/device/state")
                                                    failure_count += 1
                                                    continue
                                                
                                                if room in ROOMS and device in ROOMS[room]:
                                                    # Execute via MCP router (no LLM)
                                                    try:
                                                        print(f"[SCHEDULE CHECK] Attempting to control {room} {device} to {state}")
                                                        result = st.session_state.mcp_router.execute({
                                                            "tool": "e_device_control",
                                                            "arguments": {"room": room, "device": device, "action": state}
                                                        })
                                                        print(f"[SCHEDULE CHECK] Device control result: {result}")
                                                        
                                                        if result.get("success"):
                                                            print(f"[SCHEDULE CHECK] ✓ Successfully controlled {room} {device} to {state}")
                                                            success_count += 1
                                                            # Trigger rerun to sync UI toggles with database
                                                            st.rerun()
                                                        else:
                                                            error_msg = result.get('error', 'Unknown error')
                                                            print(f"[SCHEDULE CHECK] ✗ Failed to control {room} {device}: {error_msg}")
                                                            failure_count += 1
                                                    except Exception as e:
                                                        print(f"[SCHEDULE CHECK] ✗ Exception controlling {room} {device}: {e}")
                                                        import traceback
                                                        traceback.print_exc()
                                                        failure_count += 1
                                                else:
                                                    print(f"[SCHEDULE CHECK] ✗ Invalid device: {room} {device} (not in ROOMS config)")
                                                    print(f"[SCHEDULE CHECK] Available rooms: {list(ROOMS.keys())}")
                                                    if room in ROOMS:
                                                        print(f"[SCHEDULE CHECK] Available devices in {room}: {list(ROOMS[room])}")
                                                    failure_count += 1
                                            
                                            print(f"[SCHEDULE CHECK] Action execution summary: {success_count} succeeded, {failure_count} failed")
                                        else:
                                            print(f"[SCHEDULE CHECK] ⚠ Action field exists but devices list is empty or invalid: {devices}")
                                    else:
                                        print(f"[SCHEDULE CHECK] No action field in schedule item")
                                    
                                    # Step 3: Check location if location field exists (pure logic, no LLM)
                                    if item.get("location"):
                                        required_location = item["location"]
                                        current_location = st.session_state.state_manager.current_location
                                        
                                        print(f"[SCHEDULE CHECK] Checking location: required={required_location}, current={current_location}")
                                        
                                        if required_location != current_location:
                                            # Send location notification
                                            location_message = f"Please move to {required_location} for {item['activity']}"
                                            print(f"[SCHEDULE CHECK] Location mismatch - sending notification: {location_message}")
                                            
                                            try:
                                                if 'notification_callback' in st.session_state:
                                                    st.session_state.notification_callback(location_message)
                                                    print(f"[SCHEDULE CHECK] ✓ Location notification sent")
                                            except Exception as e:
                                                print(f"[SCHEDULE CHECK] ✗ Error sending location notification: {e}")
                                        else:
                                            print(f"[SCHEDULE CHECK] ✓ User already in correct location: {required_location}")
                                    
                                    # Trigger rerun to display notification in UI
                                    # Fragments can call st.rerun() to trigger a full script rerun
                                    print(f"[SCHEDULE CHECK] Triggering st.rerun() to display notification in UI")
                                    st.rerun()
                                    break  # Exit loop after first notification
                                except Exception as e:
                                    error_msg = f"Error calling notification_callback: {e}"
                                    print(f"[SCHEDULE CHECK] ✗ {error_msg}")
                                    log_schedule_error(error_msg)
                                    import traceback
                                    traceback.print_exc()
                            else:
                                error_msg = "notification_callback not found in session_state"
                                print(f"[SCHEDULE CHECK] ✗ ERROR: {error_msg}")
                                log_schedule_error(error_msg)
                                print(f"[SCHEDULE CHECK] Available session_state keys: {[k for k in st.session_state.keys() if 'notif' in k.lower()]}")
                        else:
                            print(f"[SCHEDULE CHECK] Notification already sent, skipping: {notification_key}")
                    except Exception as e:
                        error_msg = f"Error processing schedule item: {e}"
                        print(f"[SCHEDULE CHECK] ✗ {error_msg}")
                        log_schedule_error(error_msg)
                        import traceback
                        traceback.print_exc()
            else:
                print(f"[SCHEDULE CHECK] Minute hasn't changed ({current_minute_key}), skipping check")
    except Exception as e:
        error_msg = f"FATAL ERROR in schedule check: {e}"
        print(f"[SCHEDULE CHECK] ✗ {error_msg}")
        log_schedule_error(error_msg)
        import traceback
        traceback.print_exc()

# Call the fragment to start periodic execution
schedule_check_fragment()

# Main layout: two columns
col1, col2 = st.columns([1, 1])

# ========== LEFT COLUMN: Room Map & Device Control ==========
with col1:
    st.header("📍 Room Map & Device Control")
    
    # ========== User Information Section ==========
    with st.expander("👤 User Information", expanded=False):
        user_info = st.session_state.state_manager.get_user_info(include_one_time_events=False)
        
        # 1. Name Section
        st.subheader("1️⃣ Name")
        col_thai, col_eng = st.columns(2)
        with col_thai:
            thai_name = st.text_input(
                "Thai Name (ชื่อ)",
                value=user_info["name"].get("thai", ""),
                key="user_name_thai_input"
            )
        with col_eng:
            english_name = st.text_input(
                "English Name",
                value=user_info["name"].get("english", ""),
                key="user_name_english_input"
            )
        
        # Save name if changed
        if thai_name != user_info["name"].get("thai", "") or english_name != user_info["name"].get("english", ""):
            st.session_state.state_manager.set_user_name(thai=thai_name, english=english_name)
            if st.session_state.get("_name_saved", False) == False:
                st.session_state._name_saved = True
                st.success("Name saved!")
                st.rerun()
        else:
            st.session_state._name_saved = False
        
        st.divider()
        
        # 2. Schedule Section
        st.subheader("2️⃣ Daily Schedule")
        schedule = user_info["schedule"]
        
        # Display existing schedule items
        schedule_changed = False
        new_schedule = []
        if schedule:
            st.caption("Your daily routine:")
            for idx, item in enumerate(schedule):
                # Build expander title with indicators for optional fields
                title_parts = [f"⏰ {item.get('time', '')} - {item.get('activity', '')}"]
                if item.get("action") and item.get("action", {}).get("devices"):
                    device_count = len(item.get("action", {}).get("devices", []))
                    title_parts.append(f"⚡{device_count}")
                if item.get("location"):
                    title_parts.append(f"📍{item.get('location')}")
                expander_title = " ".join(title_parts)
                
                # Use expander for each schedule item to show optional fields
                with st.expander(expander_title, expanded=False):
                    # Basic fields: Time and Activity
                    col_time, col_activity = st.columns([1, 2])
                    with col_time:
                        new_time = st.text_input(
                            "Time",
                            value=item.get("time", ""),
                            key=f"schedule_time_{idx}",
                            help="Format: HH:MM (e.g., 07:00)"
                        )
                    with col_activity:
                        new_activity = st.text_input(
                            "Activity",
                            value=item.get("activity", ""),
                            key=f"schedule_activity_{idx}",
                            help="Activity description"
                        )
                    
                    st.divider()
                    
                    # Optional Location field
                    st.markdown("**📍 Location (Optional)**")
                    current_location = item.get("location", "")
                    location_options = ["None"] + list(ROOMS.keys())
                    location_idx = 0 if not current_location else location_options.index(current_location) if current_location in location_options else 0
                    selected_location = st.selectbox(
                        "Required Location",
                        location_options,
                        index=location_idx,
                        key=f"schedule_location_{idx}",
                        help="Select location where this activity should take place"
                    )
                    new_location = None if selected_location == "None" else selected_location
                    
                    st.divider()
                    
                    # Optional Action field (Devices to control)
                    st.markdown("**⚡ Actions (Optional)**")
                    st.caption("Specify devices to control automatically when this schedule triggers")
                    
                    # Get current action devices
                    current_action = item.get("action", {})
                    current_devices = current_action.get("devices", []) if current_action else []
                    
                    # Initialize session state for this item's devices if needed
                    if f"schedule_{idx}_devices" not in st.session_state:
                        st.session_state[f"schedule_{idx}_devices"] = current_devices.copy() if current_devices else []
                    
                    # Track devices to delete
                    device_to_delete = None
                    
                    # Display existing devices
                    devices_to_keep = st.session_state[f"schedule_{idx}_devices"].copy()
                    if devices_to_keep:
                        st.markdown("**Current devices:**")
                        # Header row
                        col_h1, col_h2, col_h3, col_h4 = st.columns([2, 2, 1, 1])
                        with col_h1:
                            st.caption("Room")
                        with col_h2:
                            st.caption("Device")
                        with col_h3:
                            st.caption("State")
                        with col_h4:
                            st.caption("Action")
                        
                        for device_idx, device_spec in enumerate(devices_to_keep):
                            col_dev_room, col_dev_device, col_dev_state, col_dev_del = st.columns([2, 2, 1, 1])
                            with col_dev_room:
                                dev_room = st.selectbox(
                                    "Room",
                                    list(ROOMS.keys()),
                                    index=list(ROOMS.keys()).index(device_spec.get("room", "Bedroom")) if device_spec.get("room") in ROOMS else 0,
                                    key=f"schedule_{idx}_device_{device_idx}_room",
                                    label_visibility="collapsed"
                                )
                            with col_dev_device:
                                room_devices = ROOMS.get(dev_room, [])
                                dev_device = st.selectbox(
                                    "Device",
                                    room_devices,
                                    index=room_devices.index(device_spec.get("device", room_devices[0])) if device_spec.get("device") in room_devices else 0,
                                    key=f"schedule_{idx}_device_{device_idx}_device",
                                    label_visibility="collapsed"
                                )
                            with col_dev_state:
                                dev_state = st.selectbox(
                                    "State",
                                    ["ON", "OFF"],
                                    index=0 if device_spec.get("state") == "ON" else 1,
                                    key=f"schedule_{idx}_device_{device_idx}_state",
                                    label_visibility="collapsed"
                                )
                            with col_dev_del:
                                if st.button("🗑️", key=f"schedule_{idx}_device_{device_idx}_delete", help="Remove device"):
                                    device_to_delete = device_idx
                            
                            # Update device in the list
                            if device_to_delete != device_idx:
                                devices_to_keep[device_idx] = {
                                    "room": dev_room,
                                    "device": dev_device,
                                    "state": dev_state
                                }
                    
                    # Handle device deletion
                    if device_to_delete is not None:
                        devices_to_keep.pop(device_to_delete)
                        st.session_state[f"schedule_{idx}_devices"] = devices_to_keep
                        st.rerun()
                    
                    # Add new device section
                    st.markdown("**Add new device:**")
                    col_new_dev_room, col_new_dev_device, col_new_dev_state, col_new_dev_btn = st.columns([2, 2, 1, 1])
                    with col_new_dev_room:
                        new_dev_room = st.selectbox(
                            "Room",
                            list(ROOMS.keys()),
                            key=f"schedule_{idx}_new_device_room"
                        )
                    with col_new_dev_device:
                        new_dev_device = st.selectbox(
                            "Device",
                            ROOMS.get(new_dev_room, []),
                            key=f"schedule_{idx}_new_device_device"
                        )
                    with col_new_dev_state:
                        new_dev_state = st.selectbox(
                            "State",
                            ["ON", "OFF"],
                            key=f"schedule_{idx}_new_device_state"
                        )
                    with col_new_dev_btn:
                        if st.button("➕ Add", key=f"schedule_{idx}_new_device_add"):
                            devices_to_keep.append({
                                "room": new_dev_room,
                                "device": new_dev_device,
                                "state": new_dev_state
                            })
                            st.session_state[f"schedule_{idx}_devices"] = devices_to_keep
                            st.rerun()
                    
                    # Update session state with current devices (after all modifications)
                    st.session_state[f"schedule_{idx}_devices"] = devices_to_keep
                    
                    # Build updated item with all fields
                    updated_item = {"time": new_time, "activity": new_activity}
                    if new_location:
                        updated_item["location"] = new_location
                    if devices_to_keep:
                        updated_item["action"] = {"devices": devices_to_keep}
                    
                    # Check if item was modified
                    item_location = updated_item.get("location")
                    old_location = item.get("location")
                    item_action_devices = updated_item.get("action", {}).get("devices", []) if updated_item.get("action") else []
                    
                    if (new_time != item.get("time", "") or 
                        new_activity != item.get("activity", "") or
                        item_location != old_location or
                        item_action_devices != current_devices):
                        schedule_changed = True
                    
                    new_schedule.append(updated_item)
                    
                    # Delete button
                    if st.button("🗑️ Delete Schedule Item", key=f"delete_schedule_{idx}", type="secondary"):
                        st.session_state.state_manager.remove_schedule_item(idx)
                        st.rerun()
        else:
            new_schedule = []
        
        # Add new schedule item
        with st.expander("➕ Add Schedule Item", expanded=False):
            # Clear form if flag is set
            if st.session_state.get("clear_schedule_form", False):
                st.session_state.pop("new_schedule_time", None)
                st.session_state.pop("new_schedule_activity", None)
                st.session_state.pop("new_schedule_location", None)
                st.session_state.pop("new_schedule_devices", None)
                st.session_state.pop("clear_schedule_form", None)
            
            col_new_time, col_new_activity = st.columns(2)
            with col_new_time:
                new_schedule_time = st.text_input(
                    "Time (e.g., 08:00)",
                    key="new_schedule_time",
                    placeholder="08:00",
                    help="Format: HH:MM"
                )
            with col_new_activity:
                new_schedule_activity = st.text_input(
                    "Activity",
                    key="new_schedule_activity",
                    placeholder="Wake up",
                    help="Activity description"
                )
            
            st.divider()
            
            # Optional Location for new item
            st.markdown("**📍 Location (Optional)**")
            new_schedule_location = st.selectbox(
                "Required Location",
                ["None"] + list(ROOMS.keys()),
                key="new_schedule_location",
                help="Select location where this activity should take place"
            )
            if new_schedule_location == "None":
                new_schedule_location = None
            
            st.divider()
            
            # Optional Actions for new item
            st.markdown("**⚡ Actions (Optional)**")
            st.caption("Specify devices to control automatically when this schedule triggers")
            
            # Manage devices for new item
            if "new_schedule_devices" not in st.session_state:
                st.session_state.new_schedule_devices = []
            
            # Display current devices
            if st.session_state.new_schedule_devices:
                st.markdown("**Devices to control:**")
                for device_idx, device_spec in enumerate(st.session_state.new_schedule_devices):
                    col_dev_room, col_dev_device, col_dev_state, col_dev_del = st.columns([2, 2, 1, 1])
                    with col_dev_room:
                        st.text(device_spec.get("room", ""))
                    with col_dev_device:
                        st.text(device_spec.get("device", ""))
                    with col_dev_state:
                        st.text(device_spec.get("state", ""))
                    with col_dev_del:
                        if st.button("🗑️", key=f"new_schedule_device_{device_idx}_delete"):
                            st.session_state.new_schedule_devices.pop(device_idx)
                            st.rerun()
            
            # Add new device
            st.markdown("**Add device:**")
            col_new_dev_room, col_new_dev_device, col_new_dev_state, col_new_dev_add = st.columns([2, 2, 1, 1])
            with col_new_dev_room:
                new_dev_room = st.selectbox(
                    "Room",
                    list(ROOMS.keys()),
                    key="new_schedule_device_room"
                )
            with col_new_dev_device:
                new_dev_device = st.selectbox(
                    "Device",
                    ROOMS.get(new_dev_room, []),
                    key="new_schedule_device_device"
                )
            with col_new_dev_state:
                new_dev_state = st.selectbox(
                    "State",
                    ["ON", "OFF"],
                    key="new_schedule_device_state"
                )
            with col_new_dev_add:
                if st.button("➕", key="new_schedule_device_add"):
                    st.session_state.new_schedule_devices.append({
                        "room": new_dev_room,
                        "device": new_dev_device,
                        "state": new_dev_state
                    })
                    st.rerun()
            
            if st.button("Add Schedule Item", key="add_schedule_btn", type="primary"):
                if new_schedule_time and new_schedule_activity:
                    # Build new item with optional fields
                    new_item = {"time": new_schedule_time, "activity": new_schedule_activity}
                    if new_schedule_location:
                        new_item["location"] = new_schedule_location
                    if st.session_state.new_schedule_devices:
                        new_item["action"] = {"devices": st.session_state.new_schedule_devices.copy()}
                    
                    # Use schedule_modifier tool to add with validation
                    result = st.session_state.mcp_server.schedule_modifier(
                        modify_type="add",
                        time=new_schedule_time,
                        activity=new_schedule_activity,
                        action=new_item.get("action"),
                        location=new_item.get("location")
                    )
                    
                    if result.get("success"):
                        # Set flag to clear form on next run
                        st.session_state.clear_schedule_form = True
                        st.success("Schedule item added!")
                        st.rerun()
                    else:
                        st.error(f"Error: {result.get('error', 'Unknown error')}")
                else:
                    st.warning("Please fill in both time and activity")
        
        # Save schedule if changed (only once to avoid rerun loops)
        if schedule_changed and not st.session_state.get("_schedule_being_saved", False):
            st.session_state._schedule_being_saved = True
            st.session_state.state_manager.set_user_schedule(new_schedule)
            st.success("Schedule updated!")
            st.rerun()
        else:
            st.session_state._schedule_being_saved = False
        
        st.divider()
        
        # 3. Condition Section
        st.subheader("3️⃣ Condition")
        condition_text = st.text_area(
            "Medical conditions or other relevant information",
            value=user_info.get("condition", ""),
            key="user_condition_input",
            height=100,
            placeholder="e.g., Has diabetes, allergic to dust, etc."
        )
        
        # Save condition if changed
        if condition_text != user_info.get("condition", ""):
            st.session_state.state_manager.set_user_condition(condition_text)
            if st.session_state.get("_condition_saved", False) == False:
                st.session_state._condition_saved = True
                st.success("Condition saved!")
                st.rerun()
        else:
            st.session_state._condition_saved = False
    
    st.divider()
    
    # ========== Schedule Reset Section ==========
    st.subheader("🔄 Schedule Reset")
    st.caption("Reset daily schedule to base schedule and clear all one-time events (for demonstrations)")
    if st.button("🔄 Reset Schedule", key="reset_schedule_button", use_container_width=True, type="secondary"):
        result = st.session_state.state_manager.reset_daily_schedule()
        st.success(f"✅ Schedule reset! Cleared {result['one_time_events_cleared']} one-time event(s).")
        st.rerun()
    
    st.divider()
    
    # Location selector
    current_location = st.session_state.state_manager.current_location
    selected_location = st.selectbox(
        "📍 Choose your current location:",
        options=list(ROOMS.keys()),
        index=list(ROOMS.keys()).index(current_location) if current_location in ROOMS else 0,
        key="location_selector"
    )
    
    # Update location if changed
    if selected_location != current_location:
        # Store old location before changing
        old_location = current_location
        st.session_state.state_manager.set_location(selected_location)
        # Update last_location to trigger house check
        st.session_state.last_location = old_location
        # Rerun to trigger location change check
        st.rerun()
    
    st.info(f"📍 **Current Location:** {current_location}")
    
    st.divider()
    
    # Display device states with interactive toggles
    devices = st.session_state.state_manager.get_all_devices()
    print(f"[UI DEBUG] Fetched devices from database: {devices}")
    
    for room, room_devices in devices.items():
        # Room header with expander
        with st.expander(f"🏠 {room}", expanded=True):
            # Highlight current room
            if room == current_location:
                st.caption(f"📍 You are here")
            
            st.markdown("---")
            
            # Display each device with toggle
            for device, state in room_devices.items():
                # Device row with toggle
                col_device, col_toggle = st.columns([3, 1])
                
                with col_toggle:
                    # Interactive toggle - Toggle is the single source of truth for display
                    base_toggle_key = f"toggle_{room}_{device}"
                    last_db_state_key = f"last_db_{room}_{device}"
                    
                    # Get widget state and last known database state
                    widget_state = st.session_state.get(base_toggle_key)
                    last_db_state = st.session_state.get(last_db_state_key)
                    
                    # Determine if this is a user interaction or external update
                    # User interaction: widget_state changed from matching last_db_state to a new value
                    # External update: widget_state doesn't match current database state (and wasn't just changed by user)
                    is_user_interaction = (last_db_state is not None and 
                                         widget_state is not None and 
                                         widget_state != last_db_state and 
                                         widget_state != state)
                    needs_sync = (widget_state is not None and 
                                widget_state != state and 
                                not is_user_interaction)
                    
                    # Store current database state for next comparison
                    st.session_state[last_db_state_key] = state
                    
                    if needs_sync:
                        # External update - sync toggle to database
                        print(f"[UI DEBUG] External update detected for {room} {device}: widget={widget_state}, db={state}")
                        import time
                        toggle_key = f"{base_toggle_key}_sync_{int(time.time()*1000)}"
                        if base_toggle_key in st.session_state:
                            del st.session_state[base_toggle_key]
                        new_state = st.toggle(
                            "",
                            value=state,  # Force database state for sync
                            key=toggle_key,
                            label_visibility="collapsed"
                        )
                    else:
                        # Normal operation or user interaction - use stable key
                        toggle_key = base_toggle_key
                        new_state = st.toggle(
                            "",
                            value=state,  # Default value (used only if widget_state doesn't exist)
                            key=toggle_key,
                            label_visibility="collapsed"
                        )
                        if is_user_interaction:
                            print(f"[UI DEBUG] User interaction detected for {room} {device}: widget_state={widget_state}, last_db={last_db_state}, current_db={state}")
                
                with col_device:
                    # Visual status indicator - use toggle's state (toggle is source of truth for display)
                    # This ensures text and toggle always match visually
                    display_state = new_state  # Use toggle's state for display
                    status_emoji = "🟢" if display_state else "⚫"
                    status_text = "**ON**" if display_state else "**OFF**"
                    st.markdown(f"{status_emoji} **{device}** - {status_text}")
                
                # Handle toggle state changes - bidirectional sync
                if new_state != state:
                    if needs_sync:
                        # Sync issue - toggle still doesn't match database after versioned key reset
                        # This shouldn't happen, but force rerun as fallback
                        print(f"[UI DEBUG] Sync incomplete for {room} {device}: toggle={new_state}, db={state}, forcing rerun")
                        st.rerun()
                    elif is_user_interaction:
                        # User interaction - user changed toggle, update database to match toggle
                        print(f"[UI DEBUG] Toggle changed by user for {room} {device}: {state} -> {new_state}")
                        action = "ON" if new_state else "OFF"
                        result = st.session_state.mcp_server.e_device_control(
                            room, device, action
                        )
                        if result["success"]:
                            # Verify database was actually updated before rerun
                            # This ensures the update is committed and visible
                            verify_state = st.session_state.state_manager.get_device_state(room, device)
                            if verify_state == new_state:
                                print(f"[UI DEBUG] Database verified: {room} {device} = {verify_state}, rerunning")
                                st.rerun()
                            else:
                                print(f"[UI DEBUG] Database verification failed: expected {new_state}, got {verify_state}, forcing rerun anyway")
                                st.rerun()
                        else:
                            print(f"[UI DEBUG] Database update failed for {room} {device}: {result.get('error', 'Unknown error')}")
                    else:
                        # Initial state mismatch (first render) - update database to match toggle
                        print(f"[UI DEBUG] Initial state mismatch for {room} {device}: toggle={new_state}, db={state}, updating database")
                        action = "ON" if new_state else "OFF"
                        result = st.session_state.mcp_server.e_device_control(
                            room, device, action
                        )
                        if result["success"]:
                            st.rerun()

# ========== RIGHT COLUMN: Chat Interface ==========
with col2:
    # Digital Clock Widget (GMT+7) - Top of Chat Interface
    st.subheader("🕐 Clock Settings")
    
    # Customize timestamp button
    if st.button("⏰ Customize TS", key="customize_ts_button", use_container_width=True):
        st.session_state.show_time_modal = True
        # Initialize modal values with current custom time/date or defaults
        if st.session_state.custom_clock_time:
            st.session_state.modal_hours = str(st.session_state.custom_clock_time[0])
            st.session_state.modal_minutes = str(st.session_state.custom_clock_time[1]).zfill(2)
        else:
            # Default to current time
            current_time = datetime.now(timezone(timedelta(hours=7)))
            st.session_state.modal_hours = str(current_time.hour)
            st.session_state.modal_minutes = str(current_time.minute).zfill(2)
        
        if st.session_state.custom_clock_date:
            st.session_state.modal_date = st.session_state.custom_clock_date
        else:
            # Default to current date
            current_date = datetime.now(timezone(timedelta(hours=7)))
            st.session_state.modal_date = current_date.strftime("%Y-%m-%d")
        st.rerun()
    
    # Modal dialog for time and date customization
    if st.session_state.show_time_modal:
        # Use expander as modal-like interface
        with st.expander("⏰ Customize Timestamp (GMT+7)", expanded=True):
            st.markdown("### Set Custom Date and Time")
            
            # Close button (X) at top right - using columns
            col_title, col_close = st.columns([10, 1])
            with col_title:
                st.markdown("**Customize Timestamp**")
            with col_close:
                if st.button("❌", key="modal_close_x", help="Close"):
                    st.session_state.show_time_modal = False
                    st.rerun()
            
            # Date input
            if 'modal_date' not in st.session_state:
                current_date = datetime.now(timezone(timedelta(hours=7)))
                st.session_state.modal_date = current_date.strftime("%Y-%m-%d")
            
            date_input = st.date_input(
                "Date",
                value=datetime.strptime(st.session_state.modal_date, "%Y-%m-%d").date(),
                key="modal_date_input"
            )
            
            # Time inputs
            col_hour, col_min = st.columns(2)
            
            with col_hour:
                if 'modal_hours' not in st.session_state:
                    current_time = datetime.now(timezone(timedelta(hours=7)))
                    st.session_state.modal_hours = str(current_time.hour)
                hours_input = st.text_input(
                    "Hours (0-23)",
                    value=st.session_state.modal_hours,
                    key="modal_hours_input",
                    help="Enter hours (0-23)"
                )
            
            with col_min:
                if 'modal_minutes' not in st.session_state:
                    current_time = datetime.now(timezone(timedelta(hours=7)))
                    st.session_state.modal_minutes = str(current_time.minute).zfill(2)
                minutes_input = st.text_input(
                    "Minutes (0-59)",
                    value=st.session_state.modal_minutes,
                    key="modal_minutes_input",
                    help="Enter minutes (0-59)"
                )
            
            # Validation function
            def validate_and_parse_datetime(date_obj, hours_str, minutes_str):
                """Validate and parse date, hours and minutes"""
                try:
                    hours = int(hours_str.strip()) if hours_str.strip() else None
                    minutes = int(minutes_str.strip()) if minutes_str.strip() else None
                    
                    if hours is None or minutes is None:
                        return None, None, "Please enter both hours and minutes"
                    
                    if not (0 <= hours <= 23):
                        return None, None, "Hours must be between 0 and 23"
                    
                    if not (0 <= minutes <= 59):
                        return None, None, "Minutes must be between 0 and 59"
                    
                    date_str = date_obj.strftime("%Y-%m-%d")
                    return date_str, (hours, minutes), None
                except ValueError:
                    return None, None, "Please enter valid numbers"
                except Exception as e:
                    return None, None, f"Error: {str(e)}"
            
            # Error message display area
            error_placeholder = st.empty()
            
            # Buttons row
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                apply_clicked = st.button("✅ Apply", use_container_width=True, type="primary", key="modal_apply")
            with col_btn2:
                original_clicked = st.button("🔄 Original", use_container_width=True, key="modal_original")
            
            # Handle button clicks
            if apply_clicked:
                date_str, parsed_time, error_msg = validate_and_parse_datetime(date_input, hours_input, minutes_input)
                
                if parsed_time and date_str:
                    # Valid input - apply it and store timestamp
                    st.session_state.custom_clock_time = parsed_time
                    st.session_state.custom_clock_date = date_str
                    st.session_state.custom_time_set_timestamp = time.time()  # Store current timestamp
                    st.session_state.modal_hours = str(parsed_time[0])
                    st.session_state.modal_minutes = str(parsed_time[1]).zfill(2)
                    st.session_state.modal_date = date_str
                    st.session_state.show_time_modal = False
                    st.rerun()
                else:
                    # Show error
                    error_placeholder.error(f"⚠️ {error_msg}")
                    # Update session state with current inputs
                    st.session_state.modal_hours = hours_input
                    st.session_state.modal_minutes = minutes_input
                    st.session_state.modal_date = date_input.strftime("%Y-%m-%d")
            
            if original_clicked:
                # Reset to real time/date
                st.session_state.custom_clock_time = None
                st.session_state.custom_clock_date = None
                st.session_state.custom_time_set_timestamp = None
                st.session_state.show_time_modal = False
                st.rerun()
            
            # Update session state with current inputs (for persistence)
            if not apply_clicked and not original_clicked:
                st.session_state.modal_hours = hours_input
                st.session_state.modal_minutes = minutes_input
                st.session_state.modal_date = date_input.strftime("%Y-%m-%d")
    
    # Calculate display time
    gmt7 = timezone(timedelta(hours=7))
    if st.session_state.custom_clock_time:
        # Use custom time - will be calculated in JavaScript with elapsed time
        custom_hours, custom_minutes = st.session_state.custom_clock_time
        # Initial display (will be updated by JavaScript)
        time_str = f"{custom_hours:02d}:{custom_minutes:02d}:00"
        # Use custom date if set, otherwise use current date
        if st.session_state.custom_clock_date:
            date_str = st.session_state.custom_clock_date
        else:
            date_str = datetime.now(gmt7).strftime("%Y-%m-%d")
        time_offset_hours = custom_hours
        time_offset_minutes = custom_minutes
        # Pass timestamp to JavaScript (in milliseconds)
        custom_time_timestamp = int(st.session_state.custom_time_set_timestamp * 1000) if st.session_state.custom_time_set_timestamp else None
        custom_date = st.session_state.custom_clock_date
    else:
        # Use real time
        current_time = datetime.now(gmt7)
        time_str = current_time.strftime("%H:%M:%S")
        date_str = current_time.strftime("%Y-%m-%d")
        time_offset_hours = None
        time_offset_minutes = None
        custom_time_timestamp = None
        custom_date = None
    
    # Display clock using components.html for reliable rendering
    clock_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Courier New', monospace;
        }}
        .clock-container {{
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        #clock-time {{
            font-size: 3rem;
            font-weight: bold;
            color: white;
            line-height: 1.2;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            margin: 0;
        }}
        #clock-date {{
            font-size: 1.1rem;
            color: rgba(255,255,255,0.95);
            margin-top: 10px;
            font-weight: 500;
        }}
    </style>
</head>
<body>
    <div class="clock-container">
        <div id="clock-time">{time_str}</div>
        <div id="clock-date">{date_str} (GMT+7)</div>
    </div>
    <script>
        const customTime = {{
            hours: {time_offset_hours if time_offset_hours is not None else 'null'},
            minutes: {time_offset_minutes if time_offset_minutes is not None else 'null'},
            timestamp: {custom_time_timestamp if custom_time_timestamp is not None else 'null'},
            date: {f"'{custom_date}'" if custom_date else 'null'}
        }};
        
        let lastMinute = null;
        
        function updateClock() {{
            let hours, minutes, seconds, year, month, day;
            
            if (customTime.hours !== null && customTime.minutes !== null && customTime.timestamp !== null) {{
                // Use custom time - calculate elapsed time and add to custom time
                const now = Date.now();
                const elapsedSeconds = Math.floor((now - customTime.timestamp) / 1000);
                
                // Start from custom time (seconds = 0)
                let totalSeconds = customTime.hours * 3600 + customTime.minutes * 60 + elapsedSeconds;
                
                // Calculate hours, minutes, seconds (handle overflow)
                hours = Math.floor(totalSeconds / 3600) % 24;
                minutes = Math.floor((totalSeconds % 3600) / 60);
                seconds = totalSeconds % 60;
                
                hours = String(hours).padStart(2, '0');
                minutes = String(minutes).padStart(2, '0');
                seconds = String(seconds).padStart(2, '0');
                
                // Use custom date if set, otherwise use today's date
                if (customTime.date !== null) {{
                    const dateParts = customTime.date.split('-');
                    year = dateParts[0];
                    month = dateParts[1];
                    day = dateParts[2];
                }} else {{
                    const nowDate = new Date();
                    year = nowDate.getFullYear();
                    month = String(nowDate.getMonth() + 1).padStart(2, '0');
                    day = String(nowDate.getDate()).padStart(2, '0');
                }}
            }} else {{
                // Use real GMT+7 time
                const now = new Date();
                const utcTime = now.getTime() + (now.getTimezoneOffset() * 60 * 1000);
                const gmt7Time = new Date(utcTime + (7 * 60 * 60 * 1000));
                
                hours = String(gmt7Time.getHours()).padStart(2, '0');
                minutes = String(gmt7Time.getMinutes()).padStart(2, '0');
                seconds = String(gmt7Time.getSeconds()).padStart(2, '0');
                year = gmt7Time.getFullYear();
                month = String(gmt7Time.getMonth() + 1).padStart(2, '0');
                day = String(gmt7Time.getDate()).padStart(2, '0');
            }}
            
            const timeElement = document.getElementById('clock-time');
            const dateElement = document.getElementById('clock-date');
            if (timeElement) {{
                timeElement.textContent = hours + ':' + minutes + ':' + seconds;
            }}
            if (dateElement) {{
                dateElement.textContent = year + '-' + month + '-' + day + ' (GMT+7)';
            }}
            
            // Check if minute has changed (for schedule checking)
            // NOTE: We used to call window.location.reload() here, but it doesn't properly
            // trigger Streamlit script execution. The schedule check now runs on every
            // script execution and uses last_schedule_check_minute to avoid duplicate checks.
            const currentMinute = hours + ':' + minutes;
            lastMinute = currentMinute;
        }}
        
        updateClock();
        setInterval(updateClock, 1000);
    </script>
</body>
</html>"""
    
    components.html(clock_html, height=140, scrolling=False)
    
    st.header("💬 Chat Interface")
    
    # Error display area (temporary - for debugging)
    if 'schedule_check_errors' in st.session_state and st.session_state.schedule_check_errors:
        with st.expander("⚠️ Schedule Check Errors (Debug)", expanded=True):
            for error in st.session_state.schedule_check_errors[-5:]:  # Show last 5 errors
                st.error(f"{error.get('time', 'Unknown')}: {error.get('message', 'Unknown error')}")
    
    # Chat history container with scrollable area
    chat_container = st.container(height=500)
    
    with chat_container:
        # Debug info (remove in production)
        if st.session_state.get('show_debug_info', False):
            with st.expander("🔍 Debug Info", expanded=False):
                st.write(f"Chat history length: {len(st.session_state.chat_history)}")
                notification_msgs = [msg for msg in st.session_state.chat_history if msg.get('is_notification')]
                st.write(f"Notification messages: {len(notification_msgs)}")
                if notification_msgs:
                    st.write("Notifications:")
                    for idx, msg in enumerate(notification_msgs):
                        st.write(f"  {idx+1}. {msg.get('content')}")
        
        if not st.session_state.chat_history:
            st.info("👋 Start a conversation! Send a message below.")
            st.caption("💡 Try: 'Turn on the bedroom lights' or 'Hello!'")
        else:
            # Display ALL chat history (both LLM responses and notifications)
            print(f"[CHAT DISPLAY] Rendering {len(st.session_state.chat_history)} messages")
            for idx, message in enumerate(st.session_state.chat_history):
                try:
                    msg_role = message.get('role', 'unknown')
                    is_notif = message.get('is_notification', False)
                    print(f"[CHAT DISPLAY] Message {idx+1}: role={msg_role}, is_notification={is_notif}")
                    
                    if msg_role == 'user':
                        with st.chat_message("user"):
                            st.write(message['content'])
                    elif msg_role == 'assistant':
                        with st.chat_message("assistant"):
                            # Use full content if available (for summarized messages), otherwise use content
                            content = message.get('content_full', message.get('content', ''))
                            # Check if this is a notification (schedule notifications, house check notifications, etc.)
                            if is_notif:
                                # Display notification with info style
                                print(f"[CHAT DISPLAY] Displaying notification: {content}")
                                st.info(content)
                            else:
                                # Display regular LLM response
                                print(f"[CHAT DISPLAY] Displaying LLM response: {content[:50]}...")
                                st.write(content)
                            
                            # Show tool execution info if available (only for non-notification messages)
                            if not is_notif:
                                if 'tool_results' in message:
                                    # Multiple tool results
                                    for tool_result in message['tool_results']:
                                        if tool_result.get('success'):
                                            if tool_result.get('tool') == 'e_device_control':
                                                st.caption(f"✅ {tool_result.get('message', '')}")
                                        else:
                                            st.caption(f"❌ Error: {tool_result.get('error', 'Unknown error')}")
                                elif 'tool_result' in message:
                                    # Single tool result (backward compatibility)
                                    tool_result = message['tool_result']
                                    if tool_result.get('success'):
                                        if tool_result.get('tool') == 'e_device_control':
                                            st.caption(f"✅ {tool_result.get('message', '')}")
                                    else:
                                        st.caption(f"❌ Error: {tool_result.get('error', 'Unknown error')}")
                    else:
                        print(f"[CHAT DISPLAY] ⚠ Unknown message role: {msg_role}")
                except Exception as e:
                    print(f"[CHAT DISPLAY] ✗ ERROR displaying message {idx}: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Chat input
    st.divider()
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to chat history first
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # Process through LLM → MCP router → state update
        with st.spinner("Thinking..."):
            # Calculate custom date/time if custom clock is enabled
            custom_date = None
            custom_time = None
            if st.session_state.custom_clock_time:
                # Calculate current custom time
                custom_hours, custom_minutes = st.session_state.custom_clock_time
                custom_time_timestamp = st.session_state.custom_time_set_timestamp if st.session_state.custom_time_set_timestamp else time.time()
                elapsed_seconds = int(time.time() - custom_time_timestamp)
                total_seconds = custom_hours * 3600 + custom_minutes * 60 + elapsed_seconds
                current_hours = (total_seconds // 3600) % 24
                current_minutes = (total_seconds % 3600) // 60
                custom_time = (current_hours, current_minutes)
                # Use custom date if set
                if st.session_state.custom_clock_date:
                    custom_date = st.session_state.custom_clock_date
                else:
                    # Use real date if custom date not set
                    gmt7 = timezone(timedelta(hours=7))
                    custom_date = datetime.now(gmt7).strftime("%Y-%m-%d")
                print(f"[APP] Using custom date/time for LLM - Date: {custom_date}, Time: {current_hours:02d}:{current_minutes:02d}")
            
            # Step 1: Get current state for context (pass custom_date to use correct date for schedule)
            current_state = st.session_state.mcp_server.get_current_state(
                custom_date=custom_date,
                current_activity=st.session_state.current_activity
            )
            
            # Step 2: Check if RAG should be called (ASYNC - Phase 1.1)
            from llm.client import should_call_rag
            user_condition = current_state.get("user_info", {}).get("condition", "")
            current_activity = current_state.get("current_activity")
            rag_context = None
            
            # Helper function to execute RAG query
            def execute_rag_query():
                """Execute RAG query and return result."""
                try:
                    rag_result = st.session_state.mcp_router.execute({
                        "tool": "rag_query",
                        "arguments": {
                            "query": user_input,
                            "user_condition": user_condition
                        }
                    })
                    return rag_result
                except Exception as e:
                    print(f"[APP ERROR] RAG query failed: {e}")
                    return {"success": False, "found": False, "error": str(e)}
            
            if should_call_rag(user_input, user_condition, st.session_state.chat_history, current_activity):
                # Start RAG query in parallel (non-blocking)
                print(f"[APP] Starting async RAG query...")
                rag_executor = ThreadPoolExecutor(max_workers=1)
                rag_future = rag_executor.submit(execute_rag_query)
                
                # Don't wait for RAG - proceed with LLM preparation
                # We'll check RAG completion before building messages
                print(f"[APP] RAG query started in background, proceeding with LLM preparation")
            else:
                print(f"[APP] RAG not called (not a health query)")
                rag_future = None
            
            # Step 3: Check RAG completion before LLM call (with 2 second timeout)
            if rag_future is not None:
                try:
                    # Wait up to 2 seconds for RAG to complete
                    rag_result = rag_future.result(timeout=2.0)
                    if rag_result.get("success") and rag_result.get("found"):
                        rag_context = {
                            "found": True,
                            "chunks": rag_result.get("chunks", [])
                        }
                        print(f"[APP] RAG completed: found {len(rag_result.get('chunks', []))} relevant chunk(s)")
                    else:
                        rag_context = {"found": False}
                        print(f"[APP] RAG completed: no relevant results found")
                except FutureTimeoutError:
                    # RAG didn't complete in time - proceed without RAG context
                    rag_context = None
                    print(f"[APP] RAG query timed out after 2 seconds, proceeding without RAG context")
                except Exception as e:
                    # RAG query failed - proceed without RAG context
                    rag_context = None
                    print(f"[APP ERROR] RAG query failed: {e}, proceeding without RAG context")
            
            # Step 4: LLM processes message and decides tool/response
            # Pass recent_notification context if user is responding to a notification
            # This allows LLM to control devices in other rooms when responding to notifications
            # Pass custom_date and custom_time so LLM uses correct timestamp for date calculations
            # Pass rag_context if RAG was called and completed
            llm_response = st.session_state.llm_client.process(
                user_input,
                current_state,
                st.session_state.chat_history,
                recent_notification=st.session_state.recent_notification if st.session_state.recent_notification else None,
                custom_date=custom_date,
                custom_time=custom_time,
                rag_context=rag_context,
                conversation_summary=st.session_state.conversation_summary if st.session_state.conversation_summary.get("summary_text") else None
            )
            
            # Step 5: Handle LLM response
            tool_results = []
            assistant_message = ""
            
            if llm_response.get("error"):
                # LLM client encountered an error
                assistant_message = llm_response.get("error", "An error occurred.")
            elif llm_response.get("tools"):
                # Multiple tool calls detected - execute all of them
                tool_calls = llm_response["tools"]
                executed_actions = []
                
                print(f"[APP DEBUG] Executing {len(tool_calls)} tool call(s)")
                for i, tool_call in enumerate(tool_calls):
                    tool_name = tool_call.get("tool")
                    tool_args = tool_call.get("arguments", {})
                    print(f"[APP DEBUG] Tool call {i+1}: tool={tool_name}, arguments={tool_args}")
                    
                    tool_result = st.session_state.mcp_router.execute({
                        "tool": tool_name,
                        "arguments": tool_args
                    }, user_message=user_input)
                    
                    print(f"[APP DEBUG] Tool result {i+1}: success={tool_result.get('success')}, error={tool_result.get('error')}")
                    if tool_result.get("success"):
                        print(f"[APP DEBUG] Tool result {i+1} details: {tool_result}")
                    tool_results.append(tool_result)
                    
                    # Collect successful actions for message
                    if tool_result.get("success"):
                        if tool_result.get("tool") == "e_device_control":
                            room = tool_result.get("room", "")
                            device = tool_result.get("device", "")
                            action = tool_result.get("action", "")
                            executed_actions.append(f"{action.lower()} {room} {device}")
                        elif tool_result.get("tool") == "chat_message":
                            if not assistant_message:
                                message_content = tool_result.get("message", "")
                                assistant_message = message_content  # Keep full message for display
                                # Summarize for chat_history (will be applied later)
                        elif tool_result.get("tool") == "schedule_modifier":
                            # Handle schedule_modifier results with verification
                            modify_type = tool_result.get("modify_type", "")
                            
                            if tool_result.get("success"):
                                # Action succeeded - verify and show explicit confirmation
                                if modify_type == "change":
                                    old_time = tool_result.get("old_time", "")
                                    time = tool_result.get("time", "")
                                    activity = tool_result.get("activity", "")
                                    
                                    # Check today's schedule
                                    current_state = st.session_state.mcp_server.get_current_state()
                                    today_schedule = current_state.get("today_active_schedule", [])
                                    schedule_item = next((item for item in today_schedule if item.get("time") == time and item.get("activity") == activity), None)
                                    old_item_same_activity = next((item for item in today_schedule if item.get("time") == old_time and item.get("activity") == activity), None)
                                    
                                    if schedule_item and not old_item_same_activity:
                                        # Change verified - show confirmation
                                        assistant_message = f"✅ Changed '{activity}' from {old_time} to {time}. The notification will now appear at {time} instead of {old_time}."
                                        print(f"[ACTION VERIFIED] Schedule change confirmed: {activity} is now at {time} (was {old_time})")
                                    elif schedule_item:
                                        # Item exists at new time - change successful (old item might still exist if multiple items at same time, which is allowed)
                                        assistant_message = f"✅ Changed '{activity}' from {old_time} to {time}. The notification will now appear at {time} instead of {old_time}."
                                        print(f"[ACTION VERIFIED] Schedule change confirmed: {activity} is now at {time}")
                                    else:
                                        # Change not found - show warning
                                        assistant_message = f"⚠️ Attempted to change '{activity}' from {old_time} to {time}, but verification failed. Please check your schedule."
                                        print(f"[ACTION WARNING] Schedule change verification failed: {activity} not found at {time}")
                                        
                                elif modify_type == "add":
                                    time = tool_result.get("time", "")
                                    activity = tool_result.get("activity", "")
                                    
                                    # Verify the add actually happened
                                    current_state = st.session_state.mcp_server.get_current_state()
                                    today_schedule = current_state.get("today_active_schedule", [])
                                    schedule_item = next((item for item in today_schedule if item.get("time") == time and item.get("activity") == activity), None)
                                    
                                    if schedule_item:
                                        assistant_message = f"✅ Added '{activity}' at {time} to your schedule. You'll receive a notification at {time}."
                                        print(f"[ACTION VERIFIED] Schedule item added: {activity} at {time}")
                                    else:
                                        assistant_message = f"⚠️ Attempted to add '{activity}' at {time}, but verification failed. Please check your schedule."
                                        print(f"[ACTION WARNING] Schedule add verification failed: {activity} not found at {time}")
                                        
                                elif modify_type == "delete":
                                    time = tool_result.get("time", "")
                                    
                                    # Verify the delete actually happened
                                    current_state = st.session_state.mcp_server.get_current_state()
                                    today_schedule = current_state.get("today_active_schedule", [])
                                    schedule_item = next((item for item in today_schedule if item.get("time") == time), None)
                                    
                                    if not schedule_item:
                                        assistant_message = f"✅ Removed schedule item at {time}. No notification will be sent for this time."
                                        print(f"[ACTION VERIFIED] Schedule item deleted at {time}")
                                    else:
                                        assistant_message = f"⚠️ Attempted to delete schedule item at {time}, but it still exists. Please check your schedule."
                                        print(f"[ACTION WARNING] Schedule delete verification failed: item still exists at {time}")
                                
                                else:
                                    # Fallback for other modify types
                                    modifier_message = tool_result.get("message", "")
                                    assistant_message = modifier_message if modifier_message else "✅ Schedule updated successfully."
                            else:
                                # Action failed
                                error_msg = tool_result.get("error", "Unknown error")
                                assistant_message = f"❌ Failed to modify schedule: {error_msg}"
                                print(f"[ACTION FAILED] Schedule modification failed: {error_msg}")
                
                # Format combined message for multiple actions
                if executed_actions:
                    if len(executed_actions) == 1:
                        assistant_message = f"I've {executed_actions[0]} for you!"
                    else:
                        actions_text = ", ".join(executed_actions[:-1]) + f", and {executed_actions[-1]}"
                        assistant_message = f"I've {actions_text} for you!"
                
                # Check for any failures
                failures = [r for r in tool_results if not r.get("success")]
                if failures:
                    error_msgs = [r.get("error", "Unknown error") for r in failures]
                    if assistant_message:
                        assistant_message += f" However, I encountered some errors: {', '.join(error_msgs)}"
                    else:
                        assistant_message = f"I encountered errors: {', '.join(error_msgs)}"
                        
            elif llm_response.get("tool"):
                # Single tool call detected - execute via MCP router (backward compatibility)
                tool_name = llm_response["tool"]
                tool_args = llm_response.get("arguments", {})
                print(f"[APP DEBUG] Executing single tool call: tool={tool_name}, arguments={tool_args}")
                
                tool_result = st.session_state.mcp_router.execute({
                    "tool": tool_name,
                    "arguments": tool_args
                }, user_message=user_input)
                
                print(f"[APP DEBUG] Tool result: success={tool_result.get('success')}, error={tool_result.get('error')}")
                if tool_result.get("success"):
                    print(f"[APP DEBUG] Tool result details: {tool_result}")
                tool_results = [tool_result]
                
                # Format assistant message based on tool result
                if tool_result.get("success"):
                    if tool_result.get("tool") == "e_device_control":
                        # Device control success
                        room = tool_result.get("room", "")
                        device = tool_result.get("device", "")
                        action = tool_result.get("action", "")
                        assistant_message = f"I've turned {action.lower()} {room} {device} for you!"
                    elif tool_result.get("tool") == "chat_message":
                        # Chat message from tool
                        assistant_message = tool_result.get("message", "")
                else:
                    # Tool execution failed
                    error_msg = tool_result.get("error", "Unknown error")
                    assistant_message = f"I encountered an error: {error_msg}"
            elif llm_response.get("content"):
                # Regular chat response (no tool call)
                content = llm_response["content"]
                
                # SAFETY CHECK: Never display raw JSON tool calls to users
                content_lower = content.lower().strip()
                if (('"tool"' in content_lower or "'tool'" in content_lower) and 
                    ('"arguments"' in content_lower or "'arguments'" in content_lower) and
                    ('{' in content or '[' in content)):
                    # This looks like a JSON tool call - don't show it to the user
                    print(f"[APP ERROR] Attempted to display JSON tool call as content: {content[:200]}")
                    assistant_message = "I encountered an issue processing that request. Could you please try again?"
                else:
                    assistant_message = content
            else:
                # Fallback
                assistant_message = "I'm not sure how to help with that. Please try again."
        
        # Step 4: Check if this is a "keep it on" response (preference update)
        # Do this AFTER LLM processes so LLM can still control if user says "turn it off"
        preference_message = None
        if st.session_state.recent_notification:
            preference_result = st.session_state.mcp_router.process_user_response_for_preferences(
                user_input,
                st.session_state.recent_notification
            )
            if preference_result.get("preference_updated"):
                # User said "keep it on" - clear notification
                st.session_state.recent_notification = None
                preference_message = preference_result.get("message", "")
        
        # Add assistant reply to chat history
        # Summarize long messages to prevent chat_history bloat
        # Store full message for display, but use summarized version in chat_history
        if assistant_message and len(assistant_message) > 500:
            # Message is too long - summarize for chat_history
            summarized_content = _summarize_long_message(assistant_message)
            message_entry = {
                'role': 'assistant',
                'content': summarized_content,  # Summarized for chat_history
                'content_full': assistant_message,  # Full message for display
                'is_summarized': True
            }
        else:
            message_entry = {
                'role': 'assistant',
                'content': assistant_message  # Full message (within limit)
            }
        
        # Include tool result(s) if available
        if tool_results:
            if len(tool_results) == 1:
                message_entry['tool_result'] = tool_results[0]
            else:
                message_entry['tool_results'] = tool_results
        
        st.session_state.chat_history.append(message_entry)
        
        # If preference was updated, add confirmation message
        if preference_message:
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': preference_message,
                'is_preference_update': True
            })
        
        # Increment turn count for summarization tracking
        st.session_state.turn_count += 1
        
        # Phase 1: Chat History Management - Limit and Summarization
        # Limit chat history to max 50 messages (keep last 50)
        MAX_CHAT_HISTORY = 50
        if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
            # Keep last 50 messages, move older ones for summarization
            messages_to_summarize = st.session_state.chat_history[:-50]
            st.session_state.chat_history = st.session_state.chat_history[-50:]
            
            # Trigger summarization in background if we have messages to summarize (Phase 3.1: Background Summarization)
            if messages_to_summarize and st.session_state.turn_count - st.session_state.conversation_summary["last_summarized_turn"] >= 10:
                print(f"[CONTEXT] Triggering background summarization: {len(messages_to_summarize)} messages, turn {st.session_state.turn_count}")
                # Start summarization in background thread (don't block)
                executor = ThreadPoolExecutor(max_workers=1)
                future = executor.submit(
                    st.session_state.llm_client.summarize_conversation,
                    messages_to_summarize,
                    st.session_state.conversation_summary
                )
                st.session_state.summarization_future = future
        
        # Check for summarization trigger (every 10 turns or when history exceeds 2K tokens estimate)
        elif st.session_state.turn_count - st.session_state.conversation_summary["last_summarized_turn"] >= 10:
            # Estimate tokens: ~4 tokens per word, average message ~20 words = ~80 tokens
            estimated_tokens = len(st.session_state.chat_history) * 80
            if estimated_tokens > 2000:
                # Summarize messages beyond last 5
                if len(st.session_state.chat_history) > 5:
                    messages_to_summarize = st.session_state.chat_history[:-5]
                    print(f"[CONTEXT] Triggering background summarization: {len(messages_to_summarize)} messages, ~{estimated_tokens} tokens")
                    # Start summarization in background thread (don't block)
                    executor = ThreadPoolExecutor(max_workers=1)
                    future = executor.submit(
                        st.session_state.llm_client.summarize_conversation,
                        messages_to_summarize,
                        st.session_state.conversation_summary
                    )
                    st.session_state.summarization_future = future
        
        # Check if previous summarization completed (Phase 3.1: Background Summarization)
        if 'summarization_future' in st.session_state:
            future = st.session_state.summarization_future
            if future.done():
                try:
                    summary = future.result()
                    if summary:
                        st.session_state.conversation_summary = summary
                        st.session_state.conversation_summary["last_summarized_turn"] = st.session_state.turn_count
                        print(f"[CONTEXT] Background summarization completed successfully")
                        # If we trimmed history, keep only last 5 messages after summarization
                        estimated_tokens_check = len(st.session_state.chat_history) * 80
                        if estimated_tokens_check > 2000 and len(st.session_state.chat_history) > 5:
                            st.session_state.chat_history = st.session_state.chat_history[-5:]
                except Exception as e:
                    print(f"[CONTEXT ERROR] Background summarization failed: {e}")
                finally:
                    # Clean up future
                    del st.session_state.summarization_future
        
        # Clear recent_notification after processing (if not already cleared)
        # This prevents it from being used in subsequent messages
        if st.session_state.recent_notification and tool_results:
            # If any device was controlled successfully, clear the notification context
            if any(r.get("success") for r in tool_results):
                st.session_state.recent_notification = None
        
        # Rerun to refresh UI with updated state
        st.rerun()

