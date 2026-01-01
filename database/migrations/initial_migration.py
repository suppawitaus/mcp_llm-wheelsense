"""
Migration script to import existing in-memory data to database.
Run this once when migrating from in-memory to database.
"""

from database.manager import DatabaseManager
from core.state import StateManager
from config import ROOMS


def migrate_from_state_manager(state_manager: StateManager, db_manager: DatabaseManager):
    """
    Migrate data from StateManager to DatabaseManager.
    
    Args:
        state_manager: Existing StateManager instance (with in-memory data)
        db_manager: DatabaseManager instance
    """
    print("[MIGRATION] Starting migration from in-memory to database...")
    
    # 1. Initialize devices
    print("[MIGRATION] Initializing device states...")
    db_manager.initialize_devices(ROOMS)
    
    # 2. Migrate device states
    print("[MIGRATION] Migrating device states...")
    all_devices = state_manager.get_all_devices()
    for room, devices in all_devices.items():
        for device, state in devices.items():
            db_manager.set_device_state(room, device, state)
    
    # 3. Migrate user info
    print("[MIGRATION] Migrating user information...")
    user_info = state_manager.get_user_info(include_one_time_events=False)
    if user_info.get("name", {}).get("thai"):
        db_manager.set_user_name(thai=user_info["name"]["thai"])
    if user_info.get("name", {}).get("english"):
        db_manager.set_user_name(english=user_info["name"]["english"])
    if user_info.get("condition"):
        db_manager.set_user_condition(user_info["condition"])
    
    # 4. Migrate current location
    db_manager.set_current_location(state_manager.current_location)
    
    # 5. Migrate schedule
    print("[MIGRATION] Migrating schedule items...")
    schedule = state_manager.get_user_schedule()
    db_manager.set_schedule_items(schedule)
    
    # 6. Migrate one-time events
    print("[MIGRATION] Migrating one-time events...")
    one_time_events = state_manager.get_schedule_addons()
    for event in one_time_events:
        db_manager.add_one_time_event(event)
    
    # 7. Migrate daily clone
    print("[MIGRATION] Migrating daily schedule clone...")
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    daily_clone = state_manager.get_daily_clone()
    if daily_clone:
        db_manager.set_daily_clone(today, daily_clone)
    
    # 8. Migrate notification preferences
    print("[MIGRATION] Migrating notification preferences...")
    prefs = state_manager.get_notification_preferences()
    for (room, device) in prefs.keys():
        if prefs[(room, device)]:
            db_manager.set_notification_preference(room, device, True)
    
    # 9. Migrate do not remind
    print("[MIGRATION] Migrating do not remind list...")
    do_not_remind = state_manager.get_do_not_remind()
    for item in do_not_remind:
        db_manager.add_to_do_not_remind(item)
    
    print("[MIGRATION] Migration completed successfully!")
    
    # Print stats
    stats = db_manager.get_database_stats()
    print(f"[MIGRATION] Database stats: {stats}")


if __name__ == "__main__":
    # Run migration
    # Note: This script is for migrating from old in-memory StateManager to database
    # For fresh installations, StateManager will automatically initialize with defaults
    from core.state import StateManager
    from database.manager import DatabaseManager
    
    print("[MIGRATION] Note: This script migrates from in-memory to database.")
    print("[MIGRATION] For fresh installations, StateManager auto-initializes with defaults.")
    
    # Create database manager
    db_manager = DatabaseManager()
    
    # Create StateManager (will use database, but may have default data)
    # If you have old in-memory data, you would need to run this before refactoring
    # For now, this will just verify the database structure
    state_manager = StateManager(db_manager=db_manager)
    
    # Check if database already has data
    stats = db_manager.get_database_stats()
    if stats["schedule_items"] > 0 or stats["device_states"] > 0:
        print("[MIGRATION] Database already contains data. Migration may not be needed.")
    else:
        print("[MIGRATION] Database is empty. StateManager will initialize with defaults.")
    
    print("[MIGRATION] Migration script ready. Run this before refactoring if you have existing in-memory data.")

