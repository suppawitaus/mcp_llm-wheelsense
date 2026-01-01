"""
Database manager for handling all database operations.
Provides a clean interface for StateManager to use.
"""

from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

from database.models import (
    Base, DeviceState, UserInfo, ScheduleItem, OneTimeEvent,
    DailyScheduleClone, NotificationPreference, DoNotRemind,
    ChatHistory, ConversationSummary
)
from config import DATABASE_PATH, ENABLE_DATABASE_LOGGING


class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file (defaults to config value)
        """
        if db_path is None:
            db_path = DATABASE_PATH
        
        # Create data directory if it doesn't exist
        db_path_obj = Path(db_path)
        db_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Create engine
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},  # Allow multi-threading
            echo=ENABLE_DATABASE_LOGGING  # SQL debugging
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        print(f"[DATABASE] Initialized database at: {db_path}")
    
    @contextmanager
    def get_session(self):
        """Context manager for database sessions"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    # ========== Device State Operations ==========
    
    def get_device_state(self, room: str, device: str) -> bool:
        """Get device state from database"""
        with self.get_session() as session:
            result = session.query(DeviceState).filter_by(room=room, device=device).first()
            return result.state if result else False
    
    def set_device_state(self, room: str, device: str, state: bool) -> bool:
        """Set device state in database"""
        with self.get_session() as session:
            result = session.query(DeviceState).filter_by(room=room, device=device).first()
            if result:
                result.state = state
                result.updated_at = datetime.utcnow()
            else:
                result = DeviceState(room=room, device=device, state=state)
                session.add(result)
            return True
    
    def get_all_devices(self) -> Dict[str, Dict[str, bool]]:
        """Get all device states"""
        with self.get_session() as session:
            results = session.query(DeviceState).all()
            devices = {}
            for result in results:
                if result.room not in devices:
                    devices[result.room] = {}
                devices[result.room][result.device] = result.state
            return devices
    
    def initialize_devices(self, rooms_config: Dict[str, List[str]]) -> None:
        """Initialize device states for all rooms/devices"""
        with self.get_session() as session:
            for room, device_list in rooms_config.items():
                for device in device_list:
                    # Only add if doesn't exist
                    existing = session.query(DeviceState).filter_by(room=room, device=device).first()
                    if not existing:
                        device_state = DeviceState(room=room, device=device, state=False)
                        session.add(device_state)
    
    # ========== User Info Operations ==========
    
    def get_user_info(self) -> Dict[str, Any]:
        """Get user information"""
        with self.get_session() as session:
            result = session.query(UserInfo).first()
            if not result:
                # Create default user info
                result = UserInfo()
                session.add(result)
                session.commit()
            
            return {
                "name": {
                    "thai": result.name_thai or "",
                    "english": result.name_english or ""
                },
                "condition": result.condition or "",
                "current_location": result.current_location or "Bedroom"
            }
    
    def set_user_name(self, thai: str = "", english: str = "") -> None:
        """Set user name"""
        with self.get_session() as session:
            result = session.query(UserInfo).first()
            if not result:
                result = UserInfo()
                session.add(result)
            
            if thai:
                result.name_thai = thai
            if english:
                result.name_english = english
            result.updated_at = datetime.utcnow()
    
    def set_user_condition(self, condition: str) -> None:
        """Set user condition"""
        with self.get_session() as session:
            result = session.query(UserInfo).first()
            if not result:
                result = UserInfo()
                session.add(result)
            
            result.condition = condition
            result.updated_at = datetime.utcnow()
    
    def get_current_location(self) -> str:
        """Get current user location"""
        with self.get_session() as session:
            result = session.query(UserInfo).first()
            return result.current_location if result else "Bedroom"
    
    def set_current_location(self, location: str) -> bool:
        """Set current user location"""
        with self.get_session() as session:
            result = session.query(UserInfo).first()
            if not result:
                result = UserInfo()
                session.add(result)
            
            result.current_location = location
            result.updated_at = datetime.utcnow()
            return True
    
    # ========== Schedule Operations ==========
    
    def get_schedule_items(self) -> List[Dict[str, Any]]:
        """Get all base schedule items"""
        with self.get_session() as session:
            results = session.query(ScheduleItem).order_by(ScheduleItem.time).all()
            return [item.to_dict() for item in results]
    
    def add_schedule_item(self, item: Dict[str, Any]) -> int:
        """Add schedule item"""
        with self.get_session() as session:
            schedule_item = ScheduleItem(
                time=item.get("time"),
                activity=item.get("activity"),
                location=item.get("location"),
                action=json.dumps(item.get("action")) if item.get("action") else None
            )
            session.add(schedule_item)
            session.flush()
            return schedule_item.id
    
    def update_schedule_item(self, item_id: int, item: Dict[str, Any]) -> bool:
        """Update schedule item"""
        with self.get_session() as session:
            result = session.query(ScheduleItem).filter_by(id=item_id).first()
            if result:
                result.time = item.get("time", result.time)
                result.activity = item.get("activity", result.activity)
                result.location = item.get("location", result.location)
                result.action = json.dumps(item.get("action")) if item.get("action") else None
                result.updated_at = datetime.utcnow()
                return True
            return False
    
    def delete_schedule_item(self, item_id: int) -> bool:
        """Delete schedule item by ID"""
        with self.get_session() as session:
            result = session.query(ScheduleItem).filter_by(id=item_id).first()
            if result:
                session.delete(result)
                return True
            return False
    
    def delete_schedule_item_by_time(self, time: str) -> bool:
        """Delete schedule item by time"""
        with self.get_session() as session:
            result = session.query(ScheduleItem).filter_by(time=time).first()
            if result:
                session.delete(result)
                return True
            return False
    
    def set_schedule_items(self, items: List[Dict[str, Any]]) -> None:
        """Replace all schedule items"""
        with self.get_session() as session:
            # Delete all existing
            session.query(ScheduleItem).delete()
            # Add new ones
            for item in items:
                schedule_item = ScheduleItem(
                    time=item.get("time"),
                    activity=item.get("activity"),
                    location=item.get("location"),
                    action=json.dumps(item.get("action")) if item.get("action") else None
                )
                session.add(schedule_item)
    
    # ========== Daily Clone Operations ==========
    
    def get_daily_clone(self, date: str) -> Optional[List[Dict[str, Any]]]:
        """Get daily schedule clone for a specific date"""
        with self.get_session() as session:
            result = session.query(DailyScheduleClone).filter_by(date=date).first()
            if result:
                return result.get_schedule_data()
            return None
    
    def set_daily_clone(self, date: str, schedule_data: List[Dict[str, Any]]) -> None:
        """Set daily schedule clone for a specific date"""
        with self.get_session() as session:
            result = session.query(DailyScheduleClone).filter_by(date=date).first()
            if result:
                result.set_schedule_data(schedule_data)
                result.updated_at = datetime.utcnow()
            else:
                result = DailyScheduleClone(date=date)
                result.set_schedule_data(schedule_data)
                session.add(result)
    
    def delete_daily_clone(self, date: str) -> bool:
        """Delete daily schedule clone"""
        with self.get_session() as session:
            result = session.query(DailyScheduleClone).filter_by(date=date).first()
            if result:
                session.delete(result)
                return True
            return False
    
    # ========== One-Time Events Operations ==========
    
    def get_one_time_events(self, date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get one-time events, optionally filtered by date"""
        with self.get_session() as session:
            query = session.query(OneTimeEvent)
            if date:
                query = query.filter_by(date=date)
            results = query.order_by(OneTimeEvent.time).all()
            return [item.to_dict() for item in results]
    
    def add_one_time_event(self, event: Dict[str, Any]) -> int:
        """Add one-time event"""
        with self.get_session() as session:
            one_time = OneTimeEvent(
                date=event.get("date"),
                time=event.get("time"),
                activity=event.get("activity"),
                location=event.get("location"),
                action=json.dumps(event.get("action")) if event.get("action") else None
            )
            session.add(one_time)
            session.flush()
            return one_time.id
    
    def delete_one_time_events(self, date: str, time: Optional[str] = None) -> int:
        """Delete one-time events"""
        with self.get_session() as session:
            query = session.query(OneTimeEvent).filter_by(date=date)
            if time:
                query = query.filter_by(time=time)
            count = query.count()
            query.delete()
            return count
    
    def cleanup_old_one_time_events(self, before_date: str) -> int:
        """Delete one-time events before a specific date"""
        with self.get_session() as session:
            count = session.query(OneTimeEvent).filter(OneTimeEvent.date < before_date).count()
            session.query(OneTimeEvent).filter(OneTimeEvent.date < before_date).delete()
            return count
    
    def delete_all_one_time_events(self) -> int:
        """Delete all one-time events (for demonstration purposes)"""
        with self.get_session() as session:
            count = session.query(OneTimeEvent).count()
            session.query(OneTimeEvent).delete()
            return count
    
    # ========== Notification Preferences Operations ==========
    
    def get_notification_preferences(self) -> List[str]:
        """Get notification preferences as list of 'room device' strings"""
        with self.get_session() as session:
            results = session.query(NotificationPreference).filter_by(do_not_notify=True).all()
            return [f"{item.room} {item.device}" for item in results]
    
    def set_notification_preference(self, room: str, device: str, do_not_notify: bool) -> bool:
        """Set notification preference"""
        with self.get_session() as session:
            result = session.query(NotificationPreference).filter_by(room=room, device=device).first()
            if result:
                result.do_not_notify = do_not_notify
            else:
                result = NotificationPreference(room=room, device=device, do_not_notify=do_not_notify)
                session.add(result)
            return True
    
    def clear_notification_preferences(self) -> None:
        """Clear all notification preferences"""
        with self.get_session() as session:
            session.query(NotificationPreference).delete()
    
    # ========== Do Not Remind Operations ==========
    
    def get_do_not_remind(self) -> List[str]:
        """Get do not remind list"""
        with self.get_session() as session:
            results = session.query(DoNotRemind).all()
            return [item.item for item in results]
    
    def add_to_do_not_remind(self, item: str) -> None:
        """Add to do not remind list"""
        with self.get_session() as session:
            # Check if already exists
            existing = session.query(DoNotRemind).filter_by(item=item).first()
            if not existing:
                do_not_remind = DoNotRemind(item=item)
                session.add(do_not_remind)
    
    def remove_from_do_not_remind(self, item: str) -> bool:
        """Remove from do not remind list"""
        with self.get_session() as session:
            result = session.query(DoNotRemind).filter_by(item=item).first()
            if result:
                session.delete(result)
                return True
            return False
    
    def clear_do_not_remind(self) -> None:
        """Clear do not remind list"""
        with self.get_session() as session:
            session.query(DoNotRemind).delete()
    
    # ========== Chat History Operations (Optional) ==========
    
    def save_chat_message(self, message: Dict[str, Any]) -> int:
        """Save chat message to database"""
        with self.get_session() as session:
            tool_result = message.get("tool_result") or message.get("tool_results")
            chat_msg = ChatHistory(
                role=message.get("role"),
                content=message.get("content", ""),
                content_full=message.get("content_full"),
                is_notification=message.get("is_notification", False),
                is_preference_update=message.get("is_preference_update", False),
                tool_result=json.dumps(tool_result) if tool_result else None
            )
            session.add(chat_msg)
            session.flush()
            return chat_msg.id
    
    def get_recent_chat_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent chat history"""
        with self.get_session() as session:
            results = session.query(ChatHistory).order_by(ChatHistory.created_at.desc()).limit(limit).all()
            return [item.to_dict() for item in reversed(results)]  # Chronological order
    
    def clear_chat_history(self) -> int:
        """Clear all chat history"""
        with self.get_session() as session:
            count = session.query(ChatHistory).count()
            session.query(ChatHistory).delete()
            return count
    
    # ========== Conversation Summary Operations ==========
    
    def get_conversation_summary(self) -> Optional[Dict[str, Any]]:
        """Get latest conversation summary"""
        with self.get_session() as session:
            result = session.query(ConversationSummary).order_by(ConversationSummary.updated_at.desc()).first()
            if result:
                return {
                    "last_summarized_turn": result.last_summarized_turn,
                    "summary_text": result.summary_text,
                    "key_events": result.get_key_events()
                }
            return None
    
    def save_conversation_summary(self, summary: Dict[str, Any]) -> None:
        """Save conversation summary"""
        with self.get_session() as session:
            # Get or create summary
            result = session.query(ConversationSummary).order_by(ConversationSummary.updated_at.desc()).first()
            if result:
                result.summary_text = summary.get("summary_text", "")
                result.set_key_events(summary.get("key_events", []))
                result.last_summarized_turn = summary.get("last_summarized_turn", 0)
                result.updated_at = datetime.utcnow()
            else:
                result = ConversationSummary(
                    summary_text=summary.get("summary_text", ""),
                    last_summarized_turn=summary.get("last_summarized_turn", 0)
                )
                result.set_key_events(summary.get("key_events", []))
                session.add(result)
    
    # ========== Utility Methods ==========
    
    def backup_database(self, backup_path: str) -> bool:
        """Create a backup of the database"""
        try:
            import shutil
            from config import DATABASE_PATH
            shutil.copy2(DATABASE_PATH, backup_path)
            print(f"[DATABASE] Backup created at: {backup_path}")
            return True
        except Exception as e:
            print(f"[DATABASE] Backup failed: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self.get_session() as session:
            return {
                "device_states": session.query(DeviceState).count(),
                "schedule_items": session.query(ScheduleItem).count(),
                "one_time_events": session.query(OneTimeEvent).count(),
                "daily_clones": session.query(DailyScheduleClone).count(),
                "notification_preferences": session.query(NotificationPreference).count(),
                "do_not_remind": session.query(DoNotRemind).count(),
                "chat_history": session.query(ChatHistory).count(),
                "conversation_summaries": session.query(ConversationSummary).count()
            }

