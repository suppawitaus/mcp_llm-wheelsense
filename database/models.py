"""
SQLAlchemy models for the smart environment database.
"""

from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

Base = declarative_base()


class DeviceState(Base):
    """Store device ON/OFF states"""
    __tablename__ = 'device_states'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    room = Column(String, nullable=False)
    device = Column(String, nullable=False)
    state = Column(Boolean, default=False, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            "room": self.room,
            "device": self.device,
            "state": self.state
        }


class UserInfo(Base):
    """Store user information"""
    __tablename__ = 'user_info'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name_thai = Column(String, default="")
    name_english = Column(String, default="")
    condition = Column(Text, default="")
    current_location = Column(String, default="Bedroom")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            "name": {
                "thai": self.name_thai or "",
                "english": self.name_english or ""
            },
            "condition": self.condition or "",
            "current_location": self.current_location or "Bedroom"
        }


class ScheduleItem(Base):
    """Store base schedule items (recurring)"""
    __tablename__ = 'schedule_items'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    time = Column(String, nullable=False)  # HH:MM format
    activity = Column(String, nullable=False)
    location = Column(String, nullable=True)
    action = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        result = {
            "time": self.time,
            "activity": self.activity
        }
        if self.location:
            result["location"] = self.location
        if self.action:
            try:
                result["action"] = json.loads(self.action)
            except:
                result["action"] = None
        return result


class OneTimeEvent(Base):
    """Store one-time schedule events"""
    __tablename__ = 'one_time_events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String, nullable=False)  # YYYY-MM-DD format
    time = Column(String, nullable=False)  # HH:MM format
    activity = Column(String, nullable=False)
    location = Column(String, nullable=True)
    action = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        result = {
            "date": self.date,
            "time": self.time,
            "activity": self.activity
        }
        if self.location:
            result["location"] = self.location
        if self.action:
            try:
                result["action"] = json.loads(self.action)
            except:
                result["action"] = None
        return result


class DailyScheduleClone(Base):
    """Store daily schedule clones"""
    __tablename__ = 'daily_schedule_clones'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String, nullable=False, unique=True)  # YYYY-MM-DD format
    schedule_data = Column(Text, nullable=False)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def get_schedule_data(self):
        """Parse JSON schedule data"""
        try:
            return json.loads(self.schedule_data)
        except:
            return []
    
    def set_schedule_data(self, data):
        """Set schedule data as JSON string"""
        self.schedule_data = json.dumps(data)


class NotificationPreference(Base):
    """Store notification preferences"""
    __tablename__ = 'notification_preferences'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    room = Column(String, nullable=False)
    device = Column(String, nullable=False)
    do_not_notify = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            "room": self.room,
            "device": self.device,
            "do_not_notify": self.do_not_notify
        }


class DoNotRemind(Base):
    """Store do not remind items"""
    __tablename__ = 'do_not_remind'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    item = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatHistory(Base):
    """Store chat history (optional)"""
    __tablename__ = 'chat_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    content_full = Column(Text, nullable=True)
    is_notification = Column(Boolean, default=False)
    is_preference_update = Column(Boolean, default=False)
    tool_result = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        result = {
            "role": self.role,
            "content": self.content,
            "is_notification": self.is_notification,
            "is_preference_update": self.is_preference_update
        }
        if self.content_full:
            result["content_full"] = self.content_full
        if self.tool_result:
            try:
                result["tool_result"] = json.loads(self.tool_result)
            except:
                pass
        return result


class ConversationSummary(Base):
    """Store conversation summaries"""
    __tablename__ = 'conversation_summaries'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    summary_text = Column(Text, nullable=False)
    key_events = Column(Text, nullable=True)  # JSON string
    last_summarized_turn = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def get_key_events(self):
        """Parse JSON key events"""
        try:
            return json.loads(self.key_events) if self.key_events else []
        except:
            return []
    
    def set_key_events(self, events):
        """Set key events as JSON string"""
        self.key_events = json.dumps(events) if events else None

