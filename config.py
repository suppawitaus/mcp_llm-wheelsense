"""
Configuration constants for the MCP Smart Environment system.
Values can be overridden via environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Model configuration
# Default model: qwen2.5:7b
# Model name can be overridden via MODEL_NAME environment variable
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:7b")
# Default to Docker service name, fallback to localhost for local development
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

# Room and device definitions
ROOMS = {
    "Bedroom": ["Light", "Alarm", "AC"],
    "Bathroom": ["Light"],
    "Kitchen": ["Light", "Alarm"],
    "Living Room": ["Light", "TV", "AC", "Fan"]
}

# Default user location
DEFAULT_USER_LOCATION = "Bedroom"

# Feature flags for optimizations
USE_COMPACT_PROMPT = os.getenv("USE_COMPACT_PROMPT", "false").lower() == "true"

# Database configuration
# Use absolute path based on project root for reliability
PROJECT_ROOT = Path(__file__).parent
DATABASE_PATH = os.getenv("DATABASE_PATH", str(PROJECT_ROOT / "data" / "smart_environment.db"))
DATABASE_BACKUP_DIR = os.getenv("DATABASE_BACKUP_DIR", str(PROJECT_ROOT / "data" / "backups"))
ENABLE_DATABASE_LOGGING = os.getenv("ENABLE_DATABASE_LOGGING", "false").lower() == "true"