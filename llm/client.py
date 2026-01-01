"""
LLM Client for interacting with language models via Ollama.
Handles MCP tool calling with safe parsing.
"""

import json
import re
from datetime import datetime
import ollama
from config import MODEL_NAME, OLLAMA_HOST, ROOMS
from llm.prompts import MCP_SYSTEM_PROMPT, MCP_SYSTEM_PROMPT_COMPACT


def should_call_rag(user_message: str, user_condition: str = None, chat_history: list = None, current_activity: dict = None) -> bool:
    """
    Determine if RAG should be called based on user message.
    
    Decision rules:
    1. Health-related keywords present (symptom, disease, medication, etc.)
    2. User has condition AND query is general health question
    3. Explicit health questions (what is, how to, tell me about)
    4. Follow-up queries (yes, please) to lifestyle questions when user has condition
    5. "What should I do?" queries with CURRENT ACTIVITY that is lifestyle-related
    
    Args:
        user_message: User's input message
        user_condition: Optional user condition context (e.g., "diabetes")
        chat_history: Optional chat history to check for context (for follow-up detection)
        current_activity: Optional dict with current activity info: {"activity": str, "time": str, ...}
    
    Returns:
        True if RAG should be called, False otherwise
    """
    if not user_message or not isinstance(user_message, str):
        return False
    
    message_lower = user_message.lower().strip()
    
    # NEW: Check if user is asking "What should I do?" with a CURRENT ACTIVITY
    if current_activity and current_activity.get("activity"):
        activity_name = current_activity.get("activity", "").lower()
        
        # Check if it's a "What should I do?" query
        what_should_queries = ["what should i do", "what should i", "what do i need to do", "what do i do", "how should i"]
        is_what_should_query = any(phrase in message_lower for phrase in what_should_queries)
        
        # Check if current activity is lifestyle-related
        lifestyle_activities = ["exercise", "workout", "breakfast", "lunch", "dinner", "meal", "sleep", "rest", "activity"]
        is_lifestyle_activity = any(keyword in activity_name for keyword in lifestyle_activities)
        
        # If user has condition AND asking about current lifestyle activity, trigger RAG
        if is_what_should_query and is_lifestyle_activity and user_condition and user_condition.strip():
            return True
    
    # Health-related keywords
    health_keywords = [
        "symptom", "symptoms", "disease", "diseases", "condition", "conditions",
        "medication", "medications", "medicine", "treatment", "treatments",
        "diagnosis", "diagnose", "therapy", "therapeutic", "health", "medical",
        "doctor", "physician", "hospital", "clinic", "patient", "illness",
        "disorder", "syndrome", "infection", "chronic", "acute", "pain",
        "blood pressure", "blood sugar", "glucose", "insulin", "heart",
        "lung", "breathing", "respiratory", "cardiac", "diabetes", "hypertension",
        "arthritis", "copd", "dementia", "depression", "stroke", "parkinson",
        "osteoporosis", "neuropathy", "vision loss", "hearing loss"
    ]
    
    # Question patterns that indicate health queries
    health_question_patterns = [
        "what is", "what are", "how to", "how do", "how should", "tell me about", "explain",
        "what causes", "what are the", "how can i", "what should i",
        "is it safe", "can i", "should i", "what happens"
    ]
    
    # Device/schedule control keywords (exclude these)
    control_keywords = [
        "turn on", "turn off", "switch", "control", "device", "light", "ac", "tv",
        "fan", "alarm", "schedule", "add", "delete", "change", "meeting",
        "appointment", "remind", "notification"
    ]
    
    # Check if message contains device/schedule control keywords (exclude)
    if any(keyword in message_lower for keyword in control_keywords):
        return False
    
    # Check for explicit health keywords
    if any(keyword in message_lower for keyword in health_keywords):
        return True
    
    # Check for health question patterns
    if any(pattern in message_lower for pattern in health_question_patterns):
        # Additional check: if user has condition, more likely to be health-related
        if user_condition and user_condition.strip():
            return True
        # If no condition but question pattern matches, still check if it's health-related
        # by looking for context clues
        health_context_words = ["eat", "food", "diet", "exercise", "manage", "prevent", "care", 
                                "meal", "breakfast", "lunch", "dinner", "snack", "sugar", "honey", 
                                "sweet", "carbohydrate", "protein", "workout", "activity", "activities",
                                "sleep", "rest", "routine", "lifestyle", "wellness", "fitness"]
        if any(word in message_lower for word in health_context_words):
            return True
    
    # If user has condition and query is about lifestyle recommendations, trigger RAG
    if user_condition and user_condition.strip():
        # Check if it's a general question (not device/schedule control)
        if not any(keyword in message_lower for keyword in ["device", "schedule", "turn", "switch", "control"]):
            # Lifestyle recommendation keywords - should trigger RAG when user has condition
            lifestyle_keywords = [
                "eat", "food", "meal", "breakfast", "lunch", "dinner", "snack",
                "exercise", "workout", "activity", "activities", "physical",
                "sleep", "rest", "routine", "lifestyle", "wellness", "fitness",
                "suggest", "recommend", "what should", "what can", "should i"
            ]
            if any(keyword in message_lower for keyword in lifestyle_keywords):
                return True
            # If it contains question words, likely health-related
            question_words = ["what", "how", "why", "when", "where", "which", "should", "can", "could"]
            if any(word in message_lower for word in question_words):
                return True
    
    # Check for follow-up responses to lifestyle questions (e.g., "yes, please" after food suggestion)
    # This helps catch cases where user responds to a lifestyle recommendation
    if user_condition and user_condition.strip() and chat_history:
        follow_up_patterns = ["yes", "please", "sure", "okay", "that sounds good", "tell me more", "go ahead"]
        if any(pattern in message_lower for pattern in follow_up_patterns):
            # Check if last assistant message was about lifestyle (food, exercise, activities)
            if len(chat_history) >= 1:
                last_assistant_msg = None
                for msg in reversed(chat_history[-5:]):  # Check last 5 messages
                    if msg.get('role') == 'assistant':
                        last_assistant_msg = msg.get('content', '').lower()
                        break
                
                if last_assistant_msg:
                    lifestyle_keywords_in_response = [
                        "eat", "food", "meal", "breakfast", "lunch", "dinner", "snack",
                        "exercise", "workout", "activity", "activities", "physical",
                        "sleep", "rest", "routine", "lifestyle", "wellness", "fitness",
                        "suggest", "recommend", "oatmeal", "nutrition", "diet"
                    ]
                    if any(keyword in last_assistant_msg for keyword in lifestyle_keywords_in_response):
                        return True
    
    return False


class LLMClient:
    """
    Client for interacting with DeepSeek-R1 via Ollama.
    Handles MCP tool calling with safe parsing.
    """
    
    def __init__(self, host: str = OLLAMA_HOST):
        """
        Initialize LLM client with Ollama.
        
        Args:
            host: Ollama server host URL
        """
        self.host = host
        self.model = MODEL_NAME
        self._connection_error = None
        
        # Try to create client, but don't fail if connection fails
        try:
            self.client = ollama.Client(host=host)
        except Exception as e:
            # Store error but don't raise - allow graceful degradation
            self._connection_error = str(e)
            self.client = None
        
        # Pre-compile regex patterns for performance (Phase 1.2: Parsing Optimization)
        self._compiled_patterns = {
            # Strategy 1: Markdown code blocks (most common format)
            'markdown_array': re.compile(r'```(?:json)?\s*(\[.*?\])\s*```', re.DOTALL),
            'markdown_object': re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL),
            # Strategy 2: Direct JSON patterns (without markdown)
            'json_array_with_tool': re.compile(r'\[[\s\S]*?\{[\s\S]*?"tool"[\s\S]*?\}[\s\S]*?\]', re.DOTALL),
            'json_object_with_tool': re.compile(r'\{[\s\S]*?"tool"[\s\S]*?\}', re.DOTALL),
            # Strategy 3: Fallback lenient patterns (only if above fail)
            'json_array_lenient': re.compile(r'\[[\s\S]*?\{[\s\S]*?\}[\s\S]*?\]', re.DOTALL),
            'json_object_lenient': re.compile(r'\{[\s\S]*?\}', re.DOTALL),
            # Structured text patterns (last resort)
            'tool_name': re.compile(r'tool["\']?\s*[:=]\s*["\']?(\w+)', re.IGNORECASE),
            'tool_arguments': re.compile(r'arguments["\']?\s*[:=]\s*(\{.*?\})', re.DOTALL),
        }
    
    def validate_connection(self) -> dict:
        """
        Validate Ollama connection and model availability.
        
        Returns:
            dict with structure:
            {
                "valid": bool,
                "ollama_accessible": bool,
                "model_available": bool,
                "error": str or None,
                "message": str
            }
        """
        # If we had an initialization error, return it
        if self._connection_error:
            return {
                "valid": False,
                "ollama_accessible": False,
                "model_available": False,
                "error": self._connection_error,
                "message": f"Failed to initialize Ollama client: {self._connection_error}"
            }
        
        # If client wasn't created, try to create it now
        if self.client is None:
            try:
                self.client = ollama.Client(host=self.host)
            except Exception as e:
                return {
                    "valid": False,
                    "ollama_accessible": False,
                    "model_available": False,
                    "error": str(e),
                    "message": f"Unable to connect to Ollama at {self.host}. Please ensure Ollama is running."
                }
        
        # Test connection by listing models
        try:
            models_response = self.client.list()
            model_names = [model.get('name', '') for model in models_response.get('models', [])]
            
            # Check if required model is available
            model_available = self.model in model_names
            
            if not model_available:
                return {
                    "valid": False,
                    "ollama_accessible": True,
                    "model_available": False,
                    "error": f"Model {self.model} not found",
                    "message": f"Model '{self.model}' is not installed. Please install it with: ollama pull {self.model}"
                }
            
            # Everything is valid
            return {
                "valid": True,
                "ollama_accessible": True,
                "model_available": True,
                "error": None,
                "message": f"Ollama connection validated. Model '{self.model}' is available."
            }
            
        except Exception as e:
            error_msg = str(e)
            if "connection" in error_msg.lower() or "failed to connect" in error_msg.lower():
                return {
                    "valid": False,
                    "ollama_accessible": False,
                    "model_available": False,
                    "error": error_msg,
                    "message": f"Unable to connect to Ollama at {self.host}. Please ensure Ollama is running."
                }
            else:
                return {
                    "valid": False,
                    "ollama_accessible": False,
                    "model_available": False,
                    "error": error_msg,
                    "message": f"Error validating Ollama connection: {error_msg}"
                }
    
    def process(self, user_message: str, current_state: dict, chat_history: list = None, recent_notification: dict = None, custom_date: str = None, custom_time: tuple = None, rag_context: dict = None, conversation_summary: dict = None) -> dict:
        """
        Process user message and get LLM response with potential tool calls.
        
        Args:
            user_message: User's input message
            current_state: Current system state (from MCP server)
            chat_history: Previous chat messages for context
            recent_notification: Optional dict with recent notification info:
                {
                    "room": str,
                    "device": str,
                    "message": str
                }
                If provided, indicates user is responding to a notification
            custom_date: Optional custom date string in YYYY-MM-DD format (for custom clock)
            custom_time: Optional custom time tuple (hours, minutes) (for custom clock)
            rag_context: Optional dict with RAG context:
                {
                    "found": bool,
                    "chunks": list or None  # List of chunk dicts with "text", "score", "metadata"
                }
                If provided, RAG knowledge will be injected into LLM context
            
        Returns:
            dict with format:
            {
                "tools": list or None,    # List of tool calls if detected: [{"tool": str, "arguments": dict}, ...]
                "tool": str or None,      # Single tool name (for backward compatibility)
                "arguments": dict,        # Single tool arguments (for backward compatibility)
                "content": str,            # Text response (if no tool call)
                "error": str or None      # Error message if parsing failed
            }
        """
        if not user_message or not isinstance(user_message, str):
            return {
                "tool": "chat_message",
                "arguments": {"message": "I didn't understand that. Could you please rephrase?"},
                "content": None,
                "error": None
            }
        
        # Check if client is available
        if self.client is None:
            # Try to create client if it wasn't created during init
            try:
                self.client = ollama.Client(host=self.host)
            except Exception as e:
                return {
                    "tool": "chat_message",
                    "arguments": {"message": f"Unable to connect to Ollama at {self.host}. Please ensure Ollama is running."},
                    "content": None,
                    "error": str(e)
                }
        
        # Build messages for LLM
        messages = self._build_messages(user_message, current_state, chat_history, recent_notification, custom_date, custom_time, rag_context, conversation_summary)
        
        try:
            # Call Ollama with streaming for better perceived latency
            # Set num_ctx to 16384 (16K) to optimize KV cache and attention computation
            # This is 2-3x the expected usage (~3-7K tokens per request)
            # NOTE: Streaming improves perceived latency - user sees response as it's generated
            response_text = ""
            stream = self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,  # Enable streaming
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_ctx": 16384  # Limit context window for better performance
                }
            )
            
            # Collect streaming response (Phase 1.3: True Streaming)
            # Note: We still collect full response for tool call parsing, but chunks are available for streaming display
            for chunk in stream:
                if chunk.get('message', {}).get('content'):
                    response_text += chunk['message']['content']
            
            # Debug: Log raw response
            print(f"[LLM DEBUG] Raw response length: {len(response_text) if response_text else 0}")
            if response_text:
                print(f"[LLM DEBUG] Raw response (first 500 chars): {response_text[:500]}")
            
            if not response_text:
                return {
                    "tool": None,
                    "arguments": {},
                    "content": "I'm sorry, I didn't receive a response. Please try again.",
                    "error": "Empty response from LLM"
                }
            
            # Preprocess: Remove reasoning markers and extract JSON if present
            # DeepSeek-R1 often includes reasoning before the actual response
            original_response = response_text
            response_text = self._preprocess_response(response_text)
            
            # Debug: Log preprocessed response
            print(f"[LLM DEBUG] Preprocessed response length: {len(response_text) if response_text else 0}")
            if response_text != original_response:
                print(f"[LLM DEBUG] Preprocessed response (first 500 chars): {response_text[:500]}")
            
            if not response_text:
                print(f"[LLM DEBUG] WARNING: Preprocessing removed all content! Original: {original_response[:200]}")
                # If preprocessing removed everything, try using original
                response_text = original_response
            
            # Try to parse tool call(s) from response
            tool_calls = self._parse_tool_calls(response_text)
            
            # Debug: Log parsed tool calls
            print(f"[LLM DEBUG] Parsed {len(tool_calls)} tool call(s)")
            for i, tc in enumerate(tool_calls):
                print(f"[LLM DEBUG] Tool call {i+1}: tool={tc.get('tool')}, arguments={tc.get('arguments')}")
            
            if tool_calls:
                # Tool call(s) detected
                # Support both single and multiple tool calls
                if len(tool_calls) == 1:
                    # Single tool call - maintain backward compatibility
                    tool_call = tool_calls[0]
                    result = {
                        "tools": tool_calls,
                        "tool": tool_call.get("tool"),
                        "arguments": tool_call.get("arguments", {}),
                        "content": None,
                        "error": None
                    }
                    print(f"[LLM DEBUG] Returning single tool call: tool={result.get('tool')}, arguments={result.get('arguments')}")
                    return result
                else:
                    # Multiple tool calls
                    result = {
                        "tools": tool_calls,
                        "tool": None,  # Multiple tools, use "tools" array instead
                        "arguments": {},
                        "content": None,
                        "error": None
                    }
                    print(f"[LLM DEBUG] Returning {len(tool_calls)} tool calls")
                    return result
            else:
                # No tool call detected - but check if response looks like JSON tool call
                # CRITICAL: Never show raw JSON tool calls to users
                looks_like_json = self._looks_like_json_tool_call(response_text)
                
                if looks_like_json:
                    print(f"[LLM ERROR] Detected JSON tool call pattern but parsing failed.")
                    print(f"[LLM ERROR] Full response (first 1000 chars): {response_text[:1000]}")
                    print(f"[LLM ERROR] Response length: {len(response_text)}")
                    # Try to show JSON parsing error details
                    json_error = None
                    try:
                        import json
                        json.loads(response_text)  # This will show the actual JSON error
                    except json.JSONDecodeError as e:
                        json_error = str(e)
                        print(f"[LLM ERROR] JSON decode error: {e}")
                        print(f"[LLM ERROR] Error at position: {e.pos if hasattr(e, 'pos') else 'unknown'}")
                        if hasattr(e, 'pos') and e.pos < len(response_text):
                            context_start = max(0, e.pos - 50)
                            context_end = min(len(response_text), e.pos + 50)
                            print(f"[LLM ERROR] Context around error: ...{response_text[context_start:context_end]}...")
                    except Exception as e:
                        json_error = str(e)
                        print(f"[LLM ERROR] Other parsing error: {e}")
                    
                    # This looks like a tool call but parsing failed - return error instead of showing raw JSON
                    return {
                        "tool": "chat_message",
                        "arguments": {"message": "I encountered an issue processing that request. Could you please try again?"},
                        "content": None,
                        "error": "Failed to parse tool call from LLM response"
                    }
                
                # No tool call - regular chat response
                return {
                    "tools": None,
                    "tool": None,
                    "arguments": {},
                    "content": response_text.strip(),
                    "error": None
                }
        
        except Exception as e:
            # Handle errors gracefully
            error_msg = str(e)
            if "connection" in error_msg.lower() or "failed to connect" in error_msg.lower():
                return {
                    "tool": "chat_message",
                    "arguments": {"message": f"Unable to connect to Ollama at {OLLAMA_HOST}. Please ensure Ollama is running."},
                    "content": None,
                    "error": error_msg
                }
            elif "not found" in error_msg.lower() or "404" in error_msg.lower():
                return {
                    "tool": "chat_message",
                    "arguments": {"message": f"Model {MODEL_NAME} not found. Please install it with: ollama pull {MODEL_NAME}"},
                    "content": None,
                    "error": error_msg
                }
            else:
                return {
                    "tool": "chat_message",
                    "arguments": {"message": "I encountered an error processing your request. Please try again."},
                    "content": None,
                    "error": error_msg
                }
    
    def _build_messages(self, user_message: str, current_state: dict, chat_history: list = None, recent_notification: dict = None, custom_date: str = None, custom_time: tuple = None, rag_context: dict = None, conversation_summary: dict = None) -> list:
        """
        Build message list for LLM with system prompt and current state.
        
        Args:
            user_message: Current user message
            current_state: Current system state
            chat_history: Previous chat messages
            recent_notification: Optional dict with recent notification info
            custom_date: Optional custom date string in YYYY-MM-DD format (for custom clock)
            custom_time: Optional custom time tuple (hours, minutes) (for custom clock)
            rag_context: Optional dict with RAG context:
                {
                    "found": bool,
                    "chunks": list or None  # List of chunk dicts with "text", "score", "metadata"
                }
            
        Returns:
            List of message dicts for Ollama
        """
        messages = []
        
        # Use system prompt (Phase 2.1: Prompt Reduction - use compact version if enabled)
        from config import USE_COMPACT_PROMPT
        system_prompt = MCP_SYSTEM_PROMPT_COMPACT if USE_COMPACT_PROMPT else MCP_SYSTEM_PROMPT
        
        # Add RAG context if available
        if rag_context:
            rag_section = self._format_rag_context(rag_context)
            if rag_section:
                system_prompt = system_prompt + "\n\n" + rag_section
        
        # Add notification context to prompt if user is responding to a notification
        if recent_notification:
            notification_message = recent_notification.get('message', '')
            devices_list = recent_notification.get('devices', [])
            
            # Build device list description for prompt
            if devices_list:
                if len(devices_list) == 1:
                    # Single device (backward compatibility)
                    device_info = f"{devices_list[0]['room']} {devices_list[0]['device']}"
                    devices_description = f"Device: {device_info}"
                    devices_list_str = f"[{{'room': '{devices_list[0]['room']}', 'device': '{devices_list[0]['device']}'}}]"
                else:
                    # Multiple devices
                    device_names = [f"{d['room']} {d['device']}" for d in devices_list]
                    devices_description = f"Devices: {', '.join(device_names)}"
                    devices_list_str = "[" + ", ".join([f"{{'room': '{d['room']}', 'device': '{d['device']}'}}" for d in devices_list]) + "]"
                notification_context = f"""

IMPORTANT CONTEXT - USER RESPONDING TO NOTIFICATION (HIGHEST PRIORITY):
The user just received a notification about: {devices_description}
Notification message was: "{notification_message}"

CRITICAL PRIORITY RULES:
- This notification context OVERRIDES any older chat history patterns
- User can control these devices even though they're in different rooms (this is one of the 2 allowed ways to control other rooms)
- If user says "yes", "yeah", "sure", "okay", "turn them off" → IMMEDIATELY call e_device_control for ONLY the devices mentioned in this notification
- You MUST call e_device_control for EACH device in the list: {devices_list_str}
- DO NOT repeat actions from older chat history - ONLY respond to this notification
- If user says "no", "keep it on", "leave it on" → Use chat_message to acknowledge (preference will be set)
- DO NOT ask for clarification - if user says "yes", take action immediately for ALL devices in this notification
- Example: If notification mentions Bedroom Light, make 1 e_device_control call: [{{"tool": "e_device_control", "arguments": {{"room": "Bedroom", "device": "Light", "action": "OFF"}}}}]
"""
            else:
                # Fallback for old format (backward compatibility)
                notification_room = recent_notification.get('room', 'Unknown')
                notification_device = recent_notification.get('device', 'Unknown')
                notification_context = f"""

IMPORTANT CONTEXT - USER RESPONDING TO NOTIFICATION (HIGHEST PRIORITY):
The user just received a notification about: {notification_room} {notification_device}
Notification message was: "{notification_message}"

CRITICAL PRIORITY RULES:
- This notification context OVERRIDES any older chat history patterns
- User can control this device even though it's in a different room (this is one of the 2 allowed ways to control other rooms)
- If user says "yes", "yeah", "sure", "okay", "turn it off" → IMMEDIATELY call e_device_control(room="{notification_room}", device="{notification_device}", action="OFF")
- DO NOT repeat actions from older chat history - ONLY respond to this notification
- If user says "no", "keep it on", "leave it on" → Use chat_message to acknowledge (preference will be set)
- DO NOT ask for clarification - if user says "yes", take action immediately
"""
            system_prompt = system_prompt + notification_context
        
        # Phase 4: Optimize Detection Methods - combine scans into single pass
        # CRITICAL: If recent_notification is present, skip chat history detection (notification context takes priority)
        if chat_history and not recent_notification:
            # Single pass through recent messages (last 6) to detect both patterns
            recent_messages = chat_history[-6:] if len(chat_history) >= 6 else chat_history
            
            # Check for information completion first (higher priority)
            info_completion_context = self._detect_information_completion_optimized(recent_messages, chat_history)
            if info_completion_context:
                completion_context = f"""

{info_completion_context}

CRITICAL: Execute the original action immediately with the provided information. Do NOT ask for clarification again.
"""
                system_prompt = system_prompt + completion_context
            else:
                # Check for recent question patterns (yes/no responses)
                recent_question_context = self._detect_recent_question_optimized(recent_messages)
                if recent_question_context:
                    question_context = f"""

IMPORTANT - RECENT CONVERSATION CONTEXT:
{recent_question_context}

Remember: If you already asked about controlling a device and the user responded, DO NOT ask the same question again.
If user said "yeah", "yes", "sure", "okay" → TAKE ACTION immediately, don't ask again.
"""
                    system_prompt = system_prompt + question_context
        
        # Phase 2: Conditional State Inclusion - only include relevant state sections
        state_info = self._format_state_info_conditional(current_state, user_message, custom_date, custom_time)
        
        # Add current timestamp and date for date calculations
        # Use custom date/time if provided, otherwise use real time
        from datetime import timedelta, timezone
        if custom_date and custom_time:
            # Use custom date and time
            custom_hours, custom_minutes = custom_time
            current_date = custom_date
            current_time = f"{current_date} {custom_hours:02d}:{custom_minutes:02d}:00"
            # Calculate tomorrow's date based on custom date
            custom_datetime = datetime.strptime(custom_date, "%Y-%m-%d")
            tomorrow_date = (custom_datetime + timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"[LLM TIMESTAMP] Using custom date/time - Date: {current_date}, Time: {custom_hours:02d}:{custom_minutes:02d}, Tomorrow: {tomorrow_date}")
        else:
            # Use real system time
            current_datetime = datetime.now()
            current_time = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            current_date = current_datetime.strftime("%Y-%m-%d")
            tomorrow_date = (current_datetime + timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"[LLM TIMESTAMP] Using real date/time - Date: {current_date}, Tomorrow: {tomorrow_date}")
        timestamp_info = f"CURRENT TIME: {current_time}\nCURRENT DATE: {current_date}\nTOMORROW'S DATE: {tomorrow_date}"
        
        # Add conversation summary if available (Phase 1: Long-term memory)
        summary_section = ""
        if conversation_summary and conversation_summary.get("summary_text"):
            summary_section = f"\n\nPREVIOUS CONVERSATION SUMMARY:\n{conversation_summary['summary_text']}"
            if conversation_summary.get("key_events"):
                events_text = "\n".join([f"- {e.get('type', 'event')}: {e.get('summary', '')[:80]}" for e in conversation_summary['key_events'][-5:]])
                summary_section += f"\n\nRecent Key Events:\n{events_text}"
        
        system_content = f"{system_prompt}{summary_section}\n\n{timestamp_info}\n\nCURRENT SYSTEM STATE:\n{state_info}"
        
        messages.append({
            "role": "system",
            "content": system_content
        })
        
        # Include chat history for context (reduced from 10 to 5 for performance)
        if chat_history:
            for msg in chat_history[-5:]:  # Last 5 messages for context
                if msg.get('role') in ['user', 'assistant']:
                    messages.append({
                        "role": msg['role'],
                        "content": msg.get('content', '')
                    })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        return messages
    
    def _format_rag_context(self, rag_context: dict) -> str:
        """
        Format RAG context for injection into system prompt.
        
        Args:
            rag_context: Dict with "found" and "chunks" keys
            
        Returns:
            Formatted string with RAG knowledge context, or empty string if no context
        """
        if not rag_context:
            return ""
        
        found = rag_context.get("found", False)
        chunks = rag_context.get("chunks", [])
        
        if not found or not chunks:
            return """HEALTH KNOWLEDGE CONTEXT (from RAG system):
No specific health knowledge was found for this query. Rely on your general knowledge,
but be cautious about providing health advice. Always recommend consulting healthcare professionals for medical concerns."""
        
        # Format chunks for LLM context (limit to top 3 chunks, truncate long chunks to 500 chars)
        chunk_texts = []
        max_chunks = 3  # Limit chunk count for performance
        max_chunk_length = 500  # Truncate long chunks
        
        for i, chunk in enumerate(chunks[:max_chunks], 1):
            chunk_text = chunk.get("text", "")
            score = chunk.get("score", 0.0)
            
            # Truncate if too long
            if len(chunk_text) > max_chunk_length:
                chunk_text = chunk_text[:max_chunk_length] + "..."
            
            chunk_texts.append(f"--- Knowledge Chunk {i} (Relevance Score: {score:.3f}) ---\n{chunk_text}")
        
        chunks_section = "\n\n".join(chunk_texts)
        
        return f"""HEALTH KNOWLEDGE CONTEXT (from RAG system):
The following health knowledge was retrieved for your reference.
CRITICAL: If this knowledge mentions wheelchair users or seated exercises, the user uses a wheelchair - use this knowledge directly.

{chunks_section}

IMPORTANT: 
- Use this knowledge to provide accurate, safe health information based on the user's specific conditions (check USER INFORMATION section).
- If the knowledge mentions "wheelchair", "seated exercises", or "wheelchair users", incorporate these specific recommendations into your answer.
- Do NOT make medical diagnoses or provide treatment advice beyond general information.
- Always recommend consulting a healthcare professional for medical concerns.
- Prioritize user safety and match recommendations to ALL aspects of the user's condition."""
    
    def _format_state_info(self, current_state: dict) -> str:
        """
        Format current state information for LLM context.
        
        Args:
            current_state: State dict from MCP server
            
        Returns:
            Formatted string with state information
        """
        if not current_state:
            return "State information unavailable."
        
        location = current_state.get("current_location", "Unknown")
        devices = current_state.get("devices", {})
        notification_prefs = current_state.get("notification_preferences", [])
        user_info = current_state.get("user_info", {})
        today_active_schedule = current_state.get("today_active_schedule", [])
        
        lines = []
        
        # Add user information
        if user_info:
            lines.append("USER INFORMATION:")
            name = user_info.get("name", {})
            if name.get("thai") or name.get("english"):
                name_parts = []
                if name.get("thai"):
                    name_parts.append(f"Thai: {name['thai']}")
                if name.get("english"):
                    name_parts.append(f"English: {name['english']}")
                lines.append(f"  Name: {', '.join(name_parts)}")
            
            # Show today's active schedule (this is what notifications use)
            if today_active_schedule:
                lines.append("  TODAY'S ACTIVE SCHEDULE (used for notifications - modify with schedule_modifier):")
                for item in today_active_schedule:
                    time = item.get("time", "")
                    activity = item.get("activity", "")
                    lines.append(f"    - {time}: {activity}")
            
            # Note: Original schedule removed to reduce token count - active schedule is sufficient
            
            condition = user_info.get("condition", "")
            if condition:
                lines.append(f"  Condition: {condition}")
            
            # One-time events (temporary schedule items added by LLM)
            one_time_events = user_info.get("one_time_events", [])
            if one_time_events:
                lines.append("  Today's Additional Schedule (one-time events):")
                from datetime import datetime
                today = datetime.now().strftime("%Y-%m-%d")
                for item in one_time_events:
                    if item.get("date") == today:
                        time = item.get("time", "")
                        activity = item.get("activity", "")
                        lines.append(f"    - {time}: {activity}")
            
            lines.append("")
        
        lines.append(f"Current Location: {location}")
        lines.append("")
        lines.append("Device States:")
        
        for room, room_devices in devices.items():
            lines.append(f"  {room}:")
            for device, state in room_devices.items():
                status = "ON" if state else "OFF"
                lines.append(f"    - {device}: {status}")
        
        return "\n".join(lines)
    
    def _format_state_info_conditional(self, current_state: dict, user_message: str, custom_date: str = None, custom_time: tuple = None) -> str:
        """
        Format state information conditionally based on query intent (with caching - Phase 2.2).
        Only includes relevant sections to reduce token count.
        
        Args:
            current_state: State dict from MCP server
            user_message: User's current message for intent detection
            custom_date: Optional custom date string
            custom_time: Optional custom time tuple
            
        Returns:
            Formatted string with only relevant state information
        """
        if not current_state:
            return "State information unavailable."
        
        # Check cache (Phase 2.2: Smart State Caching)
        import time
        import hashlib
        
        # Initialize cache if it doesn't exist (for instances created before this optimization)
        if not hasattr(self, '_state_cache'):
            self._state_cache = {}
        
        # Create cache key: hash of state dict + user_message + custom_date/time
        cache_key_data = {
            'state': current_state,
            'user_message': user_message,
            'custom_date': custom_date,
            'custom_time': custom_time
        }
        cache_key_str = str(sorted(cache_key_data.items()))
        cache_key = hashlib.md5(cache_key_str.encode()).hexdigest()
        
        # Check if cached and still valid (500ms TTL)
        if cache_key in self._state_cache:
            cached_entry = self._state_cache[cache_key]
            cache_age = time.time() - cached_entry['timestamp']
            if cache_age < 0.5:  # 500ms TTL
                return cached_entry['formatted']
            else:
                # Cache expired, remove it
                del self._state_cache[cache_key]
        
        message_lower = user_message.lower()
        location = current_state.get("current_location", "Unknown")
        devices = current_state.get("devices", {})
        user_info = current_state.get("user_info", {})
        today_active_schedule = current_state.get("today_active_schedule", [])
        
        lines = []
        
        # Always include current location (needed for device control defaults)
        lines.append(f"Current Location: {location}")
        lines.append("")
        
        # Check for "What should I do" queries - always emphasize CURRENT ACTIVITY
        is_what_should_i_do = any(phrase in message_lower for phrase in ["what should i do", "what should i", "what do i need to do", "what do i do"])
        
        # Calculate current time for activity period check
        current_time_str = None
        if custom_time:
            # Use custom time if provided
            current_hours, current_minutes = custom_time
            current_time_str = f"{current_hours:02d}:{current_minutes:02d}"
        else:
            # Use real time
            from datetime import datetime, timedelta, timezone
            gmt7 = timezone(timedelta(hours=7))
            current_time = datetime.now(gmt7)
            current_time_str = current_time.strftime("%H:%M")
        
        # Add current activity context if available and within activity period
        current_activity = current_state.get("current_activity")
        if current_activity and current_activity.get("activity"):
            activity_name = current_activity.get("activity")
            activity_time = current_activity.get("time", "")
            activity_location = current_activity.get("location")
            end_time = current_activity.get("end_time")
            
            # Check if current time is within activity period
            is_within_period = True
            if activity_time and current_time_str:
                try:
                    # Parse times
                    activity_parts = activity_time.split(":")
                    activity_hours = int(activity_parts[0])
                    activity_minutes = int(activity_parts[1])
                    activity_total = activity_hours * 60 + activity_minutes
                    
                    current_parts = current_time_str.split(":")
                    current_hours = int(current_parts[0])
                    current_minutes = int(current_parts[1])
                    current_total = current_hours * 60 + current_minutes
                    
                    # Check if current time is after activity start time
                    if current_total < activity_total:
                        is_within_period = False
                    
                    # Check if current time is before end time (if end_time exists)
                    if end_time:
                        end_parts = end_time.split(":")
                        end_hours = int(end_parts[0])
                        end_minutes = int(end_parts[1])
                        end_total = end_hours * 60 + end_minutes
                        
                        if current_total >= end_total:
                            is_within_period = False
                except (ValueError, IndexError):
                    # If parsing fails, assume within period (fallback)
                    pass
            
            # Only show as CURRENT ACTIVITY if within the activity period
            if is_within_period:
                activity_line = f"CURRENT ACTIVITY: {activity_name}"
                if activity_time:
                    activity_line += f" (scheduled at {activity_time})"
                if end_time:
                    activity_line += f" (until {end_time})"
                if activity_location:
                    activity_line += f" in {activity_location}"
                
                # For "What should I do" queries, add explicit note
                if is_what_should_i_do:
                    activity_line += " [NOTE: You are CURRENTLY doing this activity - a notification was just sent]"
                
                lines.append(activity_line)
                lines.append("")
        
        # Device queries: include device states
        device_keywords = ['device', 'light', 'ac', 'tv', 'fan', 'alarm', 'turn', 'switch', 'on', 'off', 'everything', 'all devices']
        is_device_query = any(word in message_lower for word in device_keywords)
        
        if is_device_query:
            lines.append("Device States:")
            # Include all rooms for "all rooms" queries, otherwise just current room
            if 'all rooms' in message_lower or 'everywhere' in message_lower or 'all' in message_lower:
                # Include all rooms
                for room, room_devices in devices.items():
                    lines.append(f"  {room}:")
                    for device, state in room_devices.items():
                        status = "ON" if state else "OFF"
                        lines.append(f"    - {device}: {status}")
            else:
                # Include only current room (most common case)
                if location in devices:
                    lines.append(f"  {location}:")
                    for device, state in devices[location].items():
                        status = "ON" if state else "OFF"
                        lines.append(f"    - {device}: {status}")
            lines.append("")
        
        # Schedule queries: include schedule
        schedule_keywords = ['schedule', 'appointment', 'meeting', 'time', 'activity', 'wake', 'sleep', 'breakfast', 'lunch', 'dinner', 'work', 'study', 'next', 'after', 'later', 'upcoming', 'what']
        is_schedule_query = any(word in message_lower for word in schedule_keywords)
        
        # Check for deletion keywords that need full schedule visibility
        deletion_keywords = ['awake', 'work now', 'delete', 'remove', 'cancel', 'skip', 'not doing', "won't", "will not"]
        is_deletion_query = any(keyword in message_lower for keyword in deletion_keywords)
        
        # Schedule queries OR deletion queries: include schedule with full details
        if is_schedule_query or is_deletion_query:
            if today_active_schedule:
                # Sort schedule by time to make it easier to find "next"
                sorted_schedule = sorted(today_active_schedule, key=lambda x: x.get("time", ""))
                
                # Get current time for comparison
                from datetime import datetime
                if custom_date and custom_time:
                    custom_hours, custom_minutes = custom_time
                    current_time_str = f"{custom_hours:02d}:{custom_minutes:02d}"
                else:
                    current_time_str = datetime.now().strftime("%H:%M")
                
                lines.append("TODAY'S ACTIVE SCHEDULE (sorted by time):")
                for item in sorted_schedule:
                    time = item.get("time", "")
                    activity = item.get("activity", "")
                    
                    # Build schedule line with optional fields
                    schedule_line = f"  - {time}: {activity}"
                    
                    # Add location if present
                    if "location" in item and item["location"]:
                        schedule_line += f" [Location: {item['location']}]"
                    
                    # Add action summary if present
                    if "action" in item and item.get("action", {}).get("devices"):
                        devices = item["action"]["devices"]
                        device_list = ", ".join([f"{d['room']} {d['device']} {d['state']}" for d in devices])
                        schedule_line += f" [Action: {device_list}]"
                    
                    # Mark upcoming activities (after current time)
                    if time > current_time_str:
                        schedule_line += " [UPCOMING]"
                    
                    lines.append(schedule_line)
                lines.append("")
                
                # Explicitly show next activity if query is about "next"
                if 'next' in message_lower or 'after' in message_lower or 'upcoming' in message_lower:
                    upcoming_items = [item for item in sorted_schedule if item.get("time", "") > current_time_str]
                    if upcoming_items:
                        next_item = upcoming_items[0]
                        lines.append(f"NEXT ACTIVITY: {next_item.get('time')} - {next_item.get('activity')}")
                        lines.append("")
            
            # Include one-time events if any
            one_time_events = user_info.get("one_time_events", [])
            if one_time_events:
                from datetime import datetime
                today = datetime.now().strftime("%Y-%m-%d")
                today_events = [item for item in one_time_events if item.get("date") == today]
                if today_events:
                    lines.append("Today's Additional Schedule (one-time events):")
                    for item in today_events:
                        time = item.get("time", "")
                        activity = item.get("activity", "")
                        
                        # Build schedule line with optional fields
                        schedule_line = f"  - {time}: {activity}"
                        
                        # Add location if present
                        if "location" in item and item["location"]:
                            schedule_line += f" [Location: {item['location']}]"
                        
                        # Add action summary if present
                        if "action" in item and item.get("action", {}).get("devices"):
                            devices = item["action"]["devices"]
                            device_list = ", ".join([f"{d['room']} {d['device']} {d['state']}" for d in devices])
                            schedule_line += f" [Action: {device_list}]"
                        
                        lines.append(schedule_line)
                    lines.append("")
        
        # User info queries: include user info (Phase 3: Static caching - only when needed)
        user_info_keywords = ['who', 'name', 'condition', 'about me', 'myself', 'tell me about']
        is_user_info_query = any(word in message_lower for word in user_info_keywords)
        
        # CRITICAL: Also include USER INFORMATION for lifestyle queries (food, exercise, activities, sleep, routines)
        lifestyle_keywords = ['eat', 'food', 'meal', 'breakfast', 'lunch', 'dinner', 'snack', 'exercise', 'workout', 
                             'activity', 'activities', 'sleep', 'rest', 'routine', 'lifestyle', 'wellness', 'fitness',
                             'suggest', 'recommend', 'what should', 'what can', 'should i', 'what to', "don't know what", "don't know"]
        is_lifestyle_query = any(keyword in message_lower for keyword in lifestyle_keywords)
        
        if (is_user_info_query or is_lifestyle_query) and user_info:
            lines.append("USER INFORMATION:")
            name = user_info.get("name", {})
            if name.get("thai") or name.get("english"):
                name_parts = []
                if name.get("thai"):
                    name_parts.append(f"Thai: {name['thai']}")
                if name.get("english"):
                    name_parts.append(f"English: {name['english']}")
                lines.append(f"  Name: {', '.join(name_parts)}")
            
            condition = user_info.get("condition", "")
            if condition:
                lines.append(f"  Condition: {condition}")
                # CRITICAL: Add explicit note for lifestyle queries
                if is_lifestyle_query:
                    lines.append(f"  [CRITICAL: All recommendations MUST be tailored to this condition]")
            lines.append("")
        
        # If no specific sections added, include minimal state (location + current room devices)
        if len(lines) <= 2:  # Only location and empty line
            lines.append("Device States:")
            if location in devices:
                lines.append(f"  {location}:")
                for device, state in devices[location].items():
                    status = "ON" if state else "OFF"
                    lines.append(f"    - {device}: {status}")
        
        formatted_result = "\n".join(lines)
        
        # Cache the result (Phase 2.2: Smart State Caching)
        import time
        import hashlib
        self._state_cache[cache_key] = {
            'formatted': formatted_result,
            'timestamp': time.time()
        }
        
        # Clean up old cache entries (keep only last 10)
        if len(self._state_cache) > 10:
            # Remove oldest entries
            sorted_entries = sorted(self._state_cache.items(), key=lambda x: x[1]['timestamp'])
            for old_key, _ in sorted_entries[:-10]:
                del self._state_cache[old_key]
        
        return formatted_result
    
    def _preprocess_response(self, response_text: str) -> str:
        """
        Preprocess LLM response to extract tool calls from reasoning text.
        Removes reasoning markers and tries to find JSON tool calls.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Cleaned response text with JSON tool calls
        """
        if not response_text:
            return ""
        
        # Remove common reasoning markers
        # DeepSeek-R1 uses </reasoning> or </think> tags
        # Check for </think> first (more specific)
        if "</think>" in response_text:
            # Extract everything after </think>
            parts = response_text.split("</think>")
            if len(parts) > 1:
                response_text = parts[-1].strip()
                print(f"[PREPROCESS] Found </think>, extracted: {response_text[:200]}")
        
        if "</reasoning>" in response_text:
            # Extract everything after </reasoning>
            parts = response_text.split("</reasoning>")
            if len(parts) > 1:
                response_text = parts[-1].strip()
                print(f"[PREPROCESS] Found </reasoning>, extracted: {response_text[:200]}")
        
        # Remove emoji reasoning markers and text before them
        # But be careful - only remove if we can find JSON after
        reasoning_markers = ["🛌", "💭", "🤔", "🔍", "📝"]
        for marker in reasoning_markers:
            if marker in response_text:
                # Try to find JSON after the marker
                marker_pos = response_text.find(marker)
                # Look for JSON array or object after the marker
                remaining = response_text[marker_pos + len(marker):].strip()
                # Only use remaining if we find JSON - otherwise keep original
                if "[" in remaining or "{" in remaining:
                    response_text = remaining
                    print(f"[PREPROCESS] Found {marker}, extracted JSON: {response_text[:200]}")
        
        # Try to extract JSON from the response
        # More aggressive: Find last JSON array/object in response (likely the actual tool call)
        # This handles cases where LLM repeats user message before JSON
        
        # Strategy 1: Look for JSON array patterns with "tool" keyword (multiple tool calls)
        json_arrays = list(re.finditer(r'\[[\s\S]*?\{[\s\S]*?"tool"[\s\S]*?\}[\s\S]*?\]', response_text))
        if json_arrays:
            # Get the last match (most likely the actual response)
            return json_arrays[-1].group(0).strip()
        
        # Strategy 2: Look for alternative format: ["tool_name", {...}]
        alt_format_arrays = list(re.finditer(r'\[[\s\S]*?"(?:chat_message|e_device_control|schedule_modifier|rag_query)"[\s\S]*?\{[\s\S]*?\}[\s\S]*?\]', response_text))
        if alt_format_arrays:
            # Get the last match
            return alt_format_arrays[-1].group(0).strip()
        
        # Strategy 3: Look for any JSON array that might contain tool calls (more lenient)
        any_json_arrays = list(re.finditer(r'\[[\s\S]*?\{[\s\S]*?\}[\s\S]*?\]', response_text))
        if any_json_arrays:
            # Check each match to see if it looks like a tool call
            for match in reversed(any_json_arrays):  # Start from last
                json_str = match.group(0).strip()
                # Check if it contains tool-related keywords
                if '"tool"' in json_str or '"schedule_modifier"' in json_str or '"chat_message"' in json_str or '"e_device_control"' in json_str:
                    return json_str
        
        # Strategy 4: Look for JSON object patterns with "tool" keyword (single tool call)
        json_objects = list(re.finditer(r'\{[\s\S]*?"tool"[\s\S]*?\}', response_text))
        if json_objects:
            # Get the last match
            return json_objects[-1].group(0).strip()
        
        # Strategy 5: Look for any JSON object that might be a tool call (more lenient)
        any_json_objects = list(re.finditer(r'\{[\s\S]*?\}', response_text))
        if any_json_objects:
            # Check each match to see if it looks like a tool call
            for match in reversed(any_json_objects):  # Start from last
                json_str = match.group(0).strip()
                # Check if it contains tool-related keywords
                if '"tool"' in json_str or '"schedule_modifier"' in json_str or '"chat_message"' in json_str or '"e_device_control"' in json_str:
                    return json_str
        
        # Return original if no JSON found
        return response_text.strip()
    
    def _parse_tool_calls(self, response_text: str) -> list:
        """
        Safely parse tool call(s) from LLM response (OPTIMIZED - Phase 1.2).
        Supports both single tool call and multiple tool calls in an array.
        
        Optimized parsing strategy (early exit on success):
        1. Markdown code blocks (most common format)
        2. Direct JSON patterns (without markdown)
        3. Fallback lenient patterns (only if above fail)
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            List of dicts with "tool" and "arguments", or empty list if none found
        """
        if not response_text:
            return []
        
        # Strategy 1: Markdown code blocks (most common format) - EARLY EXIT on success
        match = self._compiled_patterns['markdown_array'].search(response_text)
        if match:
            json_str = match.group(1)
            parsed = self._parse_json_array_safely(json_str)
            if parsed:
                return parsed
        
        match = self._compiled_patterns['markdown_object'].search(response_text)
        if match:
            json_str = match.group(1)
            result = self._parse_json_safely(json_str)
            if result and result.get("tool"):
                return [result]
        
        # Strategy 2: Direct JSON patterns (without markdown) - EARLY EXIT on success
        matches = self._compiled_patterns['json_array_with_tool'].finditer(response_text)
        for match in matches:
            json_str = match.group(0)
            parsed = self._parse_json_array_safely(json_str)
            if parsed:
                return parsed
        
        matches = self._compiled_patterns['json_object_with_tool'].finditer(response_text)
        for match in matches:
            json_str = match.group(0)
            result = self._parse_json_safely(json_str)
            if result and result.get("tool"):
                return [result]
        
        # Strategy 3: Fallback lenient patterns (only if above fail) - with repair attempts
        matches = self._compiled_patterns['json_array_lenient'].finditer(response_text)
        for match in matches:
            json_str = match.group(0)
            # Only repair if initial parse fails
            parsed = self._parse_json_array_safely(json_str)
            if parsed:
                return parsed
            # Try repair as fallback
            json_str = self._try_repair_json(json_str)
            parsed = self._parse_json_array_safely(json_str)
            if parsed:
                return parsed
        
        matches = self._compiled_patterns['json_object_lenient'].finditer(response_text)
        for match in matches:
            json_str = match.group(0)
            result = self._parse_json_safely(json_str)
            if result and result.get("tool"):
                return [result]
            # Try repair as fallback
            json_str = self._try_repair_json(json_str)
            result = self._parse_json_safely(json_str)
            if result and result.get("tool"):
                return [result]
        
        # Strategy 4: Structured text patterns (last resort)
        tool_match = self._compiled_patterns['tool_name'].search(response_text)
        if tool_match:
            tool_name = tool_match.group(1)
            args_match = self._compiled_patterns['tool_arguments'].search(response_text)
            if args_match:
                args_json = args_match.group(1)
                args = self._parse_json_safely(args_json)
                if args:
                    return [{
                        "tool": tool_name,
                        "arguments": args
                    }]
        
        return []
    
    def _try_repair_json(self, json_str: str) -> str:
        """
        Try to repair common JSON issues: missing quotes, trailing commas, unbalanced brackets.
        
        Args:
            json_str: Potentially malformed JSON string
            
        Returns:
            Repaired JSON string, or original if repair fails
        """
        if not json_str:
            return json_str
        
        json_str = json_str.strip()
        
        # Fix trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Fix missing quotes around keys (common LLM mistake)
        # Pattern: {key: value} -> {"key": value}
        json_str = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
        
        # Try to balance brackets/braces
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        # If it starts with [ or {, try to close it
        if json_str.startswith('[') and open_brackets > close_brackets:
            missing = open_brackets - close_brackets
            # Close any open objects first
            if open_braces > close_braces:
                json_str += '}' * (open_braces - close_braces)
            json_str += ']' * missing
        elif json_str.startswith('{') and open_braces > close_braces:
            missing = open_braces - close_braces
            json_str += '}' * missing
        
        return json_str
    
    def _repair_incomplete_json(self, json_str: str) -> str:
        """
        Try to repair incomplete JSON by adding missing closing braces/brackets.
        
        Args:
            json_str: Potentially incomplete JSON string
            
        Returns:
            Repaired JSON string, or original if repair fails
        """
        if not json_str:
            return json_str
        
        json_str = json_str.strip()
        
        # Count opening and closing braces/brackets
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        # If it looks like an incomplete JSON object
        if json_str.startswith('{') and open_braces > close_braces:
            # Try to complete it
            missing_braces = open_braces - close_braces
            # Check if it's inside an array context
            if json_str.startswith('[') or (open_brackets > 0 and open_brackets > close_brackets):
                # It's an array, need to close object then array
                json_str += '}' * missing_braces
                if open_brackets > close_brackets:
                    json_str += ']' * (open_brackets - close_brackets)
            else:
                # Just an object
                json_str += '}' * missing_braces
        
        # If it looks like an incomplete JSON array
        elif json_str.startswith('[') and open_brackets > close_brackets:
            missing_brackets = open_brackets - close_brackets
            # Close any open objects first
            if open_braces > close_braces:
                json_str += '}' * (open_braces - close_braces)
            json_str += ']' * missing_brackets
        
        return json_str
    
    def _parse_json_array_safely(self, json_str: str) -> list:
        """
        Safely parse JSON array string containing multiple tool calls.
        
        Args:
            json_str: JSON array string to parse
            
        Returns:
            List of tool call dicts if successful, empty list otherwise
        """
        if not json_str:
            return []
        
        try:
            json_str = json_str.strip()
            parsed = json.loads(json_str)
            
            if isinstance(parsed, list):
                # NEW: Handle case where the entire array IS in format ["tool_name", {...}]
                # This happens when LLM uses simplified format: ["e_device_control", {...}]
                if len(parsed) == 2 and isinstance(parsed[0], str) and isinstance(parsed[1], dict):
                    tool_name = parsed[0]
                    args = parsed[1]
                    # Validate it's a known tool name
                    if tool_name in ["e_device_control", "chat_message", "schedule_modifier", "rag_query"]:
                        return [{"tool": tool_name, "arguments": args}]
                
                # Filter out empty strings, None, and invalid entries
                parsed = [item for item in parsed if item and item != "" and item != [] and item != {}]
                
                tool_calls = []
                for item in parsed:
                    # Skip None or invalid types
                    if not item or not isinstance(item, (dict, list, str)):
                        continue
                    
                    # Handle alternative format: ["tool_name", {...}]
                    if isinstance(item, list) and len(item) == 2:
                        tool_name = item[0]
                        args = item[1]
                        if isinstance(tool_name, str) and isinstance(args, dict):
                            tool_calls.append({"tool": tool_name, "arguments": args})
                            continue
                    
                    # Handle standard format: {"tool": "...", "arguments": {...}}
                    if isinstance(item, dict) and "tool" in item:
                        tool_calls.append(item)
                return tool_calls
            
            return []
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            try:
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    # NEW: Handle case where the entire array IS in format ["tool_name", {...}]
                    # This happens when LLM uses simplified format: ["e_device_control", {...}]
                    if len(parsed) == 2 and isinstance(parsed[0], str) and isinstance(parsed[1], dict):
                        tool_name = parsed[0]
                        args = parsed[1]
                        # Validate it's a known tool name
                        if tool_name in ["e_device_control", "chat_message", "schedule_modifier", "rag_query"]:
                            return [{"tool": tool_name, "arguments": args}]
                    
                    # Filter out empty strings, None, and invalid entries
                    parsed = [item for item in parsed if item and item != "" and item != [] and item != {}]
                    
                    tool_calls = []
                    for item in parsed:
                        # Skip None or invalid types
                        if not item or not isinstance(item, (dict, list, str)):
                            continue
                        
                        # Handle alternative format: ["tool_name", {...}]
                        if isinstance(item, list) and len(item) == 2:
                            tool_name = item[0]
                            args = item[1]
                            if isinstance(tool_name, str) and isinstance(args, dict):
                                tool_calls.append({"tool": tool_name, "arguments": args})
                                continue
                        
                        # Handle standard format: {"tool": "...", "arguments": {...}}
                        if isinstance(item, dict) and "tool" in item:
                            tool_calls.append(item)
                    return tool_calls
            except:
                pass
            
            return []
        except Exception:
            return []
    
    def _detect_recent_question(self, chat_history: list) -> str:
        """
        Detect if a recent question was asked about controlling a device.
        
        Args:
            chat_history: List of chat messages
            
        Returns:
            String describing recent question context, or empty string if none detected
        """
        if not chat_history or len(chat_history) < 2:
            return ""
        
        # Look at last few messages (assistant question, then user response)
        # Check last 4 messages (2 exchanges)
        recent_messages = chat_history[-4:] if len(chat_history) >= 4 else chat_history
        
        # Look for pattern: assistant asks question, user responds
        for i in range(len(recent_messages) - 1):
            if i + 1 >= len(recent_messages):
                break
                
            assistant_msg = recent_messages[i]
            user_msg = recent_messages[i + 1]
            
            # Check if assistant message contains a question about controlling a device
            if (assistant_msg.get('role') == 'assistant' and 
                user_msg.get('role') == 'user'):
                
                assistant_content = assistant_msg.get('content', '').lower()
                user_content = user_msg.get('content', '').lower()
                
                # Check if assistant asked about controlling a device
                question_keywords = ['would you like', 'do you want', 'should i', 'can i', 'turn on', 'turn off']
                is_question = any(keyword in assistant_content for keyword in question_keywords)
                
                # Check if user gave a positive response
                positive_responses = ['yeah', 'yes', 'yep', 'sure', 'okay', 'ok', 'please', 'do it']
                is_positive = any(response in user_content for response in positive_responses)
                
                # Check if user gave a negative response
                negative_responses = ['no', 'nope', "don't", 'dont', 'not', "won't"]
                is_negative = any(response in user_content for response in negative_responses)
                
                if is_question:
                    if is_positive:
                        return f"You recently asked: \"{assistant_msg.get('content', '')[:100]}...\"\nUser responded: \"{user_msg.get('content', '')}\"\nACTION REQUIRED: User said yes/yeah/sure - you should TAKE ACTION (use e_device_control), NOT ask again."
                    elif is_negative:
                        return f"You recently asked: \"{assistant_msg.get('content', '')[:100]}...\"\nUser responded: \"{user_msg.get('content', '')}\"\nIMPORTANT: User said no - acknowledge and move on, DO NOT ask the same question again."
                    else:
                        return f"You recently asked: \"{assistant_msg.get('content', '')[:100]}...\"\nUser responded: \"{user_msg.get('content', '')}\"\nIMPORTANT: You already asked this question. If user's response is unclear, ask for clarification ONCE, then act. Do NOT repeat the same question."
        
        return ""
    
    def _detect_information_completion(self, chat_history: list) -> str:
        """
        Detect when user provides missing information in response to clarification questions.
        
        Args:
            chat_history: List of chat messages
            
        Returns:
            String describing information completion context, or empty string if none detected
        """
        if not chat_history or len(chat_history) < 3:
            return ""
        
        # Look at last 4-6 messages for pattern: Assistant clarification → User provides info
        recent_messages = chat_history[-6:] if len(chat_history) >= 6 else chat_history
        
        # Look for pattern: assistant asks clarification, user provides short response
        for i in range(len(recent_messages) - 1):
            if i + 1 >= len(recent_messages):
                break
            
            assistant_msg = recent_messages[i]
            user_msg = recent_messages[i + 1]
            
            # Check if assistant asked for clarification
            if (assistant_msg.get('role') == 'assistant' and 
                user_msg.get('role') == 'user'):
                
                assistant_content = assistant_msg.get('content', '').lower()
                user_content = user_msg.get('content', '').strip()
                
                # Check if assistant asked for clarification
                clarification_keywords = [
                    'clarify', 'clarification', 'which', 'what do you mean', 
                    'specify', 'could you', 'please specify', 'do you mean',
                    'all devices in the house', 'current location', 'which room'
                ]
                is_clarification = any(keyword in assistant_content for keyword in clarification_keywords)
                
                if is_clarification:
                    # Check if user response is a single word/phrase (likely providing missing info)
                    # Room names, device names, times, activities are usually short
                    user_words = user_content.split()
                    
                    # If user response is short (1-3 words), likely providing missing information
                    if len(user_words) <= 3 and len(user_content) < 50:
                        # Look back further for original request (6-10 messages back)
                        original_request = None
                        for j in range(max(0, len(chat_history) - 10), len(chat_history) - len(recent_messages) + i):
                            if j < 0 or j >= len(chat_history):
                                continue
                            msg = chat_history[j]
                            if msg.get('role') == 'user':
                                content = msg.get('content', '').lower()
                                # Check if it's an action request
                                action_keywords = ['turn', 'switch', 'add', 'delete', 'change', 'schedule', 'everything', 'all devices']
                                if any(keyword in content for keyword in action_keywords):
                                    original_request = msg.get('content', '')
                                    break
                        
                        if original_request:
                            # Determine what type of information was provided
                            info_type = None
                            info_value = user_content
                            
                            # Check if it's a room name
                            rooms = ['bedroom', 'bathroom', 'kitchen', 'living room', 'living']
                            if user_content.lower() in rooms:
                                info_type = 'room'
                                # Normalize room name
                                if user_content.lower() == 'living':
                                    info_value = 'Living Room'
                                else:
                                    info_value = user_content.title()
                            
                            # Check if it's a device name
                            elif user_content.lower() in ['light', 'ac', 'tv', 'fan', 'alarm', 'lgiht', 'ligth']:
                                info_type = 'device'
                                # Normalize device name
                                device_map = {
                                    'light': 'Light', 'lgiht': 'Light', 'ligth': 'Light',
                                    'ac': 'AC', 'tv': 'TV', 'fan': 'Fan', 'alarm': 'Alarm'
                                }
                                info_value = device_map.get(user_content.lower(), user_content.title())
                            
                            # Check if it's a time (HH:MM format or similar)
                            elif re.match(r'^\d{1,2}[:.]?\d{0,2}$', user_content):
                                info_type = 'time'
                            
                            # Check if it's an activity name (longer phrase)
                            elif len(user_words) > 1:
                                info_type = 'activity'
                            
                            # If we identified the info type, return completion context
                            if info_type:
                                return f"""COMPLETE PREVIOUS ACTION:
You previously asked for clarification: "{assistant_msg.get('content', '')[:150]}..."
User just provided {info_type}="{info_value}" in response.

ORIGINAL REQUEST: "{original_request}"

ACTION REQUIRED: Execute the original request using the provided {info_type}="{info_value}". 
Do NOT ask for clarification again. Do NOT just list information - TAKE ACTION immediately."""
        
        return ""
    
    def _repair_incomplete_json(self, json_str: str) -> str:
        """
        Try to repair incomplete JSON by adding missing closing braces/brackets.
        
        Args:
            json_str: Potentially incomplete JSON string
            
        Returns:
            Repaired JSON string, or original if repair fails
        """
        if not json_str:
            return json_str
        
        json_str = json_str.strip()
        
        # Count opening and closing braces/brackets
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        # If it looks like an incomplete JSON object
        if json_str.startswith('{') and open_braces > close_braces:
            # Try to complete it
            missing_braces = open_braces - close_braces
            # Check if it's inside an array context
            if json_str.startswith('[') or '[' in json_str[:10]:
                # It's an array, need to close object then array
                json_str += '}' * missing_braces
                if open_brackets > close_brackets:
                    json_str += ']' * (open_brackets - close_brackets)
            else:
                # Just an object
                json_str += '}' * missing_braces
        
        # If it looks like an incomplete JSON array
        elif json_str.startswith('[') and open_brackets > close_brackets:
            missing_brackets = open_brackets - close_brackets
            # Close any open objects first
            if open_braces > close_braces:
                json_str += '}' * (open_braces - close_braces)
            json_str += ']' * missing_brackets
        
        return json_str
    
    def _parse_json_safely(self, json_str: str) -> dict:
        """
        Safely parse JSON string with error handling.
        
        Args:
            json_str: JSON string to parse
            
        Returns:
            Parsed dict if successful, None otherwise
        """
        if not json_str:
            return None
        
        try:
            # Clean up the string
            json_str = json_str.strip()
            
            # Try to parse
            parsed = json.loads(json_str)
            
            # Validate structure
            if isinstance(parsed, dict):
                # Check if it looks like a tool call
                if "tool" in parsed:
                    return parsed
                # Maybe it's just the arguments?
                if "arguments" in parsed:
                    return parsed
            
            return None
        
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            try:
                # Remove trailing commas
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                
                # Try parsing again
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and "tool" in parsed:
                    return parsed
            except:
                pass
            
            return None
        except Exception:
            return None
    
    def _looks_like_json_tool_call(self, text: str) -> bool:
        """
        Check if text looks like a JSON tool call (even if malformed).
        This prevents showing raw JSON tool calls to users.
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be a JSON tool call, False otherwise
        """
        if not text:
            return False
        
        text_lower = text.lower().strip()
        
        # Check for common JSON tool call patterns
        # Pattern 1: Contains "tool" and "arguments" keywords
        if '"tool"' in text_lower or "'tool'" in text_lower:
            if '"arguments"' in text_lower or "'arguments'" in text_lower:
                # Also check for JSON structure markers
                if ('{' in text or '[' in text):
                    return True
        
        # Pattern 2: Looks like JSON array with tool calls
        if text.strip().startswith('[') and '"tool"' in text_lower:
            return True
        
        # Pattern 3: Looks like JSON object with tool call
        if text.strip().startswith('{') and '"tool"' in text_lower:
            return True
        
        return False
    
    def _detect_recent_question_optimized(self, recent_messages: list) -> str:
        """
        Optimized version: Detects recent questions using pre-filtered recent messages.
        Phase 4: Optimize Detection Methods
        
        Args:
            recent_messages: Pre-filtered list of last 4-6 messages
            
        Returns:
            String describing recent question context, or empty string if none detected
        """
        if not recent_messages or len(recent_messages) < 2:
            return ""
        
        # Look for pattern: assistant asks question, user responds
        # Scan from END (most recent) to BEGINNING (oldest) to prioritize latest interactions
        for i in range(len(recent_messages) - 2, -1, -1):  # Reverse iteration: start from second-to-last, go backwards
            if i + 1 >= len(recent_messages):
                continue
                
            assistant_msg = recent_messages[i]
            user_msg = recent_messages[i + 1]
            
            if (assistant_msg.get('role') == 'assistant' and 
                user_msg.get('role') == 'user'):
                
                assistant_content = assistant_msg.get('content', '').lower()
                user_content = user_msg.get('content', '').lower()
                
                question_keywords = ['would you like', 'do you want', 'should i', 'can i', 'turn on', 'turn off']
                is_question = any(keyword in assistant_content for keyword in question_keywords)
                
                positive_responses = ['yeah', 'yes', 'yep', 'sure', 'okay', 'ok', 'please', 'do it']
                is_positive = any(response in user_content for response in positive_responses)
                
                negative_responses = ['no', 'nope', "don't", 'dont', 'not', "won't"]
                is_negative = any(response in user_content for response in negative_responses)
                
                if is_question:
                    if is_positive:
                        return f"You recently asked: \"{assistant_msg.get('content', '')[:100]}...\"\nUser responded: \"{user_msg.get('content', '')}\"\nACTION REQUIRED: User said yes/yeah/sure - you should TAKE ACTION (use e_device_control), NOT ask again."
                    elif is_negative:
                        return f"You recently asked: \"{assistant_msg.get('content', '')[:100]}...\"\nUser responded: \"{user_msg.get('content', '')}\"\nIMPORTANT: User said no - acknowledge and move on, DO NOT ask the same question again."
                    else:
                        return f"You recently asked: \"{assistant_msg.get('content', '')[:100]}...\"\nUser responded: \"{user_msg.get('content', '')}\"\nIMPORTANT: You already asked this question. If user's response is unclear, ask for clarification ONCE, then act. Do NOT repeat the same question."
        
        return ""
    
    def _detect_information_completion_optimized(self, recent_messages: list, full_chat_history: list) -> str:
        """
        Optimized version: Detects information completion using pre-filtered recent messages.
        Phase 4: Optimize Detection Methods
        
        Args:
            recent_messages: Pre-filtered list of last 6 messages
            full_chat_history: Full chat history for looking back at original request
            
        Returns:
            String describing information completion context, or empty string if none detected
        """
        if not recent_messages or len(recent_messages) < 3:
            return ""
        
        # Look for pattern: assistant asks clarification, user provides short response
        # Scan from END (most recent) to BEGINNING (oldest) to prioritize latest interactions
        for i in range(len(recent_messages) - 2, -1, -1):  # Reverse iteration: start from second-to-last, go backwards
            if i + 1 >= len(recent_messages):
                continue
            
            assistant_msg = recent_messages[i]
            user_msg = recent_messages[i + 1]
            
            if (assistant_msg.get('role') == 'assistant' and 
                user_msg.get('role') == 'user'):
                
                assistant_content = assistant_msg.get('content', '').lower()
                user_content = user_msg.get('content', '').strip()
                
                clarification_keywords = [
                    'clarify', 'clarification', 'which', 'what do you mean', 
                    'specify', 'could you', 'please specify', 'do you mean',
                    'all devices in the house', 'current location', 'which room'
                ]
                is_clarification = any(keyword in assistant_content for keyword in clarification_keywords)
                
                if is_clarification:
                    user_words = user_content.split()
                    if len(user_words) <= 3 and len(user_content) < 50:
                        info_type = None
                        info_value = user_content
                        
                        # Check if it's a room name
                        rooms = ['bedroom', 'bathroom', 'kitchen', 'living room', 'living']
                        if user_content.lower() in rooms:
                            info_type = 'room'
                            if user_content.lower() == 'living':
                                info_value = 'Living Room'
                            else:
                                info_value = user_content.title()
                        
                        # Check if it's a device name
                        devices = ['light', 'ac', 'tv', 'fan', 'alarm', 'lgiht', 'ligth']
                        if user_content.lower() in devices:
                            info_type = 'device'
                            device_map = {
                                'light': 'Light', 'lgiht': 'Light', 'ligth': 'Light',
                                'ac': 'AC', 'tv': 'TV', 'fan': 'Fan', 'alarm': 'Alarm'
                            }
                            info_value = device_map.get(user_content.lower(), user_content.title())
                        
                        # Check if it's a time
                        if re.match(r'^\d{1,2}[:.]?\d{0,2}$', user_content):
                            info_type = 'time'
                        
                        # Check if it's an activity name
                        if len(user_words) > 1:
                            info_type = 'activity'
                        
                        if info_type:
                            # Look back for original request (up to 10 messages back)
                            original_request_msg = None
                            start_idx = max(0, len(full_chat_history) - 10)
                            for j in range(start_idx, len(full_chat_history)):
                                if j < 0 or j >= len(full_chat_history):
                                    continue
                                msg = full_chat_history[j]
                                if msg.get('role') == 'user':
                                    content = msg.get('content', '').lower()
                                    action_keywords = ['turn', 'switch', 'add', 'delete', 'change', 'schedule', 'everything', 'all devices']
                                    if any(keyword in content for keyword in action_keywords):
                                        original_request_msg = full_chat_history[j]
                                        break
                            
                            if original_request_msg:
                                original_request_content = original_request_msg.get('content', '')
                                return f"""COMPLETE PREVIOUS ACTION:
You previously asked for clarification: "{assistant_msg.get('content', '')[:150]}..."
User just provided {info_type}="{info_value}" in response.
Original request was: "{original_request_content[:150]}..."
ACTION REQUIRED: Combine the original request with the provided information and execute the action immediately. Do NOT ask for clarification again."""
        
        return ""
    
    def summarize_conversation(self, old_messages: list, existing_summary: dict = None) -> dict:
        """
        Summarize old conversation messages to compress context.
        
        Args:
            old_messages: List of messages to summarize (beyond recent window)
            existing_summary: Existing summary dict to merge with
            
        Returns:
            Updated summary dict with:
            {
                "last_summarized_turn": int,
                "summary_text": str,
                "key_events": list
            }
        """
        if not old_messages:
            return existing_summary or {
                "last_summarized_turn": 0,
                "summary_text": "",
                "key_events": []
            }
        
        # Extract key events (device controls, schedule changes, preferences)
        key_events = []
        for msg in old_messages:
            content = msg.get('content', '').lower()
            role = msg.get('role', '')
            
            # Extract device control events
            if 'turned' in content or 'turned on' in content or 'turned off' in content:
                key_events.append({
                    "type": "device_control",
                    "summary": msg.get('content', '')[:100]
                })
            # Extract schedule changes
            elif any(word in content for word in ['schedule', 'appointment', 'meeting', 'added', 'changed', 'deleted']):
                key_events.append({
                    "type": "schedule_change",
                    "summary": msg.get('content', '')[:100]
                })
            # Extract preferences
            elif 'preference' in content or 'keep it on' in content:
                key_events.append({
                    "type": "preference_set",
                    "summary": msg.get('content', '')[:100]
                })
        
        # Format messages for summarization
        conversation_text = ""
        for msg in old_messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            # Skip notifications and preference updates in summary
            if msg.get('is_notification') or msg.get('is_preference_update'):
                continue
            conversation_text += f"{role.upper()}: {content}\n"
        
        if not conversation_text.strip():
            # No meaningful content to summarize
            return existing_summary or {
                "last_summarized_turn": 0,
                "summary_text": "",
                "key_events": key_events
            }
        
        # Generate summary using LLM (compact prompt)
        summary_prompt = f"""Summarize this conversation focusing on:
- User preferences and important decisions
- Device control patterns
- Schedule changes
- Key information the user shared

Conversation:
{conversation_text[:2000]}

Provide a concise summary (max 200 words):"""
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": summary_prompt}],
                options={
                    "temperature": 0.3,  # Lower temperature for more factual summaries
                    "num_ctx": 4096,  # Smaller context for summarization
                    "num_predict": 300  # Limit summary length
                }
            )
            summary_text = response.get('message', {}).get('content', '').strip()
        except Exception as e:
            print(f"[CONTEXT] Summarization failed: {e}")
            # Fallback: create simple summary from key events
            summary_text = f"Previous conversation included: {len(key_events)} key events (device controls, schedule changes, preferences)."
        
        # Merge with existing summary
        if existing_summary and existing_summary.get("summary_text"):
            combined_summary = f"{existing_summary['summary_text']}\n\n{summary_text}"
            combined_events = existing_summary.get("key_events", []) + key_events
        else:
            combined_summary = summary_text
            combined_events = key_events
        
        return {
            "last_summarized_turn": existing_summary.get("last_summarized_turn", 0) if existing_summary else 0,
            "summary_text": combined_summary[:500],  # Limit summary size
            "key_events": combined_events[-20:]  # Keep last 20 key events
        }
