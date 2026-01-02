"""
Test examples for MCP Router.

These tests demonstrate the MCP router functionality with various tool calls.
Run with: python -m tests.test_mcp_router
"""

from core.state import StateManager
from mcp.server import MCPServer
from mcp.router import MCPRouter


if __name__ == "__main__":
    # Initialize components
    state_manager = StateManager()
    mcp_server = MCPServer(state_manager)
    router = MCPRouter(mcp_server)
    
    print("=" * 60)
    print("MCP Router Test Examples")
    print("=" * 60)
    print()
    
    # Test 1: Chat message
    print("Test 1: chat_message")
    print("-" * 60)
    llm_response_1 = {
        "tool": "chat_message",
        "arguments": {
            "message": "Hello! How can I help you today?"
        }
    }
    result_1 = router.execute(llm_response_1)
    print(f"Input: {llm_response_1}")
    print(f"Result: {result_1}")
    print()
    
    # Test 2: Device control - turn on bedroom Light
    print("Test 2: e_device_control - Turn ON Bedroom Light")
    print("-" * 60)
    llm_response_2 = {
        "tool": "e_device_control",
        "arguments": {
            "room": "Bedroom",
            "device": "Light",
            "action": "ON"
        }
    }
    result_2 = router.execute(llm_response_2)
    print(f"Input: {llm_response_2}")
    print(f"Result: {result_2}")
    print()
    
    # Test 3: Device control - turn off living room TV
    print("Test 3: e_device_control - Turn OFF Living Room TV")
    print("-" * 60)
    llm_response_3 = {
        "tool": "e_device_control",
        "arguments": {
            "room": "Living Room",
            "device": "TV",
            "action": "OFF"
        }
    }
    result_3 = router.execute(llm_response_3)
    print(f"Input: {llm_response_3}")
    print(f"Result: {result_3}")
    print()
    
    # Test 4: Invalid tool name
    print("Test 4: Invalid tool name")
    print("-" * 60)
    llm_response_4 = {
        "tool": "unknown_tool",
        "arguments": {}
    }
    result_4 = router.execute(llm_response_4)
    print(f"Input: {llm_response_4}")
    print(f"Result: {result_4}")
    print()
    
    # Test 5: Missing arguments
    print("Test 5: Missing arguments")
    print("-" * 60)
    llm_response_5 = {
        "tool": "e_device_control",
        "arguments": {
            "room": "Bedroom"
            # Missing device and action
        }
    }
    result_5 = router.execute(llm_response_5)
    print(f"Input: {llm_response_5}")
    print(f"Result: {result_5}")
    print()
    
    # Test 6: Invalid device
    print("Test 6: Invalid device")
    print("-" * 60)
    llm_response_6 = {
        "tool": "e_device_control",
        "arguments": {
            "room": "Bedroom",
            "device": "InvalidDevice",  # Doesn't exist
            "action": "ON"
        }
    }
    result_6 = router.execute(llm_response_6)
    print(f"Input: {llm_response_6}")
    print(f"Result: {result_6}")
    print()
    
    # Test 7: Show current state
    print("Test 7: Current state after changes")
    print("-" * 60)
    current_state = mcp_server.get_current_state()
    print(f"Current State: {current_state}")
    print()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)

