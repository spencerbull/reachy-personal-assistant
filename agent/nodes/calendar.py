"""
Calendar node for handling Google Calendar operations.

This specialized sub-agent handles all calendar-related requests with its own
system prompt and only calendar tools for better accuracy.
"""

import os
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool

from agent.state import ReachyAgentState, StateUpdate
from agent.config import AgentConfig, REACHY_IDENTITY, REACHY_OUTPUT_RULES

# Get calendar ID from environment
DEFAULT_CALENDAR_ID = os.getenv("CALENDAR_ID", "primary")


def get_calendar_system_prompt() -> str:
    """Build the calendar-specific system prompt with current environment settings."""
    calendar_id = DEFAULT_CALENDAR_ID
    now = datetime.now()
    today_iso = now.strftime("%Y-%m-%dT00:00:00Z")
    today_display = now.strftime("%Y-%m-%d %H:%M")
    two_weeks = now + timedelta(days=14)
    two_weeks_iso = two_weeks.strftime("%Y-%m-%dT23:59:59Z")
    
    return f"""{REACHY_IDENTITY}

{REACHY_OUTPUT_RULES}

You are a calendar assistant. Your job is to help the user manage their Google Calendar.

Current Date/Time: {today_display}

AVAILABLE CALENDAR TOOLS (use EXACT names with HYPHENS):

1. "get-current-time" - Get the current date/time. Call this first if you need the current time.

2. "list-events" - List upcoming calendar events.
   REQUIRED parameters (use these exact values):
   {{
     "calendarId": "{calendar_id}",
     "timeMin": "{today_iso}",
     "timeMax": "{two_weeks_iso}",
     "maxResults": 10
   }}

3. "list-calendars" - List all available calendars. No parameters required.

4. "search-events" - Search for events by text query.
   Required: calendarId ("{calendar_id}"), q (search query text)

5. "create-event" - Create a new calendar event.
   Required parameters:
   {{
     "calendarId": "{calendar_id}",
     "summary": "Event title",
     "start": {{"dateTime": "2024-01-15T14:00:00", "timeZone": "America/Los_Angeles"}},
     "end": {{"dateTime": "2024-01-15T15:00:00", "timeZone": "America/Los_Angeles"}}
   }}

6. "update-event" - Update an existing event.
   Required: calendarId, eventId, plus fields to update.

IMPORTANT RULES:
- Tool names use HYPHENS: "list-events" NOT "list_events"
- Always use calendarId: "{calendar_id}"
- For list-events, always include timeMin, timeMax, and maxResults
- After getting results, summarize them naturally in speech
- Say things like "You have a meeting at 2pm with John" - not raw data
- If no events found, say "Your calendar is clear for that time"

RESPONSE FORMAT:
After tool results, respond conversationally. Examples:
- "You have 3 events today: a standup at 9am, lunch at noon, and a team meeting at 3pm."
- "Your calendar is free tomorrow afternoon. Want me to schedule something?"
- "I've created your meeting with John for Friday at 2pm."
"""


def create_calendar_llm(config: AgentConfig, tools: list[BaseTool]) -> ChatOpenAI:
    """Create the calendar LLM instance with tool bindings."""
    llm = ChatOpenAI(
        model=config.models.main_model,
        base_url=config.models.main_model_base_url,
        api_key=config.models.api_key,
        temperature=config.models.main_temperature,
    )
    
    if tools:
        return llm.bind_tools(tools)
    return llm


def build_calendar_messages(state: ReachyAgentState) -> list:
    """Build the message list for the calendar LLM."""
    messages = [SystemMessage(content=get_calendar_system_prompt())]
    
    # Add conversation history
    for msg in state.get("messages", []):
        messages.append(msg)
    
    return messages


def filter_calendar_tools(all_tools: list[BaseTool]) -> list[BaseTool]:
    """Filter to only include calendar-related tools."""
    calendar_tool_names = {
        "list-events", "list-calendars", "search-events", 
        "create-event", "update-event", "get-current-time"
    }
    return [tool for tool in all_tools if tool.name in calendar_tool_names]


async def execute_tools(tool_calls: list[dict], tools: list[BaseTool]) -> list[dict]:
    """Execute the requested tool calls."""
    results = []
    tool_map = {tool.name: tool for tool in tools}
    
    for tc in tool_calls:
        tool_name = tc.get("name", "")
        tool_args = tc.get("args", {})
        tool_id = tc.get("id", "")
        
        if tool_name in tool_map:
            try:
                tool = tool_map[tool_name]
                result = await tool.ainvoke(tool_args)
                results.append({
                    "id": tool_id,
                    "name": tool_name,
                    "result": result,
                    "success": True,
                })
                logger.info(f"Calendar tool '{tool_name}' executed successfully")
            except Exception as e:
                logger.error(f"Calendar tool '{tool_name}' failed: {e}")
                results.append({
                    "id": tool_id,
                    "name": tool_name,
                    "result": str(e),
                    "success": False,
                })
        else:
            logger.warning(f"Unknown calendar tool requested: {tool_name}")
            results.append({
                "id": tool_id,
                "name": tool_name,
                "result": f"Unknown tool: {tool_name}. Available: {list(tool_map.keys())}",
                "success": False,
            })
    
    return results


async def calendar_node(
    state: ReachyAgentState,
    config: AgentConfig,
    tools: Optional[list[BaseTool]] = None
) -> StateUpdate:
    """
    Calendar node for handling calendar-related requests.
    
    Uses an agentic loop that continues making tool calls until the LLM
    determines it has enough information to respond.
    
    Args:
        state: Current agent state
        config: Agent configuration
        tools: List of all available tools (will be filtered to calendar tools)
        
    Returns:
        State update with calendar response
    """
    tools = tools or []
    calendar_tools = filter_calendar_tools(tools)
    
    logger.info(f"Calendar node: Processing with {len(calendar_tools)} calendar tools: {[t.name for t in calendar_tools]}")
    
    if not calendar_tools:
        logger.warning("Calendar node: No calendar tools available!")
        return {
            "messages": [AIMessage(content="I'm sorry, I don't have access to your calendar right now.")],
            "emotional_state": "apologetic",
        }
    
    # Maximum iterations to prevent infinite loops
    MAX_ITERATIONS = 5
    all_tool_results = []
    
    try:
        llm = create_calendar_llm(config, calendar_tools)
        messages = build_calendar_messages(state)
        
        # Agentic loop - continue until no more tool calls or max iterations
        for iteration in range(MAX_ITERATIONS):
            logger.info(f"Calendar node: Iteration {iteration + 1}/{MAX_ITERATIONS}")
            
            # Call LLM
            response = await llm.ainvoke(messages)
            
            # Check for tool calls
            tool_calls = []
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tc in response.tool_calls:
                    tool_calls.append({
                        "id": tc.get("id", ""),
                        "name": tc.get("name", ""),
                        "args": tc.get("args", {}),
                    })
                logger.info(f"Calendar node: Tool calls requested: {[tc['name'] for tc in tool_calls]}")
            
            if not tool_calls:
                # No more tool calls - LLM is ready to respond
                response_content = response.content
                if not response_content or response_content.strip() == "":
                    response_content = "I checked your calendar but couldn't format the response properly."
                
                logger.info(f"Calendar node: Final response after {iteration + 1} iterations: {response_content[:100]}...")
                
                return {
                    "messages": [AIMessage(content=response_content)],
                    "tool_results": all_tool_results,
                    "emotional_state": "helpful",
                }
            
            # Execute tool calls
            logger.info(f"Calendar node: Executing {len(tool_calls)} tool calls")
            tool_results = await execute_tools(tool_calls, calendar_tools)
            all_tool_results.extend(tool_results)
            
            # Add assistant message with tool calls and tool results to message history
            messages.append(response)
            for result in tool_results:
                messages.append(ToolMessage(
                    content=str(result.get("result", "")),
                    tool_call_id=result.get("id", ""),
                ))
        
        # Max iterations reached - force a response
        logger.warning(f"Calendar node: Max iterations ({MAX_ITERATIONS}) reached, forcing response")
        
        # One final call without tools to get a summary
        final_llm = ChatOpenAI(
            model=config.models.main_model,
            base_url=config.models.main_model_base_url,
            api_key=config.models.api_key,
            temperature=config.models.main_temperature,
        )
        
        # Add instruction to summarize
        messages.append(HumanMessage(content="Please summarize what you found and respond to the user's request."))
        final_response = await final_llm.ainvoke(messages)
        
        return {
            "messages": [AIMessage(content=final_response.content or "I checked your calendar.")],
            "tool_results": all_tool_results,
            "emotional_state": "helpful",
        }
            
    except Exception as e:
        logger.error(f"Calendar node error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "messages": [AIMessage(content="I had trouble accessing your calendar. Please try again.")],
            "emotional_state": "apologetic",
        }
