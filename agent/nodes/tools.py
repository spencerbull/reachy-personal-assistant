"""
Tools node for handling physical robot actions and memory operations.

This node processes requests that require calling tools like:
- Reachy movement controls (look, turn, dance)
- Memory operations (remember/recall object locations)
- Image generation

Note: Calendar and email are handled by dedicated sub-agents.
"""

from typing import Optional

from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool

from agent.state import ReachyAgentState, StateUpdate
from agent.config import AgentConfig, REACHY_IDENTITY, REACHY_OUTPUT_RULES

def get_tools_system_prompt() -> str:
    """Build the tools system prompt for robot actions and memory."""
    from datetime import datetime
    now = datetime.now()
    today_display = now.strftime("%Y-%m-%d %H:%M")
    
    return f"""{REACHY_IDENTITY}

{REACHY_OUTPUT_RULES}

Current Date/Time: {today_display}

You are handling physical robot actions and memory operations.
Use tools by making tool calls - do NOT write out function syntax as text.

AVAILABLE TOOLS:

1. MOVEMENT - Control the robot's head and body:
   - look_at_tool: Move head to look in a direction (left, right, up, down, forward)
   - turn_body_tool: Turn the robot's body
   - enable_face_tracking_tool: Enable/disable automatic face tracking
   - get_current_pose_tool: Get current head/body position
   - scan_room_tool: Scan the room by looking around
   - nod_tool: Nod the head (yes/no gesture)

2. DANCE - Make the robot dance:
   - dance_tool: Perform a dance. Available dances:
     groovy_sway_and_roll, chicken_peck, headbanger_combo, dizzy_spin,
     jackson_square, stumble_and_recover, side_to_side_sway, simple_nod, yeah_nod

3. EMOTIONS - Express emotions:
   - express_emotion_tool: Show emotion through body language

4. MEMORY - Remember and recall object locations:
   - remember_location_tool: Save where an object is located
   - recall_location_tool: Retrieve where an object was stored
   - list_remembered_objects_tool: List all remembered objects
   - forget_object_tool: Forget an object's location
   - set_location_context_tool: Set current location (office, home, etc.)

5. IMAGE GENERATION - Transform camera images:
   - generate_image_tool: Transform the current camera view into different styles
   - Returns a URL - ALWAYS include it in your response so user can view it

RESPONSE RULES:
- After actions complete, confirm them naturally: "Looking left now." or "I remember your keys are on the desk."
- NEVER output raw data or JSON
- Speak naturally and briefly

IMPORTANT: You must USE the tools via the tool calling interface. Do not write function calls as text."""


def create_tools_llm(config: AgentConfig, tools: list[BaseTool]) -> ChatOpenAI:
    """Create the tools LLM instance with tool bindings."""
    llm = ChatOpenAI(
        model=config.models.main_model,
        base_url=config.models.main_model_base_url,
        api_key=config.models.api_key,
        temperature=config.models.main_temperature,
    )
    
    if tools:
        return llm.bind_tools(tools)
    return llm


def build_tools_messages(state: ReachyAgentState, config: AgentConfig) -> list:
    """Build the message list for the tools LLM."""
    messages = [SystemMessage(content=get_tools_system_prompt())]
    
    # Add relevant context from state
    reachy_state = state.get("reachy_state", {})
    spatial_memory = state.get("spatial_memory", {})
    
    context_parts = []
    
    # Add current robot state context
    if reachy_state:
        emotion = reachy_state.get("emotion", "neutral")
        face_tracking = reachy_state.get("face_tracking_enabled", False)
        context_parts.append(f"Current state: emotion={emotion}, face_tracking={'on' if face_tracking else 'off'}")
    
    # Add spatial memory context if relevant
    if spatial_memory:
        objects = list(spatial_memory.keys())[:5]  # Limit to 5 most recent
        if objects:
            context_parts.append(f"Objects in memory: {', '.join(objects)}")
    
    # Add current location
    current_location = state.get("current_location", "unknown")
    if current_location != "unknown":
        context_parts.append(f"Current location: {current_location}")
    
    if context_parts:
        context = "\n".join(context_parts)
        messages.append(SystemMessage(content=f"Current context:\n{context}"))
    
    # Add conversation history
    for msg in state.get("messages", []):
        messages.append(msg)
    
    return messages


def extract_tool_calls(response) -> list[dict]:
    """Extract tool calls from LLM response."""
    tool_calls = []
    
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            tool_calls.append({
                "id": tc.get("id", ""),
                "name": tc.get("name", ""),
                "args": tc.get("args", {}),
            })
    
    return tool_calls


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
                logger.info(f"Tool '{tool_name}' executed successfully: {result}")
            except Exception as e:
                logger.error(f"Tool '{tool_name}' failed: {e}")
                results.append({
                    "id": tool_id,
                    "name": tool_name,
                    "result": str(e),
                    "success": False,
                })
        else:
            logger.warning(f"Unknown tool requested: {tool_name}")
            results.append({
                "id": tool_id,
                "name": tool_name,
                "result": f"Unknown tool: {tool_name}",
                "success": False,
            })
    
    return results


def parse_reachy_commands(tool_results: list[dict]) -> list[dict]:
    """Parse tool results for Reachy robot commands."""
    commands = []
    
    for result in tool_results:
        tool_name = result.get("name", "")
        tool_result = result.get("result", "")
        
        # Check for command tokens in the result
        # Format: [CMD_LOOK_LEFT], [CMD_TURN_RIGHT], etc.
        import re
        cmd_pattern = r'\[CMD_([A-Z_]+)\]'
        matches = re.findall(cmd_pattern, str(tool_result))
        
        for match in matches:
            commands.append({
                "command": match.lower(),
                "source_tool": tool_name,
            })
    
    return commands


async def tools_node(
    state: ReachyAgentState, 
    config: AgentConfig,
    tools: Optional[list[BaseTool]] = None
) -> StateUpdate:
    """
    Tools node for handling tool execution.
    
    Args:
        state: Current agent state
        config: Agent configuration
        tools: List of available tools (injected at graph creation)
        
    Returns:
        State update with tool results and response
    """
    tools = tools or []
    
    try:
        llm = create_tools_llm(config, tools)
        messages = build_tools_messages(state, config)
        
        # Log available tools
        tool_names = [t.name for t in tools]
        logger.info(f"Tools node: Processing with {len(tools)} tools: {tool_names}")
        
        # Extract and log user message
        user_msg = ""
        for msg in state.get("messages", []):
            if hasattr(msg, "type") and msg.type == "human":
                user_msg = msg.content[:100]
                break
        logger.info(f"Tools node: User request: {user_msg}")
        
        # First call - let LLM decide which tools to use
        response = await llm.ainvoke(messages)
        
        # Log raw response details
        logger.info(f"Tools node: LLM response type: {type(response)}")
        if hasattr(response, "tool_calls"):
            logger.info(f"Tools node: Raw tool_calls: {response.tool_calls}")
        logger.info(f"Tools node: Response content: {response.content[:200] if response.content else '(empty)'}")
        
        # Check for tool calls
        tool_calls = extract_tool_calls(response)
        
        if tool_calls:
            logger.info(f"Tools node: Executing {len(tool_calls)} tool calls: {[tc.get('name') for tc in tool_calls]}")
            
            # Execute the tools
            tool_results = await execute_tools(tool_calls, tools)
            
            # Parse for Reachy commands
            reachy_commands = parse_reachy_commands(tool_results)
            
            # Build tool messages for second LLM call
            tool_messages = messages + [response]
            for result in tool_results:
                tool_messages.append(ToolMessage(
                    content=str(result.get("result", "")),
                    tool_call_id=result.get("id", ""),
                ))
            
            # Second call - generate response incorporating tool results
            final_response = await llm.ainvoke(tool_messages)
            
            # Log the final response details
            logger.info(f"Tools node: Final response type: {type(final_response)}")
            logger.info(f"Tools node: Final response content: '{final_response.content[:200] if final_response.content else '(empty)'}'")
            
            # Check if the model made more tool calls instead of responding
            if hasattr(final_response, "tool_calls") and final_response.tool_calls:
                logger.warning(f"Tools node: Model made additional tool calls instead of responding: {final_response.tool_calls}")
            
            # Handle empty response - use the first LLM response's content or generate a fallback
            response_content = final_response.content
            if not response_content or response_content.strip() == "":
                # Try to use the initial response content
                if response.content and response.content.strip():
                    response_content = response.content
                    logger.info(f"Tools node: Using initial response content: {response_content[:100]}")
                else:
                    # Generate a fallback based on tool results
                    tool_summaries = []
                    for result in tool_results:
                        tool_name = result.get("name", "unknown")
                        tool_output = str(result.get("result", ""))[:200]
                        tool_summaries.append(f"{tool_name}: {tool_output}")
                    response_content = f"I checked that for you. Here's what I found: {'; '.join(tool_summaries)}"
                    logger.info(f"Tools node: Generated fallback response")
            
            # Determine emotional state based on actions
            emotional_state = "helpful"
            for cmd in reachy_commands:
                if "look" in cmd.get("command", ""):
                    emotional_state = "attentive"
                elif "dance" in cmd.get("command", ""):
                    emotional_state = "excited"
            
            # Update reachy state
            reachy_state = state.get("reachy_state", {})
            reachy_state["emotion"] = emotional_state
            
            return {
                "messages": [AIMessage(content=response_content)],
                "tool_results": tool_results,
                "pending_reachy_commands": reachy_commands,
                "emotional_state": emotional_state,
                "reachy_state": reachy_state,
            }
        else:
            # No tool calls - this is unexpected for the tools node!
            # The LLM should have called a tool for physical actions
            logger.warning(f"Tools node: NO TOOL CALLS made! Response was: {response.content[:200]}")
            logger.warning("Tools node: The LLM responded with text instead of using tools. Check tool binding.")
            
            return {
                "messages": [AIMessage(content=response.content)],
                "emotional_state": "helpful",
            }
        
    except Exception as e:
        logger.error(f"Tools node error: {e}")
        return {
            "messages": [AIMessage(
                content="I had trouble with that action. Could you try asking in a different way?"
            )],
            "emotional_state": "neutral",
        }
