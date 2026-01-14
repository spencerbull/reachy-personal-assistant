"""
Email node for handling Gmail operations.

This specialized sub-agent handles all email-related requests with its own
system prompt and only email tools for better accuracy.
"""

from typing import Optional

from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool

from agent.state import ReachyAgentState, StateUpdate
from agent.config import AgentConfig, REACHY_IDENTITY, REACHY_OUTPUT_RULES


def get_email_system_prompt() -> str:
    """Build the email-specific system prompt."""
    return f"""{REACHY_IDENTITY}

{REACHY_OUTPUT_RULES}

You are an email assistant. Your job is to help the user manage their Gmail.

AVAILABLE EMAIL TOOLS (use EXACT names with UNDERSCORES):

1. "search_emails" - Search for emails using Gmail query syntax.
   Parameters: query (string)
   Query examples:
   - "is:unread" - unread emails
   - "from:john@example.com" - emails from John
   - "subject:meeting" - emails with "meeting" in subject
   - "is:unread after:2024/01/15" - unread emails after a date
   - "has:attachment" - emails with attachments
   - "in:inbox is:unread" - unread emails in inbox

2. "read_email" - Read a specific email by ID.
   Parameters: messageId (string from search results)

3. "send_email" - Send a new email.
   Parameters: to, subject, body

4. "draft_email" - Create a draft email (not sent).
   Parameters: to, subject, body

5. "modify_email" - Modify email labels (move, archive, etc.)
   Parameters: messageId, addLabelIds, removeLabelIds

6. "list_email_labels" - List all Gmail labels.
   No parameters required.

IMPORTANT RULES:
- Tool names use UNDERSCORES: "search_emails" NOT "search-emails"
- For checking unread emails, use: search_emails with query "is:unread in:inbox"
- After getting results, summarize them naturally in speech
- Say things like "You have 3 unread emails" - not raw JSON
- Summarize email content briefly: "John sent you a message about the project deadline"
- For privacy, don't read out full email bodies unless asked

RESPONSE FORMAT:
After tool results, respond conversationally. Examples:
- "You have 5 unread emails. The most recent is from your boss about tomorrow's meeting."
- "I found 3 emails from John this week. Want me to read the latest one?"
- "I've sent your email to Sarah with the subject 'Meeting Tomorrow'."
- "Your inbox is clear - no unread emails right now!"

COMMON TASKS:
- "Check my email" → search_emails with query "is:unread in:inbox"
- "Emails from John" → search_emails with query "from:john"
- "Send email to X about Y" → send_email with to, subject, body
"""


def create_email_llm(config: AgentConfig, tools: list[BaseTool]) -> ChatOpenAI:
    """Create the email LLM instance with tool bindings."""
    llm = ChatOpenAI(
        model=config.models.main_model,
        base_url=config.models.main_model_base_url,
        api_key=config.models.api_key,
        temperature=config.models.main_temperature,
    )
    
    if tools:
        return llm.bind_tools(tools)
    return llm


def build_email_messages(state: ReachyAgentState) -> list:
    """Build the message list for the email LLM."""
    messages = [SystemMessage(content=get_email_system_prompt())]
    
    # Add conversation history
    for msg in state.get("messages", []):
        messages.append(msg)
    
    return messages


def filter_email_tools(all_tools: list[BaseTool]) -> list[BaseTool]:
    """Filter to only include email-related tools."""
    email_tool_names = {
        "send_email", "draft_email", "read_email", "search_emails",
        "modify_email", "delete_email", "list_email_labels",
        "batch_modify_emails", "batch_delete_emails",
        "create_label", "update_label", "delete_label", "get_or_create_label",
        "create_filter", "list_filters", "get_filter", "delete_filter",
        "create_filter_from_template", "download_attachment"
    }
    return [tool for tool in all_tools if tool.name in email_tool_names]


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
                logger.info(f"Email tool '{tool_name}' executed successfully")
            except Exception as e:
                logger.error(f"Email tool '{tool_name}' failed: {e}")
                results.append({
                    "id": tool_id,
                    "name": tool_name,
                    "result": str(e),
                    "success": False,
                })
        else:
            logger.warning(f"Unknown email tool requested: {tool_name}")
            results.append({
                "id": tool_id,
                "name": tool_name,
                "result": f"Unknown tool: {tool_name}. Available: {list(tool_map.keys())}",
                "success": False,
            })
    
    return results


async def email_node(
    state: ReachyAgentState,
    config: AgentConfig,
    tools: Optional[list[BaseTool]] = None
) -> StateUpdate:
    """
    Email node for handling email-related requests.
    
    Uses an agentic loop that continues making tool calls until the LLM
    determines it has enough information to respond.
    
    Args:
        state: Current agent state
        config: Agent configuration
        tools: List of all available tools (will be filtered to email tools)
        
    Returns:
        State update with email response
    """
    tools = tools or []
    email_tools = filter_email_tools(tools)
    
    logger.info(f"Email node: Processing with {len(email_tools)} email tools: {[t.name for t in email_tools]}")
    
    if not email_tools:
        logger.warning("Email node: No email tools available!")
        return {
            "messages": [AIMessage(content="I'm sorry, I don't have access to your email right now.")],
            "emotional_state": "apologetic",
        }
    
    # Maximum iterations to prevent infinite loops
    MAX_ITERATIONS = 5
    all_tool_results = []
    
    try:
        llm = create_email_llm(config, email_tools)
        messages = build_email_messages(state)
        
        # Agentic loop - continue until no more tool calls or max iterations
        for iteration in range(MAX_ITERATIONS):
            logger.info(f"Email node: Iteration {iteration + 1}/{MAX_ITERATIONS}")
            
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
                logger.info(f"Email node: Tool calls requested: {[tc['name'] for tc in tool_calls]}")
            
            if not tool_calls:
                # No more tool calls - LLM is ready to respond
                response_content = response.content
                if not response_content or response_content.strip() == "":
                    response_content = "I checked your email but couldn't format the response properly."
                
                logger.info(f"Email node: Final response after {iteration + 1} iterations: {response_content[:100]}...")
                
                return {
                    "messages": [AIMessage(content=response_content)],
                    "tool_results": all_tool_results,
                    "emotional_state": "helpful",
                }
            
            # Execute tool calls
            logger.info(f"Email node: Executing {len(tool_calls)} tool calls")
            tool_results = await execute_tools(tool_calls, email_tools)
            all_tool_results.extend(tool_results)
            
            # Add assistant message with tool calls and tool results to message history
            messages.append(response)
            for result in tool_results:
                messages.append(ToolMessage(
                    content=str(result.get("result", "")),
                    tool_call_id=result.get("id", ""),
                ))
        
        # Max iterations reached - force a response
        logger.warning(f"Email node: Max iterations ({MAX_ITERATIONS}) reached, forcing response")
        
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
            "messages": [AIMessage(content=final_response.content or "I checked your email.")],
            "tool_results": all_tool_results,
            "emotional_state": "helpful",
        }
            
    except Exception as e:
        logger.error(f"Email node error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "messages": [AIMessage(content="I had trouble accessing your email. Please try again.")],
            "emotional_state": "apologetic",
        }
