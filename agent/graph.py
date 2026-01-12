"""
Main LangGraph definition for the Reachy Personal Assistant.

This module defines the StateGraph that orchestrates the agent's behavior,
including routing, conversation, vision understanding, and tool execution.
"""

import logging
from typing import Callable, Optional
from functools import partial

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from agent.state import ReachyAgentState, create_initial_state
from agent.config import AgentConfig
from agent.nodes.router import router_node, get_next_node
from agent.nodes.conversation import conversation_node
from agent.nodes.vision import vision_node
from agent.nodes.tools import tools_node
from agent.tools.reachy_tools import get_all_reachy_tools
from agent.tools.memory_tools import get_all_memory_tools

logger = logging.getLogger(__name__)


def create_graph(
    config: Optional[AgentConfig] = None,
    checkpointer: Optional[MemorySaver] = None,
    additional_tools: Optional[list] = None,
) -> CompiledStateGraph:
    """
    Create the LangGraph for the Reachy Personal Assistant.
    
    Args:
        config: Agent configuration (uses defaults if not provided)
        checkpointer: LangGraph checkpointer for persistence (uses MemorySaver if not provided)
        additional_tools: Additional tools to include (e.g., MCP tools)
        
    Returns:
        Compiled StateGraph ready for invocation
    """
    config = config or AgentConfig.from_env()
    
    # Collect all tools
    all_tools = get_all_reachy_tools() + get_all_memory_tools()
    if additional_tools:
        all_tools.extend(additional_tools)
    
    logger.info(f"Creating graph with {len(all_tools)} tools")
    
    # Create node functions with config and tools bound
    async def router_with_config(state: ReachyAgentState) -> dict:
        return await router_node(state, config)
    
    async def conversation_with_config(state: ReachyAgentState) -> dict:
        return await conversation_node(state, config)
    
    async def vision_with_config(state: ReachyAgentState) -> dict:
        return await vision_node(state, config)
    
    async def tools_with_config(state: ReachyAgentState) -> dict:
        return await tools_node(state, config, all_tools)
    
    # Build the graph
    builder = StateGraph(ReachyAgentState)
    
    # Add nodes
    builder.add_node("router", router_with_config)
    builder.add_node("conversation", conversation_with_config)
    builder.add_node("vision", vision_with_config)
    builder.add_node("tools", tools_with_config)
    
    # Add edges
    # Start -> Router
    builder.add_edge(START, "router")
    
    # Router -> Conditional routing based on intent
    builder.add_conditional_edges(
        "router",
        get_next_node,
        {
            "conversation": "conversation",
            "vision": "vision",
            "tools": "tools",
        }
    )
    
    # All processing nodes -> END
    builder.add_edge("conversation", END)
    builder.add_edge("vision", END)
    builder.add_edge("tools", END)
    
    # Compile the graph
    # Note: When running via LangGraph CLI (langgraph dev), persistence is 
    # handled automatically by the platform. Only use checkpointer for 
    # standalone/local usage.
    if checkpointer is not None:
        graph = builder.compile(checkpointer=checkpointer)
    else:
        graph = builder.compile()
    
    logger.info("LangGraph compiled successfully")
    return graph


def create_graph_with_mcp(
    config: Optional[AgentConfig] = None,
    mcp_tools: Optional[list] = None,
) -> CompiledStateGraph:
    """
    Create the LangGraph with MCP server tools included.
    
    This is a convenience function that initializes MCP tools and
    creates the graph with them included.
    
    Args:
        config: Agent configuration
        mcp_tools: Pre-loaded MCP tools (if not provided, loads from config)
        
    Returns:
        Compiled StateGraph with MCP tools
    """
    config = config or AgentConfig.from_env()
    
    # MCP tools would be loaded here if not provided
    # For now, just pass them through
    additional_tools = mcp_tools or []
    
    return create_graph(config=config, additional_tools=additional_tools)


# Create a default graph instance for simple usage
# This will be used when the module is imported directly or by langgraph CLI
# Note: No checkpointer for CLI usage - the platform handles persistence
_default_config = AgentConfig.from_env()

# Create the default graph (without checkpointer for CLI compatibility)
graph = create_graph(config=_default_config, checkpointer=None)


async def invoke_agent(
    message: str,
    image: Optional[str] = None,
    thread_id: str = "default",
    graph_instance: Optional[CompiledStateGraph] = None,
) -> dict:
    """
    Convenience function to invoke the agent with a message.
    
    Args:
        message: User message text
        image: Optional base64-encoded image from camera
        thread_id: Thread ID for conversation persistence
        graph_instance: Optional specific graph instance (uses default if not provided)
        
    Returns:
        Agent response including messages and state updates
    """
    from langchain_core.messages import HumanMessage
    
    graph_to_use = graph_instance or graph
    
    # Build input state
    input_state = {
        "messages": [HumanMessage(content=message)],
        "current_image": image,
    }
    
    # Invoke the graph
    config = {"configurable": {"thread_id": thread_id}}
    
    result = await graph_to_use.ainvoke(input_state, config)
    
    return result


def get_graph_visualization() -> str:
    """
    Get a Mermaid diagram representation of the graph.
    
    Returns:
        Mermaid diagram string
    """
    try:
        return graph.get_graph().draw_mermaid()
    except Exception as e:
        logger.warning(f"Could not generate graph visualization: {e}")
        return """
graph TD
    START --> router
    router -->|conversation| conversation
    router -->|vision| vision
    router -->|tools| tools
    conversation --> END
    vision --> END
    tools --> END
"""


# Export for langgraph.json configuration
__all__ = ["graph", "create_graph", "create_graph_with_mcp", "invoke_agent"]
