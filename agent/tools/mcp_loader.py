"""
MCP Server Integration for LangGraph.

Provides functionality to load tools from MCP (Model Context Protocol) servers
and integrate them with the LangGraph agent.

Supported MCP servers:
- @modelcontextprotocol/server-memory - Persistent memory storage
- @modelcontextprotocol/server-filesystem - Local file access
- Custom MCP servers via configuration
"""

import logging
import asyncio
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    
    name: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    description: str = ""


# Default MCP server configurations
DEFAULT_MCP_SERVERS = [
    MCPServerConfig(
        name="memory",
        command=["npx", "-y", "@modelcontextprotocol/server-memory"],
        description="Persistent memory storage for facts and preferences",
        enabled=True,
    ),
    MCPServerConfig(
        name="filesystem",
        command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "/home"],
        description="Access to local filesystem for reading files",
        enabled=False,  # Disabled by default for security
    ),
]


class MCPToolLoader:
    """
    Loads and manages tools from MCP servers.
    
    Uses langchain-mcp-adapters to convert MCP tools to LangChain format.
    """
    
    def __init__(self, server_configs: Optional[list[MCPServerConfig]] = None):
        """
        Initialize MCP tool loader.
        
        Args:
            server_configs: List of MCP server configurations
        """
        self._configs = server_configs or DEFAULT_MCP_SERVERS
        self._clients = {}
        self._tools = []
        self._loaded = False
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    @property
    def tools(self) -> list:
        return self._tools
    
    async def load_tools(self) -> list:
        """
        Load tools from all configured MCP servers.
        
        Returns:
            List of LangChain-compatible tools
        """
        if self._loaded:
            return self._tools
        
        self._tools = []
        
        # Check if langchain-mcp-adapters is available
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError:
            logger.warning(
                "langchain-mcp-adapters not installed. "
                "MCP tools will not be available. "
                "Install with: pip install langchain-mcp-adapters"
            )
            return self._tools
        
        # Build server configurations for the client
        enabled_servers = [
            cfg for cfg in self._configs 
            if cfg.enabled
        ]
        
        if not enabled_servers:
            logger.info("No MCP servers enabled")
            return self._tools
        
        logger.info(f"Loading tools from {len(enabled_servers)} MCP servers...")
        
        try:
            # Configure servers for MultiServerMCPClient
            server_params = {}
            for config in enabled_servers:
                server_params[config.name] = {
                    "command": config.command[0],
                    "args": config.command[1:] if len(config.command) > 1 else [],
                    "env": config.env,
                }
            
            # Create client and get tools
            async with MultiServerMCPClient(server_params) as client:
                self._tools = client.get_tools()
                logger.info(f"Loaded {len(self._tools)} tools from MCP servers")
                
                for tool in self._tools:
                    logger.debug(f"  - {tool.name}: {tool.description[:50]}...")
            
            self._loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load MCP tools: {e}")
            self._tools = []
        
        return self._tools
    
    async def close(self):
        """Clean up MCP client connections."""
        self._clients.clear()
        self._loaded = False


async def load_mcp_tools(
    configs: Optional[list[MCPServerConfig]] = None
) -> list:
    """
    Convenience function to load MCP tools.
    
    Args:
        configs: Optional list of MCP server configurations
        
    Returns:
        List of LangChain-compatible tools
    """
    loader = MCPToolLoader(configs)
    return await loader.load_tools()


def create_calendar_mcp_config(oauth_token: Optional[str] = None) -> MCPServerConfig:
    """
    Create configuration for Google Calendar MCP server.
    
    Note: This requires setting up OAuth credentials.
    
    Args:
        oauth_token: OAuth token for Google Calendar API
        
    Returns:
        MCP server configuration
    """
    env = {}
    if oauth_token:
        env["GOOGLE_OAUTH_TOKEN"] = oauth_token
    
    return MCPServerConfig(
        name="google_calendar",
        command=["npx", "-y", "@anthropic/mcp-google-calendar"],
        env=env,
        description="Google Calendar access for scheduling and reminders",
        enabled=bool(oauth_token),
    )


def create_github_mcp_config(token: Optional[str] = None) -> MCPServerConfig:
    """
    Create configuration for GitHub MCP server.
    
    Args:
        token: GitHub personal access token
        
    Returns:
        MCP server configuration
    """
    env = {}
    if token:
        env["GITHUB_TOKEN"] = token
    
    return MCPServerConfig(
        name="github",
        command=["npx", "-y", "@anthropic/mcp-github"],
        env=env,
        description="GitHub repository access",
        enabled=bool(token),
    )


# Directory for custom MCP servers
MCP_SERVERS_DIR = "mcp_servers"


def get_all_mcp_configs() -> list[MCPServerConfig]:
    """Get all available MCP server configurations."""
    configs = list(DEFAULT_MCP_SERVERS)
    
    # Add custom configurations from environment
    import os
    
    if os.getenv("GOOGLE_OAUTH_TOKEN"):
        configs.append(create_calendar_mcp_config(os.getenv("GOOGLE_OAUTH_TOKEN")))
    
    if os.getenv("GITHUB_TOKEN"):
        configs.append(create_github_mcp_config(os.getenv("GITHUB_TOKEN")))
    
    return configs
