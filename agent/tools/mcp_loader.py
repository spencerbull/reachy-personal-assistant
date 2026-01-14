"""
MCP Server Integration for LangGraph.

Provides functionality to load tools from MCP (Model Context Protocol) servers
and integrate them with the LangGraph agent.

Supported MCP servers:
- @modelcontextprotocol/server-memory - Persistent memory storage
- @modelcontextprotocol/server-filesystem - Local file access
- Custom MCP servers via configuration
"""

import asyncio
from typing import Optional
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    
    name: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    description: str = ""
    enabled_tools: Optional[list[str]] = None  # If set, only these tools will be loaded from this server


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
    The client connection is kept alive for the lifetime of this loader.
    """
    
    def __init__(self, server_configs: Optional[list[MCPServerConfig]] = None):
        """
        Initialize MCP tool loader.
        
        Args:
            server_configs: List of MCP server configurations
        """
        self._configs = server_configs or DEFAULT_MCP_SERVERS
        self._clients: list = []  # One client per server for proper filtering
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
        
        Each server is loaded individually so we can filter its tools based on
        the enabled_tools config before combining them.
        
        The client connections are kept alive - call close() when done.
        
        Returns:
            List of LangChain-compatible tools
        """
        if self._loaded:
            return self._tools
        
        self._tools = []
        self._clients = []  # Track all clients for cleanup
        
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
        
        # Load each server individually so we can filter per-server
        for config in enabled_servers:
            try:
                # Build single-server config
                server_config = {
                    "transport": "stdio",
                    "command": config.command[0],
                    "args": config.command[1:] if len(config.command) > 1 else [],
                }
                if config.env:
                    server_config["env"] = config.env
                
                server_params = {config.name: server_config}
                
                logger.info(f"Loading tools from server: {config.name}")
                
                # Create client and get tools for this server
                client = MultiServerMCPClient(server_params)
                server_tools = await client.get_tools()
                self._clients.append(client)  # Keep reference for cleanup
                
                logger.info(f"  Server '{config.name}' returned {len(server_tools)} tools")
                
                # Apply filtering if enabled_tools is set for this server
                if config.enabled_tools is not None:
                    enabled_set = set(config.enabled_tools)
                    filtered_tools = [t for t in server_tools if t.name in enabled_set]
                    excluded = [t.name for t in server_tools if t.name not in enabled_set]
                    
                    logger.info(f"  Filtering '{config.name}': keeping {len(filtered_tools)}, excluding {len(excluded)}")
                    if excluded:
                        logger.info(f"  Excluded tools: {excluded}")
                    
                    self._tools.extend(filtered_tools)
                else:
                    # No filtering - include all tools from this server
                    self._tools.extend(server_tools)
                
            except Exception as e:
                logger.error(f"Failed to load tools from server '{config.name}': {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info(f"Final tool count: {len(self._tools)} tools")
        
        for tool in self._tools:
            desc = tool.description[:50] if tool.description else "No description"
            logger.info(f"  - {tool.name}: {desc}...")
        
        self._loaded = True
        
        return self._tools
    
    async def close(self):
        """Clean up MCP client connections."""
        for client in self._clients:
            try:
                # New API doesn't require explicit close
                if hasattr(client, 'close'):
                    await client.close()
            except Exception as e:
                logger.warning(f"Error closing MCP client: {e}")
        self._clients = []
        self._tools = []
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


def create_calendar_mcp_config(credentials_path: Optional[str] = None, enabled: bool = True) -> MCPServerConfig:
    """
    Create configuration for Google Calendar MCP server.
    
    Uses the @cocal/google-calendar-mcp package which provides:
    - list-calendars: List all available calendars
    - list-events: List events with date filtering
    - get-event: Get details of a specific event by ID
    - search-events: Search events by text query
    - create-event: Create new calendar events
    - update-event: Update existing events
    - delete-event: Delete events
    - respond-to-event: Respond to event invitations
    - get-freebusy: Check availability across calendars
    - get-current-time: Get current date and time
    - list-colors: List available event colors
    - manage-accounts: Add, list, or remove connected Google accounts
    
    Prerequisites:
    1. Set up Google Cloud project with Calendar API enabled
    2. Create OAuth credentials (Desktop app)
    3. Run: npx @cocal/google-calendar-mcp auth
    4. Complete OAuth flow in browser
    5. Tokens saved to ~/.config/google-calendar-mcp/tokens.json
    
    See: https://github.com/nspady/google-calendar-mcp
    
    Environment Variables:
        GCAL_ENABLED_TOOLS: Comma-separated list of tool names to enable.
            Example: GCAL_ENABLED_TOOLS=list-events,create-event,get-current-time,update-event
            If not set, all tools from the server will be available.
    
    Args:
        credentials_path: Path to OAuth credentials file (optional, uses default location)
        enabled: Whether the server is enabled
        
    Returns:
        MCP server configuration
    """
    import os
    
    env = {}
    if credentials_path:
        env["GOOGLE_OAUTH_CREDENTIALS"] = credentials_path
    
    # Parse GCAL_ENABLED_TOOLS environment variable for tool filtering
    # If not set or empty, all tools will be available (no filtering)
    enabled_tools = None
    gcal_tools_env = os.getenv("GCAL_ENABLED_TOOLS")
    if gcal_tools_env:
        parsed_tools = [tool.strip() for tool in gcal_tools_env.split(",") if tool.strip()]
        if parsed_tools:  # Only set if we have actual tool names
            enabled_tools = parsed_tools
            logger.info(f"Google Calendar MCP tools filtered to: {enabled_tools}")
    
    return MCPServerConfig(
        name="google_calendar",
        command=["npx", "-y", "@cocal/google-calendar-mcp"],
        env=env,
        description="Google Calendar access for scheduling, events, and reminders",
        enabled=enabled,
        enabled_tools=enabled_tools,
    )


def is_calendar_configured() -> bool:
    """
    Check if Google Calendar MCP credentials are configured.
    
    The @cocal/google-calendar-mcp package stores tokens at:
    ~/.config/google-calendar-mcp/tokens.json
    
    Returns:
        True if tokens exist (authenticated)
    """
    import os
    from pathlib import Path
    
    home = Path.home()
    
    # Primary location used by @cocal/google-calendar-mcp
    config_dir = home / ".config" / "google-calendar-mcp"
    tokens_file = config_dir / "tokens.json"
    
    if tokens_file.exists():
        logger.info(f"Google Calendar tokens found at {tokens_file}")
        return True
    
    # Check environment variable for custom credentials path
    if os.getenv("GOOGLE_OAUTH_CREDENTIALS"):
        creds_path = Path(os.getenv("GOOGLE_OAUTH_CREDENTIALS"))
        if creds_path.exists():
            logger.info(f"Google Calendar credentials found at {creds_path}")
            return True
    
    # Check custom token path if set
    if os.getenv("GOOGLE_CALENDAR_MCP_TOKEN_PATH"):
        token_path = Path(os.getenv("GOOGLE_CALENDAR_MCP_TOKEN_PATH"))
        if token_path.exists():
            return True
    
    return False


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


def create_gmail_mcp_config(enabled: bool = True) -> MCPServerConfig:
    """
    Create configuration for Gmail MCP server.
    
    Uses the @gongrzhe/server-gmail-autoauth-mcp package which provides:
    - send_email: Send emails with attachments
    - draft_email: Create email drafts
    - read_email: Read email content by ID
    - search_emails: Search using Gmail query syntax
    - list_labels: List all Gmail labels
    - create_label, update_label, delete_label: Manage labels
    
    Prerequisites:
    1. Set up Google Cloud project with Gmail API enabled
    2. Create OAuth credentials (Desktop app)
    3. Save credentials to ~/.gmail-mcp/gcp-oauth.keys.json
    4. Run: npx @gongrzhe/server-gmail-autoauth-mcp auth
    
    Environment Variables:
        GMAIL_ENABLED_TOOLS: Comma-separated list of tool names to enable.
            Example: GMAIL_ENABLED_TOOLS=send_email,draft_email,read_email,search_emails
            If not set, all tools from the server will be available.
    
    Returns:
        MCP server configuration
    """
    import os
    
    # Parse GMAIL_ENABLED_TOOLS environment variable for tool filtering
    # If not set or empty, all tools will be available (no filtering)
    enabled_tools = None
    gmail_tools_env = os.getenv("GMAIL_ENABLED_TOOLS")
    if gmail_tools_env:
        parsed_tools = [tool.strip() for tool in gmail_tools_env.split(",") if tool.strip()]
        if parsed_tools:  # Only set if we have actual tool names
            enabled_tools = parsed_tools
            logger.info(f"Gmail MCP tools filtered to: {enabled_tools}")
    
    return MCPServerConfig(
        name="gmail",
        command=["npx", "-y", "@gongrzhe/server-gmail-autoauth-mcp"],
        env={},  # Uses credentials from ~/.gmail-mcp/
        description="Gmail access for reading, sending, searching emails and managing labels",
        enabled=enabled,
        enabled_tools=enabled_tools,
    )


def is_gmail_configured() -> bool:
    """
    Check if Gmail MCP credentials are configured.
    
    Returns:
        True if credentials exist in ~/.gmail-mcp/
    """
    import os
    from pathlib import Path
    
    gmail_dir = Path.home() / ".gmail-mcp"
    credentials_file = gmail_dir / "credentials.json"
    oauth_keys_file = gmail_dir / "gcp-oauth.keys.json"
    
    # Check if authenticated (credentials.json exists)
    if credentials_file.exists():
        return True
    
    # Check if OAuth keys exist (can authenticate)
    if oauth_keys_file.exists():
        logger.info("Gmail OAuth keys found but not authenticated. Run: npx @gongrzhe/server-gmail-autoauth-mcp auth")
        return False
    
    return False


# Directory for custom MCP servers
MCP_SERVERS_DIR = "mcp_servers"


def get_all_mcp_configs() -> list[MCPServerConfig]:
    """Get all available MCP server configurations."""
    configs = list(DEFAULT_MCP_SERVERS)
    
    # Add custom configurations from environment
    import os
    
    # Add Google Calendar MCP if configured
    if is_calendar_configured():
        creds_path = os.getenv("GOOGLE_OAUTH_CREDENTIALS")
        configs.append(create_calendar_mcp_config(credentials_path=creds_path, enabled=True))
        logger.info("Google Calendar MCP server enabled")
    elif os.getenv("CALENDAR_MCP_ENABLED", "").lower() == "true":
        # Allow enabling via environment variable even if not authenticated yet
        creds_path = os.getenv("GOOGLE_OAUTH_CREDENTIALS")
        configs.append(create_calendar_mcp_config(credentials_path=creds_path, enabled=True))
        logger.warning("Google Calendar MCP enabled via env var but may not be authenticated")
    
    if os.getenv("GITHUB_TOKEN"):
        configs.append(create_github_mcp_config(os.getenv("GITHUB_TOKEN")))
    
    # Add Gmail MCP if configured
    if is_gmail_configured():
        configs.append(create_gmail_mcp_config(enabled=True))
        logger.info("Gmail MCP server enabled")
    elif os.getenv("GMAIL_MCP_ENABLED", "").lower() == "true":
        # Allow enabling via environment variable even if not authenticated yet
        configs.append(create_gmail_mcp_config(enabled=True))
        logger.warning("Gmail MCP enabled via env var but may not be authenticated")
    
    return configs
