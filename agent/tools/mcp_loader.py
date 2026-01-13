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
        self._client = None
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
        
        The client connection is kept alive - call close() when done.
        
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
            # New API in langchain-mcp-adapters 0.2+ requires 'transport' key
            server_params = {}
            for config in enabled_servers:
                server_config = {
                    "transport": "stdio",  # Required in new API
                    "command": config.command[0],
                    "args": config.command[1:] if len(config.command) > 1 else [],
                }
                if config.env:
                    server_config["env"] = config.env
                server_params[config.name] = server_config
            
            logger.info(f"MCP server configs: {list(server_params.keys())}")
            
            # Create client and get tools (new API - no context manager)
            self._client = MultiServerMCPClient(server_params)
            self._tools = await self._client.get_tools()
            logger.info(f"Loaded {len(self._tools)} tools from MCP servers")
            
            for tool in self._tools:
                desc = tool.description[:50] if tool.description else "No description"
                logger.info(f"  - {tool.name}: {desc}...")
            
            self._loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load MCP tools: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self._tools = []
            self._client = None
        
        return self._tools
    
    async def close(self):
        """Clean up MCP client connections."""
        if self._client:
            try:
                # New API doesn't require explicit close
                if hasattr(self._client, 'close'):
                    await self._client.close()
            except Exception as e:
                logger.warning(f"Error closing MCP client: {e}")
            self._client = None
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
    
    Args:
        credentials_path: Path to OAuth credentials file (optional, uses default location)
        enabled: Whether the server is enabled
        
    Returns:
        MCP server configuration
    """
    env = {}
    if credentials_path:
        env["GOOGLE_OAUTH_CREDENTIALS"] = credentials_path
    
    return MCPServerConfig(
        name="google_calendar",
        command=["npx", "-y", "@cocal/google-calendar-mcp"],
        env=env,
        description="Google Calendar access for scheduling, events, and reminders",
        enabled=enabled,
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
    - read_email: Read email content by ID
    - search_emails: Search using Gmail query syntax
    - list_labels: List all Gmail labels
    - create_label, update_label, delete_label: Manage labels
    
    Prerequisites:
    1. Set up Google Cloud project with Gmail API enabled
    2. Create OAuth credentials (Desktop app)
    3. Save credentials to ~/.gmail-mcp/gcp-oauth.keys.json
    4. Run: npx @gongrzhe/server-gmail-autoauth-mcp auth
    
    Returns:
        MCP server configuration
    """
    return MCPServerConfig(
        name="gmail",
        command=["npx", "-y", "@gongrzhe/server-gmail-autoauth-mcp"],
        env={},  # Uses credentials from ~/.gmail-mcp/
        description="Gmail access for reading, sending, searching emails and managing labels",
        enabled=enabled,
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
