# Custom MCP Servers

This directory contains custom MCP (Model Context Protocol) servers for the Reachy Personal Assistant.

## Overview

MCP servers provide tools that the agent can use to interact with external services and data sources. The agent integrates with MCP servers using the `langchain-mcp-adapters` library.

## Available Servers

### Built-in (via npm)

These servers are installed automatically via npm:

- `@modelcontextprotocol/server-memory` - Persistent memory storage
- `@modelcontextprotocol/server-filesystem` - File system access

### Custom Servers

Place custom MCP server implementations in subdirectories here.

## Creating a Custom MCP Server

1. Create a new directory for your server (e.g., `personal_data/`)
2. Implement the MCP protocol using the official SDK
3. Add configuration in `agent/tools/mcp_loader.py`

### Example: Personal Data Server

```python
# personal_data/server.py
from mcp import Server
from mcp.types import Tool, Resource

server = Server("personal-data")

@server.tool()
async def get_contacts(query: str) -> str:
    """Search personal contacts."""
    # Implementation here
    pass

if __name__ == "__main__":
    server.run()
```

## Configuration

MCP servers are configured in `agent/tools/mcp_loader.py`:

```python
MCPServerConfig(
    name="personal_data",
    command=["python", "mcp_servers/personal_data/server.py"],
    description="Access to personal data",
    enabled=True,
)
```

## Security Notes

- The filesystem server is disabled by default for security
- OAuth tokens should be stored in `.env` file, not committed
- Review all MCP server code before enabling
