# Windows Filesystem MCP Server

A standalone MCP (Model Context Protocol) server that provides file system operations for AI agents. This server enables semantic file organization by exposing tools to list, read, move, rename, and manage files on Windows systems.

## Overview

This MCP server is designed to run on a Windows machine and be accessed remotely by LangGraph agents via SSE (Server-Sent Events) transport. The agent uses its LLM to understand file contents semantically and make intelligent organization decisions, while this server handles the actual file operations.

## Features

- **List directories** with rich metadata (sizes, dates, types, text previews)
- **Read file contents** for semantic analysis by the agent
- **Move and rename files** with safety checks
- **Create directories** for folder organization
- **Delete files** with confirmation safeguards
- **Security controls**: Only operates within configured root directories

## Prerequisites

- Windows 10/11 or Windows Server
- Python 3.11 or higher
- Network connectivity to the agent machine (Tailscale, VPN, or direct)

## Installation

### Option 1: Using uv (Recommended)

1. **Install uv** (if not already installed):
   ```cmd
   pip install uv
   ```

2. **Copy this folder** to your Windows machine:
   ```
   mcp_servers/windows_filesystem/
   ```

3. **Configure allowed directories** (optional):
   ```cmd
   set MCP_ALLOWED_ROOTS=C:\Users\YourName\Desktop,C:\Users\YourName\Downloads,C:\Users\YourName\Documents
   ```

4. **Run the server**:
   ```cmd
   start_server.bat
   ```

### Option 2: Manual Installation

1. **Create virtual environment**:
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```cmd
   pip install -e .
   ```

3. **Run the server**:
   ```cmd
   python server.py
   ```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_SERVER_PORT` | `8765` | Port the server listens on |
| `MCP_ALLOWED_ROOTS` | Desktop, Downloads, Documents | Comma-separated list of allowed root directories |

### Example Configuration

```cmd
set MCP_SERVER_PORT=8765
set MCP_ALLOWED_ROOTS=C:\Users\Alice\Desktop,C:\Users\Alice\Downloads,D:\Projects
start_server.bat
```

## Security

The server includes several security measures:

1. **Root Directory Restriction**: Only operates within directories specified in `MCP_ALLOWED_ROOTS`
2. **System File Protection**: Skips hidden files, system directories, and common protected patterns
3. **Delete Confirmation**: Requires explicit `confirm=true` parameter to delete files
4. **Non-Empty Directory Protection**: Cannot delete directories that contain files
5. **Audit Logging**: All operations are logged to `mcp_server.log`

### Protected Patterns

The following are automatically skipped:
- `$RECYCLE.BIN`, `System Volume Information`
- `.git`, `__pycache__`, `node_modules`
- Hidden files (starting with `.`)
- System files like `Thumbs.db`, `desktop.ini`

## API / Tools

### list_directory

List files and folders with metadata.

```json
{
  "path": "C:\\Users\\Name\\Desktop",
  "recursive": false,
  "include_hidden": false,
  "max_depth": 3,
  "include_preview": true
}
```

### read_file_content

Read file content for semantic analysis.

```json
{
  "path": "C:\\Users\\Name\\Desktop\\document.txt",
  "max_bytes": 1048576,
  "encoding": "utf-8"
}
```

### get_file_info

Get detailed metadata for a file or directory.

```json
{
  "path": "C:\\Users\\Name\\Desktop\\folder"
}
```

### move_file

Move a file or directory to a new location.

```json
{
  "source": "C:\\Users\\Name\\Desktop\\file.txt",
  "destination": "C:\\Users\\Name\\Documents\\file.txt",
  "overwrite": false
}
```

### rename_file

Rename a file or directory in place.

```json
{
  "path": "C:\\Users\\Name\\Desktop\\old_name.txt",
  "new_name": "new_name.txt"
}
```

### create_directory

Create a new directory.

```json
{
  "path": "C:\\Users\\Name\\Desktop\\NewFolder",
  "parents": true
}
```

### delete_file

Delete a file or empty directory.

```json
{
  "path": "C:\\Users\\Name\\Desktop\\file_to_delete.txt",
  "confirm": true
}
```

## Agent-Side Configuration

On the machine running the LangGraph agent, set these environment variables:

```bash
export WINDOWS_FS_MCP_HOST=192.168.1.100  # Windows machine IP or Tailscale hostname
export WINDOWS_FS_MCP_PORT=8765
export WINDOWS_FS_MCP_ENABLED=true

# Optional: Limit which tools are available
export WINDOWS_FS_ENABLED_TOOLS=list_directory,move_file,create_directory,rename_file
```

## Network Setup

### Using Tailscale (Recommended)

1. Install Tailscale on both machines
2. Use the Tailscale hostname or IP as `WINDOWS_FS_MCP_HOST`
3. No firewall configuration needed

### Direct Connection

1. Open port 8765 in Windows Firewall:
   ```cmd
   netsh advfirewall firewall add rule name="MCP Server" dir=in action=allow protocol=tcp localport=8765
   ```

2. Use the Windows machine's IP address as `WINDOWS_FS_MCP_HOST`

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/sse` | GET | SSE connection for MCP protocol |
| `/messages` | POST | Message endpoint for MCP |
| `/health` | GET | Health check (returns JSON status) |

## Logging

Logs are written to:
- **Console**: INFO level and above
- **File**: `mcp_server.log` (DEBUG level, rotates at 10MB, keeps 7 days)

## Troubleshooting

### Server won't start

1. Check Python version: `python --version` (must be 3.11+)
2. Check if port is in use: `netstat -an | findstr :8765`
3. Try a different port: `set MCP_SERVER_PORT=9000`

### Agent can't connect

1. Verify server is running: `curl http://<ip>:8765/health`
2. Check firewall settings
3. Verify network connectivity with `ping`
4. Check Tailscale status if using it

### Permission errors

1. Verify the target directory is in `MCP_ALLOWED_ROOTS`
2. Check Windows file permissions
3. Run server as Administrator if needed (not recommended for security)

### Files not appearing

1. Check if files are hidden (use `include_hidden: true`)
2. Verify path is correct (use full Windows paths)
3. Check `mcp_server.log` for errors

## Running as a Windows Service

To run the server automatically on startup:

1. Install NSSM (Non-Sucking Service Manager):
   ```cmd
   winget install nssm
   ```

2. Create the service:
   ```cmd
   nssm install MCPFilesystem "C:\path\to\windows_filesystem\start_server.bat"
   nssm set MCPFilesystem DisplayName "MCP Filesystem Server"
   nssm set MCPFilesystem Description "File organization MCP server for AI agents"
   ```

3. Start the service:
   ```cmd
   nssm start MCPFilesystem
   ```

## License

This project is part of the Reachy Personal Assistant and follows the same license.
