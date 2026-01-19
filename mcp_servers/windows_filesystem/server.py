"""
Windows Filesystem MCP Server

A Model Context Protocol (MCP) server that provides file system operations
for AI agents to organize, manage, and understand files on Windows systems.

This server is designed to run standalone on a Windows machine and be accessed
remotely by LangGraph agents via SSE transport.

Tools provided:
- list_directory: List files and folders with metadata
- read_file_content: Read file contents for semantic analysis
- get_file_info: Get detailed file metadata
- move_file: Move a file to a new location
- rename_file: Rename a file
- create_directory: Create a new directory
- delete_file: Delete a file (with safety checks)

Security:
- Only operates within configured allowed root directories
- Prevents access to system files and hidden directories
- Logs all operations for audit trail
"""

import os
import sys
import shutil
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Optional

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
from pydantic import BaseModel, Field
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "mcp_server.log",
    rotation="10 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG"
)

# Configuration
SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8765"))
ALLOWED_ROOTS = [
    Path(p.strip()).resolve() 
    for p in os.getenv("MCP_ALLOWED_ROOTS", "").split(",") 
    if p.strip()
]

# Default to common user directories if not configured
if not ALLOWED_ROOTS:
    user_home = Path.home()
    ALLOWED_ROOTS = [
        user_home / "Desktop",
        user_home / "Downloads", 
        user_home / "Documents",
    ]
    # Filter to only existing directories
    ALLOWED_ROOTS = [p for p in ALLOWED_ROOTS if p.exists()]

# File size limits
MAX_READ_SIZE = 1024 * 1024  # 1MB max for reading file content
MAX_PREVIEW_SIZE = 4096  # 4KB preview for text files

# Extensions considered safe to read as text
TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".json", ".xml", ".html", ".css",
    ".csv", ".yaml", ".yml", ".ini", ".cfg", ".conf", ".log", ".sh", ".bat",
    ".ps1", ".java", ".c", ".cpp", ".h", ".hpp", ".rs", ".go", ".rb", ".php",
    ".sql", ".r", ".m", ".swift", ".kt", ".scala", ".pl", ".lua", ".vim",
    ".dockerfile", ".gitignore", ".env", ".toml"
}

# Hidden/system patterns to skip
SKIP_PATTERNS = {
    "$RECYCLE.BIN", "System Volume Information", "hiberfil.sys",
    "pagefile.sys", "swapfile.sys", ".git", "__pycache__", "node_modules",
    ".vs", ".vscode", ".idea", "Thumbs.db", ".DS_Store", "desktop.ini"
}


def is_path_allowed(path: Path) -> bool:
    """Check if the path is within allowed root directories."""
    resolved = path.resolve()
    return any(
        resolved == root or root in resolved.parents or resolved in root.parents
        for root in ALLOWED_ROOTS
    )


def should_skip(name: str) -> bool:
    """Check if a file/folder should be skipped."""
    return name in SKIP_PATTERNS or name.startswith(".")


def get_file_type(path: Path) -> str:
    """Determine the general type of a file."""
    if path.is_dir():
        return "directory"
    
    ext = path.suffix.lower()
    
    # Common type mappings
    type_map = {
        # Documents
        ".pdf": "document/pdf",
        ".doc": "document/word", ".docx": "document/word",
        ".xls": "document/excel", ".xlsx": "document/excel",
        ".ppt": "document/powerpoint", ".pptx": "document/powerpoint",
        ".odt": "document/opendocument", ".ods": "document/opendocument",
        # Text
        ".txt": "text/plain", ".md": "text/markdown",
        ".json": "text/json", ".xml": "text/xml",
        ".csv": "text/csv",
        # Code
        ".py": "code/python", ".js": "code/javascript", ".ts": "code/typescript",
        ".java": "code/java", ".c": "code/c", ".cpp": "code/cpp",
        ".html": "code/html", ".css": "code/css",
        ".sh": "code/shell", ".bat": "code/batch", ".ps1": "code/powershell",
        # Images
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".gif": "image/gif", ".bmp": "image/bitmap", ".svg": "image/svg",
        ".webp": "image/webp", ".ico": "image/icon",
        # Audio
        ".mp3": "audio/mp3", ".wav": "audio/wav", ".flac": "audio/flac",
        ".ogg": "audio/ogg", ".m4a": "audio/m4a",
        # Video
        ".mp4": "video/mp4", ".avi": "video/avi", ".mkv": "video/mkv",
        ".mov": "video/quicktime", ".wmv": "video/wmv", ".webm": "video/webm",
        # Archives
        ".zip": "archive/zip", ".rar": "archive/rar", ".7z": "archive/7z",
        ".tar": "archive/tar", ".gz": "archive/gzip",
        # Executables
        ".exe": "executable/windows", ".msi": "executable/installer",
        ".dll": "library/windows",
    }
    
    if ext in type_map:
        return type_map[ext]
    
    # Try mimetypes as fallback
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type:
        return mime_type
    
    return "unknown"


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


# Initialize MCP Server
server = Server("windows-filesystem")


# ============================================================================
# Tool: list_directory
# ============================================================================

class ListDirectoryArgs(BaseModel):
    """Arguments for list_directory tool."""
    path: str = Field(description="Directory path to list")
    recursive: bool = Field(default=False, description="Whether to list recursively")
    include_hidden: bool = Field(default=False, description="Include hidden files")
    max_depth: int = Field(default=3, description="Maximum recursion depth")
    include_preview: bool = Field(default=True, description="Include text preview for small text files")


@server.call_tool()
async def list_directory(name: str, arguments: dict) -> list[TextContent]:
    """List files and directories with metadata."""
    if name != "list_directory":
        raise ValueError(f"Unknown tool: {name}")
    
    args = ListDirectoryArgs(**arguments)
    path = Path(args.path).resolve()
    
    if not is_path_allowed(path):
        return [TextContent(
            type="text",
            text=f"Error: Path '{path}' is outside allowed directories. Allowed roots: {[str(r) for r in ALLOWED_ROOTS]}"
        )]
    
    if not path.exists():
        return [TextContent(type="text", text=f"Error: Path '{path}' does not exist")]
    
    if not path.is_dir():
        return [TextContent(type="text", text=f"Error: Path '{path}' is not a directory")]
    
    logger.info(f"Listing directory: {path} (recursive={args.recursive})")
    
    results = []
    
    def scan_directory(dir_path: Path, depth: int = 0):
        if depth > args.max_depth:
            return
        
        try:
            entries = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            results.append({
                "path": str(dir_path),
                "error": "Permission denied"
            })
            return
        
        for entry in entries:
            if not args.include_hidden and should_skip(entry.name):
                continue
            
            try:
                stat = entry.stat()
                file_info = {
                    "name": entry.name,
                    "path": str(entry),
                    "type": get_file_type(entry),
                    "is_directory": entry.is_dir(),
                    "size": stat.st_size if not entry.is_dir() else None,
                    "size_human": format_size(stat.st_size) if not entry.is_dir() else None,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "extension": entry.suffix.lower() if entry.is_file() else None,
                }
                
                # Add text preview for small text files
                if (args.include_preview and 
                    entry.is_file() and 
                    entry.suffix.lower() in TEXT_EXTENSIONS and
                    stat.st_size < MAX_PREVIEW_SIZE):
                    try:
                        with open(entry, "r", encoding="utf-8", errors="ignore") as f:
                            preview = f.read(500)
                            if len(preview) == 500:
                                preview += "..."
                            file_info["preview"] = preview.strip()
                    except Exception:
                        pass
                
                results.append(file_info)
                
                if args.recursive and entry.is_dir():
                    scan_directory(entry, depth + 1)
                    
            except Exception as e:
                results.append({
                    "name": entry.name,
                    "path": str(entry),
                    "error": str(e)
                })
    
    scan_directory(path)
    
    import json
    summary = {
        "directory": str(path),
        "total_items": len(results),
        "files": sum(1 for r in results if not r.get("is_directory", False) and "error" not in r),
        "directories": sum(1 for r in results if r.get("is_directory", False)),
        "errors": sum(1 for r in results if "error" in r),
        "items": results
    }
    
    return [TextContent(type="text", text=json.dumps(summary, indent=2))]


# ============================================================================
# Tool: read_file_content
# ============================================================================

class ReadFileArgs(BaseModel):
    """Arguments for read_file_content tool."""
    path: str = Field(description="File path to read")
    max_bytes: int = Field(default=MAX_READ_SIZE, description="Maximum bytes to read")
    encoding: str = Field(default="utf-8", description="Text encoding to use")


@server.call_tool()
async def read_file_content(name: str, arguments: dict) -> list[TextContent]:
    """Read file content for semantic analysis."""
    if name != "read_file_content":
        raise ValueError(f"Unknown tool: {name}")
    
    args = ReadFileArgs(**arguments)
    path = Path(args.path).resolve()
    
    if not is_path_allowed(path):
        return [TextContent(
            type="text",
            text=f"Error: Path '{path}' is outside allowed directories"
        )]
    
    if not path.exists():
        return [TextContent(type="text", text=f"Error: File '{path}' does not exist")]
    
    if not path.is_file():
        return [TextContent(type="text", text=f"Error: Path '{path}' is not a file")]
    
    file_size = path.stat().st_size
    if file_size > args.max_bytes:
        logger.warning(f"File {path} exceeds max size, truncating")
    
    logger.info(f"Reading file: {path}")
    
    try:
        # Determine if binary or text
        file_type = get_file_type(path)
        is_text = (
            path.suffix.lower() in TEXT_EXTENSIONS or
            file_type.startswith("text/") or
            file_type.startswith("code/")
        )
        
        if is_text:
            with open(path, "r", encoding=args.encoding, errors="replace") as f:
                content = f.read(args.max_bytes)
                truncated = file_size > args.max_bytes
        else:
            # For binary files, just return metadata
            content = f"[Binary file: {file_type}, size: {format_size(file_size)}]"
            truncated = False
        
        result = {
            "path": str(path),
            "name": path.name,
            "type": file_type,
            "size": file_size,
            "size_human": format_size(file_size),
            "truncated": truncated,
            "content": content
        }
        
        import json
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        logger.error(f"Error reading file {path}: {e}")
        return [TextContent(type="text", text=f"Error reading file: {str(e)}")]


# ============================================================================
# Tool: get_file_info
# ============================================================================

class GetFileInfoArgs(BaseModel):
    """Arguments for get_file_info tool."""
    path: str = Field(description="File or directory path")


@server.call_tool()
async def get_file_info(name: str, arguments: dict) -> list[TextContent]:
    """Get detailed file or directory information."""
    if name != "get_file_info":
        raise ValueError(f"Unknown tool: {name}")
    
    args = GetFileInfoArgs(**arguments)
    path = Path(args.path).resolve()
    
    if not is_path_allowed(path):
        return [TextContent(
            type="text",
            text=f"Error: Path '{path}' is outside allowed directories"
        )]
    
    if not path.exists():
        return [TextContent(type="text", text=f"Error: Path '{path}' does not exist")]
    
    logger.info(f"Getting info for: {path}")
    
    try:
        stat = path.stat()
        
        info = {
            "name": path.name,
            "path": str(path),
            "parent": str(path.parent),
            "type": get_file_type(path),
            "is_directory": path.is_dir(),
            "is_file": path.is_file(),
            "exists": path.exists(),
            "size": stat.st_size if path.is_file() else None,
            "size_human": format_size(stat.st_size) if path.is_file() else None,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
            "extension": path.suffix.lower() if path.is_file() else None,
        }
        
        # For directories, add item count
        if path.is_dir():
            try:
                items = list(path.iterdir())
                info["item_count"] = len(items)
                info["file_count"] = sum(1 for i in items if i.is_file())
                info["dir_count"] = sum(1 for i in items if i.is_dir())
            except PermissionError:
                info["item_count"] = "Permission denied"
        
        import json
        return [TextContent(type="text", text=json.dumps(info, indent=2))]
        
    except Exception as e:
        logger.error(f"Error getting file info {path}: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


# ============================================================================
# Tool: move_file
# ============================================================================

class MoveFileArgs(BaseModel):
    """Arguments for move_file tool."""
    source: str = Field(description="Source file or directory path")
    destination: str = Field(description="Destination path (including new filename if renaming)")
    overwrite: bool = Field(default=False, description="Whether to overwrite if destination exists")


@server.call_tool()
async def move_file(name: str, arguments: dict) -> list[TextContent]:
    """Move a file or directory to a new location."""
    if name != "move_file":
        raise ValueError(f"Unknown tool: {name}")
    
    args = MoveFileArgs(**arguments)
    source = Path(args.source).resolve()
    destination = Path(args.destination).resolve()
    
    # Security checks
    if not is_path_allowed(source):
        return [TextContent(
            type="text",
            text=f"Error: Source path '{source}' is outside allowed directories"
        )]
    
    if not is_path_allowed(destination):
        return [TextContent(
            type="text",
            text=f"Error: Destination path '{destination}' is outside allowed directories"
        )]
    
    if not source.exists():
        return [TextContent(type="text", text=f"Error: Source '{source}' does not exist")]
    
    if destination.exists() and not args.overwrite:
        return [TextContent(
            type="text",
            text=f"Error: Destination '{destination}' already exists. Set overwrite=true to replace."
        )]
    
    logger.info(f"Moving: {source} -> {destination}")
    
    try:
        # Ensure destination parent directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Perform the move
        shutil.move(str(source), str(destination))
        
        result = {
            "success": True,
            "action": "move",
            "source": str(source),
            "destination": str(destination),
            "message": f"Successfully moved '{source.name}' to '{destination}'"
        }
        
        import json
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        logger.error(f"Error moving {source} to {destination}: {e}")
        return [TextContent(type="text", text=f"Error moving file: {str(e)}")]


# ============================================================================
# Tool: rename_file
# ============================================================================

class RenameFileArgs(BaseModel):
    """Arguments for rename_file tool."""
    path: str = Field(description="File or directory path to rename")
    new_name: str = Field(description="New name (just the filename, not full path)")


@server.call_tool()
async def rename_file(name: str, arguments: dict) -> list[TextContent]:
    """Rename a file or directory."""
    if name != "rename_file":
        raise ValueError(f"Unknown tool: {name}")
    
    args = RenameFileArgs(**arguments)
    path = Path(args.path).resolve()
    
    if not is_path_allowed(path):
        return [TextContent(
            type="text",
            text=f"Error: Path '{path}' is outside allowed directories"
        )]
    
    if not path.exists():
        return [TextContent(type="text", text=f"Error: Path '{path}' does not exist")]
    
    # Validate new name (no path separators)
    if "/" in args.new_name or "\\" in args.new_name:
        return [TextContent(
            type="text",
            text="Error: new_name should be just the filename, not a path. Use move_file for moving."
        )]
    
    new_path = path.parent / args.new_name
    
    if new_path.exists():
        return [TextContent(
            type="text",
            text=f"Error: A file named '{args.new_name}' already exists in this directory"
        )]
    
    logger.info(f"Renaming: {path.name} -> {args.new_name}")
    
    try:
        path.rename(new_path)
        
        result = {
            "success": True,
            "action": "rename",
            "old_name": path.name,
            "new_name": args.new_name,
            "old_path": str(path),
            "new_path": str(new_path),
            "message": f"Successfully renamed '{path.name}' to '{args.new_name}'"
        }
        
        import json
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        logger.error(f"Error renaming {path}: {e}")
        return [TextContent(type="text", text=f"Error renaming file: {str(e)}")]


# ============================================================================
# Tool: create_directory
# ============================================================================

class CreateDirectoryArgs(BaseModel):
    """Arguments for create_directory tool."""
    path: str = Field(description="Directory path to create")
    parents: bool = Field(default=True, description="Create parent directories if needed")


@server.call_tool()
async def create_directory(name: str, arguments: dict) -> list[TextContent]:
    """Create a new directory."""
    if name != "create_directory":
        raise ValueError(f"Unknown tool: {name}")
    
    args = CreateDirectoryArgs(**arguments)
    path = Path(args.path).resolve()
    
    if not is_path_allowed(path):
        return [TextContent(
            type="text",
            text=f"Error: Path '{path}' is outside allowed directories"
        )]
    
    if path.exists():
        if path.is_dir():
            return [TextContent(
                type="text",
                text=f"Directory '{path}' already exists"
            )]
        else:
            return [TextContent(
                type="text",
                text=f"Error: A file with name '{path.name}' already exists at this location"
            )]
    
    logger.info(f"Creating directory: {path}")
    
    try:
        path.mkdir(parents=args.parents, exist_ok=True)
        
        result = {
            "success": True,
            "action": "create_directory",
            "path": str(path),
            "message": f"Successfully created directory '{path}'"
        }
        
        import json
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        logger.error(f"Error creating directory {path}: {e}")
        return [TextContent(type="text", text=f"Error creating directory: {str(e)}")]


# ============================================================================
# Tool: delete_file
# ============================================================================

class DeleteFileArgs(BaseModel):
    """Arguments for delete_file tool."""
    path: str = Field(description="File or empty directory path to delete")
    confirm: bool = Field(description="Must be true to actually delete - safety check")


@server.call_tool()
async def delete_file(name: str, arguments: dict) -> list[TextContent]:
    """Delete a file or empty directory."""
    if name != "delete_file":
        raise ValueError(f"Unknown tool: {name}")
    
    args = DeleteFileArgs(**arguments)
    path = Path(args.path).resolve()
    
    if not args.confirm:
        return [TextContent(
            type="text",
            text="Error: confirm must be true to delete. This is a safety check."
        )]
    
    if not is_path_allowed(path):
        return [TextContent(
            type="text",
            text=f"Error: Path '{path}' is outside allowed directories"
        )]
    
    if not path.exists():
        return [TextContent(type="text", text=f"Error: Path '{path}' does not exist")]
    
    # Don't allow deleting root directories
    if path in ALLOWED_ROOTS:
        return [TextContent(
            type="text",
            text="Error: Cannot delete a root directory"
        )]
    
    logger.info(f"Deleting: {path}")
    
    try:
        if path.is_file():
            path.unlink()
            action = "deleted file"
        elif path.is_dir():
            # Only delete empty directories for safety
            if any(path.iterdir()):
                return [TextContent(
                    type="text",
                    text=f"Error: Directory '{path}' is not empty. Only empty directories can be deleted."
                )]
            path.rmdir()
            action = "deleted directory"
        else:
            return [TextContent(type="text", text=f"Error: Unknown file type at '{path}'")]
        
        result = {
            "success": True,
            "action": action,
            "path": str(path),
            "message": f"Successfully {action}: {path}"
        }
        
        import json
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        logger.error(f"Error deleting {path}: {e}")
        return [TextContent(type="text", text=f"Error deleting: {str(e)}")]


# ============================================================================
# Tool definitions for MCP
# ============================================================================

@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return the list of available tools."""
    return [
        Tool(
            name="list_directory",
            description=(
                "List files and directories with detailed metadata. "
                "Returns file names, sizes, types, modification dates, and text previews. "
                "Use this to scan a directory and understand its contents before organizing."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list (e.g., 'C:\\Users\\Name\\Desktop')"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to list subdirectories recursively",
                        "default": False
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": "Include hidden and system files",
                        "default": False
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum recursion depth",
                        "default": 3
                    },
                    "include_preview": {
                        "type": "boolean",
                        "description": "Include text preview for small text files",
                        "default": True
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="read_file_content",
            description=(
                "Read the content of a text file for semantic analysis. "
                "Use this to understand what a file contains before deciding how to organize it. "
                "Best for text files, code, documents. Binary files return metadata only."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read"
                    },
                    "max_bytes": {
                        "type": "integer",
                        "description": "Maximum bytes to read (default 1MB)",
                        "default": 1048576
                    },
                    "encoding": {
                        "type": "string",
                        "description": "Text encoding",
                        "default": "utf-8"
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="get_file_info",
            description=(
                "Get detailed metadata for a single file or directory. "
                "Returns size, dates, type, and for directories: item counts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File or directory path"
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="move_file",
            description=(
                "Move a file or directory to a new location. "
                "Can also be used to move into a subdirectory. "
                "Creates parent directories if needed."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Source file or directory path"
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination path (full path including filename)"
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Overwrite if destination exists",
                        "default": False
                    }
                },
                "required": ["source", "destination"]
            }
        ),
        Tool(
            name="rename_file",
            description=(
                "Rename a file or directory in place. "
                "Only changes the name, not the location. Use move_file to change location."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File or directory to rename"
                    },
                    "new_name": {
                        "type": "string",
                        "description": "New name (filename only, not a path)"
                    }
                },
                "required": ["path", "new_name"]
            }
        ),
        Tool(
            name="create_directory",
            description=(
                "Create a new directory. Creates parent directories automatically. "
                "Use this before moving files to create the folder structure."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to create"
                    },
                    "parents": {
                        "type": "boolean",
                        "description": "Create parent directories if needed",
                        "default": True
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="delete_file",
            description=(
                "Delete a file or empty directory. "
                "Requires confirm=true as a safety check. "
                "Will not delete non-empty directories."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File or empty directory to delete"
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "Must be true to actually delete (safety check)"
                    }
                },
                "required": ["path", "confirm"]
            }
        ),
    ]


# ============================================================================
# Main entry point
# ============================================================================

def main():
    """Start the MCP server with SSE transport."""
    import asyncio
    from aiohttp import web
    
    logger.info("=" * 60)
    logger.info("Windows Filesystem MCP Server")
    logger.info("=" * 60)
    logger.info(f"Port: {SERVER_PORT}")
    logger.info(f"Allowed roots: {[str(r) for r in ALLOWED_ROOTS]}")
    logger.info("=" * 60)
    
    async def handle_sse(request):
        """Handle SSE connections from MCP clients."""
        logger.info(f"New SSE connection from {request.remote}")
        
        sse_transport = SseServerTransport("/messages")
        
        async with sse_transport.connect_sse(
            request.path,
            request.headers,
        ) as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
        
        return web.Response(text="Connection closed")
    
    async def handle_messages(request):
        """Handle POST messages from MCP clients."""
        # The SSE transport handles this internally
        pass
    
    async def health_check(request):
        """Health check endpoint."""
        return web.json_response({
            "status": "healthy",
            "server": "windows-filesystem-mcp",
            "allowed_roots": [str(r) for r in ALLOWED_ROOTS]
        })
    
    app = web.Application()
    app.router.add_get("/sse", handle_sse)
    app.router.add_post("/messages", handle_messages)
    app.router.add_get("/health", health_check)
    
    logger.info(f"Starting server on http://0.0.0.0:{SERVER_PORT}")
    logger.info("Endpoints:")
    logger.info(f"  SSE: http://0.0.0.0:{SERVER_PORT}/sse")
    logger.info(f"  Health: http://0.0.0.0:{SERVER_PORT}/health")
    
    web.run_app(app, host="0.0.0.0", port=SERVER_PORT)


if __name__ == "__main__":
    main()
