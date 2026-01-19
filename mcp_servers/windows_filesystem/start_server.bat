@echo off
REM Windows Filesystem MCP Server Startup Script
REM 
REM This script starts the MCP server for file organization operations.
REM The server exposes tools for listing, reading, moving, and organizing files.
REM
REM Prerequisites:
REM   1. Python 3.11+ installed
REM   2. uv installed: pip install uv
REM
REM Configuration:
REM   Set environment variables before running:
REM     set MCP_ALLOWED_ROOTS=C:\Users\YourName\Desktop,C:\Users\YourName\Downloads
REM     set MCP_SERVER_PORT=8765
REM

echo ========================================
echo  Windows Filesystem MCP Server
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11 or higher
    pause
    exit /b 1
)

REM Check if uv is installed
uv --version >nul 2>&1
if errorlevel 1 (
    echo uv not found. Installing uv...
    pip install uv
)

REM Set default configuration if not already set
if not defined MCP_SERVER_PORT set MCP_SERVER_PORT=8765
if not defined MCP_ALLOWED_ROOTS set MCP_ALLOWED_ROOTS=%USERPROFILE%\Desktop,%USERPROFILE%\Downloads,%USERPROFILE%\Documents

echo Configuration:
echo   Port: %MCP_SERVER_PORT%
echo   Allowed Roots: %MCP_ALLOWED_ROOTS%
echo.

REM Navigate to script directory
cd /d "%~dp0"

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo Creating virtual environment...
    uv venv
)

REM Install dependencies
echo Installing dependencies...
uv pip install -e .

REM Start the server
echo.
echo Starting MCP server on port %MCP_SERVER_PORT%...
echo Press Ctrl+C to stop the server.
echo.

uv run python server.py

pause
