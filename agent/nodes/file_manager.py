"""
File Manager node for handling file organization operations.

This specialized sub-agent handles all file management requests with its own
system prompt and only file operation tools for better accuracy.

The node uses the remote Windows Filesystem MCP server to:
- List and analyze files on the Windows machine
- Understand file contents semantically using the LLM
- Propose logical folder structures based on file types and content
- Execute moves, renames, and folder creation operations
"""

from typing import Optional

from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool

from agent.state import ReachyAgentState, StateUpdate
from agent.config import AgentConfig, REACHY_IDENTITY, REACHY_OUTPUT_RULES


def get_file_manager_system_prompt() -> str:
    """Build the file manager-specific system prompt."""
    return f"""{REACHY_IDENTITY}

{REACHY_OUTPUT_RULES}

You are a file organization assistant with access to a Windows filesystem.
You help users organize their files by analyzing, categorizing, and moving them into logical folder structures.

AVAILABLE FILE MANAGEMENT TOOLS:

1. "list_directory" - List files and folders with metadata
   Parameters:
   - path (string): Directory path like "C:\\Users\\Name\\Desktop"
   - recursive (boolean): Whether to include subdirectories (default: false)
   - include_preview (boolean): Include text file previews (default: true)
   
   Returns: JSON with file names, sizes, types, dates, and previews

2. "read_file_content" - Read file contents for semantic understanding
   Parameters:
   - path (string): File path to read
   - max_bytes (integer): Maximum bytes to read (default: 1MB)
   
   Use this to understand what a file contains when the name is ambiguous

3. "get_file_info" - Get detailed metadata for a single file
   Parameters:
   - path (string): File or directory path
   
   Returns: Detailed info including size, dates, type

4. "move_file" - Move a file or folder to a new location
   Parameters:
   - source (string): Current file/folder path
   - destination (string): New full path including filename
   - overwrite (boolean): Whether to overwrite existing (default: false)

5. "rename_file" - Rename a file or folder in place
   Parameters:
   - path (string): File/folder to rename
   - new_name (string): New name (filename only, not a path)

6. "create_directory" - Create a new folder
   Parameters:
   - path (string): Full path of directory to create
   - parents (boolean): Create parent directories if needed (default: true)

7. "delete_file" - Delete a file or empty directory
   Parameters:
   - path (string): Path to delete
   - confirm (boolean): Must be true to actually delete (safety check)

==============================================================================
FILE ORGANIZATION WORKFLOW
==============================================================================

When asked to organize files, follow this process:

**PHASE 1: DISCOVERY**
1. Use list_directory to scan the target folder
2. Review the file list, noting types, names, and sizes
3. For ambiguous files, use read_file_content to understand their purpose

**PHASE 2: ANALYSIS & PLANNING**
4. Group files into logical categories based on:
   - File type (documents, images, code, media, archives)
   - Project/topic (if files relate to specific projects or subjects)
   - Date patterns (if organizing by time makes sense)
   - Content themes (work vs personal, by client/project name)

5. Create a proposed folder structure. Common patterns:
   - By type: Documents/, Images/, Code/, Media/, Archives/
   - By project: ProjectA/, ProjectB/, Personal/, Work/
   - Hybrid: Documents/Work/, Documents/Personal/, Media/Photos/, Media/Videos/

6. Consider renaming files for clarity:
   - Use consistent naming: YYYY-MM-DD_ProjectName_Description.ext
   - Remove special characters and spaces if problematic
   - Make names descriptive but concise

**PHASE 3: CONFIRMATION**
7. Present your plan to the user BEFORE executing:
   - Show proposed folder structure
   - List which files will move where
   - Highlight any renames
   - Ask for confirmation

Example confirmation message:
"I'll organize your Desktop into these folders:
- Documents/ (5 files: reports, notes)
- Images/ (12 files: screenshots, photos)  
- Code/ (3 files: Python scripts)
- Misc/ (2 files: couldn't categorize)

Should I proceed?"

**PHASE 4: EXECUTION**
8. Only after user confirms:
   - Create directories first (create_directory)
   - Move files one by one (move_file)
   - Apply renames if needed (rename_file)
   - Report progress

**PHASE 5: SUMMARY**
9. After completion, summarize what was done:
   - Number of files organized
   - New folder structure created
   - Any issues encountered

==============================================================================
SAFETY RULES
==============================================================================

- NEVER delete files without explicit user request and confirmation
- NEVER move system files or hidden files (starting with .)
- NEVER overwrite files - if destination exists, ask user what to do
- ALWAYS show plan before executing bulk operations
- Skip files that fail and continue with others - report failures at end
- Handle duplicate names by appending numbers: file.txt -> file_1.txt

==============================================================================
CATEGORY TAXONOMY (Suggestions)
==============================================================================

Documents:
- .pdf, .doc, .docx, .txt, .md, .odt - text documents
- .xls, .xlsx, .csv - spreadsheets
- .ppt, .pptx - presentations

Images:
- .jpg, .jpeg, .png, .gif, .bmp, .svg, .webp - images
- .psd, .ai, .sketch - design files

Code:
- .py, .js, .ts, .java, .c, .cpp, .html, .css - source code
- .json, .xml, .yaml, .yml - config files
- .sh, .bat, .ps1 - scripts

Media:
- .mp3, .wav, .flac, .ogg - audio
- .mp4, .avi, .mkv, .mov - video

Archives:
- .zip, .rar, .7z, .tar, .gz - compressed files

==============================================================================
NAMING CONVENTIONS
==============================================================================

When renaming files for clarity:
- Format: YYYY-MM-DD_Category_Description.ext
- Examples:
  - 2024-03-15_Invoice_Acme_Corp.pdf
  - 2024-03-14_Meeting_Notes_Project_Alpha.docx
  - Screenshot_2024-03-15_Dashboard.png

Only rename if:
- Names are meaningless (like "Document (1).pdf")
- User explicitly requests renaming
- Multiple files have confusing similar names

==============================================================================
RESPONSE STYLE
==============================================================================

Be conversational and helpful:
- "I found 47 files on your Desktop. Let me categorize them..."
- "Most files are documents and images. I suggest creating Documents/ and Images/ folders."
- "Done! I organized 47 files into 4 folders. Your Desktop is much cleaner now."

Report issues clearly:
- "I couldn't move 'report.pdf' because a file with that name already exists in Documents/"
- "Skipped 3 hidden files as requested"
"""


def create_file_manager_llm(config: AgentConfig, tools: list[BaseTool]) -> ChatOpenAI:
    """Create the file manager LLM instance with tool bindings."""
    llm = ChatOpenAI(
        model=config.models.main_model,
        base_url=config.models.main_model_base_url,
        api_key=config.models.api_key,
        temperature=config.models.main_temperature,
    )
    
    if tools:
        return llm.bind_tools(tools)
    return llm


def build_file_manager_messages(state: ReachyAgentState) -> list:
    """Build the message list for the file manager LLM."""
    messages = [SystemMessage(content=get_file_manager_system_prompt())]
    
    # Add conversation history
    for msg in state.get("messages", []):
        messages.append(msg)
    
    return messages


def filter_file_manager_tools(all_tools: list[BaseTool]) -> list[BaseTool]:
    """Filter to only include file management tools."""
    file_tool_names = {
        # Windows Filesystem MCP tools
        "list_directory",
        "read_file_content", 
        "get_file_info",
        "move_file",
        "rename_file",
        "create_directory",
        "delete_file",
    }
    return [tool for tool in all_tools if tool.name in file_tool_names]


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
                logger.info(f"File manager tool '{tool_name}' executed successfully")
            except Exception as e:
                logger.error(f"File manager tool '{tool_name}' failed: {e}")
                results.append({
                    "id": tool_id,
                    "name": tool_name,
                    "result": str(e),
                    "success": False,
                })
        else:
            logger.warning(f"Unknown file manager tool requested: {tool_name}")
            results.append({
                "id": tool_id,
                "name": tool_name,
                "result": f"Unknown tool: {tool_name}. Available: {list(tool_map.keys())}",
                "success": False,
            })
    
    return results


async def file_manager_node(
    state: ReachyAgentState,
    config: AgentConfig,
    tools: Optional[list[BaseTool]] = None
) -> StateUpdate:
    """
    File manager node for handling file organization requests.
    
    Uses an agentic loop that continues making tool calls until the LLM
    determines it has enough information to respond or complete the task.
    
    Args:
        state: Current agent state
        config: Agent configuration
        tools: List of all available tools (will be filtered to file tools)
        
    Returns:
        State update with file manager response
    """
    tools = tools or []
    file_tools = filter_file_manager_tools(tools)
    
    logger.info(f"File manager node: Processing with {len(file_tools)} file tools: {[t.name for t in file_tools]}")
    
    if not file_tools:
        logger.warning("File manager node: No file management tools available!")
        return {
            "messages": [AIMessage(content=(
                "I'm sorry, I don't have access to the Windows filesystem right now. "
                "Please make sure the Windows MCP server is running and the connection is configured. "
                "You can check by setting WINDOWS_FS_MCP_HOST and WINDOWS_FS_MCP_PORT environment variables."
            ))],
            "emotional_state": "apologetic",
        }
    
    # Maximum iterations to prevent infinite loops
    MAX_ITERATIONS = 15  # Higher limit for complex file operations
    all_tool_results = []
    
    try:
        llm = create_file_manager_llm(config, file_tools)
        messages = build_file_manager_messages(state)
        
        # Agentic loop - continue until no more tool calls or max iterations
        for iteration in range(MAX_ITERATIONS):
            logger.info(f"File manager node: Iteration {iteration + 1}/{MAX_ITERATIONS}")
            
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
                logger.info(f"File manager node: Tool calls requested: {[tc['name'] for tc in tool_calls]}")
            
            if not tool_calls:
                # No more tool calls - LLM is ready to respond
                response_content = response.content
                if not response_content or response_content.strip() == "":
                    response_content = "I've completed the file organization task."
                
                logger.info(f"File manager node: Final response after {iteration + 1} iterations")
                
                return {
                    "messages": [AIMessage(content=response_content)],
                    "tool_results": all_tool_results,
                    "emotional_state": "helpful",
                }
            
            # Execute tool calls
            logger.info(f"File manager node: Executing {len(tool_calls)} tool calls")
            tool_results = await execute_tools(tool_calls, file_tools)
            all_tool_results.extend(tool_results)
            
            # Add assistant message with tool calls and tool results to message history
            messages.append(response)
            for result in tool_results:
                messages.append(ToolMessage(
                    content=str(result.get("result", "")),
                    tool_call_id=result.get("id", ""),
                ))
        
        # Max iterations reached - force a response
        logger.warning(f"File manager node: Max iterations ({MAX_ITERATIONS}) reached, forcing response")
        
        # One final call without tools to get a summary
        final_llm = ChatOpenAI(
            model=config.models.main_model,
            base_url=config.models.main_model_base_url,
            api_key=config.models.api_key,
            temperature=config.models.main_temperature,
        )
        
        # Add instruction to summarize
        messages.append(HumanMessage(content="Please summarize what you've done and any remaining issues."))
        final_response = await final_llm.ainvoke(messages)
        
        return {
            "messages": [AIMessage(content=final_response.content or "File organization task completed.")],
            "tool_results": all_tool_results,
            "emotional_state": "helpful",
        }
            
    except Exception as e:
        logger.error(f"File manager node error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "messages": [AIMessage(content=(
                "I encountered an error while managing files. "
                "Please check that the Windows MCP server is running and try again."
            ))],
            "emotional_state": "apologetic",
        }
