"""Shared fixtures for all tests.

Ensures the agent package is importable and provides common test helpers.

IMPORTANT: The agent/__init__.py eagerly imports agent.graph which cascades
into langgraph, langchain, pydantic, etc.  For unit tests that only need the
soul subsystem (agent.soul.*) we stub out the heavy top-level imports before
any test module can trigger them.
"""

import os
import sys
import types
from unittest.mock import MagicMock

# ── Path setup ────────────────────────────────────────────────────────────────

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_BOT_DIR = os.path.join(_REPO_ROOT, "bot")
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)


# ── Stub heavy dependencies that agent/__init__.py pulls in ──────────────────
# We only need agent.soul.* for unit tests; avoid importing the full LangGraph
# stack by providing lightweight stubs for modules that aren't installed.


def _ensure_module(name: str) -> None:
    """Insert a lightweight stub module if *name* isn't already importable."""
    if name not in sys.modules:
        try:
            __import__(name)
        except (ImportError, ModuleNotFoundError):
            sys.modules[name] = types.ModuleType(name)


# Packages that agent.graph, agent.state, agent.config etc. try to import:
_OPTIONAL_STUBS = [
    "langgraph",
    "langgraph.graph",
    "langgraph.checkpoint",
    "langgraph.checkpoint.memory",
    "langchain",
    "langchain.chat_models",
    "langchain_openai",
    "langchain_core",
    "langchain_core.messages",
    "pydantic",
]

for _mod in _OPTIONAL_STUBS:
    _ensure_module(_mod)

# Add mock classes to stub modules so `from X import Y` works.
# This avoids import failures when agent.nodes.* tries to import from
# langchain_openai, langchain_core.messages, etc.
_lc_openai = sys.modules.get("langchain_openai")
if _lc_openai is not None and not hasattr(_lc_openai, "ChatOpenAI"):
    _lc_openai.ChatOpenAI = MagicMock()  # type: ignore[attr-defined]

_lc_core_msgs = sys.modules.get("langchain_core.messages")
if _lc_core_msgs is not None:
    for _cls in (
        "HumanMessage",
        "AIMessage",
        "SystemMessage",
        "BaseMessage",
        "ToolMessage",
        "FunctionMessage",
        "ChatMessage",
    ):
        if not hasattr(_lc_core_msgs, _cls):
            setattr(_lc_core_msgs, _cls, MagicMock())

# Stub other langchain modules that agent.nodes may import
for _sub in [
    "langchain_core.tools",
    "langchain_core.runnables",
    "langchain_core.output_parsers",
]:
    _ensure_module(_sub)
    _m = sys.modules.get(_sub)
    if _m is not None:
        for _attr in ("tool", "BaseTool", "StructuredTool", "Runnable"):
            if not hasattr(_m, _attr):
                setattr(_m, _attr, MagicMock())

# Provide a no-op stub for agent.graph so agent/__init__.py doesn't crash.
# This must happen *before* anything imports `agent`.
if "agent.graph" not in sys.modules:
    _graph_stub = types.ModuleType("agent.graph")
    _graph_stub.create_graph = MagicMock()  # type: ignore[attr-defined]
    _graph_stub.graph = MagicMock()  # type: ignore[attr-defined]
    sys.modules["agent.graph"] = _graph_stub

if "agent.state" not in sys.modules:
    _state_stub = types.ModuleType("agent.state")
    _state_stub.ReachyAgentState = MagicMock()  # type: ignore[attr-defined]
    _state_stub.ReachyState = MagicMock()  # type: ignore[attr-defined]
    _state_stub.StateUpdate = MagicMock()  # type: ignore[attr-defined]
    sys.modules["agent.state"] = _state_stub


# ── Pytest fixtures ──────────────────────────────────────────────────────────

import pytest  # noqa: E402  (must come after sys.path / module stubs)


@pytest.fixture
def soul_config():
    """Create a default SoulConfig for testing."""
    from agent.soul.config import SoulConfig

    return SoulConfig(
        debug_logging=False,
        log_emotions=False,
        log_events=False,
        log_decisions=False,
        log_movements=False,
    )


@pytest.fixture
def soul_config_verbose():
    """SoulConfig with all logging enabled (for debugging test failures)."""
    from agent.soul.config import SoulConfig

    return SoulConfig(debug_logging=True, log_emotions=True, log_events=True)
