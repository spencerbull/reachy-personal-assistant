"""Unit tests for v3 router: LLM-first routing, flow exit, keyword fast-path."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from agent.nodes.router import (
    router_node,
    extract_user_message,
    parse_router_response,
    _is_clearly_image_related,
)
from agent.config import AgentConfig


def _make_state(user_msg: str, **extra) -> dict:
    """Build a minimal ReachyAgentState-like dict."""
    msg = MagicMock()
    msg.type = "human"
    msg.content = user_msg
    state = {"messages": [msg]}
    state.update(extra)
    return state


def _make_config() -> AgentConfig:
    return AgentConfig()


class TestFlowExit:
    """Test universal flow-exit detection."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "cancel",
            "stop",
            "never mind",
            "nevermind",
            "forget it",
            "let's move on",
            "change topic",
            "something else",
            "no thanks",
            "skip",
            "abort",
        ],
    )
    async def test_exit_phrase_routes_to_conversation(self, phrase):
        state = _make_state(phrase)
        result = await router_node(state, _make_config())
        assert result["route"] == "conversation"

    async def test_exit_clears_image_gen_context(self):
        state = _make_state("never mind", image_gen_context={"phase": "awaiting_image"})
        result = await router_node(state, _make_config())
        assert result["route"] == "conversation"
        assert result.get("image_gen_context") is None
        assert result.get("captured_source_image") is None

    async def test_exit_phrase_in_longer_sentence(self):
        state = _make_state("you know what, forget it")
        result = await router_node(state, _make_config())
        assert result["route"] == "conversation"


class TestUnambiguousCommands:
    """Test fast-path for unambiguous physical commands."""

    @pytest.mark.parametrize(
        "command",
        [
            "look left",
            "look right",
            "look up",
            "look down",
            "look at me",
            "turn left",
            "turn right",
            "turn around",
            "dance for me",
            "do a dance",
        ],
    )
    async def test_physical_commands_go_to_tools(self, command):
        state = _make_state(command)
        result = await router_node(state, _make_config())
        assert result["route"] == "tools"


class TestImageGenContinuation:
    """Test image_gen flow continuation vs topic change."""

    async def test_image_related_message_stays_in_flow(self):
        state = _make_state(
            "make it 3d",
            image_gen_context={"phase": "collecting_details"},
        )
        result = await router_node(state, _make_config())
        assert result["route"] == "image_gen"

    async def test_transform_stays_in_flow(self):
        state = _make_state(
            "transform it into a painting",
            image_gen_context={"phase": "collecting_details"},
        )
        result = await router_node(state, _make_config())
        assert result["route"] == "image_gen"

    async def test_yes_confirmation_stays_in_flow(self):
        """'yes' should continue the image_gen flow, not fall to LLM."""
        state = _make_state(
            "yes",
            image_gen_context={"phase": "awaiting_image"},
        )
        result = await router_node(state, _make_config())
        assert result["route"] == "image_gen"

    async def test_ready_stays_in_flow(self):
        state = _make_state(
            "ready",
            image_gen_context={"phase": "awaiting_image"},
        )
        result = await router_node(state, _make_config())
        assert result["route"] == "image_gen"

    async def test_here_it_is_stays_in_flow(self):
        state = _make_state(
            "here it is",
            image_gen_context={"phase": "awaiting_image"},
        )
        result = await router_node(state, _make_config())
        assert result["route"] == "image_gen"

    async def test_unrelated_message_still_goes_to_image_gen(self):
        """When in image_gen context, even unrelated messages route to image_gen
        so the node can decide whether to exit the flow."""
        state = _make_state(
            "what's the weather like",
            image_gen_context={"phase": "collecting_details"},
        )
        result = await router_node(state, _make_config())
        assert result["route"] == "image_gen"

    async def test_exit_phrase_escapes_image_gen_flow(self):
        """Exit phrases should break out of image_gen context."""
        state = _make_state(
            "never mind",
            image_gen_context={"phase": "collecting_details"},
        )
        result = await router_node(state, _make_config())
        assert result["route"] == "conversation"
        assert result.get("image_gen_context") is None

    async def test_captured_image_keeps_routing_to_image_gen(self):
        """Even without image_gen_context, captured_source_image means we're in the flow."""
        state = _make_state(
            "cyberpunk style",
            captured_source_image="base64data",
        )
        result = await router_node(state, _make_config())
        assert result["route"] == "image_gen"


class TestLLMRouterFallback:
    """Test LLM router is primary path for non-command messages."""

    async def test_ambiguous_message_goes_to_llm(self):
        """Messages like 'remember when we met' should go to LLM, not keyword match."""
        state = _make_state("remember when we first met?")

        # Mock the LLM to return conversation
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(
            content='{"route": "conversation", "reason": "casual reminiscing"}'
        )

        with patch("agent.nodes.router.create_router_llm", return_value=mock_llm):
            result = await router_node(state, _make_config())
        assert result["route"] == "conversation"

    async def test_llm_error_defaults_to_conversation(self):
        """When LLM fails, route to conversation as safe default."""
        state = _make_state("what's the meaning of life?")

        with patch(
            "agent.nodes.router.create_router_llm",
            side_effect=Exception("connection refused"),
        ):
            result = await router_node(state, _make_config())
        assert result["route"] == "conversation"


class TestParseRouterResponse:
    """Test JSON response parsing."""

    def test_valid_json(self):
        route, reason = parse_router_response(
            '{"route": "vision", "reason": "user asked what they see"}'
        )
        assert route == "vision"

    def test_json_with_extra_text(self):
        route, reason = parse_router_response(
            'I think the route should be {"route": "tools", "reason": "movement"} based on context'
        )
        assert route == "tools"

    def test_invalid_route_defaults_to_conversation(self):
        route, reason = parse_router_response(
            '{"route": "dance_party", "reason": "fun"}'
        )
        assert route == "conversation"

    def test_invalid_json_defaults_to_conversation(self):
        route, reason = parse_router_response("I don't know what to do")
        assert route == "conversation"


class TestIsClearlyImageRelated:
    """Test _is_clearly_image_related helper."""

    def test_render_is_image_related(self):
        assert _is_clearly_image_related("can you render this") is True

    def test_3d_is_image_related(self):
        assert _is_clearly_image_related("make it 3d") is True

    def test_hello_is_not_image_related(self):
        assert _is_clearly_image_related("hello how are you") is False

    def test_painting_is_image_related(self):
        assert _is_clearly_image_related("oil painting style") is True


class TestExtractUserMessage:
    """Test user message extraction from state."""

    def test_extract_from_langchain_message(self):
        msg = MagicMock()
        msg.type = "human"
        msg.content = "Hello there"
        assert extract_user_message({"messages": [msg]}) == "Hello there"

    def test_extract_from_dict_message(self):
        msg = {"role": "user", "content": "Hi robot"}
        assert extract_user_message({"messages": [msg]}) == "Hi robot"

    def test_empty_messages_returns_empty(self):
        assert extract_user_message({"messages": []}) == ""

    def test_latest_user_message_extracted(self):
        ai_msg = MagicMock()
        ai_msg.type = "ai"
        ai_msg.content = "I'm Reachy"
        user_msg = MagicMock()
        user_msg.type = "human"
        user_msg.content = "What can you do?"
        assert (
            extract_user_message({"messages": [ai_msg, user_msg]}) == "What can you do?"
        )
