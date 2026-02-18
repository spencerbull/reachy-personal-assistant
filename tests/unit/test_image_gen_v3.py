"""Unit tests for v3 image_gen: exit detection, phase helpers."""

import pytest

from agent.nodes.image_gen import (
    _seems_image_related,
    is_image_confirmation,
    has_style_keyword,
    PHASE_AWAITING_IMAGE,
    PHASE_COLLECTING_DETAILS,
)


class TestSeemsImageRelated:
    """Test the _seems_image_related helper for topic-change detection."""

    def test_style_keyword_in_collecting_phase(self):
        assert _seems_image_related("make it 3d", PHASE_COLLECTING_DETAILS) is True

    def test_confirmation_in_awaiting_phase(self):
        assert _seems_image_related("here it is", PHASE_AWAITING_IMAGE) is True

    def test_unrelated_message_in_collecting_phase(self):
        assert (
            _seems_image_related("what's the weather like?", PHASE_COLLECTING_DETAILS)
            is False
        )

    def test_unrelated_message_in_awaiting_phase(self):
        assert _seems_image_related("what time is it", PHASE_AWAITING_IMAGE) is False

    def test_yes_is_related(self):
        assert _seems_image_related("yes", PHASE_COLLECTING_DETAILS) is True

    def test_empty_message_not_related(self):
        assert _seems_image_related("", PHASE_AWAITING_IMAGE) is False

    def test_render_is_related(self):
        assert (
            _seems_image_related("render it cyberpunk", PHASE_COLLECTING_DETAILS)
            is True
        )

    def test_calendar_question_not_related(self):
        assert (
            _seems_image_related(
                "what's on my calendar today?", PHASE_COLLECTING_DETAILS
            )
            is False
        )


class TestIsImageConfirmation:
    """Test image confirmation detection."""

    def test_here_it_is(self):
        assert is_image_confirmation("here it is") is True

    def test_yes(self):
        assert is_image_confirmation("yes") is True

    def test_ready(self):
        assert is_image_confirmation("ready") is True

    def test_random_message_not_confirmation(self):
        assert is_image_confirmation("tell me about robots") is False

    def test_empty_not_confirmation(self):
        assert is_image_confirmation("") is False

    def test_none_not_confirmation(self):
        assert is_image_confirmation(None) is False


class TestHasStyleKeyword:
    """Test style keyword detection."""

    def test_3d(self):
        assert has_style_keyword("make it 3d") is True

    def test_cyberpunk(self):
        assert has_style_keyword("cyberpunk style") is True

    def test_watercolor(self):
        assert has_style_keyword("watercolor painting") is True

    def test_no_style(self):
        assert has_style_keyword("hello there") is False

    def test_empty(self):
        assert has_style_keyword("") is False

    def test_none(self):
        assert has_style_keyword(None) is False
