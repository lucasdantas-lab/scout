"""Unit tests for agents/context_agent.py."""

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.context_agent import (
    _NEUTRAL_CONTEXT,
    process_context_with_claude,
)


class TestProcessContextWithClaude:
    """Tests for process_context_with_claude using mocked Claude API."""

    def test_returns_neutral_when_no_api_key(self) -> None:
        """Without API key, should return neutral context."""
        with patch("agents.context_agent.ANTHROPIC_API_KEY", ""):
            result = process_context_with_claude(
                "Some news text", "Flamengo", "Palmeiras"
            )
        assert result == _NEUTRAL_CONTEXT

    def test_returns_neutral_when_empty_text(self) -> None:
        """With empty text, should return neutral context."""
        with patch("agents.context_agent.ANTHROPIC_API_KEY", "test-key"):
            result = process_context_with_claude("", "Flamengo", "Palmeiras")
        assert result == _NEUTRAL_CONTEXT

    def test_parses_valid_claude_response(self) -> None:
        """Should correctly parse a valid JSON response from Claude."""
        mock_response = {
            "home": {
                "ausencias_confirmadas": ["Pedro"],
                "duvidas": ["Arrascaeta"],
                "confirmados_importantes": ["Gabigol"],
                "lambda_delta": -0.15,
                "confianca": 0.8,
                "notas": "Pedro lesionado, duvida sobre Arrascaeta.",
            },
            "away": {
                "ausencias_confirmadas": [],
                "duvidas": ["Endrick"],
                "confirmados_importantes": ["Raphael Veiga"],
                "lambda_delta": 0.0,
                "confianca": 0.5,
                "notas": "Sem desfalques confirmados.",
            },
        }

        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=json.dumps(mock_response))]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg

        with (
            patch("agents.context_agent.ANTHROPIC_API_KEY", "test-key"),
            patch("anthropic.Anthropic", return_value=mock_client),
        ):
            result = process_context_with_claude(
                "Pedro lesionado no treino", "Flamengo", "Palmeiras"
            )

        assert result["home"]["ausencias_confirmadas"] == ["Pedro"]
        assert result["home"]["lambda_delta"] == -0.15
        assert result["away"]["confianca"] == 0.5

    def test_clamps_lambda_delta(self) -> None:
        """Should clamp lambda_delta to [-0.40, 0.10]."""
        mock_response = {
            "home": {
                "lambda_delta": -0.80,
                "confianca": 1.0,
            },
            "away": {
                "lambda_delta": 0.50,
                "confianca": 1.0,
            },
        }

        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=json.dumps(mock_response))]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg

        with (
            patch("agents.context_agent.ANTHROPIC_API_KEY", "test-key"),
            patch("anthropic.Anthropic", return_value=mock_client),
        ):
            result = process_context_with_claude(
                "Some news", "TimeA", "TimeB"
            )

        assert result["home"]["lambda_delta"] == -0.40
        assert result["away"]["lambda_delta"] == 0.10

    def test_returns_neutral_on_invalid_json(self) -> None:
        """Should return neutral context when Claude returns invalid JSON."""
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="This is not JSON")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg

        with (
            patch("agents.context_agent.ANTHROPIC_API_KEY", "test-key"),
            patch("anthropic.Anthropic", return_value=mock_client),
        ):
            result = process_context_with_claude(
                "Some news", "TimeA", "TimeB"
            )

        assert result == _NEUTRAL_CONTEXT

    def test_returns_neutral_on_api_error(self) -> None:
        """Should return neutral context when Claude API raises an error."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API down")

        with (
            patch("agents.context_agent.ANTHROPIC_API_KEY", "test-key"),
            patch("anthropic.Anthropic", return_value=mock_client),
        ):
            result = process_context_with_claude(
                "Some news", "TimeA", "TimeB"
            )

        assert result == _NEUTRAL_CONTEXT
