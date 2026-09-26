from unittest.mock import Mock, patch

import requests

from app.models import Message
from app.services.ollama import OllamaService


def make_service():
    with patch("app.services.ollama.create_ollama_client", return_value=Mock()):
        return OllamaService("llama3.2:latest", "http://localhost:11434")


@patch("app.services.ollama.requests.get")
def test_check_connection_reports_installed_model(mock_get):
    mock_get.return_value.raise_for_status.return_value = None
    mock_get.return_value.json.return_value = {"models": [{"name": "llama3.2:latest"}]}
    result = make_service().check_connection()
    assert result["connected"] is True
    assert result["model_available"] is True
    mock_get.assert_called_once_with("http://localhost:11434/api/tags", timeout=5)


@patch("app.services.ollama.requests.get", side_effect=requests.ConnectionError("connection refused"))
def test_check_connection_reports_unavailable_host(_mock_get):
    # The implementation should convert request failures into a health result.
    service = make_service()
    result = service.check_connection()
    assert result["connected"] is False


def test_message_conversion_preserves_roles():
    converted = OllamaService._to_autogen_messages([
        Message(role="system", content="Be concise"),
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi"),
    ])
    assert [message.__class__.__name__ for message in converted] == [
        "SystemMessage", "UserMessage", "AssistantMessage"
    ]


@patch("app.services.ollama.requests.get")
def test_check_connection_can_validate_a_selected_model(mock_get):
    mock_get.return_value.raise_for_status.return_value = None
    mock_get.return_value.json.return_value = {"models": [{"name": "llama3.2:latest"}]}
    result = make_service().check_connection("llama3.2:latest")
    assert result["model_available"] is True


@patch("app.services.ollama.create_ollama_client")
@patch("app.services.ollama.requests.get")
def test_completion_creates_a_fresh_client_for_each_request(mock_get, mock_factory):
    mock_get.return_value.raise_for_status.return_value = None
    mock_get.return_value.json.return_value = {"models": [{"name": "llama3.2:latest"}]}

    first_client = Mock()
    first_client.create = Mock(return_value=None)
    second_client = Mock()
    second_client.create = Mock(return_value=None)
    mock_factory.side_effect = [first_client, second_client]

    service = OllamaService("llama3.2:latest", "http://localhost:11434")
    # The mocked async methods are enough to verify client lifecycle selection;
    # the real integration test verifies the actual response content.
    import asyncio
    first_client.create = lambda messages: asyncio.sleep(0)
    second_client.create = lambda messages: asyncio.sleep(0)
    service.complete([Message(role="user", content="one")])
    service.complete([Message(role="user", content="two")])
    assert mock_factory.call_count == 2
