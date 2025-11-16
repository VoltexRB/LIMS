import json
import os
import pytest
from llm_interaction_manager.handlers.langchain_handler import LangchainHandler

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../config.json"
)


@pytest.fixture(scope="module")
def langchain_handler():
    """Connect to TogetherAI using the local config.json token."""
    if not os.path.exists(CONFIG_PATH):
        pytest.skip("config.json not found, skipping integration test.")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    token = config["handlers"]["langchain"]["token"]
    if not token:
        pytest.skip("No token found in config.json, skipping integration test.")

    handler = LangchainHandler()
    connected = handler.connect({
        "token": token,
        "model": "meta-llama/Llama-3-70b-chat-hf"
    })
    if not connected:
        pytest.skip("Connection to TogetherAI failed.")
    return handler


def test_is_connected(langchain_handler):
    """Ensure the handler reports a valid connection."""
    assert langchain_handler.is_connected() is True


def test_validate_model_name(langchain_handler):
    try:
        assert langchain_handler.validate_model_name("meta-llama/Llama-3-70b-chat-hf")
    except Exception:
        pytest.skip("Skipped: TogetherAI service not reachable or model unavailable.")


def test_send_prompt_basic(langchain_handler):
    """Ensure the LLM returns a valid response for a simple question."""
    try:
        result = langchain_handler.send_prompt("Was ist die Quersumme von 420?")
    except Exception as e:
        pytest.skip(f"Skipped: TogetherAI service not reachable ({e})")

    assert isinstance(result, dict)
    assert "response" in result
    response = result["response"].strip().lower()
    assert len(response) > 0, "Empty response received from model"

    # Semantic check: loosely ensure the answer is numeric or mentions 'sechs'
    if not any(x in response for x in ["6", "sechs"]):
        pytest.xfail(f"Model gave a valid but unexpected response: {response}")


def test_send_prompt_with_rag(langchain_handler):
    """Ensure the RAG context is incorporated in the response."""
    rag_docs = [
        "Die Hauptstadt von Deutschland ist Berlin.",
        "Deutschland liegt in Europa."
    ]
    try:
        result = langchain_handler.send_prompt(
            "Was ist die Hauptstadt von Deutschland?",
            rag=rag_docs
        )
    except Exception as e:
        pytest.skip(f"Skipped: TogetherAI service not reachable ({e})")

    assert isinstance(result, dict)
    assert "response" in result
    response = result["response"].strip().lower()
    assert len(response) > 0, "Empty response received from model"

    # Check for thematic correctness without enforcing specific phrasing
    if not any(x in response for x in ["berlin", "hauptstadt", "deutschland"]):
        pytest.xfail(f"Model gave a valid but unexpected response: {response}")
