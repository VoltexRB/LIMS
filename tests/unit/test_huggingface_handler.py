import pytest
from llm_interaction_manager.handlers.huggingface_handler import HuggingfaceHandler

@pytest.fixture()
def hf_handler():
    handler = HuggingfaceHandler()
    handler.connect({"model": "meta-llama/Llama-3.1-8B-Instruct"})
    return handler

def test_is_connected(hf_handler):
    assert hf_handler.is_connected() is True

def test_send_prompt_basic(hf_handler):
    prompt = "Write a short poem about AI."
    result = hf_handler.send_prompt(prompt)
    assert "response" in result
    assert isinstance(result["response"], str)
    assert len(result["response"]) > 0

def test_send_prompt_with_rag(hf_handler):
    prompt = "Summarize the content."
    rag_docs = ["AI is the field of creating intelligent machines.", "It can perform tasks that normally require human intelligence."]
    result = hf_handler.send_prompt(prompt, rag=rag_docs)
    response = result["response"]
    assert isinstance(response, str)
    assert len(response.strip()) > 0

def test_connect_without_model_raises():
    handler = HuggingfaceHandler()
    with pytest.raises(ValueError):
        handler.connect({})  # missing "model"

def test_send_prompt_without_connect_raises():
    handler = HuggingfaceHandler()
    with pytest.raises(ValueError):
        handler.send_prompt("Hello")
