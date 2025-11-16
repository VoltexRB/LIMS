import pytest
from pathlib import Path
import json
from llm_interaction_manager.core.interaction_manager import InteractionManager
from llm_interaction_manager.utils import RAGMode


@pytest.fixture(scope="module")
def create_settings():
    config_path = Path(__file__).parent / "test_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        test_config = json.load(f)

    from llm_interaction_manager.utils.settings import Settings
    settings = Settings(
        handlers=test_config.get("handlers", {}),
        default_handlers=test_config.get("default_handlers", {}),
        **test_config.get("general", {})
    )
    return settings


@pytest.fixture(scope="module")
def interaction_manager(create_settings):
    im = InteractionManager(settings=create_settings)
    return im


def test_send_prompt_with_llm(interaction_manager):
    """
    Sends a prompt to the LLM handler and verifies a response is returned.
    """
    prompt = "Hello from integration test"
    interaction_manager.start_conversation()
    response = interaction_manager.send_prompt(prompt)

    assert isinstance(response, dict)
    assert "content" in response
    assert prompt in response["prompt"]
    assert isinstance(response["content"], str)


def test_rag_data_integration(interaction_manager):
    """
    Checks that RAG data is correctly included in the LLM prompt.
    """
    # Add some on-the-fly RAG data
    interaction_manager.settings.on_the_fly_data = {"doc1": "Test vector content"}
    interaction_manager.settings.use_rag_data = RAGMode.VOLATILE

    prompt = "Test RAG prompt"
    interaction_manager.start_conversation()
    response = interaction_manager.send_prompt(prompt)

    assert isinstance(response, dict)
    assert "content" in response
    assert "RAG-Data" in response
    assert isinstance(response["RAG-Data"], list)
    assert isinstance(response["content"], str)


def test_conversation_history(interaction_manager):
    """
    Verifies that conversation history is correctly stored.
    """
    prompt = "Store this in history"
    interaction_manager.start_conversation()
    interaction_manager.send_prompt(prompt)

    history = interaction_manager.conversation.conversation_history
    assert isinstance(history, list)
    assert any(msg["prompt"] == prompt for msg in history)


def test_vector_storage_chroma(interaction_manager):
    """
    Checks that the last LLM response is saved in ChromaDB (vector handler).
    """
    prompt = "Store this in vector aswell"
    interaction_manager.start_conversation()
    interaction_manager.send_prompt(prompt)
    last_msg = interaction_manager.conversation.get_last_response()
    vectors = interaction_manager.vector_handler.nearest_search(last_msg, top_k=1, table="lims_embeddings")

    # The last message content should appear in vector search results
    assert any(last_msg in v for v in vectors)


def test_persistent_storage_mongodb(interaction_manager):
    """
    Verifies that the last message is stored in the persistent database.
    """
    prompt = "Third, store this persistently"
    interaction_manager.start_conversation()
    interaction_manager.send_prompt(prompt)
    last_msg = interaction_manager.conversation.conversation_history[-1]

    collection_name = "conversations"
    filters = {"message_id": last_msg["message_id"]}
    records = interaction_manager.persistent_handler.get_data(filters)

    # Since each record is a conversation:
    assert isinstance(records, list)
    assert len(records) == 1

    conversation = records[0]
    assert "messages" in conversation
    messages = conversation["messages"]

    assert len(messages) == 1  # message_id is unique
    message = messages[0]

    # Now you can assert the message content
    assert message["llm_response"] == last_msg["content"]
    assert message["user_prompt"] == last_msg["prompt"]
