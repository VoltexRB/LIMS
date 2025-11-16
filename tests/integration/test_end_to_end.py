import pytest
from pathlib import Path
import json
from unittest.mock import patch
import llm_interaction_manager.api.lims_interface as api
from llm_interaction_manager.api.interaction_manager_factory import LLMEnum, PersistentEnum, VectorEnum
from llm_interaction_manager.core.interaction_manager import ConnectionType


@pytest.fixture(scope="module", autouse=True)
def setup_api():
    """
    Initialize the API for all end-to-end tests.
    """
    api.initialize(llm=LLMEnum.LANGCHAIN, vector=VectorEnum.CHROMADB, persistent=PersistentEnum.MONGODB)

    # Connection data
    llm_data = {
        "token": "[REMOVED]",
        "model": "moonshotai/Kimi-K2-Instruct-0905"
    }
    vector_data = {
        "client_type": "PERSISTENT",
        "persistent_client_db_path": "D:/chroma"
    }
    persistent_data = {
        "host": "localhost",
        "port": 27017,
        "database": "promptDB"
    }

    api.connect(ConnectionType.LLM, llm_data)
    api.connect(ConnectionType.VECTOR, vector_data)
    api.connect(ConnectionType.PERSISTENT, persistent_data)

    # Ensure all handlers are connected
    assert api.is_connected(ConnectionType.LLM)
    assert api.is_connected(ConnectionType.PERSISTENT)
    assert api.is_connected(ConnectionType.VECTOR)

    yield


def test_conversation_workflow():
    """
    Full conversation test: start conversation, send prompt, store and retrieve message.
    """
    api.start_conversation()
    api.set_rag_data({"doc1": "Test vector content"}, volatile=True)

    prompt = "Hello from end-to-end test"

    # Patch input to always return "default comment"
    with patch("builtins.input", return_value="default comment"):
        response = api.send_prompt(prompt)

    assert isinstance(response, dict)
    assert "content" in response and len(response["content"]) > 0

    # Add metadata to last message
    api.add_metadata(to_conversation=False, data={"test_meta": "meta_value"})

    # Remove RAG data
    api.delete_rag_data()


def test_persistent_storage(tmp_path):
    """
    Test adding conversation/messages to persistent storage and retrieving them.
    """
    api.start_conversation()
    prompt = "Store this persistently"
    with patch("builtins.input", return_value="default comment"):
        api.send_prompt(prompt)

    conv = api.interaction_manager.conversation
    conversation_id = conv.conversation_id
    conversation_history = conv.conversation_history

    # Add to persistent storage
    api.add_persistent_data(
        {"conversation_id": conversation_id},
        conversation_history
    )

    # Retrieve and check stored data
    filters = {"message_id": conversation_history[-1]["message_id"]}
    records = api.interaction_manager.persistent_handler.get_data(filters)

    assert isinstance(records, list) and len(records) > 0
    conv_record = records[0]
    messages = conv_record["messages"]
    assert any(m["message_id"] == conversation_history[-1]["message_id"] for m in messages)



def test_vector_database_search():
    """
    Test vector DB operations: add vector data and run nearest search.
    DOES NOT FUNCTION WITH POSTGRES, AS POSTGRES HAS A FOREIGN KEY RESTRAINT LINKING LIMS_EMBEDDINGS TO MESSAGES
    """
    data = {
        "prompt": "Vector prompt",
        "response": "Vector response",
        "message_id": "vec_msg_1"
    }
    api.add_vector_data(data, table="lims_embeddings")
    results = api.nearest_search_vector("Vector prompt", top_k=1, table="lims_embeddings")

    assert isinstance(results, list)
    assert len(results) > 0
    assert any("Vector" in r for r in results)


def test_export_data(tmp_path):
    """
    Test exporting persistent data to a JSON file.
    """
    api.export_data(str(tmp_path))

    # There should be exactly one JSON file in tmp_path
    exported_files = list(tmp_path.glob("lims_export_*.json"))
    assert len(exported_files) == 1

    export_file = exported_files[0]
    assert export_file.exists()

    with open(export_file, "r", encoding="utf-8") as f:
        content = json.load(f)

    assert isinstance(content, list)
    assert len(content) > 0
