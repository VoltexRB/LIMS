import uuid
import pytest
from llm_interaction_manager.handlers.postgres_handler import PostgresHandler

@pytest.fixture
def pg_handler():
    """
    Fixture to initialize a PostgresqlHandler for testing.
    Adjust host/port/auth as needed for your test database.
    """
    handler = PostgresHandler()
    handler.connect(
        host="localhost",
        port=5432,
        auth={"database": "promptDB", "user": "postgres", "password": "postgres"}
    )
    return handler

def test_connection(pg_handler):
    """Test that the PostgreSQL connection is alive"""
    assert pg_handler.is_connected()

def test_save_and_get_record(pg_handler):
    """Test saving a conversation and retrieving it with messages"""
    conversation = {
        "conversation_id": "test_conv_001",
        "name": "test_llm",
        "description": "Test conversation",
        "metadata": {}
    }
    messages = [
        {
            "message_id": "test_msg_001",
            "user_prompt": "Hello",
            "llm_response": "Hi!",
            "metadata": {}
        },
        {
            "message_id": "test_msg_002",
            "user_prompt": "How are you?",
            "llm_response": "I am good, thanks!",
            "metadata": {}
        }
    ]

    pg_handler.save_record(conversation, messages)

    results = pg_handler.get_data({"conversation_id": "test_conv_001"})
    assert len(results) == 1
    conv = results[0]
    assert conv["conversation_id"] == "test_conv_001"
    assert conv["name"] == "test_llm"
    assert conv["description"] == "Test conversation"
    assert conv["metadata"] == {}
    assert len(conv["messages"]) == 2

    message_ids = {msg["message_id"] for msg in conv["messages"]}
    assert "test_msg_001" in message_ids
    assert "test_msg_002" in message_ids

def test_save_vector_and_load(pg_handler):
    """Test saving a vector and loading it"""
    conversation_id = f"conv_test_{uuid.uuid4()}"
    message_id = f"vec_test_{uuid.uuid4()}"

    conversation = {
        "conversation_id": conversation_id,
        "name": "test_llm",
        "description": "Test conversation",
        "metadata": {}
    }
    message = [
        {
            "message_id": message_id,
            "user_prompt": "What is the capital of France?",
            "llm_response": "Paris",
            "metadata": {}
        }
    ]
    pg_handler.save_record(conversation, message)

    data = {
        "message_id": message_id,
        "prompt": "What is the capital of France?",
        "response": "Paris"
    }
    pg_handler.save_vector(data, table="message_vectors")

    result = pg_handler.load_vector({"id": message_id}, table="message_vectors")

    # Alte Assertion ersetzen:
    # assert result["message_id"] == message_id
    # Stattdessen prüfen, dass die geladenen Daten stimmen:
    assert result["user_prompt"] == data["prompt"]
    assert result["llm_response"] == data["response"]
    assert "prompt_embedding" in result
    assert "response_embedding" in result


def test_nearest_search(pg_handler):
    """Test nearest search returns expected top-k messages"""
    conv_id = f"conv_ns_{uuid.uuid4()}"
    conversation = {
        "conversation_id": conv_id,
        "name": "test_llm",
        "description": "Test conversation",
        "metadata": {}
    }

    messages = [
        {
            "message_id": f"ns_{i}",
            "user_prompt": f"Prompt {i}",
            "llm_response": f"Response {i}",
            "metadata": {}
        }
        for i in range(5)
    ]
    pg_handler.save_record(conversation, messages)

    for m in messages:
        pg_handler.save_vector({
            "message_id": m["message_id"],
            "prompt": m["user_prompt"],
            "response": m["llm_response"]
        }, table="message_vectors")

    results = pg_handler.nearest_search(input="Prompt 2", top_k=3, table="message_vectors")

    assert isinstance(results, list)
    assert len(results) <= 3
    # Check that the results contain the expected prompt/response text
    assert any("Prompt 2" in r or "Response 2" in r for r in results)

def test_imported_data(pg_handler):
    """Test importing external data as a conversation"""
    data = {"text": "Imported test sentence banana"}

    pg_handler.import_vectors(table="message_vectors", data=data)

    results = pg_handler.nearest_search(input="banana", top_k=1, table="message_vectors")
    assert len(results) == 1
    assert "Imported test sentence banana" in results[0]
