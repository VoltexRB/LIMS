import pytest
from llm_interaction_manager.handlers.mongodb_handler import MongodbHandler

@pytest.fixture()
def mongo_handler():
    """
    Fixture that initializes the MongoDB handler and connects to a local instance.
    """
    handler = MongodbHandler()
    connected = handler.connect(host="localhost", port=27017, auth={"database": "promptDB"})
    handler.select_database("promptDB")
    handler.db["conversations"].delete_many({})
    yield handler

def test_connection(mongo_handler):
    """
    Verify MongoDB connection is active.
    """
    assert mongo_handler.is_connected() is True


def test_save_and_retrieve_conversation(mongo_handler):
    """
    Test saving a new conversation and retrieving it.
    """
    conversation = {
        "conversation_id": "conv_001",
        "created_at": "2025-11-03T10:00:00Z",
        "metadata": {"topic": "test_conversation"}
    }
    messages = [
        {"message_id": "msg_001", "role": "user", "content": "Hello"},
        {"message_id": "msg_002", "role": "assistant", "content": "Hi there"}
    ]

    mongo_handler.save_record(conversation, messages)

    # Retrieve and verify
    data = mongo_handler.get_data({"conversation_id": "conv_001"})
    assert len(data) == 1
    conv = data[0]
    assert conv["conversation_id"] == "conv_001"
    assert len(conv["messages"]) == 2


def test_update_existing_message(mongo_handler):
    """
    Test partial update of an existing message.
    """
    updated_msg = {"message_id": "msg_001", "role": "user", "content": "Hello, updated!"}
    mongo_handler.save_record({"conversation_id": "conv_001"}, [updated_msg])

    # Verify update applied
    data = mongo_handler.get_data({"conversation_id": "conv_001"})
    user_msg = next((m for m in data[0]["messages"] if m["message_id"] == "msg_001"), None)
    assert user_msg["content"] == "Hello, updated!"


def test_append_new_message(mongo_handler):
    """
    Test appending a new message to existing conversation.
    """
    new_msg = {"message_id": "msg_003", "role": "assistant", "content": "Adding a new message"}
    mongo_handler.save_record({"conversation_id": "conv_001"}, [new_msg])

    data = mongo_handler.get_data({"conversation_id": "conv_001"})
    msg_ids = [m["message_id"] for m in data[0]["messages"]]
    assert "msg_003" in msg_ids


def test_filtering_by_message_content(mongo_handler):
    """
    Verify filtering works at message level.
    """

    # Insert mock data for this test
    conversation = {"conversation_id": "conv_filter_test"}
    messages = [
        {"message_id": "msg_001", "role": "user", "content": "Hello, updated!"},
        {"message_id": "msg_002", "role": "assistant", "content": "No match here"}
    ]
    mongo_handler.save_record(conversation, messages)

    # Fetch filtered data
    result = mongo_handler.get_data({"content": "updated"})

    # Debug print to see what is returned
    print("Filtered result:", result)
    for conv in result:
        print("Conversation ID:", conv.get("conversation_id"))
        print("Messages:", conv.get("messages"))

    # Assertions
    assert len(result) == 1
    assert len(result[0]["messages"]) == 1
    assert result[0]["messages"][0]["message_id"] == "msg_001"



def test_reconnect_and_ping(mongo_handler):
    """
    Ensure reconnection logic and ping check work correctly.
    """
    assert mongo_handler.is_connected()
    mongo_handler.client.close()
    assert mongo_handler.is_connected() is False

    # Reconnect
    connected = mongo_handler.connect(host="localhost", port=27017, auth={"database": "promptDB"})
    assert connected
    assert mongo_handler.is_connected() is True