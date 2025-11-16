import pytest
from unittest.mock import MagicMock
from llm_interaction_manager.core.conversation import Conversation
from llm_interaction_manager.utils.settings import Settings, RAGMode


@pytest.fixture
def mock_handlers():
    llm = MagicMock()
    persistent = MagicMock()
    vector = MagicMock()
    return llm, persistent, vector


@pytest.fixture
def default_settings():
    return Settings(
        use_rag_data=RAGMode.NONE,
        on_the_fly_data={"doc1": "text1"},
        default_rag_data={"doc2": "text2"},
        wait_for_manual_data=False
    )


def test_send_prompt_basic(mock_handlers, default_settings):
    llm, persistent, vector = mock_handlers
    llm.send_prompt.return_value = {"response": "Hello World"}

    conv = Conversation(llm, persistent, vector, default_settings)
    response = conv.send_prompt("Hi LLM")

    # LLM called correctly
    llm.send_prompt.assert_called_once_with("Hi LLM")
    # History updated
    assert conv.get_last_response() == "Hello World"
    # Handlers called
    persistent.save_record.assert_called_once()
    vector.save_vector.assert_called_once()
    # RAG data not added
    assert "RAG-Data" not in response


@pytest.mark.parametrize("rag_mode", [RAGMode.VOLATILE, RAGMode.PERSISTENT])
def test_send_prompt_rag_modes(mock_handlers, default_settings, rag_mode):
    llm, persistent, vector = mock_handlers
    llm.send_prompt.return_value = {"response": "RAG Response"}

    settings = default_settings
    settings.use_rag_data = rag_mode

    conv = Conversation(llm, persistent, vector, settings)
    response = conv.send_prompt("Query")

    # LLM called with rag_list
    if rag_mode == RAGMode.VOLATILE:
        llm.send_prompt.assert_called_with("Query", list(settings.on_the_fly_data.values()))
        assert response["RAG-Data"] == list(settings.on_the_fly_data.values())
    else:
        llm.send_prompt.assert_called_with("Query", list(settings.default_rag_data.values()))
        assert response["RAG-Data"] == list(settings.default_rag_data.values())


def test_metadata_add_remove(mock_handlers, default_settings):
    llm, persistent, vector = mock_handlers
    llm.send_prompt.return_value = {"response": "Test"}

    conv = Conversation(llm, persistent, vector, default_settings)
    conv.send_prompt("Prompt1")

    # Add conversation metadata
    conv.add_metadata(conversation=True, data={"user": "Alice"})
    assert conv.conversation_metadata["user"] == "Alice"

    # Add message metadata
    conv.add_metadata(conversation=False, data={"topic": "test"})
    last_msg = conv.conversation_history[-1]
    assert last_msg["metadata"]["topic"] == "test"

    # Remove conversation metadata
    conv.remove_metadata(conversation=True, key="user")
    assert "user" not in conv.conversation_metadata

    # Removing protected key raises ValueError
    with pytest.raises(ValueError):
        conv.remove_metadata(conversation=True, key="prompt")


def test_change_comment(mock_handlers, default_settings):
    llm, persistent, vector = mock_handlers
    llm.send_prompt.return_value = {"response": "Hi"}

    conv = Conversation(llm, persistent, vector, default_settings)
    conv.send_prompt("Hello")
    conv.change_comment("New Comment")
    assert conv.conversation_history[-1]["comment"] == "New Comment"
