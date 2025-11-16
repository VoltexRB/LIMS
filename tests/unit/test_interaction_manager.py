import pytest
from unittest.mock import MagicMock, patch
from llm_interaction_manager.core.interaction_manager import InteractionManager, ConnectionType
from llm_interaction_manager.utils.settings_handler import SettingsHandler


@pytest.fixture
def mock_handlers():
    llm = MagicMock()
    persistent = MagicMock()
    vector = MagicMock()

    # Connection status
    llm.is_connected.return_value = True
    persistent.is_connected.return_value = True
    vector.is_connected.return_value = True

    # LLM response
    llm.send_prompt.return_value = {"response": "Mock Response"}

    # Names and info
    llm.get_name.return_value = "mock_llm"
    llm.get_info.return_value = {"param": "value"}

    persistent.get_name.return_value = "mock_persistent"
    persistent.get_info.return_value = {"param": "value"}

    vector.get_name.return_value = "mock_vector"
    vector.get_info.return_value = {"param": "value"}

    return llm, persistent, vector


@pytest.fixture(autouse=True)
def patch_settings_and_write(mock_handlers):
    llm, persistent, vector = mock_handlers

    # Create a settings object with pre-populated handlers
    settings = SettingsHandler.initialize_settings_object()
    settings.handlers = {
        "mock_llm": {},
        "mock_persistent": {},
        "mock_vector": {}
    }

    with patch(
        "llm_interaction_manager.utils.settings_handler.SettingsHandler.initialize_settings_object",
        return_value=settings
    ), patch(
        "llm_interaction_manager.utils.settings_handler.SettingsHandler.write_setting"
    ):
        yield


def test_interaction_manager_init_with_mocks(mock_handlers):
    llm, persistent, vector = mock_handlers
    im = InteractionManager(
        llm_handler=llm,
        persistent_handler=persistent,
        vector_handler=vector
    )

    assert im.llm_handler.is_connected()
    assert im.persistent_handler.is_connected()
    assert im.vector_handler.is_connected()


def test_start_conversation_and_send_prompt(mock_handlers):
    llm, persistent, vector = mock_handlers
    im = InteractionManager(
        llm_handler=llm,
        persistent_handler=persistent,
        vector_handler=vector
    )

    im.start_conversation({"topic": "test"})
    assert im.conversation is not None

    with patch("builtins.input", return_value=""):
        response = im.send_prompt("Hello")

    assert response["content"] == "Mock Response"
    assert len(im.conversation.conversation_history) == 1


def test_rag_data_settings_applied(mock_handlers):
    llm, persistent, vector = mock_handlers
    im = InteractionManager(
        llm_handler=llm,
        persistent_handler=persistent,
        vector_handler=vector
    )

    im.set_rag_data({"key": "value"}, volatile=True)
    assert im.settings.use_rag_data.name == "VOLATILE"

    im.set_rag_data({"key2": "value2"}, volatile=False)
    assert im.settings.use_rag_data.name == "PERSISTENT"


def test_delete_rag_data(mock_handlers):
    llm, persistent, vector = mock_handlers
    im = InteractionManager(
        llm_handler=llm,
        persistent_handler=persistent,
        vector_handler=vector
    )

    im.set_rag_data({"key": "value"}, volatile=True)
    im.delete_rag_data()
    assert im.settings.use_rag_data.name == "NONE"


def test_connection_methods(mock_handlers):
    llm, persistent, vector = mock_handlers
    im = InteractionManager(
        llm_handler=llm,
        persistent_handler=persistent,
        vector_handler=vector
    )

    assert im.is_connected(ConnectionType.LLM)
    assert im.is_connected(ConnectionType.PERSISTENT)
    assert im.is_connected(ConnectionType.VECTOR)

    im.connect(ConnectionType.LLM, {"token": "abc"})
    llm.connect.assert_called_once_with({"token": "abc"})


def test_read_write_setting(mock_handlers):
    llm, persistent, vector = mock_handlers
    im = InteractionManager(
        llm_handler=llm,
        persistent_handler=persistent,
        vector_handler=vector
    )

    im.write_setting("wait_for_manual_data", True)
    value = im.settings.wait_for_manual_data
    assert value is True
