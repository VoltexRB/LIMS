import pytest
import tempfile
from pathlib import Path
from llm_interaction_manager.utils.settings_handler import SettingsHandler, SettingsSection
from llm_interaction_manager.utils.settings import Settings, RAGMode


@pytest.fixture
def temp_config(monkeypatch):
    # Create temporary config file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    # Patch the CONFIG_PATH
    monkeypatch.setattr(SettingsHandler, "CONFIG_PATH", temp_path)

    # Start with empty JSON
    SettingsHandler._write_json({})

    yield temp_path

    # Cleanup
    temp_path.unlink(missing_ok=True)


def test_write_and_read_general_section(temp_config):
    general_data = {
        "default_export_path": "/tmp/export",
        "wait_for_manual_data": False,
        "use_rag_data": "PERSISTENT"
    }
    SettingsHandler.write_setting(SettingsSection.GENERAL, general_data)
    read_data = SettingsHandler.read_setting(SettingsSection.GENERAL)
    assert read_data["default_export_path"] == "/tmp/export"
    assert read_data["wait_for_manual_data"] is False
    assert read_data["use_rag_data"] == "PERSISTENT"


def test_write_and_read_specific_key(temp_config):
    SettingsHandler.write_setting(SettingsSection.GENERAL, {"default_export_path": "/tmp/export"})
    value = SettingsHandler.read_setting(SettingsSection.GENERAL, "default_export_path")
    assert value == "/tmp/export"
    assert SettingsHandler.read_setting(SettingsSection.GENERAL, "nonexistent") is None


def test_set_and_get_connection(temp_config):
    hf_config = {"model_name": "hf-test", "use_gpu": False}
    SettingsHandler.set_connection("huggingface", hf_config)
    retrieved = SettingsHandler.get_connection("huggingface")
    assert retrieved["model_name"] == "hf-test"
    assert retrieved["use_gpu"] is False

    # Update connection
    hf_update = {"model_name": "hf-updated"}
    SettingsHandler.set_connection("huggingface", hf_update)
    retrieved = SettingsHandler.get_connection("huggingface")
    assert retrieved["model_name"] == "hf-updated"
    assert retrieved["use_gpu"] is False  # previous key remains


def test_initialize_settings_object(temp_config):
    # General settings
    general_data = {
        "default_export_path": "/tmp/export",
        "wait_for_manual_data": False,
        "use_rag_data": "VOLATILE",
        "default_system_prompt": "Hello"
    }
    SettingsHandler.write_setting(SettingsSection.GENERAL, general_data)

    # Handlers
    handlers_data = {
        "huggingface": {"model_name": "hf-test", "use_gpu": True},
        "postgres": {"host": "localhost", "port": 5432}
    }
    SettingsHandler.write_setting(SettingsSection.HANDLERS, handlers_data)

    # Default handlers
    default_handlers_data = {"llm": "huggingface", "vector": "postgres"}
    SettingsHandler.write_setting(SettingsSection.DEFAULT_HANDLERS, default_handlers_data)

    # Initialize settings object
    settings_obj = SettingsHandler.initialize_settings_object()

    # Assertions
    assert settings_obj.default_export_path == "/tmp/export"
    assert settings_obj.wait_for_manual_data is False
    assert settings_obj.use_rag_data == RAGMode.VOLATILE
    assert settings_obj.default_system_prompt == "Hello"
    assert "huggingface" in settings_obj.handlers
    assert "postgres" in settings_obj.handlers
    assert settings_obj.default_handlers["llm"] == "huggingface"

def test_initialize_with_missing_file(monkeypatch):
    """
    Simulate first run: config file does not exist.
    SettingsHandler should initialize Settings object with placeholders.
    """
    temp_path = Path(tempfile.mktemp())
    monkeypatch.setattr(SettingsHandler, "CONFIG_PATH", temp_path)

    # Ensure file does not exist
    if temp_path.exists():
        temp_path.unlink()

    settings_obj = SettingsHandler.initialize_settings_object()

    # Placeholders should be used
    assert settings_obj.default_export_path == "-1"
    assert settings_obj.wait_for_manual_data is True
    assert settings_obj.use_rag_data == RAGMode.NONE
    assert settings_obj.default_system_prompt == "-1"
    assert settings_obj.handlers == {}
    assert settings_obj.default_handlers == {}
    assert settings_obj.on_the_fly_data == {}
    assert settings_obj.default_rag_data == {}

    # Cleanup
    temp_path.unlink(missing_ok=True)


def test_initialize_with_empty_file(temp_config):
    """
    Simulate an empty config.json file.
    SettingsHandler should still initialize Settings object with placeholders.
    """
    # Empty file already handled by temp_config fixture
    SettingsHandler._write_json({})

    settings_obj = SettingsHandler.initialize_settings_object()

    # Placeholders should be used
    assert settings_obj.default_export_path == "-1"
    assert settings_obj.wait_for_manual_data is True
    assert settings_obj.use_rag_data == RAGMode.NONE
    assert settings_obj.default_system_prompt == "-1"
    assert settings_obj.handlers == {}
    assert settings_obj.default_handlers == {}
    assert settings_obj.on_the_fly_data == {}
    assert settings_obj.default_rag_data == {}

def test_read_missing_section_returns_empty(temp_config):
    """
    Reading a section that does not exist should return an empty dict.
    """
    result = SettingsHandler.read_setting(SettingsSection.HANDLERS)
    assert result == {}
    result = SettingsHandler.read_setting(SettingsSection.DEFAULT_HANDLERS)
    assert result == {}

def test_get_connection_missing_handler_raises(temp_config):
    """
    Attempting to get a handler that is not set should raise ValueError.
    """
    with pytest.raises(ValueError):
        SettingsHandler.get_connection("nonexistent_handler")


def test_write_creates_file_if_missing(monkeypatch):
    """
    Writing to a section should create the file if it does not exist.
    """
    temp_path = Path(tempfile.mktemp())
    monkeypatch.setattr(SettingsHandler, "CONFIG_PATH", temp_path)

    SettingsHandler.write_setting(SettingsSection.GENERAL, {"default_export_path": "/tmp/export"})
    assert temp_path.exists()

    data = SettingsHandler._read_json()
    assert data["general"]["default_export_path"] == "/tmp/export"

    # Cleanup
    temp_path.unlink(missing_ok=True)