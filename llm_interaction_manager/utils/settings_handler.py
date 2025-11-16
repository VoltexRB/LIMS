import json
from pathlib import Path
from enum import Enum
from typing import Any, Dict, Optional
from llm_interaction_manager.utils.settings import Settings, RAGMode


class SettingsSection(Enum):
    GENERAL = "general"
    HANDLERS = "handlers"
    DEFAULT_HANDLERS = "default_handlers"


class SettingsHandler:
    """
    Handles reading/writing of dynamic Settings objects to config.json.
    """
    CONFIG_PATH = Path(__file__).parents[2] / "config.json"

    # -------------------------
    # Generic JSON read/write
    # -------------------------
    @staticmethod
    def _read_json() -> Dict[str, Any]:
        """
        Reads the entire data from the config.json file
        :return: Data from config.json in a single dict
        """
        if not SettingsHandler.CONFIG_PATH.exists():
            raise FileNotFoundError(f"{SettingsHandler.CONFIG_PATH} not found")
        try:
            with open(SettingsHandler.CONFIG_PATH, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    # Empty or invalid JSON returns empty dict
                    return {}
        except OSError as e:
            raise OSError(f"Cannot read {SettingsHandler.CONFIG_PATH}: {e}")

    @staticmethod
    def _write_json(data: Dict[str, Any]):
        """
        Writes the entire data dict into the config.json file
        :param data: new settings to be saved in config.json
        """
        try:
            SettingsHandler.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(SettingsHandler.CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except OSError as e:
            raise OSError(f"Cannot write to {SettingsHandler.CONFIG_PATH}: {e}")

    # -------------------------
    # Public read/write
    # -------------------------
    @staticmethod
    def read_setting(section: SettingsSection, key: Optional[str] = None) -> Any:
        """
        Reads a specific section or a single key within that section.
        :param section: SettingsSection selection which setting to read
        :param key: Which Key to read from the section. If omitted, returns all keys
        """
        if not isinstance(section, SettingsSection):
            raise TypeError(f"Expected SettingsSection, got {type(section).__name__}")

        all_settings = SettingsHandler._read_json()
        section_data = all_settings.get(section.value, {})

        if key is not None:
            return section_data.get(key)
        return section_data

    @staticmethod
    def write_setting(section: SettingsSection, value: dict):
        """
        Writes or updates a section in config.json by merging the provided dictionary.
        Creates the file if it does not exist.
        :param section: SettingsSection selection which setting to write to
        :param value: Which key and value to write into the section
        """
        if not isinstance(section, SettingsSection):
            raise TypeError(f"Expected SettingsSection, got {type(section).__name__}")
        if not isinstance(value, dict):
            raise TypeError(f"Expected dict for 'value', got {type(value).__name__}")

        try:
            all_settings = SettingsHandler._read_json()
        except FileNotFoundError:
            all_settings = {}

        section_dict = all_settings.get(section.value, {})
        section_dict.update(value)
        all_settings[section.value] = section_dict
        SettingsHandler._write_json(all_settings)

    # -------------------------
    # Connection handling
    # -------------------------
    @staticmethod
    def get_connection(handler_name: str) -> dict:
        """
        Retrieves stored configuration for a specific handler.
        Raises ValueError if handler not found.
        :param handler_name: Name of the handler to get the connection information for
        """
        handlers_block = SettingsHandler.read_setting(SettingsSection.HANDLERS)
        if handler_name not in handlers_block:
            raise ValueError(f"No handler configuration found for '{handler_name}'")
        return handlers_block[handler_name]

    @staticmethod
    def set_connection(handler_name: str, config: dict):
        """
        Stores or updates the connection data for a specific handler.
        Existing keys are preserved if not overwritten.
        :param handler_name: Name of the handler to set the connection information for
        :param config: Config-Data to save
        """
        if not isinstance(config, dict):
            raise TypeError(f"Expected dict, got {type(config).__name__}")

        try:
            handlers_block = SettingsHandler.read_setting(SettingsSection.HANDLERS)
        except FileNotFoundError:
            handlers_block = {}

        existing_config: dict = handlers_block.get(handler_name, {})
        existing_config.update(config)
        handlers_block[handler_name] = existing_config
        SettingsHandler.write_setting(SettingsSection.HANDLERS, handlers_block)

    # -------------------------
    # Initialization
    # -------------------------
    @staticmethod
    def initialize_settings_object() -> Settings:
        """
        Builds a Settings object dynamically from config.json.
        Missing sections or keys are initialized with "not set" placeholder values.
        :returns: Settings-Object instantiated with the settings from config.json
        """
        try:
            data = SettingsHandler._read_json()
        except FileNotFoundError:
            data = {}

        # Ensure all sections exist
        handlers_block = data.get(SettingsSection.HANDLERS.value, {})
        default_handlers = data.get(SettingsSection.DEFAULT_HANDLERS.value, {})
        general_block = data.get(SettingsSection.GENERAL.value, {})

        # Initialize any missing general settings
        placeholders = {
            "use_rag_data": RAGMode.NONE,
            "on_the_fly_data": {},
            "default_rag_data": {},
            "default_system_prompt": "-1",
            "wait_for_manual_data": False,
            "default_export_path": "-1"
        }
        for key, default_value in placeholders.items():
            if key not in general_block:
                general_block[key] = default_value

        # Convert use_rag_data string to RAGMode if necessary
        if isinstance(general_block.get("use_rag_data"), str):
            try:
                general_block["use_rag_data"] = RAGMode[general_block["use_rag_data"].upper()]
            except KeyError:
                general_block["use_rag_data"] = RAGMode.NONE

        return Settings(
            handlers=handlers_block,
            default_handlers=default_handlers,
            **general_block
        )
