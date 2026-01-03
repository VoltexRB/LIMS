import importlib
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from sympy.strategies.core import switch

from llm_interaction_manager.handlers.persistent_data_handler_base import PersistentDataHandlerBase
from llm_interaction_manager.handlers.llm_handler_base import LLMHandlerBase
from llm_interaction_manager.handlers.vector_data_handler_base import VectorDataHandlerBase
from llm_interaction_manager.utils.settings import Settings
from llm_interaction_manager.utils.settings_handler import SettingsHandler, SettingsSection
from .conversation import Conversation
from ..utils import RAGMode


class ConnectionType(Enum):
    """
    Which Handler-Type to use for generic methods like connect()
    """
    LLM = "llm",
    VECTOR = "vector",
    PERSISTENT = "persistent"

class InteractionManager:
    """
    Interaction Keystone for the whole interface. All interaction should be done over the Interaction Manager. If more granular approaches are necessary,
    Individual handlers and classes within the Interface can also be accessed separately, but consistency is not secured
    """
    llm_handler: LLMHandlerBase
    persistent_handler: PersistentDataHandlerBase
    vector_handler: VectorDataHandlerBase
    settings: Settings
    conversation: Optional[Conversation]

    def _dynamic_handler_factory(self, handler_name: str):
        """
        Dynamically loads a llm handler class from llm_interaction_manager/handlers and initializes it with the provided configuration.
        Mostly used in __init__

        :param handler_name: Name of the handler (eg. huggingface) that should come from saved settings
        :return: Instantiated handler object
        """
        module_path = f"llm_interaction_manager.handlers.{handler_name}_handler"
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError:
            raise ImportError(f"Cannot find module for {handler_name}_handler in llm_interaction_manager.handlers")
        class_name = f"{handler_name.capitalize()}Handler"

        try:
            handler_cls = getattr(module, class_name)
        except AttributeError:
            raise ImportError(f"Module '{module_path}' does not define class '{class_name}'")
        return handler_cls()

    def __init__(self, llm_handler:LLMHandlerBase = None, persistent_handler:PersistentDataHandlerBase = None, vector_handler:VectorDataHandlerBase = None, settings: Settings = None):
        """
        Initializes an InteractionManager object with Handlers that are either provided or received from the settings and created.
        Throws ValueErrors if Handlers are not provided and connection information is not found in the settings

        :param llm_handler: Optional LLMHandler to use
        :param persistent_handler: Optional PersistenDataHandler to use
        :param vector_handler: Optional VectorDataHandler to use
        """
        if settings is None:
            self.settings = SettingsHandler.initialize_settings_object()
        else:
            self.settings = settings

        # LLM Handler not handed over
        if llm_handler is None:
            default_llm_name = self.settings.default_handlers.get("llm", None)
            if not default_llm_name:
                raise ValueError("No LLM Handler handed over on init and no default LLM connection saved yet")

            #retrieve configuration from the settings
            llm_config = self.settings.handlers.get(default_llm_name, None)
            if llm_config is None:
                raise ValueError(f"No configuration found in settings.handlers for '{default_llm_name}'.")
            llm_handler: LLMHandlerBase = self._dynamic_handler_factory(default_llm_name)

            #Connect with the specific connection method
            llm_connected = llm_handler.connect(llm_config)

        # if handed over, save connection settings
        elif llm_handler is not None and llm_handler.is_connected():
            name = llm_handler.get_name()
            info = llm_handler.get_info()
            self.settings.default_handlers["llm"] = name
            SettingsHandler.write_setting(SettingsSection.DEFAULT_HANDLERS, {"llm": name})

            self.settings.handlers[name].update(info)
            SettingsHandler.set_connection(name,info)

        # Persistent Data Handler not handed over
        if persistent_handler is None:
            default_persistent_name = self.settings.default_handlers.get("persistent")
            if not default_persistent_name:
                raise ValueError(
                    "No Persistent Handler handed over on init and no default Persistent connection saved yet"
                )

            persistent_config: dict = self.settings.handlers.get(default_persistent_name, None)
            if persistent_config is None:
                raise ValueError(f"No configuration found in settings.handlers for '{default_persistent_name}'.")

            persistent_handler: PersistentDataHandlerBase = self._dynamic_handler_factory(default_persistent_name)

            #Connect with the specific connection method
            host: str = persistent_config.pop("host", "-1")
            port: int = persistent_config.pop("port", -1)
            persistent_connected = persistent_handler.connect(host, port, persistent_config)

        # if handed over, save connection settings
        elif persistent_handler is not None and persistent_handler.is_connected():
            name = persistent_handler.get_name()
            info = persistent_handler.get_info()
            self.settings.default_handlers["persistent"] = name
            SettingsHandler.write_setting(SettingsSection.DEFAULT_HANDLERS, {"persistent": name})

            self.settings.handlers[name].update(info)
            SettingsHandler.set_connection(name,info)

        #Vector Data Handler not handed over
        if vector_handler is None:
            default_vector_name = self.settings.default_handlers.get("vector")
            if not default_vector_name:
                raise ValueError(
                    "No Vector Handler handed over on init and no default Vector connection saved yet"
                )

            vector_config: dict = self.settings.handlers.get(default_vector_name, None)
            if vector_config is None:
                raise ValueError(f"No configuration found in settings.handlers for '{default_vector_name}'.")

            vector_handler: VectorDataHandlerBase = self._dynamic_handler_factory(default_vector_name)

            #Connect with the specific connection method
            host: str = vector_config.pop("host", "-1")
            port: int = vector_config.pop("port", -1)
            vector_connected = vector_handler.connect(host=host, port=port, auth=vector_config)

        # if handed over, save connection settings
        elif vector_handler is not None and vector_handler.is_connected():
            name = vector_handler.get_name()
            info = vector_handler.get_info()
            self.settings.default_handlers["vector"] = name
            SettingsHandler.write_setting(SettingsSection.DEFAULT_HANDLERS, {"vector": name})

            self.settings.handlers[name].update(info)
            SettingsHandler.set_connection(name,info)

        # Assign handlers to instance
        self.llm_handler = llm_handler
        self.persistent_handler = persistent_handler
        self.vector_handler = vector_handler
        self.conversation = None

    def start_conversation(self, conversation_metadata: dict = None):
        """
        Creates a new conversation object and possibly adds conversation-metadata to it

        :param conversation_metadata: data to add to the conversation-object
        """
        if not all([
            self.llm_handler is not None and self.llm_handler.is_connected(),
            self.persistent_handler is not None and self.persistent_handler.is_connected(),
            self.vector_handler is not None and self.vector_handler.is_connected()
        ]):
            raise RuntimeError(f"Conversations can only be started when all 3 handlers are instantiated and connected")
        try:
            self.conversation = Conversation(llm= self.llm_handler, vector=self.vector_handler, persistent=self.persistent_handler, settings=self.settings, conversation_metadata=conversation_metadata)
        except Exception as e:
            raise RuntimeError(f"Conversation could not be initialized: {e}")

    def set_rag_data(self, data: dict, volatile: bool):
        """
        Adds new RAG-Data to the current run and sets the RAG-Mode accordingly, if volatile is set to false, RAG-data added will be saved to the config aswell

        :param data: data to be used as RAG-Data
        :param volatile: If the RAG-Data should be used on the fly or saved in the settings for further use
        """
        if volatile:
            self.settings.on_the_fly_data = data
            self.settings.use_rag_data = RAGMode.VOLATILE
        else:
            self.settings.default_rag_data = data
            self.settings.use_rag_data = RAGMode.PERSISTENT
            SettingsHandler.write_setting(SettingsSection.GENERAL, {"default_rag_data": data})

    def set_rag_mode(self, mode: RAGMode):
        """
        Sets the RAG-Mode of the application

        :param mode: The mode to set it to
        """
        if mode == RAGMode.VOLATILE:
            if not self.settings.on_the_fly_data:
                raise ValueError("Cannot set RAG mode to VOLATILE because no on-the-fly RAG data is available. Use set_rag_data() instead.")

        elif mode == RAGMode.PERSISTENT:
            if not self.settings.default_rag_data:
                raise ValueError("Cannot set RAG mode to PERSISTENT because no persistent RAG data is available. Use set_rag_data() instead.")
        self.settings.use_rag_data = mode

    def delete_rag_data(self):
        """
        Turns off RAG-Data usage and deletes any RAG-Data, persistent and volatile
        """
        self.settings.use_rag_data = RAGMode.NONE
        self.settings.on_the_fly_data = {}
        self.settings.default_rag_data = {}
        SettingsHandler.write_setting(SettingsSection.GENERAL, {"default_rag_data": {}})

    def nearest_search_vector(self, input: str, top_k: int, table: str) -> list[str]:
        """
        Searches the vector database for close matches of the input and returns the top_k results. May result in a combination of Prompt and Response lines depending on Interface

        :param input: String data to generate an embedding from to use nearest search for on the vector database.
        :param top_k: How many results should be returned
        :param table: Which table to do the nearest search on
        :return: List of strings where the embeddings are matched near the input embedding
        """
        if self.vector_handler is None or not self.vector_handler.is_connected():
            raise RuntimeError("Vector Handler not connected or not initialized")
        return self.vector_handler.nearest_search(input, top_k, table)

    def export_data(self, to: str, filters: dict[str, Any] | None = None):
        """
        Exports data from the persistent database to the specified path

        :param to: Path to export the data to
        :param filters: Filters to apply to the data. If no filters are specified, all available data will be exported
        """
        if self.persistent_handler is None or not self.persistent_handler.is_connected():
            raise RuntimeError("Persistent Handler not connected or not initialized")

        data: list[dict] =  self.persistent_handler.get_data(filters)

        if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
            raise TypeError("Persistent handler returned invalid data, expected List[Dict]")

        #convert datetime object to ISO-8601 strings so they can be saved in json
        data = self._recursive_convert(data)

        export_dir = Path(to)
        export_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"lims_export_{timestamp}.json"
        filepath = export_dir / filename
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except OSError as e:
            raise RuntimeError(f"Failed to write JSON file to {filepath}: {e}")

    def _recursive_convert(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: self._recursive_convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._recursive_convert(i) for i in obj]
        return obj

    def add_persistent_data(self, conversation: dict, messages: list[dict]):
        """
        Adds a new persistent entry into the database handler.
        Must contain a conversation_id and at least one message with messageID. View data_schemas_example.json in utils for reference

        :param conversation: Conversation to be added to the database
        :param messages: list of Messages to be added to the database and linked to the conversation
        """
        if self.persistent_handler is None or not self.persistent_handler.is_connected():
            raise RuntimeError("Persistent Handler not connected or not initialized")
        self.persistent_handler.save_record(conversation, messages)

    def add_vector_data(self, data: dict, table: str):
        """
        Adds a new entry into the vector database in reference to a persistent data entry

        :param data: Dict of data, must at least contain: message_id, prompt, and response
        :param table: Table to save the entry into (or collection in chromadb)
        """
        if self.vector_handler is None or not self.vector_handler.is_connected():
            raise RuntimeError("Vector Handler not connected or not initialized")
        self.vector_handler.save_vector(data, table)

    def is_connected(self, target: ConnectionType) -> bool:
        """
        Check if the specified handler is connected

        :param target: ConnectionType-Enum, which handler should be checked
        :return: True if specified handler is connected, False else
        """
        if target == ConnectionType.PERSISTENT:
            return self.persistent_handler.is_connected()
        elif target == ConnectionType.VECTOR:
            return self.vector_handler.is_connected()
        else:
            return self.llm_handler.is_connected()

    def connect(self, target: ConnectionType, data: dict):
        """
        Connect to the specified handlers endpoint

        :param target: ConnectionType-Enum, which handler should be checked
        :param data: Data to use for connection (eg. API tokens, routing information)
        """

        # prevents call by reference destruction
        d = data.copy()

        if target == ConnectionType.PERSISTENT:
            host: str = d.pop("host", "-1")
            port: int = d.pop("port", -1)
            self.persistent_handler.connect(host, port, d)

        elif target == ConnectionType.VECTOR:
            host: str = d.pop("host", "-1")
            port: int = d.pop("port", -1)
            self.vector_handler.connect(host, port, d)

        else:
            self.llm_handler.connect(d)

    def read_setting(self, key: str):
        """
        Reads a setting from the current Settings-Object

        :param key: Which setting to read
        :return: Requested Object from Settings
        """
        if not hasattr(Settings, key):
            raise KeyError(f"Setting {key} does not exist in Settings")
        return getattr(self, key)

    def write_setting(self, key: str, value):
        """
        Updates a setting in the Settings Object and propagates to config.json

        :param key: Key to write into the settings
        :param value: Value to be changed
        """
        if not hasattr(Settings, key):
            raise KeyError(f"Setting {key} does not exist in Settings")
        setattr(self.settings, key, value)
        SettingsHandler.write_setting(SettingsSection.GENERAL, {key: value})

    def add_metadata(self, to_conversation: bool, data: dict):
        """
        Adds metadata to the last message that was sent/received or the current conversation-object

        :param to_conversation: If the metadata should be added to the conversation-object or the last message
        :param data: Data to add to the metadata
        """
        if self.conversation is None:
            raise RuntimeError("No conversation initialized yet. Use start_conversation() first.")
        self.conversation.add_metadata(to_conversation, data)

    def send_prompt(self, prompt: str) -> dict:
        """
        Sends a prompt to the LLM-Handler and thus to the specified LLM. Receives a response, saves the response persistently in the databases

        :param prompt: Prompt to send to the LLM-Handler. If RAG-Data is specified, Prompt and RAG-Data will be combined
        :return: Answer from LLM
        """
        full_prompt = prompt
        if self.conversation is None:
            raise RuntimeError("No conversation initialized yet. Use start_conversation() first.")

        if self.settings.default_system_prompt != "-1":
            full_prompt = "SYSTEM PROMPT: " + self.settings.default_system_prompt + " PROMPT: " + prompt
        return self.conversation.send_prompt(full_prompt)

    def change_comment(self, comment: str):
        """
        Changes the comment on the last sent message

        :param comment: Comment to add to the last sent message
        """
        if self.conversation is None:
            raise RuntimeError("No conversation initialized yet. Use start_conversation() first.")
        return self.conversation.change_comment(comment)



