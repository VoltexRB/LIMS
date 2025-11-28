from llm_interaction_manager.handlers.data_handler_base import DataHandlerBase
from abc import ABC, abstractmethod
from typing import Any

class PersistentDataHandlerBase (DataHandlerBase, ABC):

    @abstractmethod
    def save_record(self, conversation: dict, messages: list[dict]):
        """
        Save conversation and messages to the persistent storage.
        Always requires both items to guarantee reference between messages and their conversation

        :param conversation: dict containing conversation-level data. Must contain "conversation_id"
        :param messages: list of dicts containing message-level data. Must contain "message_id"
        """

    @abstractmethod
    def get_data(self, filters: dict[str, Any] | None = None) -> list[dict]:
        """
        Gets one or more records from the database.

        :param filters: Optional filters to select specific records.
                        Example:
                            {"conversation_id": "conv123"}
                            {"created_at": {"$gte": "2025-01-01"}}
                        If None, all records are returned.
        :type filters: dict[str, Any] | None
        :return: A list of recovered records.
        :rtype: list[dict]
        """

    @abstractmethod
    def select_database(self, db_name: str):
        """Selects or switches the active database after connection."""
        pass

