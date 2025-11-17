from datetime import datetime, timezone
from llm_interaction_manager.handlers.persistent_data_handler_base import PersistentDataHandlerBase
import pymongo
from pymongo import *
from pymongo.errors import *

class MongodbHandler(PersistentDataHandlerBase):

    def __init__(self):
        self.client = None
        self.db = None

    def get_info(self) -> dict:
        """
        Returns the connection information to be saved in the settings
        :return: dict with the connection information
        """
        return {**self.auth, "host": self.host, "port": self.port}

    def get_name(self) -> str:
        """
        Returns the name of the Handler for dynamic binding
        :return: String-Object "mongodb"
        """
        return "mongodb"

    def save_record(self, conversation: dict, messages: list[dict]):
        """
        Save a conversation and its messages to MongoDB.
        :param conversation: dict containing conversation-level data. Must contain "conversation_id"
        :param messages: list of dicts containing message-level data. Must contain "message_id"
        """
        if not self._fully_initialized():
            raise ConnectionError("MongoDB client/database not initialized. Call connect first and select a database.")

        conv_id = conversation.get("conversation_id")
        if not conv_id:
            raise ValueError("conversation dict must have 'conversation_id'")

        self.db["conversations"].update_one(
            {"_id": conv_id},
            {"$set": conversation},
            upsert=True)

        # Process each message
        for msg in messages:
            message_id = msg.get("message_id")
            if not message_id:
                raise ValueError("Each message dict must have a 'message_id'")

            # Try to find the existing message
            existing_conv = self.db["conversations"].find_one(
                {"_id": conv_id, "messages.message_id": message_id},
                {"messages.$": 1}
            )

            if existing_conv and existing_conv.get("messages"):
                # Merge: update only the keys provided
                merged = existing_conv["messages"][0].copy()
                merged.update(msg)  # new keys overwrite old keys, existing keys not in msg stay
                self.db["conversations"].update_one(
                    {"_id": conv_id, "messages.message_id": message_id},
                    {"$set": {"messages.$": merged}}
                )
            else:
                # Append new message
                self.db["conversations"].update_one(
                    {"_id": conv_id},
                    {"$push": {"messages": msg}}
                )

    def get_data(self, filters: dict[str, Any] | None = None) -> list[dict]:
        """
        Dynamically retrieve conversation and message data from MongoDB
        consistent with PostgresHandler.get_data() structure. May result in different attributes, depending on the keys in the database.
        Only returns conversations that have messages matching message-level filters if any are applied.
        :param filters: Optional dictionary of filter conditions.
        :return: List of conversations, each with a 'messages' list
        """
        if not self._fully_initialized():
            raise ConnectionError(
                "MongoDB client/database not initialized. Call connect() first and select a database."
            )

        mongo_filter = {}

        # Top-level (conversation-level) filters
        if filters and "conversation_id" in filters:
            mongo_filter["_id"] = filters["conversation_id"]

        cursor = self.db["conversations"].find(mongo_filter)
        conversations = list(cursor)
        results = []

        for conv in conversations:
            conv_copy = {k: v for k, v in conv.items() if k != "messages"}
            conv_copy["conversation_id"] = conv_copy.pop("_id", None)

            messages_out = []
            for msg in conv.get("messages", []):
                if filters:
                    skip = False
                    for key, value in filters.items():
                        if key == "conversation_id":
                            continue
                        if key in msg:
                            msg_value = str(msg[key]).lower() if msg[key] is not None else ""
                            if str(value).lower() not in msg_value:
                                skip = True
                                break
                        elif key in conv_copy:
                            conv_value = str(conv_copy[key]).lower() if conv_copy[key] is not None else ""
                            if str(value).lower() not in conv_value:
                                skip = True
                                break
                    if skip:
                        continue

                messages_out.append(dict(msg))

            # Only include conversation if messages_out is non-empty
            if messages_out:
                conv_copy["messages"] = messages_out
                results.append(conv_copy)

        return results

    def is_connected(self) -> bool:
        """
        Returns True if the MongoDB client is connected and reachable.
        """
        if not getattr(self, "client", None):
            return False

        try:
            # Ping the server to test connectivity
            self.client.admin.command("ping")
            return True
        except:
            return False

    def connect(self, host: str, port: int, auth: dict = None) -> bool:
        """
        Connects to the mongodb instance specified by host, port and the authentication data
        :param host: Host to the MongoDB instance
        :param port: Port the MongoDB instance is running under
        :param auth: Authentication-Data like a mandatory database or optional Username and Password
        :return: True if connection successfully established
        """
        # Ensure a database name is always provided
        if not auth or "database" not in auth:
            raise ValueError("A 'database' key must be provided in auth, even without authentication")

        try:
            # Authenticated connection
            if all(k in auth for k in ("username", "password", "database")):
                self.client = MongoClient(
                    host=host,
                    port=port,
                    username=auth["username"],
                    password=auth["password"],
                    authSource=auth["database"],
                    serverSelectionTimeoutMS=3000
                )
            # Unauthenticated connection
            else:
                self.client = MongoClient(
                    host=host,
                    port=port,
                    serverSelectionTimeoutMS=3000
                )

            # Select the database
            self.db = self.client[auth["database"]]
            # Test connection
            self.client.admin.command("ping")
            self.host = host
            self.port = port
            self.auth = auth
            return True
        except ConnectionFailure as e:
            raise Exception(f"Failed to connect to MongoDB: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error during MongoDB connection: {e}")

    def select_database(self, db_name: str):
        """
        Selects or switches the active MongoDB database.
        Must be called after a successful connection.
        :param db_name: The name of the database to be selected
        """
        if not self.client:
            raise ConnectionError("MongoDB client not initialized. Call connect() first.")

        try:
            self.db = self.client.get_database(db_name)
            return True
        except Exception as e:
            raise Exception(f"Failed to select database '{db_name}': {e}")

    def _fully_initialized(self) -> bool:
        """
        Check if the client is initialized and a database is selected
        :return: True if client connected and database selected
        """
        if self.client is not None and self.db is not None: return True
        else: return False
