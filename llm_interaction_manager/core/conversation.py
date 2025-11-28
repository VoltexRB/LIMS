from datetime import datetime, timezone
from llm_interaction_manager.handlers.persistent_data_handler_base import PersistentDataHandlerBase
from llm_interaction_manager.handlers.vector_data_handler_base import VectorDataHandlerBase
from llm_interaction_manager.handlers.llm_handler_base import LLMHandlerBase
from llm_interaction_manager.utils.settings import Settings, RAGMode
import uuid

class Conversation:
    """
    Saves Conversations and their History, sent and received messages aswell as additional metadata
    """
    conversation_id: str
    conversation_metadata: dict
    conversation_history: list[dict]
    created_at: float
    llm_handler: LLMHandlerBase
    persistent_handler: PersistentDataHandlerBase
    vector_handler: VectorDataHandlerBase
    settings: Settings

    def __init__(self, llm: LLMHandlerBase, persistent: PersistentDataHandlerBase, vector: VectorDataHandlerBase, settings: Settings, conversation_metadata: dict = None):
        """
        Initializes a new conversation with the three handlers, the current settings object, and default configurations.
        If conversation metadata is provided during initialization, it will also be stored.
        Additionally, the conversation is created with its own unique conversation ID.

        :param llm: LLM handler object
        :param persistent: Persistent handler object
        :param vector: Vector handler object
        :param settings: Settings object
        :param conversation_metadata: Optional conversation metadata to be added to the conversation object
        """
        self.llm_handler = llm
        self.persistent_handler = persistent
        self.vector_handler = vector
        self.settings = settings
        self.conversation_metadata = conversation_metadata or {}
        self.conversation_id = "conv_" + str(uuid.uuid4())[5:]
        self.conversation_history = []
        self.created_at = datetime.now(timezone.utc).timestamp()

    def send_prompt(self, prompt: str) -> dict:
        """
        Sends the prompt to the LLM handler and retrieves the response.
        If the usage of RAG-data is defined in the settings, the specified RAG-Data will also be sent.
        If manual user comments are enabled, the program flow will pause to allow comment input.
        Depending on the setting "send_conversation_history" previous messages in the conversation will also be sent to the LLM

        :param prompt: Prompt to be sent to the LLM handler
        :return: Dictionary object containing the prompt, response, optional user comment, RAG data, and additional metadata
        """
        #Since Rag-Data can be saved as both dict or list of strings, make sure both are handed over as a list
        def _normalize_rag_data(data):
            if isinstance(data, dict):
                return list(data.values())
            elif isinstance(data, list):
                return data
            else:
                return []

        rag_list = []
        # Include previous conversation messages if enabled
        if self.settings.send_conversation_history:
            for msg in self.conversation_history[-10:]:
                # Add both user prompt and LLM response to context
                rag_list.append("PREVIOUS PROMPT: " + msg["prompt"])
                rag_list.append("PREVIOUS RESPONSE: " + msg["content"])

        # Include other RAG data if enabled
        if self.settings.use_rag_data == RAGMode.VOLATILE:
            rag_list += _normalize_rag_data(self.settings.on_the_fly_data)

        elif self.settings.use_rag_data == RAGMode.PERSISTENT:
            rag_list += _normalize_rag_data(self.settings.default_rag_data)

        elif self.settings.use_rag_data == RAGMode.DYNAMIC:
            #get the 10 nearest vectors to the prompt that are saved in the "lims_embeddings" Table of the Vector Database
            rag_list += self.vector_handler.nearest_search(prompt, 10, "lims_embeddings")
            self.vector_handler.get_info()

        # Send to LLM
        if rag_list:
            response = self.llm_handler.send_prompt(prompt, rag_list)
        else:
            response = self.llm_handler.send_prompt(prompt)

        if "response" not in response:
            raise ValueError("No 'response' found in response from LLM")

        # User Comment
        comment = ""
        if self.settings.wait_for_manual_data:
            print("LLM Response:\n", response["response"])
            try:
                comment = input("Comment: ").strip()
            except EOFError:
                comment = ""

        # IDs and Metadata
        message_id = "msg_" + str(uuid.uuid4())[4:]
        response_obj = {
            "message_id": message_id,
            "prompt": prompt,
            "content": response["response"],
            "comment": comment,
            "metadata": {k: v for k, v in response.items() if k not in ("content", "prompt")}
        }
        if self.settings.use_rag_data == RAGMode.VOLATILE:
            response_obj["RAG-Data"] = _normalize_rag_data(self.settings.on_the_fly_data)

        elif self.settings.use_rag_data == RAGMode.PERSISTENT:
            response_obj["RAG-Data"] = _normalize_rag_data(self.settings.default_rag_data)

        # Saving the Data in the conversation History
        self.conversation_history.append(response_obj)

        # Saving the data in the Databases
        self._save_last_message_in_data()
        return response_obj

    def get_last_response(self) -> str:
        """
        Returns the content of the last message-object as a strong, so the last Response from the LLM

        :return: "Content"-Value of the last Message-Object
        """
        if not self.conversation_history:
            return ""
        return self.conversation_history[-1]["content"]

    def get_conversation_id(self) -> str:
        """
        Returns the current conversation-id
        :return: String-Object containing the current conversation-id
        """
        return self.conversation_id

    def add_metadata(self, conversation: bool, data: dict):
        """
        Adds Metadata either to the conversation-object or to the last received response

        :param conversation: If it should be saved to the conversation or the last response
        :param data: The data that should be updated in the metadata
        """
        if conversation:
            self.conversation_metadata.update(data)
        else:
            last_msg = self.conversation_history[-1]
            last_msg["metadata"].update(data)
            # Update in DB
            self._save_last_message_in_data()

    def change_comment(self, comment: str):
        """
        Changes the comment to the last received response

        :param comment: The comment to be updated
        """
        if len(self.conversation_history) == 0:
            raise IndexError("No message sent or received yet")
        self.conversation_history[-1]["comment"] = comment

    def get_metadata(self, conversation: bool, id: int = None) -> dict:
        """
        Returns Metadata for the conversation or for a specific message ID

        :param conversation: If metadata should be taken from the conversation or one message.
        :param id: Message ID to filter by
        :return: dict of Metadata extracted
        """
        if conversation:
            return self.conversation_metadata
        else:
            for item in self.conversation_history:
                if item["id"] == id:
                    return item["metadata"]
            raise ValueError(f"No item found with id {id}")

    def remove_metadata(self, conversation: bool,key: str,  id: str = None):
        """
        Removes specified metadata from either the conversation or a specific message indicated by the id

        :param conversation: True, if the metadata should be removed from the conversation
        :param key: Which metadata-entry should be removed
        :param id: Which message-id the metadata should be removed from
        """
        protected_keys = {"prompt", "content", "data", "id"}
        if key in protected_keys:
            raise ValueError(f"Cannot remove protected key '{key}'")

        target_list = self.conversation_history
        if conversation:
            if key in self.conversation_metadata:
                del self.conversation_metadata[key]
                self._save_last_message_in_data()
            else:
                raise KeyError(f"Key '{key}' not found in conversation metadata")
        else:
            for item in target_list:
                if item.get("id") == id:
                    if key in item["metadata"]:
                        del item[key]
                    else:
                        raise KeyError(f"Key '{key}' not found in item w0ith id {id}'s metadata")
                    self._save_last_message_in_data()
                    return  # exit after removing
            raise ValueError(f"No item found with id {id}")


    def _save_last_message_in_data(self):
        """
        Saves the last message in the conversation's message list in the databases
        """
        last_msg = self.conversation_history[-1]
        self.persistent_handler.save_record(
            conversation={"conversation_id": self.conversation_id, "created_at": self.created_at, **self.conversation_metadata},
            messages=[{
                "message_id": last_msg["message_id"],
                "user_prompt": last_msg["prompt"],
                "llm_response": last_msg["content"],
                "timestamp": datetime.now(timezone.utc).timestamp(),
                "user_comment": last_msg.get("comment", ""),
                "metadata": last_msg["metadata"]
            }]
        )
        self.vector_handler.save_vector({
            "message_id": last_msg["message_id"],
            "prompt": last_msg["prompt"],
            "response": last_msg["content"],
            **last_msg["metadata"]
        }, "lims_embeddings")
