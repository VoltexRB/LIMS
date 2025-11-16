from llm_interaction_manager.core.interaction_manager import *
from llm_interaction_manager.api.interaction_manager_factory import *
import llm_interaction_manager.api.interaction_manager_factory as imf
from dataclasses import field

# Kapselt objektorientierte Konzepte aus der Schnittstelle in ein einzelnes Skript. Durch Initialize wird ein Interaction-Manager Objekt erzeugt und intern gespeichert.
# Der Nutzer kann mit der Schnittstelle rein durch Methodenaufrufe kommunizieren

interaction_manager: InteractionManager = None

def initialize(llm: LLMEnum = LLMEnum.SETTINGS, persistent: PersistentEnum = PersistentEnum.SETTINGS, vector: VectorEnum = VectorEnum.SETTINGS):
    """
    Initializes a new InteractionManager Object with the 3 specified interfaces. For enum values, see InteractionManagerFactory
    :param llm: Enum which LLM interface should be used
    :param persistent: Enum which Persistent Data interface should be used
    :param vector: Enum which Vector Data interface should be used
    :return: InteractionManager Object with the 3 Interfaces
    """
    global interaction_manager
    interaction_manager = imf.initialize(llm, persistent, vector)

def _check_initialized()-> bool:
    """
    Checks if an InteractionManager Object was already initialized and is available locally
    :return: bool if InteractionManager Object exists
    """
    if interaction_manager is None:
        raise RuntimeError("Interaction Manager not initialized. Use initialize() first.")
    return True

def start_conversation(conversation_metadata: dict = None):
    """
    Starts a new conversation in the InteractionManager with the Handlers of the interactionManager.
    :param conversation_metadata: Optional metadata that can immediately be added to the conversation
    """
    if not _check_initialized(): return
    interaction_manager.start_conversation()

def set_rag_data(data: dict, volatile: bool):
    """
    Adds new RAG data to use for sending prompts. RAD-Data can either be saved in the config or just be used in the current session
    :param data: Data to be used as RAG data
    :param volatile: If the data should just persist for the current session
    """
    if not _check_initialized(): return
    interaction_manager.set_rag_data(data, volatile)

def delete_rag_data():
    """
    Stops using RAG-Data in conversations with the LLMs from now on and deletes any RAG-Data, persistent or volatile
    :return:
    """
    if not _check_initialized(): return
    interaction_manager.delete_rag_data()

def nearest_search_vector(input: str, top_k: int, table: str) -> list[str]:
    """
    Searches the vector database for close matches of the input and returns the top_k results. May result in a combination of Prompt and Response lines depending on Interface
    :param input: String data to generate an embedding from to use nearest search for on the vector database.
    :param top_k: How many results should be returned
    :param table: Which table to do the nearest search on
    :return: List of strings where the embeddings are matched near the input embedding
    """
    if not _check_initialized(): return []
    return interaction_manager.nearest_search_vector(input, top_k, table)


def export_data(to: str, filters: dict[str, Any] | None = None):
    """
    Exports data from the persistent database to the specified path
    :param to: Path to export the data to
    :param filters: Filters to apply to the data. If no filters are specified, all available data will be exported
    """
    if not _check_initialized(): return
    interaction_manager.export_data(to, filters)


def add_persistent_data(conversation: dict, messages: list[dict]):
    """
    Adds new persistent data to the database. Data has to contain messages and conversations that align with the internal datastructure.
    Conversations must contain a conversation_id, messages must each contain a message_id
    :param conversation: Conversation Object with conversation_id that should be inserted
    :param messages: List of messages with message_ids that should be inserted and linked to the conversation
    """
    if not _check_initialized(): return
    interaction_manager.add_persistent_data(conversation, messages)

def add_vector_data(data: dict, table: str):
    """
    Adds new data to the vector-database
    :param data: Data to be added, must contain at least "prompt", "response" and "message_id" as message_id to link the data to a persistent message
    :param table: Which table the data should be saved in
    """
    if not _check_initialized(): return
    interaction_manager.add_vector_data(data, table)

def is_connected(target: ConnectionType) -> bool:
    """
    Check if the specified handler is connected
    :param target: ConnectionType-Enum, which handler should be checked
    :return: True if specified handler is connected, False else
    """
    if not _check_initialized(): return False
    return interaction_manager.is_connected(target)

def connect(target: ConnectionType, data: dict):
    """
    Connect to the specified handlers endpoint
    :param target: ConnectionType-Enum, which handler should be checked
    :param data: Data to use for connection (eg. API tokens, routing information)
    """
    if not _check_initialized(): return
    interaction_manager.connect(target, data)

def read_setting(key: str):
    """
    Reads a setting from the current Settings-Object
    :param key: Which setting to read
    :return: Requested Object from Settings
    """
    if not _check_initialized(): return field(default_factory=dict)
    return interaction_manager.read_setting(key)

def write_setting(key: str, value):
    """
    Updates a setting in the Settings Object and propagates to config.json
    :param key: Key to write into the settings
    :param value: Value to be changed
    """
    if not _check_initialized(): return
    interaction_manager.write_setting(key, value)

def set_rag_mode(mode: RAGMode):
    """
    Sets the RAG-Mode of the application
    :param mode: The mode to set it to
    """
    if not _check_initialized(): return
    interaction_manager.set_rag_mode(mode)



def add_metadata(to_conversation: bool, data: dict):
    """
    Adds metadata to the last message that was sent/received or the current conversation-object
    :param to_conversation: If the metadata should be added to the conversation-object or the last message
    :param data: Data to add to the metadata
    """
    if not _check_initialized(): return
    interaction_manager.add_metadata(to_conversation, data)

def send_prompt(prompt: str) -> dict:
    """
    Sends a prompt to the LLM-Handler and thus to the specified LLM. Receives a response, saves the response persistently in the databases
    :param prompt: Prompt to send to the LLM-Handler. If RAG-Data is specified, Prompt and RAG-Data will be combined
    :return: Answer from LLM
    """
    if not _check_initialized(): return field(default_factory=dict)
    return interaction_manager.send_prompt(prompt)

def change_comment(comment: str):
    """
    Changes the comment on the last sent message
    :param comment: Comment to add to the last sent message
    """
    if not _check_initialized(): return
    interaction_manager.change_comment(comment)
