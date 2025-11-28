from llm_interaction_manager.handlers.data_handler_base import DataHandlerBase
from abc import ABC, abstractmethod

class VectorDataHandlerBase (DataHandlerBase, ABC):

    @abstractmethod
    def save_vector(self, data: dict, table: str):
        """
        Save a single vector (prompt + response) to a collection/table.
        Expected keys in data:
            - "prompt": str
            - "response": str
            - "id": str if the data is from the interface and from a specific message. External data does not have an ID
            - other keys: optional metadata

        :param data: Data to be saved
        :param table: Table to save the data to
        """
        pass

    @abstractmethod
    def load_vector(self, query: dict, table: str) -> dict:
        """
        finds vector by at least its id, can also provide more data in the query

        :param query: dict with values to query against, must at least contain an id
        :param table: chromadb collection
        :return: returns the matched vector
        """
        pass

    @abstractmethod
    def nearest_search(self, input: str, top_k: int, table: str) -> list[str]:
        """
        Gets vectors from the connected database and specified connection based on the input prompt

        :param table: Which chromadb collection to search in
        :param input: prompt that the vector collection is searching against
        :param top_k: how many results to return.
        returns
            list[str]: of Vector 'document' values
        """
        pass

    @abstractmethod
    def import_vectors(self,  table: str, data: dict = None, path: str = None):
        """
        Adds vectors to the vector database, either from a dict structure or a file specified by a path.
        If they are imported without an ID, a special "imported" ID will be added

        :param table: Table to import the data into
        :param data: Data to be imported directly
        :param path: Path to the data that should be imported from external files
        """
        pass