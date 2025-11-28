import os
import pathlib
import uuid

import httpx
from llm_interaction_manager.handlers.vector_data_handler_base import VectorDataHandlerBase
import chromadb
from chromadb.errors import *
from enum import Enum

class ClientType(Enum):
    """
    different types of Chroma Client locations. Either create a volatile local database, address or create a persistent local database or use webservices
    Attributes:
        VOLATILE: An in-memory, ephemeral client. Data is not persistent.
        PERSISTENT: A local persistent database on disk, addressed by path.
        HTTP_SERVER: Connects to a self-hosted or otherwise accessible Chroma server over HTTP.
        CHROMA_CLOUD: Connects to the Chroma Cloud service if you have an active subscription.
    """
    VOLATILE = "VOLATILE"
    PERSISTENT = "PERSISTENT"
    HTTP_SERVER = "HTTP_SERVER"
    CHROMA_CLOUD = "CHROMA_CLOUD"

class ChromadbHandler(VectorDataHandlerBase):
    """
    Handles connection with ChromaDB
    """
    _client = None

    def get_info(self) -> dict:
        """
        Returns the connection information to be saved in the settings

        :return: dict with the connection information
        """
        return {**self.auth, "host": self.host, "port": self.port}

    def get_name(self) -> str:
        """
        Returns the name "chromadb" for dynamic binding purposes

        :return: String "chromadb"
        """
        return "chromadb"


    def save_vector(self, data: dict, table: str):
        """
        Save a single vector (prompt + response) to a collection.

        Expected keys in data:
            - "prompt": str
            - "response": str
            - "id": str if the data is from the interface and from a specific message. External data does not have an ID
            - other keys: optional metadata
        """
        if not data.get("prompt") and not data.get("response"):
            raise ValueError("Data must contain at least 'prompt' or 'response'")
        # Combine prompt and response into a single string for storage as chromadb does not allow multi-section entries
        vector_content = f"PROMPT: {data.get('prompt', '')}\nRESPONSE: {data.get('response', '')}".strip()

        # Extract ID
        id_value = data.get("id")

        # Everything else goes into metadata
        metadata = {k: v for k, v in data.items() if k not in ("prompt", "response", "id")}

        # wrap everything in lists of length 1 as _add_to_collection expects lists
        metadatas = [metadata] if metadata else None
        vectors = [vector_content]
        ids_list = [id_value] if id_value else None

        return self._add_to_collection(
            _vectors=vectors,
            _collection_name=table,
            _metadata=metadatas,
            ids=ids_list
        )

    def load_vector(self, query: dict, table: str) -> dict:
        """
        finds vector by at least its id, can also provide more data in the query

        :param query: dict with values to query against, must at least contain an id
        :param table: chromadb collection
        :return: returns the matched vector
        """
        client = self._client
        if not (client and self.is_connected() and self._collection_exists(table)):
            raise RuntimeError("Client not connected or a collection with that name doesn't exist")

        if "id" not in query:
            raise ValueError("Query must contain an 'id' key.")

        vector_id = query["id"]
        metadata_filter = query.get("metadata", {})
        collection = client.get_collection(table)

        # Fetch the vector using ID
        results = collection.get(ids=[vector_id], where=metadata_filter if metadata_filter else None)

        if not results['ids']:
            raise ValueError(f"No vector found with ID '{vector_id}' and matching metadata.")

        #separate prompt and response
        vector_text = results['documents'][0]
        prompt_start = vector_text.index("PROMPT:") + len("PROMPT:")
        response_start = vector_text.index("RESPONSE:")

        prompt = vector_text[prompt_start:response_start].strip()
        response = vector_text[response_start + len("RESPONSE:"):].strip()

        # Return the first (and only) matching vector
        return {
            "id": results['ids'][0],
            "prompt": prompt,
            "response": response,
            "metadata": results['metadatas'][0],
        }

    def nearest_search(self, input: str, top_k: int, table: str) -> list[str]:
        """
        Gets vectors from the connected database and specified connection based on the input prompt

        :param table: Which chromadb collection to search in
        :param input: prompt that the vector collection is searching against
        :param top_k: how many results to return.
        returns
            list[str]: of Vector 'document' values
        """
        # if client not connected
        results: list[str] = []
        client = self._client
        if not (client and self.is_connected() and self._collection_exists(table)):
            raise RuntimeError("Client not connected or a collection with that name doesn't exist")

        collection = client.get_collection(name=table)
        response = collection.query(
            query_texts=[
                input
            ],
            n_results=top_k
        )

        for query_result in response["documents"]:
            results.extend(query_result)

        return results

    def is_connected(self) -> bool:
        """
        Tries to check for a valid connection to chromaDB

        :return: True if chromadb instance can be reached via the builtin heartbeat() function
        """
        if self._client is None: return False
        try:
            self._client.heartbeat()
        except:
            return False
        return True

    def connect(self, host: str = "-1", port: int = "-1", auth = None) -> bool:
        """
        Connects to ChromaDB.
        The required keys in `auth` depend on the chosen client type:

        - ClientType.VOLATILE: no keys required
        - ClientType.PERSISTENT: requires 'persistent_client_db_path'
        - ClientType.HTTP_SERVER: no keys required
        - ClientType.CHROMA_CLOUD: requires 'cloud_tenant', 'cloud_database', 'cloud_key'

        Raises KeyError if ClientType can not be found or the corresponding Keys for the selected ClientType are missing.
        Only connects to HTTP-Server-instances without authentication

        :param auth: Dictionary containing the required parameters to use ChromaDB.
        :param host: if using http-service to connect to a ChromaDB instance, supply hostname
        :param port: same usage as hostname, supply port
        :return: bool if connected successfully
        """
        try:
            client_type = ClientType(auth["client_type"])
        except KeyError:
            raise KeyError("Missing 'client_type' in auth dictionary")

        if client_type == ClientType.PERSISTENT:
            if "persistent_client_db_path" not in auth:
                raise KeyError("key 'persistent_client_db_path' required for PERSISTENT client")
            persistent_client_db_path = auth["persistent_client_db_path"]
        elif client_type == ClientType.CHROMA_CLOUD:
            required_keys = ["cloud_tenant", "cloud_database", "cloud_key"]
            missing = [k for k in required_keys if k not in auth]
            if missing:
                raise KeyError(f"Missing keys in auth for CHROMA_CLOUD client: {missing}")
            cloud_tenant = auth["cloud_tenant"]
            cloud_database = auth["cloud_database"]
            cloud_key = auth["cloud_key"]

        try:
            match client_type:
                case ClientType.VOLATILE:
                    self._client = chromadb.Client()

                case ClientType.PERSISTENT:
                    if "persistent_client_db_path" not in auth:
                        raise KeyError("key 'persistent_client_db_path' required for PERSISTENT client")

                    path = pathlib.Path(auth["persistent_client_db_path"])

                    # Create the path if it doesn't exist
                    if not path.exists():
                        if path.suffix:  # has file extension
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.touch(exist_ok=True)
                        else:
                            path.mkdir(parents=True, exist_ok=True)

                    # Check write access
                    if not os.access(path, os.W_OK):
                        raise PermissionError(f"No write access to {path}")

                    # Connect to ChromaDB
                    self._client = chromadb.PersistentClient(str(path))

                case ClientType.HTTP_SERVER:
                        self._client = chromadb.HttpClient(host = host, port= port)

                case ClientType.CHROMA_CLOUD:
                        self._client = chromadb.CloudClient(tenant=cloud_tenant, database=cloud_database, api_key=cloud_key)

            self._client.heartbeat() # check connection since client is addressed lazily
        except (PermissionError, httpx.HTTPError) as e:
            raise PermissionError(f"Could not connect to Chromadb: {e}")
        self.auth = auth
        self.host = host
        self.port = port
        return True

    def import_vectors(self, table: str, data: dict = None, path: str = None):
        """
        Adds vectors to the vector database, either from a dict structure or a file specified by a path.
        If they are imported without an ID, a special "imported" ID will be added

        :param table: Table to import the data into
        :param data: Data to be imported directly
        :param path: Path to the data that should be imported from external files
        """
        if not self.is_connected():
            raise ConnectionError("Client is not connected. Use 'connect' first")

        if (data is None and path is None) or (data is not None and path is not None):
            raise ValueError("You must provide exactly one of 'data' or 'file_path'.")

        if data is not None:
            texts = [entry["text"] for entry in data]  # data is a list of dicts
            metadata = [
                {k: v for k, v in entry.items() if k != "text"}
                for entry in data
            ]
            self._add_to_collection(texts, _collection_name=table, _metadata=metadata)

        if path is not None:
            if not os.path.isfile(path):
                print("no valid file selected, exiting function")
                return False

            # read the Vectors into a list
            vectors: list[str] = []
            with open(path, "r", encoding="utf-8") as f:
                    vectors.extend(f.read().splitlines())

            # add Vectors to collection
            self._add_to_collection(vectors, table)

    def _collection_exists(self, name: str):
        """
        Check if the collection, ChromaDBs version of a Table, exists already

        :param name: Name of the collection to search for
        :return: True if the collection exists
        """
        try:
            client = self._client
            if client is None: # client not yet connected
                return "-1"
            client.get_collection(name) #no method in chroma to check if collection exists
            return True
        except NotFoundError:
            return False

    def _add_to_collection(
            self,
            _vectors: list[str],
            _collection_name: str,
            _metadata: list[dict] | None = None,
            ids: list[str] | None = None
    ) -> bool:
        """
        Adds vectors to a collection, optionally with metadata and/or custom IDs.
        - If `ids` is None, random UUIDs are generated for each vector.
        - If `_metadata` is None, no metadata is stored.
        """
        client = self._client
        if client is None:
            return False

        collection = client.get_or_create_collection(name=_collection_name)

        # Ensure IDs exist
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in _vectors]

        # Add vectors with optional metadata
        if _metadata is None:
            collection.add(
                ids=ids,
                documents=_vectors
            )
        else:
            # Metadata must match vector length and be flattened to one dimension
            _metadata = self._flatten_metadata_list(_metadata)
            if len(_metadata) != len(_vectors):
                raise ValueError("Length of metadata must match number of vectors")
            collection.add(
                ids=ids,
                documents=_vectors,
                metadatas=_metadata
            )
        return True

    def _flatten_metadata(self, metadata: dict, parent_key: str = "", sep: str = "_") -> dict:
        flat = {}
        for k, v in metadata.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                flat.update(self._flatten_metadata(v, new_key, sep))
            elif v is not None:  # Skip None values
                flat[new_key] = v
        return flat

    def _flatten_metadata_list(self, metadatas: list[dict]) -> list[dict]:
        return [self._flatten_metadata(meta) for meta in metadatas]