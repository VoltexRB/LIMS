import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any
import psycopg2
from psycopg2 import *
from psycopg2.extras import Json
from sentence_transformers import SentenceTransformer
from llm_interaction_manager.handlers.persistent_data_handler_base import PersistentDataHandlerBase
from llm_interaction_manager.handlers.vector_data_handler_base import VectorDataHandlerBase


class PostgresHandler(PersistentDataHandlerBase, VectorDataHandlerBase):
    """
    Class to connect to a Postgres Instance. Offers persistent Database functionality.
    If PGVector is installed as well, Vector-Embedding functionality is also available.
    """
    def __init__(self):
        self.conn = None
        self.db = None
        self.embedding_model = None

    # --- DATABASE

    def get_info(self) -> dict:
        return {**self.auth, "host": self.host, "port": self.port}

    def get_name(self) -> str:
        return "postgres"

    def save_record(self, conversation: dict, messages: list[dict]):
        if not self.conn:
            raise ConnectionError("PostgreSQL connection not initialized. Call connect() first.")

        conv_id = conversation.get("conversation_id")
        if not conv_id:
            raise ValueError("Conversation dict must contain 'conversation_id'")

        # --- convert conversation created_at ---
        conv_created = conversation.get("created_at")
        if isinstance(conv_created, (int, float)):
            conv_created = datetime.fromtimestamp(conv_created, tz=timezone.utc)

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversations (conversation_id, created_at, name, description, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (conversation_id) DO UPDATE
                SET name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    metadata = EXCLUDED.metadata;
            """, (
                conv_id,
                conv_created,
                conversation.get("name"),
                conversation.get("description"),
                json.dumps(conversation.get("metadata") or {})
            ))

            for msg in messages:
                msg_id = msg.get("message_id")
                if not msg_id:
                    raise ValueError("Each message dict must contain 'message_id'")

                # --- convert message timestamp ---
                ts = msg.get("timestamp")
                if isinstance(ts, (int, float)):
                    ts = datetime.fromtimestamp(ts, tz=timezone.utc)

                cur.execute("""
                    INSERT INTO messages (message_id, conversation_id, user_prompt, llm_response, timestamp, user_comment, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (message_id) DO UPDATE
                    SET user_prompt = EXCLUDED.user_prompt,
                        llm_response = EXCLUDED.llm_response,
                        timestamp = EXCLUDED.timestamp,
                        user_comment = EXCLUDED.user_comment,
                        metadata = EXCLUDED.metadata;
                """, (
                    msg_id,
                    conv_id,
                    msg.get("user_prompt"),
                    msg.get("llm_response"),
                    ts,
                    msg.get("user_comment"),
                    json.dumps(msg.get("metadata") or {})
                ))

        self.conn.commit()

    def get_data(self, filters: dict[str, Any] | None = None) -> list[dict]:
        if not self.is_connected():
            raise ConnectionError("PostgreSQL connection not initialized. Call connect() first.")

        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                base_query = """
                    SELECT
                        c.conversation_id,
                        c.name AS llm_name,
                        c.created_at AS conversation_created_at,
                        c.description,
                        c.metadata AS conversation_metadata,
                        m.message_id,
                        m.user_prompt,
                        m.llm_response,
                        m.timestamp AS message_timestamp,
                        m.user_comment,
                        m.metadata AS message_metadata
                    FROM conversations c
                    LEFT JOIN messages m ON c.conversation_id = m.conversation_id
                """
                conditions = []
                params = []

                if filters:
                    if "conversation_id" in filters:
                        conditions.append("c.conversation_id = %s")
                        params.append(filters["conversation_id"])
                    if "message_id" in filters:
                        conditions.append("m.message_id = %s")
                        params.append(filters["message_id"])
                    if "user_prompt" in filters:
                        conditions.append("m.user_prompt ILIKE %s")
                        params.append(f"%{filters['user_prompt']}%")
                    if "llm_response" in filters:
                        conditions.append("m.llm_response ILIKE %s")
                        params.append(f"%{filters['llm_response']}%")

                if conditions:
                    base_query += " WHERE " + " AND ".join(conditions)

                cur.execute(base_query, params)
                results = cur.fetchall()

                conversations = {}
                for row in results:
                    conv_id = row["conversation_id"]
                    if conv_id not in conversations:
                        conversations[conv_id] = {
                            "conversation_id": conv_id,
                            "name": row["llm_name"],
                            "description": row["description"],
                            "created_at": row["conversation_created_at"],
                            "metadata": row["conversation_metadata"],
                            "messages": []
                        }

                    if row["message_id"]:
                        message = {
                            "message_id": row["message_id"],
                            "user_prompt": row["user_prompt"],
                            "llm_response": row["llm_response"],
                            "timestamp": row["message_timestamp"],
                            "user_comment": row["user_comment"],
                            "metadata": row["message_metadata"]
                        }
                        conversations[conv_id]["messages"].append(message)

                return list(conversations.values())

        except Exception as e:
            raise RuntimeError(f"Failed to fetch data from PostgreSQL: {e}")

    def select_database(self, db_name: str):
        if self.conn is None:
            raise ConnectionError("No existing connection. Call connect() first.")
        params = self.conn.get_dsn_parameters()
        host = params["host"]
        port = params["port"]
        user = params.get("user", None)
        password = params.get("password", None)

        auth = {"database": db_name}
        if user:
            auth["user"] = user
        if password:
            auth["password"] = password

        return self.connect(host, port, auth)

    # --- VECTOR ---

    def save_vector(self, data: dict, table: str = "lims_embeddings"):
        if not self.conn or self.conn.closed:
            raise ConnectionError("PostgreSQL connection is not initialized. Call connect() first.")
        if not self._vector_extension():
            raise ConnectionError("PostgreSQL does not have the pgvector extension installed.")

        message_id = data.get("id") or data.get("message_id")
        prompt_text = data.get("prompt")
        response_text = data.get("response")
        if not message_id or prompt_text is None or response_text is None:
            raise ValueError("Data dict must contain 'id'/'message_id', 'prompt', and 'response'.")

        prompt_embedding = self._generate_embedding(prompt_text)
        response_embedding = self._generate_embedding(response_text)

        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {table} (message_id, prompt_embedding, response_embedding)
                VALUES (%s, %s::vector, %s::vector)
                ON CONFLICT (message_id) DO UPDATE
                SET prompt_embedding = EXCLUDED.prompt_embedding,
                    response_embedding = EXCLUDED.response_embedding
                """,
                (message_id, prompt_embedding, response_embedding)
            )
        self.conn.commit()

    def _generate_embedding(self, text: str):
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self.embedding_model.encode(text).tolist()

    def load_vector(self, query: dict, table: str = "lims_embeddings") -> dict:
        if not self.is_connected():
            raise RuntimeError("PostgreSQL connection not initialized.")
        if "id" not in query:
            raise ValueError("Query must contain an 'id' key.")
        message_id = query["id"]

        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT mv.message_id, mv.prompt_embedding, mv.response_embedding,
                       m.user_prompt, m.llm_response, m.metadata
                FROM {table} mv
                JOIN messages m ON mv.message_id = m.message_id
                WHERE mv.message_id = %s
            """, [message_id])
            result = cur.fetchone()
            if not result:
                raise ValueError(f"No vector found with ID '{message_id}'.")

            vector_content = f"PROMPT: {result['user_prompt']}\nRESPONSE: {result['llm_response']}"
            return {
                "id": result["message_id"],
                "vector": vector_content,
                "prompt_embedding": result["prompt_embedding"],
                "response_embedding": result["response_embedding"],
                "user_prompt": result["user_prompt"],
                "llm_response": result["llm_response"],
                "metadata": result.get("metadata", {})
            }

    def nearest_search(self, input: str, top_k: int, table: str = "lims_embeddings") -> list[str]:
        if not self.is_connected():
            raise RuntimeError("PostgreSQL connection not initialized.")

        embedding = self._generate_embedding(input)

        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT m.user_prompt, m.llm_response
                FROM {table} mv
                JOIN messages m ON mv.message_id = m.message_id
                ORDER BY LEAST(mv.prompt_embedding <-> %s::vector, mv.response_embedding <-> %s::vector)
                LIMIT %s
                """,
                [embedding, embedding, top_k]
            )
            rows = cur.fetchall()
            results = [f"PROMPT: {row[0]}\nRESPONSE: {row[1]}" for row in rows]
            return results

    def import_vectors(self, table: str = "lims_embeddings", data: dict = None, path: str = None):
        if not self.is_connected():
            raise ConnectionError("Client is not connected. Use 'connect' first.")
        if not self._vector_extension():
            raise ConnectionError("Client needs pgvector installed for this operation.")
        if (data is None and path is None) or (data is not None and path is not None):
            raise ValueError("Provide exactly one of 'data' or 'path'.")

        if data is not None and isinstance(data, dict):
            data = [data]

        if path is not None:
            if not os.path.isfile(path):
                raise ValueError(f"File {path} does not exist.")
            with open(path, "r", encoding="utf-8") as f:
                data = [{"text": line.strip()} for line in f if line.strip()]

        imported_conv = {
            "conversation_id": "imported_conversation",
            "name": "external_import",
            "description": "Imported data without original LLM context",
            "metadata": {}
        }
        self.save_record(imported_conv, [])

        for entry in data:
            rand_uuid = str(uuid.uuid4())
            message_id = "imported_" + rand_uuid[9:]
            user_prompt = "imported_data"
            llm_response = entry["text"]

            self.save_record(
                conversation=imported_conv,
                messages=[{
                    "message_id": message_id,
                    "user_prompt": user_prompt,
                    "llm_response": llm_response,
                    "metadata": {}
                }]
            )

            self.save_vector(
                data={
                    "message_id": message_id,
                    "prompt": user_prompt,
                    "response": llm_response
                },
                table=table
            )

    # --- CONNECTION ---

    def is_connected(self) -> bool:
        if not self.conn:
            return False
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1;")
                return True
        except:
            return False

    def connect(self, host: str, port: int, auth: dict = None) -> bool:
        if not auth or "database" not in auth:
            raise ValueError("Authentication dictionary must contain at least 'database' key.")
        self.db = auth["database"]
        conn_params = {"host": host, "port": port, "dbname": auth["database"]}
        if "user" in auth:
            conn_params["user"] = auth["user"]
        if "password" in auth:
            conn_params["password"] = auth["password"]

        try:
            self.conn = psycopg2.connect(**conn_params)
            self._initialize_schema()
            self.host = host
            self.port = port
            self.auth = auth
            return True
        except OperationalError as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during PostgreSQL connection: {e}")

    def _vector_extension(self) -> bool:
        if not self.conn:
            raise ConnectionError("PostgreSQL connection not established. Call connect() first.")

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT 1 
                        FROM pg_extension 
                        WHERE extname = 'vector'
                    );
                """)
                return bool(cur.fetchone()[0])
        except:
            return False

    def _initialize_schema(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT NOW(),
                    name TEXT,
                    description TEXT,
                    metadata JSONB
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT REFERENCES conversations(conversation_id) ON DELETE CASCADE,
                    user_prompt TEXT,
                    llm_response TEXT,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    user_comment TEXT,
                    metadata JSONB
                );
            """)
            if self._vector_extension():
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS lims_embeddings (
                        message_id TEXT REFERENCES messages(message_id) ON DELETE CASCADE PRIMARY KEY,
                        prompt_embedding VECTOR(384),
                        response_embedding VECTOR(384)
                    );
                """)
        self.conn.commit()
