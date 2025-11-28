import pytest
from llm_interaction_manager.handlers.chromadb_handler import ChromadbHandler, ClientType

@pytest.fixture
def chroma_handler():
    """Fixture to provide a fresh ChromadbHandler with an in-memory client."""
    handler = ChromadbHandler()
    auth = {"client_type": ClientType.VOLATILE}
    assert handler.connect(auth=auth)
    yield handler
    # No teardown needed for in-memory client

def test_save_and_load_vector(chroma_handler):
    """Test saving a vector and retrieving it by ID."""
    table = "test_collection"
    data = {
        "id": "vec_001",
        "prompt": "Hello",
        "response": "World",
        "extra_meta": "meta_value"
    }

    # Save the vector
    assert chroma_handler.save_vector(data, table=table) is True

    # Load the vector
    result = chroma_handler.load_vector({"id": "vec_001"}, table=table)
    assert result["id"] == "vec_001"
    assert "PROMPT: Hello" in result["vector"]
    assert "RESPONSE: World" in result["vector"]
    assert result["metadata"]["extra_meta"] == "meta_value"

def test_nearest_search(chroma_handler):
    """Test nearest search returns saved vectors."""
    table = "test_collection_nn"
    data1 = {"prompt": "Good morning", "response": "Have a nice day", "id": "vec_001"}
    data2 = {"prompt": "Hello there", "response": "General Kenobi", "id": "vec_002"}

    chroma_handler.save_vector(data1, table)
    chroma_handler.save_vector(data2, table)

    # Search for a prompt similar to "Good morning"
    results = chroma_handler.nearest_search("Good morning", top_k=2, table=table)
    assert any("Good morning" in r for r in results)
    assert len(results) <= 2

def test_load_nonexistent_vector_raises(chroma_handler):
    """Attempting to load a vector that doesn't exist should raise ValueError."""
    table = "test_collection"
    # Ensure collection exists (can be empty)
    chroma_handler._add_to_collection(
        _vectors=["dummy"],
        _collection_name=table,
        ids=["dummy_id"]
    )
    with pytest.raises(ValueError):
        chroma_handler.load_vector({"id": "nonexistent"}, table=table)

def test_save_vector_without_prompt_or_response_raises(chroma_handler):
    """Save should fail if neither prompt nor response is provided."""
    table = "test_collection"
    data = {"id": "vec_003"}
    with pytest.raises(ValueError):
        chroma_handler.save_vector(data, table=table)

def test_metadata_preservation(chroma_handler):
    """Ensure metadata is stored and retrieved correctly."""
    table = "test_collection_meta"
    data = {
        "id": "vec_004",
        "prompt": "Check",
        "response": "Metadata",
        "category": "test",
        "source": "unit_test"
    }
    # Save the vector
    assert chroma_handler.save_vector(data, table=table)
    # Load the vector by ID
    result = chroma_handler.load_vector({"id": "vec_004"}, table=table)
    # Assertions to verify metadata is preserved
    assert result["metadata"] is not None
    assert result["metadata"]["category"] == "test"
    assert result["metadata"]["source"] == "unit_test"
    assert result["vector"].startswith("PROMPT: Check")
    assert "RESPONSE: Metadata" in result["vector"]