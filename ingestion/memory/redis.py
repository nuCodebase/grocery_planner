import os
from typing import Any, Optional
import redis


def init_redis_connection(decode_responses: bool = True) -> redis.Redis:
    """Initialize and return a Redis client using environment configuration.
    
    Args:
        decode_responses: If True, decode Redis responses to strings. If False, return bytes.
                         Use False when retrieving binary fields like embeddings.
    """
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        return redis.from_url(redis_url, decode_responses=decode_responses)

    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    db = int(os.getenv("REDIS_DB", "0"))
    password = os.getenv("REDIS_PASSWORD")

    return redis.Redis(
        host=host,
        port=port,
        db=db,
        password=password,
        decode_responses=decode_responses,
    )


def add_key_value(client: redis.Redis, key: str, value: Any, ex: Optional[int] = None) -> bool:
    """Add a key-value pair to Redis. Returns True if the value was set."""
    return client.set(name=key, value=value, ex=ex)


def get_key_value(client: redis.Redis, key: str) -> Optional[str]:
    """Retrieve the value for a key from Redis. Returns None if the key does not exist."""
    return client.get(name=key)


def update_key_value(client: redis.Redis, key: str, value: Any, ex: Optional[int] = None) -> bool:
    """Update the value for an existing key in Redis. Returns True if updated, False when the key is missing."""
    if client.exists(key) != 1:
        return False
    return client.set(name=key, value=value, ex=ex)


def remove_key(client: redis.Redis, key: str) -> int:
    """Remove a key from Redis. Returns the number of deleted keys (0 or 1)."""
    return client.delete(key)


def is_redis_connection_active(client: Optional[redis.Redis] = None, timeout: float = 1.0) -> bool:
    """Check whether a Redis connection is active.

    If `client` is None this will initialize a client using
    :func:`init_redis_connection()`. Returns True when a PING succeeds,
    otherwise False.

    Args:
        client: Optional Redis client to use for the check.
        timeout: Unused for redis-py ping (kept for API compatibility).

    Returns:
        bool: True if Redis responds to PING, False on error/timeout.
    """
    if client is None:
        try:
            client = init_redis_connection()
        except Exception:
            return False

    try:
        # redis.Redis.ping() returns True on success
        return bool(client.ping())
    except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
        return False
    except Exception:
        return False


from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType

INDEX_NAME = "chunk-index"
KEY_PREFIX = "chunk:"
EMBEDDING_DIMENSION = 384

def create_schema(client: redis.Redis):
    """Create the Redis schema for storing chunks with embeddings."""
    schema = (
        TextField("text"),
        TagField("document_id"),
        TagField("section"),
        VectorField(
            "embedding",
            "HNSW",
            {
                "TYPE": "FLOAT32",
                "DIM": EMBEDDING_DIMENSION,
                "DISTANCE_METRIC": "COSINE",
            },
        ),
    )

    definition = IndexDefinition(
        prefix=[KEY_PREFIX],
        index_type=IndexType.HASH,
    )

    try:
        client.ft(INDEX_NAME).create_index(
            fields=schema,
            definition=definition,
        )
        print("Vector index created")
    except redis.ResponseError as exc:
        if "Index already exists" in str(exc):
            print("Vector index already exists")
        else:
            raise

from sentence_transformers import SentenceTransformer
import numpy as np
from redis.commands.search.query import Query

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def store_chunk(
    client: redis.Redis,    
    chunk_id: str,
    text: str,
    document_id: str="",
    section: str = "",
) -> None:
    embedding = embedding_model.encode(
        text,
        normalize_embeddings=True,
    )

    embedding_bytes = np.asarray(
        embedding,
        dtype=np.float32,
    ).tobytes()

    redis_key = f"chunk:{chunk_id}"

    client.hset(
        redis_key,
        mapping={
            "text": text,
            "document_id": document_id,
            "section": section,
            "embedding": embedding_bytes,
        },
    )

def search_similar_chunks(
    redis_client: redis.Redis,
    question: str,
    top_k: int = 3,
) -> list[dict]:
    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True,
    )

    query_vector = np.asarray(
        query_embedding,
        dtype=np.float32,
    ).tobytes()

    redis_query = (
        Query(
            f"*=>[KNN {top_k} "
            f"@embedding $query_vector AS vector_distance]"
        )
        .sort_by("vector_distance")
        .return_fields(
            "text",
            "document_id",
            "section",
            "vector_distance",
        )
        .dialect(2)
    )

    results = redis_client.ft(INDEX_NAME).search(
        redis_query,
        query_params={
            "query_vector": query_vector,
        },
    )

    matches = []

    for document in results.docs:
        matches.append(
            {
                "redis_key": document.id,
                "text": document.text,
                "document_id": document.document_id,
                "section": document.section,
                "distance": float(document.vector_distance),
            }
        )

    return matches