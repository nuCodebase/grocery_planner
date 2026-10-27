import sys
from pathlib import Path

# Ensure the repository root is on sys.path when running this file directly
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import os
import json
from typing import Optional
import streamlit as st
import numpy as np

# Redis and embedding imports
from ingestion.memory.redis import (
    init_redis_connection,
    is_redis_connection_active,
)

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError as exc:
        raise ImportError(
            "HuggingFaceEmbeddings requires either the langchain-huggingface or langchain-community package. "
            "Install one with `pip install langchain-huggingface` or `pip install langchain-community`."
        ) from exc

@st.cache_resource
def load_embeddings():
    """Cache the embedding model to avoid reloading on each interaction."""
    model_name = os.getenv("HUGGING_FACE_EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model_name)


@st.cache_resource
def get_redis_client():
    """Cache the Redis client connection."""
    return init_redis_connection()


@st.cache_resource
def get_raw_redis_client():
    """Cache a raw Redis client (no auto-decoding) for safe chunk retrieval."""
    return init_redis_connection(decode_responses=False)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)


def search_redis_chunks(
    query: str,
    embeddings,
    client,
    top_k: int = 5,
    similarity_threshold: float = 0.3,
) -> list[dict]:
    """Search Redis for chunks relevant to the query using semantic similarity.
    
    Args:
        query: Natural language query string
        embeddings: Embedding model instance
        client: Redis client
        top_k: Number of top results to return
        similarity_threshold: Minimum similarity score (0-1)
        
    Returns:
        List of relevant chunks with similarity scores
    """
    # Embed the query
    query_embedding = embeddings.embed_query(query)
    query_vec = np.array(query_embedding, dtype=np.float32)
    
    # Retrieve all chunks from Redis
    chunks_data = []
    
    try:
        # Use raw client to avoid auto-decoding binary fields
        raw_client = get_raw_redis_client()
        
        # Scan all keys in Redis
        cursor = 0
        while True:
            cursor, keys = raw_client.scan(cursor, match="chunk:*", count=100)
            for key in keys:
                try:
                    chunk_data_raw = raw_client.hgetall(key)
                    if chunk_data_raw:
                        chunk_data = {}
                        for field, value in chunk_data_raw.items():
                            # Decode field name
                            if isinstance(field, bytes):
                                field = field.decode("utf-8", errors="ignore")
                            
                            # Skip binary embedding field
                            if field == "embedding":
                                continue
                            
                            # Decode text fields safely
                            if isinstance(value, bytes):
                                try:
                                    value = value.decode("utf-8")
                                except UnicodeDecodeError:
                                    # Skip fields that can't be decoded as text
                                    continue
                            
                            chunk_data[field] = value
                        
                        # Only add if has text content
                        if "text" in chunk_data and chunk_data["text"]:
                            chunk_data["chunk_id"] = key if isinstance(key, str) else key.decode("utf-8", errors="ignore")
                            chunk_data["source_file"] = "redis"
                            chunks_data.append(chunk_data)
                except Exception as chunk_err:
                    # Skip chunks that fail to parse
                    continue
            if cursor == 0:
                break
    except Exception as e:
        st.error(f"Error retrieving chunks from Redis: {e}")
    
    if not chunks_data:
        return []
    
    # Compute similarity scores for each chunk
    scored_chunks = []
    for chunk in chunks_data:
        text = chunk.get("text", "")
        if not text:
            continue
            
        try:
            chunk_embedding = embeddings.embed_query(text)
            chunk_vec = np.array(chunk_embedding, dtype=np.float32)
            similarity = cosine_similarity(query_vec, chunk_vec)
            
            if similarity >= similarity_threshold:
                scored_chunks.append({
                    "chunk_id": chunk.get("chunk_id", "unknown"),
                    "text": text[:500],  # Truncate for display
                    "similarity": round(similarity, 4),
                    "full_text": text,
                    "metadata": chunk.get("metadata", {}),
                    "source_file": chunk.get("source_file", "unknown"),
                })
        except Exception as e:
            st.warning(f"Error embedding chunk {chunk.get('chunk_id', 'unknown')}: {e}")
            continue
    
    # Sort by similarity and return top_k
    scored_chunks.sort(key=lambda x: x["similarity"], reverse=True)
    return scored_chunks[:top_k]


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Grocery Planner - Redis Chunk Retriever",
        page_icon="🛒",
        layout="wide"
    )
    
    st.title("🛒 Grocery Planner - Redis Semantic Search")
    
    st.markdown("""
    Search through document chunks stored in Redis using natural language.
    The app will find the most semantically relevant chunks based on your query.
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check Redis connection
        try:
            client = get_redis_client()
            is_active = is_redis_connection_active(client)
            if is_active:
                st.success("✅ Redis Connected")
            else:
                st.error("❌ Redis Connection Failed")
        except Exception as e:
            st.error(f"❌ Redis Error: {e}")
            return
        
        # Configuration sliders
        top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
        similarity_threshold = st.slider(
            "Minimum similarity score",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05
        )
    
    # Main content
    st.header("Search Chunks")
    
    # User query input
    query = st.text_area(
        "Enter your natural language query:",
        placeholder="e.g., What are the best grocery items for a health-conscious diet?",
        height=100
    )
    
    # Search button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        search_button = st.button("🔍 Search", use_container_width=True)
    with col2:
        clear_button = st.button("🧹 Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.clear()
        st.rerun()
    
    # Perform search if button clicked or query entered
    if search_button and query.strip():
        try:
            with st.spinner("🔄 Searching and embedding..."):
                embeddings = load_embeddings()
                results = search_redis_chunks(
                    query=query,
                    embeddings=embeddings,
                    client=client,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                )
            
            st.markdown("---")
            
            if results:
                st.success(f"✅ Found {len(results)} relevant chunk(s)")
                
                for idx, result in enumerate(results, 1):
                    with st.expander(
                        f"📄 Result #{idx} | Similarity: {result['similarity']} | "
                        f"Source: {result['source_file']}",
                        expanded=(idx == 1)
                    ):
                        st.markdown(f"**Chunk ID:** `{result['chunk_id']}`")
                        st.markdown(f"**Similarity Score:** `{result['similarity']}`")
                        st.markdown(f"**Source File:** `{result['source_file']}`")
                        
                        if result["metadata"]:
                            st.markdown("**Metadata:**")
                            st.json(result["metadata"])
                        
                        st.markdown("**Content:**")
                        st.markdown(result["text"])
                        
                        # Option to view full text
                        if len(result["full_text"]) > 500:
                            if st.checkbox("Show full text", key=f"full_{idx}"):
                                st.text(result["full_text"])
            else:
                st.info("ℹ️ No chunks found matching your query. Try adjusting the similarity threshold.")
        
        except Exception as e:
            st.error(f"❌ Search failed: {e}")
            st.exception(e)
    
    elif search_button:
        st.warning("⚠️ Please enter a query before searching.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **How it works:**
    1. Your query is converted into embeddings using HuggingFace's `all-MiniLM-L6-v2` model
    2. All chunks stored in Redis are embedded and compared to your query embeddings
    3. Chunks with similarity scores above the threshold are returned
    4. Results are sorted by semantic relevance (highest similarity first)
    """)


if __name__ == "__main__":
    main()
