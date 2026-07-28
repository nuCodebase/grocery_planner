from pathlib import Path
import json

# LangChain specific text splitter engines
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)

# Add this new import to your import block
from langchain_experimental.text_splitter import SemanticChunker

# Alternatively for local offline runs, uncomment below:
from langchain_community.embeddings import HuggingFaceEmbeddings

# This module expects callers to pass `output_dir: Path` (where parsed.md lives)
def _ensure_output_dir(output_dir: Path | None) -> Path:
    if output_dir is None:
        output_dir = Path.cwd() / "parsed_output"
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir


def langchain_markdown_header_chunk(max_tokens: int = 500, output_dir: Path | None = None) -> None:
    """
    Strategy 1: Structural Markdown Chunker
    Splits the document at physical heading changes (#, ##, ###) so that sections
    stay together. Then splits large sections by character tokens.
    """
    print("\n" + "="*20 + " LANGCHAIN MARKDOWN HEADER CHUNKS " + "="*20)
    
    output_dir = _ensure_output_dir(output_dir)
    markdown_file = output_dir / "parsed.md"
    if not markdown_file.exists():
        print(f"[ERROR] Required file {markdown_file} missing. Run Docling exports first.")
        return

    with open(markdown_file, "r", encoding="utf-8") as f:
        md_text = f.read()

    # 1. Define which markdown headers to track and inject into metadata
    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]
    
    # 2. Split structurally by headers
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False )
    md_header_splits = md_splitter.split_text(md_text)

    # 3. Sub-split large heading blocks to enforce strict token/character maximums
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens * 4,  # Rough conversion: 1 token ≈ 4 characters
        chunk_overlap=50
    )
    
    final_docs = text_splitter.split_documents(md_header_splits)
    
    # 4. Save to JSON using your sequential schema
    chunks_dict = {}
    for i, doc in enumerate(final_docs):
        chunk_key = f"chunk_{i+1}"
        chunks_dict[chunk_key] = {
            "text": doc.page_content,
            "metadata": {
                "source_framework": "langchain_markdown",
                **doc.metadata  # Automatically carries over inherited header paths!
            }
        }
        print(f"\n[{chunk_key.upper()}] | Inherited Context: {doc.metadata}")
        print(f"  Content snippet:\n{doc.page_content[:100]}...")
        print("-" * 60)

    output_path = output_dir / "lc_markdown_chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_dict, f, indent=2, ensure_ascii=False)
    print(f"[SUCCESS] Saved LangChain Markdown chunks to: {output_path}")


def langchain_recursive_chunk(chunk_size_tokens: int = 300, chunk_overlap_tokens: int = 30, output_dir: Path | None = None) -> None:
    """
    Strategy 2: Standard Semantic Recursive Chunker
    Uses an ordered list of delimiters ("\n\n", "\n", " ", "") to split text.
    It prioritizes keeping paragraphs, then sentences, then words together.
    """
    print("\n" + "="*20 + " LANGCHAIN RECURSIVE CHARACTER CHUNKS " + "="*20)
    
    output_dir = _ensure_output_dir(output_dir)
    markdown_file = output_dir / "parsed.md"
    with open(markdown_file, "r", encoding="utf-8") as f:
        text_content = f.read()

    # Approximate token size using standard character length scaling
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_tokens * 4,
        chunk_overlap=chunk_overlap_tokens * 4,
        length_function=len,
    )
    
    chunks = splitter.split_text(text_content)
    
    chunks_dict = {}
    for i, chunk in enumerate(chunks):
        chunk_key = f"chunk_{i+1}"
        chunks_dict[chunk_key] = {
            "text": chunk,
            "metadata": {
                "source_framework": "langchain_recursive",
                "chunk_size_characters": len(chunk)
            }
        }
        print(f"\n[{chunk_key.upper()}] | Char Length: {len(chunk)}")
        print(f"  Content snippet:\n{chunk[:100]}...")
        print("-" * 60)

    output_path = output_dir / "lc_recursive_chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_dict, f, indent=2, ensure_ascii=False)
    print(f"[SUCCESS] Saved LangChain Recursive chunks to: {output_path}")


def langchain_parent_child_simulation_chunk(output_dir: Path | None = None) -> None:
    """
    Strategy 3: Multi-Vector / Parent-Child Mapping Simulation
    Creates large "Parent" context blocks (e.g., full sections) for LLM synthesis,
    and cuts smaller nested "Child" chunks (e.g., sentences) to generate high-accuracy embeddings.
    """
    print("\n" + "="*20 + " LANGCHAIN PARENT-CHILD CHUNKS " + "="*20)
    
    output_dir = _ensure_output_dir(output_dir)
    markdown_file = output_dir / "parsed.md"
    with open(markdown_file, "r", encoding="utf-8") as f:
        text_content = f.read()

    # Define Parent (broad) and Child (narrow) boundaries
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    
    parent_docs = parent_splitter.split_text(text_content)
    
    chunks_dict = {}
    global_counter = 1
    
    for p_id, parent_text in enumerate(parent_docs):
        # Generate smaller sub-chunks from the parent text
        child_sub_chunks = child_splitter.split_text(parent_text)
        
        for c_id, child_text in enumerate(child_sub_chunks):
            chunk_key = f"chunk_{global_counter}"
            chunks_dict[chunk_key] = {
                "text": child_text,  # Sent to vector database for embedding search
                "metadata": {
                    "source_framework": "langchain_parent_child",
                    "parent_id": f"parent_{p_id+1}",
                    "parent_full_context": parent_text  # Returned to LLM window when child matches
                }
            }
            global_counter += 1
            
    output_path = output_dir / "lc_parent_child_chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_dict, f, indent=2, ensure_ascii=False)
    print(f"[SUCCESS] Saved LangChain Parent-Child simulation chunks to: {output_path}")


def langchain_semantic_chunk(output_dir: Path | None = None) -> None:
    """
    Strategy 4: Embedding-Distance Semantic Chunker
    Analyzes sentence-by-sentence embeddings and creates boundaries 
    only when the difference in semantic meaning exceeds a calculated threshold.
    """
    print("\n" + "="*20 + " LANGCHAIN SEMANTIC EMBEDDING CHUNKS " + "="*20)
    output_dir = _ensure_output_dir(output_dir)
    markdown_file = output_dir / "parsed.md"
    
    if not markdown_file.exists():
        print(f"[ERROR] Required file {markdown_file} missing. Run Docling exports first.")
        return

    with open(markdown_file, "r", encoding="utf-8") as f:
        text_content = f.read()

    # 1. Initialize your targeted embedding model 
    # # (Note: OpenAIEmbeddings requires your OPENAI_API_KEY environment variable set)
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Alternative local model (no API key needed):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Configure the Semantic Chunker
    # 'percentile' means it splits text when differences cross a statistical percentile boundary.
    # Alternatives: 'standard_deviation', 'interquartile', or 'gradient'
    splitter = SemanticChunker(
        embeddings=embeddings, 
        breakpoint_threshold_type="percentile"
    )

    # 3. Process the file string into LangChain documents
    semantic_docs = splitter.create_documents([text_content])

    # 4. Save to JSON using your sequential schema
    chunks_dict = {}
    for i, doc in enumerate(semantic_docs):
        chunk_key = f"chunk_{i+1}"
        chunks_dict[chunk_key] = {
            "text": doc.page_content,
            "metadata": {
                "source_framework": "langchain_semantic",
                "character_length": len(doc.page_content)
            }
        }
        print(f"\n[{chunk_key.upper()}] | Size: {len(doc.page_content)} characters")
        print(f"  Content snippet:\n{doc.page_content[:120]}...")
        print("-" * 60)

    output_path = output_dir / "lc_semantic_chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_dict, f, indent=2, ensure_ascii=False)
        
    print(f"[SUCCESS] Saved LangChain Semantic chunks to: {output_path}")