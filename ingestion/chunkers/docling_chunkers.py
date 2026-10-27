from pathlib import Path
import json

from docling_core.types.doc.document import DoclingDocument

# Docling Chunker imports
from docling_core.transforms.chunker.hierarchical_chunker import (
    HierarchicalChunker,
    ChunkingDocSerializer
)
from docling_core.transforms.chunker.chunk_expander import PageChunkExpander, TreeChunkExpander
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.line_chunker import LineBasedTokenChunker
from ingestion.models import Chunk, ChunkCollection, chunks_to_dict

def _save_chunks_to_json(
    chunk_iterator,
    doc: DoclingDocument,
    filename: str,
    context_flag: bool,
    output_dir: Path,
) -> ChunkCollection:
    """Internal helper to process, print, and write out chunk data uniformly."""
    chunks: ChunkCollection = {}
    
    for i, chunk in enumerate(chunk_iterator):
        chunk_key = f"chunk_{i+1}"
        
        # NOTE: For hybrid chunks, chunker.contextualize(chunk) can be used to extract 
        # text strings that pre-attach heading paths directly into the output content.
        chunk_text = chunk.text
        
        # Extract metadata
        headings = getattr(chunk.meta, "headings", [])
        page_no = "N/A"
        element_type = "UNKNOWN"
        
        if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
            items = chunk.meta.doc_items if isinstance(chunk.meta.doc_items, list) else [chunk.meta.doc_items]
            if items:
                primary_item = items[0]
                element_type = type(primary_item).__name__
                
                if hasattr(primary_item, "prov") and primary_item.prov:
                    prov_data = primary_item.prov
                    if isinstance(prov_data, list) and len(prov_data) > 0:
                        page_no = getattr(prov_data[0], "page_no", "N/A")
                    else:
                        page_no = getattr(prov_data, "page_no", "N/A")
        
        # Build payload structure
        chunks[chunk_key] = Chunk(
            id=chunk_key,
            text=chunk_text,
            metadata={
                "page_number": page_no,
                "element_type": element_type,
                "headings_path": headings,
                "context_processed": context_flag
            },
        )
        
        # # Print snippet to terminal
        # print(f"\n[{chunk_key.upper()}] (Page {page_no}) | Type: {element_type} | Tree Path: {' > '.join(headings) if headings else 'Root'}")
        # print(f"  Content snippet:\n{chunk_text[:120]}...")
        # print("-" * 60)
        
    # Write JSON out to provided directory
    output_dir.mkdir(exist_ok=True, parents=True)
    json_output_path = output_dir / filename
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_to_dict(chunks), f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] Saved chunks to JSON format -> {json_output_path}")
    return chunks

def hierarchical_chunk_tree_expander_and_print(doc: DoclingDocument, expand_context: bool = False, output_dir: Path | None = None) -> ChunkCollection:
    """
    Split the document purely by its structural and hierarchical boundaries,
    optionally expanding small fragments to their parent semantic units using 
    TreeChunkExpander, and filters out duplicate chunks before saving.
    """
    print("\n" + "="*20 + " HIERARCHICAL CHUNKS " + "="*20)
    
    # 1. Initialize Docling's pure structure-aware chunker
    chunker = HierarchicalChunker(merge_list_items=True)
    chunk_iterator = chunker.chunk(dl_doc=doc)
    
    # 2. Context Expansion within the flag condition using TreeChunkExpander
    if expand_context:
        print("[INFO] TreeChunkExpander flag is active. Enhancing fragments to full DocItem units...")
        
        # FIXED: Instantiated TreeChunkExpander
        expander = TreeChunkExpander()
        serializer = ChunkingDocSerializer(doc=doc)
        
        base_chunks = list(chunk_iterator)
        expanded_chunks = []
        
        for chunk in base_chunks:
            # Expand structural components up to their native tree-container boundaries
            enriched_chunk = expander.expand(chunk=chunk, dl_doc=doc, serializer=serializer)
            expanded_chunks.append(enriched_chunk)
            
        chunk_iterator = expanded_chunks

    # 3. Deduplicate the resulting tree chunks before routing to save files
    chunks_list = list(chunk_iterator)
    unique_chunks = []
    seen_texts = set()
    duplicate_count = 0

    for chunk in chunks_list:
        chunk_text_stripped = chunk.text.strip()
        
        # Skip processing if this specific semantic layout text was already seen
        if chunk_text_stripped in seen_texts:
            duplicate_count += 1
            continue
            
        seen_texts.add(chunk_text_stripped)
        unique_chunks.append(chunk)

    if duplicate_count > 0:
        print(f"[INFO] Deduplication complete. Removed {duplicate_count} duplicate chunks.")

    # 4. Save out only the unique semantically-expanded chunks using provided output_dir
    if output_dir is None:
        output_dir = Path.cwd() / "parsed_output"
    return _save_chunks_to_json(unique_chunks, doc, filename="hierarchical_chunks.json", context_flag=expand_context, output_dir=output_dir)

def hierarchical_chunk_and_print(doc: DoclingDocument, expand_context: bool = False, output_dir: Path | None = None) -> ChunkCollection:
    """
    Split the document purely by its structural and hierarchical boundaries,
    print the output elements, and save the chunks to a JSON file.
    
    Args:
        doc (DoclingDocument): The parsed docling document object.
        expand_context (bool): If True, applies PageChunkExpander to enrich 
                               the structural chunks with surrounding page context.
    """
    print("\n" + "="*20 + " HIERARCHICAL CHUNKS " + "="*20)
    
    # 1. Initialize Docling's pure structure-aware chunker
    chunker = HierarchicalChunker(merge_list_items=True)
    chunk_iterator = chunker.chunk(dl_doc=doc)
    
    # 2. Add implementation code for Context Expansion within the flag condition
    if expand_context:
        print("[INFO] PageChunkExpander flag is active. Enhancing item contextual horizons...")
        expander = PageChunkExpander()
        
        # FIXED: Initialize the required serializer instance for the document
        serializer = ChunkingDocSerializer(doc=doc)
        
        base_chunks = list(chunk_iterator)
        expanded_chunks = []
        
        for chunk in base_chunks:
            # FIXED: Added 'serializer=serializer' to fulfill positional layout arguments
            enriched_chunk = expander.expand(chunk=chunk, dl_doc=doc, serializer=serializer)
            expanded_chunks.append(enriched_chunk)
            
        chunk_iterator = expanded_chunks

    # 3. Deduplicate the resulting chunks before saving
    # Convert chunk_iterator to a list so we can process and prune it
    chunks_list = list(chunk_iterator)
    unique_chunks = []
    seen_texts = set()
    duplicate_count = 0

    for chunk in chunks_list:
        # Strip trailing white spaces to prevent identical strings with extra spaces from slipping through
        chunk_text_stripped = chunk.text.strip()
        
        if chunk_text_stripped in seen_texts:
            duplicate_count += 1
            continue  # Skip this chunk entirely
            
        seen_texts.add(chunk_text_stripped)
        unique_chunks.append(chunk)

    if duplicate_count > 0:
        print(f"[INFO] Deduplication complete. Removed {duplicate_count} duplicate chunks.")

    if output_dir is None:
        output_dir = Path.cwd() / "parsed_output"
    return _save_chunks_to_json(unique_chunks, doc, filename="hierarchical_chunks.json", context_flag=expand_context, output_dir=output_dir)


def hybrid_chunk_and_print(doc: DoclingDocument, max_tokens: int = 400, output_dir: Path | None = None) -> ChunkCollection:
    """
    Split the document using token-size and structural thresholds.
    Enriches text blocks by auto-prepending parent path headings to the chunk.
    """
    print("\n" + "="*20 + " HYBRID CHUNKS (TOKEN + LAYOUT AWARE) " + "="*20)
    print(f"[INFO] Hybrid Chunker active. Enforcing max token limit of: {max_tokens}")
    
    # 1. Initialize Hybrid Chunker with a fixed token threshold
    # You can also pass a specific 'tokenizer' path here (e.g., "sentence-transformers/all-MiniLM-L6-v2")
    chunker = HybridChunker(max_tokens=max_tokens)
    
    # 2. Generate chunks using 'dl_doc' parameter
    chunk_iterator = chunker.chunk(dl_doc=doc)
    
    if output_dir is None:
        output_dir = Path.cwd() / "parsed_output"
    return _save_chunks_to_json(chunk_iterator, doc, filename="hybrid_chunks.json", context_flag=True, output_dir=output_dir)

def line_based_chunk_and_print(doc: DoclingDocument, max_tokens: int = 256, output_dir: Path | None = None) -> ChunkCollection:
    """
    Split the document using a line-oriented tokenization workflow.
    Ensures complete sentence lines are preserved together where possible.
    """
    print("\n" + "="*20 + " LINE-BASED TOKEN CHUNKS " + "="*20)
    print(f"[INFO] LineBasedTokenChunker active. Target Max Tokens: {max_tokens}")
    
    # 1. Initialize LineBased Chunker (respects absolute line breaks)
    chunker = LineBasedTokenChunker(max_tokens=max_tokens)
    
    # 2. Extract line elements via chunk generation stream
    chunk_iterator = chunker.chunk(dl_doc=doc)
    
    if output_dir is None:
        output_dir = Path.cwd() / "parsed_output"
    return _save_chunks_to_json(chunk_iterator, doc, filename="line_chunks.json", context_flag=False, output_dir=output_dir)
