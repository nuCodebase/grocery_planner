"""Parse a PDF with docling and dump the parsed document + layout for inspection."""
import sys
from pathlib import Path
from docling.datamodel.base_models import InputFormat, DocItemLabel
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument

# Make sure the repository root is on sys.path when running this file directly.
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from config import Config

## import the chunking functions from docling_chunkers.py
from ingestion.chunkers.docling_chunkers import (
    hierarchical_chunk_and_print,
    hierarchical_chunk_tree_expander_and_print,
    hybrid_chunk_and_print,
    line_based_chunk_and_print
)
## import the chunking functions from langchain_chunkers.py
from ingestion.chunkers.langchain_chunkers import (
    langchain_markdown_header_chunk,
    langchain_recursive_chunk,
    langchain_parent_child_simulation_chunk,
    langchain_semantic_chunk
)

from ingestion.helpers import (
    extract_and_save_pictures,
    export_document_formats,
    save_layout_visualizations,
    print_table_data_json,
    print_node_metadata_dashboard,
)
from ingestion.models import ChunkCollection

def print_parsing_summary(doc: DoclingDocument, output_dir: Path) -> None:
    """Print the final stats and output file locations to the console."""
    layout_dir = output_dir / "layout"
    print(f"\nPages parsed: {len(doc.pages)}")
    print(f"Text items: {len(doc.texts)}")
    print(f"Tables: {len(doc.tables)}")
    print(f"Pictures: {len(doc.pictures)}")
    print(f"Markdown -> {output_dir / 'parsed.md'}")
    print(f"JSON -> {output_dir / 'parsed.json'}")
    print(f"HTML -> {output_dir / 'parsed.html'}")
    print(f"Layout images -> {layout_dir}")


def main() -> None:
    # 1. Initialize Pipeline Settings
    pipeline_options = PdfPipelineOptions(
        generate_page_images=True,
        generate_picture_images=True,
        images_scale=2.0,
    )
    
    #### Other possible options
    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = True
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    # pipeline_options.generate_page_images = True
    # pipeline_options.generate_picture_images = True
    # pipeline_options.images_scale = 2.0

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Configuration
    config = Config()
    SRC = config.pdf_src
    OUT_DIR = config.parsed_output

    # 2. Parse Document
    result = converter.convert(SRC)
    doc = result.document

    # # 3. Process Visualizations & Data Splits
    # extract_and_save_pictures(doc, OUT_DIR)
    # export_document_formats(doc, OUT_DIR)
    # save_layout_visualizations(doc, OUT_DIR)
    # print_table_data_json(doc)
    
    # Optional metadata inspection (uncomment if you want the layout tree logging)
    # print_node_metadata_dashboard(doc)

    chunks: ChunkCollection = {}

    if config.chunking_lib == "docling":
    # ----------------------------------------------------
    # Execute and experiment with Docling chunkers
    # ----------------------------------------------------
        if config.chunking_type == "hierarchical":
            # Choice A: Run Hierarchical Chunker
            chunks = hierarchical_chunk_and_print(
                doc,
                expand_context=True,
                output_dir=OUT_DIR,
            )
            # hierarchical_chunk_tree_expander_and_print(doc, expand_context=True, output_dir=OUT_DIR) ## for TreeChunkExpander
        elif config.chunking_type == "hybrid":
            # Choice B: Run Token-Bounded Hybrid Chunker
            chunks = hybrid_chunk_and_print(doc, max_tokens=1024, output_dir=OUT_DIR)
        elif config.chunking_type == "line":
            # Choice C: Run Line-Aware Token Chunker
            chunks = line_based_chunk_and_print(doc, max_tokens=256, output_dir=OUT_DIR)
    elif config.chunking_lib == "langchain":
    # ----------------------------------------------------
    # Run LangChain Chunking Strategy Experiments
    # ----------------------------------------------------
        if config.chunking_type == "markdown":
            # Experiment 1: Split strictly on Markdown structural headers
            chunks = langchain_markdown_header_chunk(max_tokens=400, output_dir=OUT_DIR)
        elif config.chunking_type == "recursive":
            # Experiment 2: Classic recursive character structural grouping
            chunks = langchain_recursive_chunk(chunk_size_tokens=256, chunk_overlap_tokens=32, output_dir=OUT_DIR)
        elif config.chunking_type == "parent_child":
            # Experiment 3: Parent/Child dense context retrieval tracking
            chunks = langchain_parent_child_simulation_chunk(output_dir=OUT_DIR)
        elif config.chunking_type == "semantic":
            # Experiment 4: Semantic chunking with embeddings (uses hugging face embeddings model)
            chunks = langchain_semantic_chunk(output_dir=OUT_DIR)
    else:
        print(f"Unknown chunking library: {config.chunking_lib}. Please set CHUNKING_LIB in .env to 'docling' or 'langchain'.")
        return


    if chunks:
        from ingestion.memory.redis import init_redis_connection, create_schema, store_chunk, search_similar_chunks

        # Initialize Redis connection
        redis_client = init_redis_connection()
        create_schema(redis_client)

        # Save each chunk to Redis
        for chunk in chunks.values():
            headings = chunk.metadata.get("headings_path", [])
            section = " > ".join(headings) if isinstance(headings, list) else ""
            store_chunk(
                redis_client,
                chunk_id=chunk.id,
                text=chunk.text,
                document_id=SRC.name,
                section=section,
            )


        ## Testing a search query
        question = "what are microwave oven safety precautions?"
        similar_chunks = search_similar_chunks(redis_client, question, top_k=3)
        for match in similar_chunks:
            print(match)

    else:
        print("[INFO] No chunks were generated to save to Redis.")



    # 4. Final Summary
    print_parsing_summary(doc, OUT_DIR)


if __name__ == "__main__":
    main()
