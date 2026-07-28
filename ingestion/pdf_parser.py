"""Parse a PDF with docling and dump the parsed document + layout for inspection."""
import json
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

    # Execute and experiment with Docling chunkers
    # ----------------------------------------------------
    # Choice A: Run Hierarchical Chunker
    # hierarchical_chunk_and_print(doc, expand_context=True) ## For PageChunkExpander
    # hierarchical_chunk_tree_expander_and_print(doc, expand_context=True, output_dir=OUT_DIR) ## for TreeChunkExpander
    
    # Choice B: Run Token-Bounded Hybrid Chunker
    # hybrid_chunk_and_print(doc, max_tokens=1024, output_dir=OUT_DIR)

    # Choice C: Run Line-Aware Token Chunker
    # line_based_chunk_and_print(doc, max_tokens=256, output_dir=OUT_DIR)
    # ----------------------------------------------------


    # ----------------------------------------------------
    # Run LangChain Chunking Strategy Experiments
    # ----------------------------------------------------
    # Experiment 1: Split strictly on Markdown structural headers
    # langchain_markdown_header_chunk(max_tokens=400, output_dir=OUT_DIR)
    
    # Experiment 2: Classic recursive character structural grouping
    # langchain_recursive_chunk(chunk_size_tokens=256, chunk_overlap_tokens=32, output_dir=OUT_DIR)
    
    # Experiment 3: Parent/Child dense context retrieval tracking
    # langchain_parent_child_simulation_chunk(output_dir=OUT_DIR)

    # Experiment 4: Semantic chunking with embeddings (uses hugging face embeddings model)
    langchain_semantic_chunk(output_dir=OUT_DIR)
    # ----------------------------------------------------

    # 4. Final Summary
    print_parsing_summary(doc, OUT_DIR)


if __name__ == "__main__":
    main()
