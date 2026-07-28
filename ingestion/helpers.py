import json
from pathlib import Path
from docling_core.types.doc.document import DoclingDocument
from docling.datamodel.base_models import DocItemLabel


def extract_and_save_pictures(doc: DoclingDocument, output_dir: Path) -> None:
    """Extract individual embedded images/figures and save them as PNGs."""
    for i, element in enumerate(doc.pictures):
        pil_image = element.get_image(doc)
        if pil_image:
            image_path = output_dir / f"extracted_figure_{i+1}.png"
            pil_image.save(image_path, "PNG")


def export_document_formats(doc: DoclingDocument, output_dir: Path) -> None:
    """Save the document into structured Markdown, JSON, and HTML formats."""
    doc.save_as_markdown(output_dir / "parsed.md")
    doc.save_as_json(output_dir / "parsed.json")
    doc.save_as_html(output_dir / "parsed.html")


def save_layout_visualizations(doc: DoclingDocument, output_dir: Path) -> None:
    """Generate and save full-page images overlaid with labeled bounding boxes."""
    layout_dir = output_dir / "layout"
    layout_dir.mkdir(exist_ok=True)
    
    page_images = doc.get_visualization(show_label=True)
    for page_no, img in page_images.items():
        img.save(layout_dir / f"page_{page_no:03d}.png")


def print_table_data_json(doc: DoclingDocument) -> None:
    """Find the first available table and print its rows in JSON format."""
    for node, level in doc.iterate_items():
        if getattr(node, "label", None) == DocItemLabel.TABLE:
            print(f"\n[Found Table] ID: {node.self_ref}")
            
            df = node.export_to_dataframe()
            table_json_str = df.to_json(orient="records", force_ascii=False)
            table_json_data = json.loads(table_json_str)
            
            print(json.dumps(table_json_data, indent=2, ensure_ascii=False))


def print_node_metadata_dashboard(doc: DoclingDocument) -> None:
    """Linearly loop through all nodes to display text, types, and spatial bounding boxes."""
    for node, level in doc.iterate_items():
        node_type = type(node).__name__
        label = getattr(node, "label", "UNKNOWN")
        text_content = getattr(node, "text", "")[:50].replace("\n", " ")
        
        page_no = "N/A"
        bbox_coords = "N/A"
        
        if hasattr(node, "prov") and node.prov:
            primary_prov = node.prov[0]
            page_no = primary_prov.page_no
            if hasattr(primary_prov, "bbox") and primary_prov.bbox:
                b = primary_prov.bbox
                bbox_coords = f"[L:{b.l:.1f}, T:{b.t:.1f}, R:{b.r:.1f}, B:{b.b:.1f}]"

        print(f"[{node_type} | {label}] (Page {page_no})")
        print(f"  ├─ Text: \"{text_content}...\"")
        print(f"  └─ BBox: {bbox_coords}")
        print("-" * 50)
        break
