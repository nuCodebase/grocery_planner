import json
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from PIL import ImageDraw, ImageFont
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


@dataclass(frozen=True)
class TextBoundingBox:
    """A text bounding box in top-left-origin PDF coordinates."""

    left: float
    top: float
    right: float
    bottom: float
    label: str = ""
    text: str = ""


@dataclass(frozen=True)
class BoundingBoxGroup:
    """The enclosing box and source boxes for an overlap-connected group."""

    left: float
    top: float
    right: float
    bottom: float
    boxes: tuple[TextBoundingBox, ...]


def _distance_between_boxes(
    first: TextBoundingBox,
    second: TextBoundingBox,
) -> float:
    """Return the shortest edge-to-edge distance between two boxes."""

    horizontal_gap = max(
        first.left - second.right,
        second.left - first.right,
        0.0,
    )
    vertical_gap = max(
        first.top - second.bottom,
        second.top - first.bottom,
        0.0,
    )
    return hypot(horizontal_gap, vertical_gap)


def group_overlapping_bounding_boxes(
    boxes: list[TextBoundingBox],
    proximity_threshold: float = 0.0,
) -> list[BoundingBoxGroup]:
    """
    Group overlapping or nearby boxes and return their enclosing rectangles.

    ``proximity_threshold`` is the maximum edge-to-edge distance, in the same
    PDF coordinate units as the boxes. A value of 0 merges only overlapping or
    touching boxes. Grouping is transitive, and isolated boxes are preserved.
    """

    if proximity_threshold < 0:
        raise ValueError("proximity_threshold must be greater than or equal to 0")

    if not boxes:
        return []

    parent = list(range(len(boxes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first_index: int, second_index: int) -> None:
        first_root = find(first_index)
        second_root = find(second_index)
        if first_root != second_root:
            parent[second_root] = first_root

    for first_index, first_box in enumerate(boxes):
        for second_index in range(first_index + 1, len(boxes)):
            if (
                _distance_between_boxes(first_box, boxes[second_index])
                <= proximity_threshold
            ):
                union(first_index, second_index)

    grouped_boxes: dict[int, list[TextBoundingBox]] = {}
    for index, box in enumerate(boxes):
        grouped_boxes.setdefault(find(index), []).append(box)

    groups = []
    for source_boxes in grouped_boxes.values():
        groups.append(
            BoundingBoxGroup(
                left=min(box.left for box in source_boxes),
                top=min(box.top for box in source_boxes),
                right=max(box.right for box in source_boxes),
                bottom=max(box.bottom for box in source_boxes),
                boxes=tuple(source_boxes),
            )
        )

    return sorted(groups, key=lambda group: (group.top, group.left))


def get_group_text(group: BoundingBoxGroup) -> str:
    """Combine a group's text in top-to-bottom, left-to-right order."""

    boxes_in_reading_order = sorted(
        group.boxes,
        key=lambda box: (box.top, box.left),
    )
    text_parts = [
        box.text.strip()
        for box in boxes_in_reading_order
        if box.text and box.text.strip()
    ]
    # An element can have multiple provenance boxes; avoid repeating its text.
    return "\n".join(dict.fromkeys(text_parts))


# 1. Setup paths
source_pdf = "/Users/nikileshurella/Documents/GitHub/grocery_planner/ingestion/data/bimart_weekly.pdf"  # Change to your input file
# source_pdf = "/Users/nikileshurella/Documents/GitHub/grocery_planner/ingestion/data/weekly_flyer.pdf"  # Change to your input file
output_dir = Path("ingestion/flyer_output")
output_dir.mkdir(parents=True, exist_ok=True)

# Maximum edge-to-edge gap between boxes that should be merged, in PDF points.
# Set to 0 to merge only overlapping/touching boxes. Increase gradually if
# nearby text regions should be combined (for example, try 5, 10, or 15).
box_proximity_threshold = 5.0

# 2. Configure Docling to explicitly generate and preserve page images
pipeline_options = PdfPipelineOptions()
pipeline_options.generate_page_images = True  # Crucial for backend rendering
pipeline_options.images_scale = 2.0           # Increases image resolution for crispness

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

print("Parsing document and rendering pages...")
result = converter.convert(source_pdf)
doc = result.document

# 3. Process each page's image and overlay bounding boxes
for page_no, page_item in doc.pages.items():
    if not page_item.image:
        print(f"Skipping page {page_no}: No visual image layer available.")
        continue
        
    # Get the underlying PIL image and its canvas dimensions
    page_image = page_item.image.pil_image.copy()
    img_w, img_h = page_image.size
    
    # Get standard PDF canvas dimensions to compute the scaling ratio
    pdf_w = page_item.size.width
    pdf_h = page_item.size.height
    
    scale_x = img_w / pdf_w
    scale_y = img_h / pdf_h
    
    # Initialize Pillow drawing tool
    draw = ImageDraw.Draw(page_image)
    try:
        font = ImageFont.load_default()
    except IOError:
        font = None

    print(f"Drawing bounding boxes for Page {page_no}...")

    # Collect all text boxes for this page in top-left-origin PDF coordinates.
    page_boxes = []
    for element in doc.texts:
        for prov in element.prov:
            if prov.page_no != page_no:
                continue

            # Standardize system to Top-Left origin to match image space
            bbox = prov.bbox.to_top_left_origin(page_height=pdf_h)
            element_text = getattr(element, "text", "") or ""
            page_boxes.append(
                TextBoundingBox(
                    left=bbox.l,
                    top=bbox.t,
                    right=bbox.r,
                    bottom=bbox.b,
                    label=str(element.label) if element.label else "",
                    text=str(element_text),
                )
            )

    # Post-process the page before mapping the boxes to image coordinates.
    bounding_box_groups = group_overlapping_bounding_boxes(
        page_boxes,
        proximity_threshold=box_proximity_threshold,
    )

    # Remove empty and single-word groups before numbering or saving outputs.
    groups_with_text = [
        (group, group_text)
        for group in bounding_box_groups
        if len((group_text := get_group_text(group)).split()) > 1
    ]
    removed_group_count = len(bounding_box_groups) - len(groups_with_text)

    # Each JSON key matches the number drawn on the corresponding merged box.
    page_text = {
        str(group_number): group_text
        for group_number, (_, group_text) in enumerate(groups_with_text, start=1)
    }
    json_output_path = output_dir / f"page_{page_no}_text.json"
    with json_output_path.open("w", encoding="utf-8") as json_file:
        json.dump(page_text, json_file, ensure_ascii=False, indent=2)

    for group_number, (group, _) in enumerate(groups_with_text, start=1):
        # Map standard PDF dimensions to the scaled image dimensions
        x0 = group.left * scale_x
        y0 = group.top * scale_y
        x1 = group.right * scale_x
        y1 = group.bottom * scale_y

        # Draw the enclosing box for the overlap-connected group.
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)

        if font:
            labels = list(dict.fromkeys(box.label for box in group.boxes if box.label))
            label_text = ", ".join(labels)
            if len(group.boxes) > 1:
                label_text = f"{label_text} ({len(group.boxes)} boxes)".strip()
            marker_text = f"{group_number}: {label_text}".rstrip(": ")
            draw.text((x0 + 2, y0 - 12), marker_text, fill="blue", font=font)

    print(
        f"Page {page_no}: combined {len(page_boxes)} text boxes into "
        f"{len(bounding_box_groups)} bounding boxes, then removed "
        f"{removed_group_count} boxes containing fewer than 2 words."
    )
    print(f"Saved: {json_output_path}")
                
    # 4. Save visual output per page
    output_path = output_dir / f"page_{page_no}_overlay.png"
    page_image.save(output_path, "PNG")
    print(f"Saved: {output_path}")

print("\nProcessing complete! Check the 'flyer_output' folder.")
