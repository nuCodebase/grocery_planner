"""Load Costco warehouse savings products into a pandas DataFrame.

Usage:
    python webpage_to_data_code/costco_weekly_savings.py
    python webpage_to_data_code/costco_weekly_savings.py --csv costco_savings.csv
    python webpage_to_data_code/costco_weekly_savings.py --debug-text costco_debug.txt
"""

from __future__ import annotations

import argparse
import re
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup


DEFAULT_URL = "https://www.costco.com/o/-/warehouse-savings"
CATEGORY_ORDER = [
    "Apparel",
    "Appliances",
    "Automotive",
    "Baby",
    "Beauty",
    "Electronics",
    "Furniture",
    "Grocery",
    "Health & Personal Care",
    "Home & Kitchen",
    "Home Improvement",
    "Household Items",
    "Jewelry",
    "Mattresses",
    "Optical",
    "Patio, Lawn & Garden",
    "Pets",
    "Pharmacy",
    "Sports & Fitness",
]
STOP_CATEGORIES = {"Find a Warehouse", "Customer Service", "Locations & Services"}
STOP_LINES = {
    "Image",
    "Buy Online",
    "Terms Apply*.",
    "Terms Apply.",
    "Phone not included.",
}
DETAIL_PREFIXES = (
    "Item ",
    "Limit ",
    "Selection varies",
    "Pricing shown",
    "Online-Only.",
    "Warehouse pricing",
    "Available in ",
    "Search ",
    "Valid ",
)


def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def normalize_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if line:
            lines.append(line)
    return lines


def html_to_text_lines(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")
    return normalize_lines(text)


def is_category_line(line: str) -> bool:
    return line.startswith("## ") and len(line) > 3


def normalize_availability(lines: list[str], index: int) -> tuple[str | None, int]:
    line = lines[index]

    if line == "Warehouse Only":
        return "Warehouse Only", index + 1
    if line == "Online Only":
        return "Online Only", index + 1
    if index + 2 < len(lines) and lines[index : index + 3] == ["Warehouse", "&", "Online"]:
        return "Warehouse & Online", index + 3

    return None, index


def is_price_line(line: str) -> bool:
    return bool(re.fullmatch(r"\$\d[\d,]*(?:\.\d+)?", line))


def looks_like_detail(line: str) -> bool:
    if line in STOP_LINES:
        return True
    if any(line.startswith(prefix) for prefix in DETAIL_PREFIXES):
        return True
    if re.fullmatch(r"\d+ ct", line):
        return True
    if re.fullmatch(r"[\d,/+. -]+(?:oz|lb|lbs|pk|pack|vials|sprays|fl oz|sq ft|BTU|mL|ct)", line):
        return True
    return False


def looks_like_product_name(line: str) -> bool:
    if not line:
        return False
    if is_category_line(line):
        return False
    if line in STOP_LINES:
        return False
    if line in {"Warehouse", "Online", "Warehouse Only", "Online Only", "&", "Save"}:
        return False
    if is_price_line(line):
        return False
    if line.startswith("After $") or line.startswith("Delivered After $"):
        return False
    if looks_like_detail(line):
        return False
    return True


def collect_save_discount(lines: list[str], index: int) -> tuple[str, int]:
    parts = ["Save"]
    index += 1

    while index < len(lines):
        line = lines[index]
        if line == "Buy Online" or is_category_line(line):
            break
        if normalize_availability(lines, index)[0] is not None:
            break
        if line in {"-", "&"} or is_price_line(line) or re.fullmatch(r"\d[\d,]*(?:\.\d+)?", line):
            parts.append(line)
            index += 1
            continue
        break

    return " ".join(parts), index


def parse_savings_from_lines(lines: Iterable[str]) -> list[dict[str, str]]:
    lines = list(lines)
    rows: list[dict[str, str]] = []
    category: str | None = None
    i = 0

    while i < len(lines):
        line = lines[i]

        if is_category_line(line):
            category = line[3:].strip()
            if category in STOP_CATEGORIES:
                break
            i += 1
            continue

        if not category:
            i += 1
            continue

        availability, next_index = normalize_availability(lines, i)
        if availability is None:
            i += 1
            continue

        product = None
        for back in range(i - 1, max(-1, i - 8), -1):
            candidate = lines[back]
            if looks_like_product_name(candidate):
                product = candidate
                break

        if not product:
            i = next_index
            continue

        i = next_index
        discount = None

        while i < len(lines):
            line = lines[i]

            if is_category_line(line):
                break
            if normalize_availability(lines, i)[0] is not None:
                break

            if line.startswith("After $") or line.startswith("Delivered After $"):
                discount = line
                i += 1
                break

            if line == "Save":
                discount, i = collect_save_discount(lines, i)
                break

            i += 1

        if product and discount:
            rows.append(
                {
                    "category": category,
                    "product": product,
                    "discount": discount,
                    "availability": availability,
                }
            )

    return dedupe_rows(rows)


def extract_categories_from_flat_heading(lines: list[str]) -> list[str]:
    categories: list[str] = []
    joined = "\n".join(lines[:120])
    match = re.search(
        r"Apparel\s*Appliances\s*Automotive\s*Baby\s*Beauty\s*Electronics\s*Furniture\s*Grocery\s*"
        r"Health\s*&\s*Personal\s*Care\s*Home\s*&\s*Kitchen\s*Home\s*Improvement\s*Household\s*Items\s*"
        r"Jewelry\s*Mattresses\s*Optical\s*Patio,\s*Lawn\s*&\s*Garden\s*Pets\s*Pharmacy\s*Sports\s*&\s*Fitness",
        joined,
    )
    if match:
        categories = CATEGORY_ORDER[:]
    return categories


def inject_missing_category_markers(lines: list[str]) -> list[str]:
    categories = set(extract_categories_from_flat_heading(lines)) | set(CATEGORY_ORDER) | STOP_CATEGORIES
    rebuilt: list[str] = []
    in_savings_section = False

    for line in lines:
        if line == "Warehouse Savings":
            in_savings_section = True
            rebuilt.append(line)
            continue

        # Skip the flattened category nav line like:
        # ApparelAppliancesAutomotive...
        if "Apparel" in line and "Sports" in line and "Fitness" in line and "Buy Online" not in line:
            continue

        if in_savings_section and line in categories:
            rebuilt.append(f"## {line}")
            continue

        rebuilt.append(line)

    return rebuilt


def parse_savings_from_html(html: str) -> list[dict[str, str]]:
    lines = html_to_text_lines(html)
    lines = inject_missing_category_markers(lines)
    return parse_savings_from_lines(lines)


def dedupe_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    deduped: list[dict[str, str]] = []
    for row in rows:
        key = (row["category"], row["product"], row["discount"], row["availability"])
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def load_costco_savings(url: str = DEFAULT_URL) -> pd.DataFrame:
    html = fetch_html(url)
    rows = parse_savings_from_html(html)
    return pd.DataFrame(rows, columns=["category", "product", "discount", "availability"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="Costco savings page URL.")
    parser.add_argument("--csv", help="Optional path to save the parsed dataframe as CSV.")
    parser.add_argument(
        "--debug-text",
        help="Optional path to write the normalized page text for debugging.",
    )
    args = parser.parse_args()

    html = fetch_html(args.url)
    lines = inject_missing_category_markers(html_to_text_lines(html))
    rows = parse_savings_from_lines(lines)
    df = pd.DataFrame(rows, columns=["category", "product", "discount", "availability"])

    if args.debug_text:
        with open(args.debug_text, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Saved {len(df)} rows to {args.csv}")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
