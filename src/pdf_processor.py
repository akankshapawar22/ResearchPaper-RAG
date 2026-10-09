
from pathlib import Path
import json
import re

import pymupdf
import pymupdf4llm


PDF_FOLDER = Path("data/papers")
OUTPUT_FOLDER = Path("data/processed")

CHUNK_SIZE = 450
CHUNK_OVERLAP = 75


def extract_pdf(pdf_path):
    with pymupdf.open(pdf_path) as doc:
        pdf_metadata = doc.metadata or {}

        title = (pdf_metadata.get("title") or "").strip()
        authors = (pdf_metadata.get("author") or "").strip()

        if not title:
            title = pdf_path.stem

        page_results = pymupdf4llm.to_markdown(
            doc,
            page_chunks=True
        )

        pages = []
        for index, result in enumerate(page_results):
            text = result.get("text", "").strip()

            if text:
                pages.append({
                    "page": index + 1,
                    "text": text
                })

    return title, authors, pages


def split_into_chunks(text):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end == len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def is_heading(line):
    line = line.strip()

    match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)

    if match:
        heading = match.group(1).strip()
        heading = re.sub(r"^\*\*(.*?)\*\*$", r"\1", heading)
        return heading.strip()

    match = re.match(r"^\*\*(.+?)\*\*$", line)

    if match:
        return match.group(1).strip()

    return None


def split_page_by_section(page_text, previous_section="Unknown"):
    """
    Split a page into section-specific text blocks.
    Section headings are retained in the text.
    """
    lines = page_text.splitlines()

    sections = []
    current_section = previous_section
    current_lines = []

    for line in lines:
        heading = is_heading(line)

        if heading:
            # Save text belonging to the previous section.
            text = "\n".join(current_lines).strip()

            if text:
                sections.append({
                    "section": current_section,
                    "text": text
                })

            current_section = heading
            current_lines = [line]

        else:
            current_lines.append(line)

    # Save the final section on this page.
    text = "\n".join(current_lines).strip()

    if text:
        sections.append({
            "section": current_section,
            "text": text
        })

    return sections, current_section


def process_pdf(pdf_path):
    title, authors, pages = extract_pdf(pdf_path)

    records = []
    chunk_number = 0
    current_section = "Unknown"

    for page_data in pages:
        page_number = page_data["page"]
        page_text = page_data["text"]

        sections, current_section = split_page_by_section(
            page_text,
            current_section
        )

        for section_data in sections:
            section_name = section_data["section"]
            section_text = section_data["text"]

            chunks = split_into_chunks(section_text)

            for chunk_text in chunks:
                records.append({
                    "chunk_id": (
                        f"{pdf_path.stem}_{chunk_number:05d}"
                    ),
                    "title": title,
                    "authors": authors,
                    "filename": pdf_path.name,
                    "page": page_number,
                    "section": section_name,
                    "text": chunk_text
                })

                chunk_number += 1

    return records, len(pages)


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {PDF_FOLDER.resolve()}")
        return

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")

        records, page_count = process_pdf(pdf_path)

        output_path = OUTPUT_FOLDER / f"{pdf_path.stem}.jsonl"

        with output_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

        print(
            f"Title: "
            f"{records[0]['title'] if records else pdf_path.stem}"
        )
        print(f"Pages extracted: {page_count}")
        print(f"Chunks created: {len(records)}")
        print(f"Saved: {output_path}")

        print("\nSample chunk metadata:")

        for record in records[:5]:
            print(
                f"Chunk: {record['chunk_id']} | "
                f"Page: {record['page']} | "
                f"Section: {record['section']}"
            )


if __name__ == "__main__":
    main()