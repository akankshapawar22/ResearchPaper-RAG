from pathlib import Path
from pypdf import PdfReader


def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    return "\n".join(pages)


if __name__ == "__main__":
    pdf_folder = Path("data/papers")

    pdf_files = list(pdf_folder.glob("*.pdf"))

    if not pdf_files:
        print("No PDF found in data/papers")
    else:
        pdf_path = pdf_files[0]

        text = load_pdf(pdf_path)

        print(f"Loaded: {pdf_path.name}")
        print(f"Characters extracted: {len(text)}")
        print("\n--- First 2000 characters ---\n")
        print(text[:2000])