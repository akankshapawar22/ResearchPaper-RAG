if __name__ == "__main__":
    pdf_path = Path("data/papers/2296_AnastasiaMoumtzidou_etal2020 (1).pdf")

    with pymupdf.open(pdf_path) as doc:
        markdown = pymupdf4llm.to_markdown(doc)

    output_path = Path("data/processed/markdown_preview.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    print(f"Markdown saved to: {output_path}")