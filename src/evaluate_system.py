import json
from pathlib import Path

import faiss

from hybrid_retriever import HybridRetriever


ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "indexes"
CHUNKS_FILE = INDEX_DIR / "chunks.jsonl"
FAISS_FILE = INDEX_DIR / "dense.faiss"


def load_chunks():
    chunks = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                chunks.append(json.loads(line))

    return chunks


def check_index_consistency(chunks):
    print("\n[1] Checking FAISS/chunk consistency...")

    index = faiss.read_index(str(FAISS_FILE))

    faiss_count = index.ntotal
    chunk_count = len(chunks)

    print(f"    FAISS vectors: {faiss_count}")
    print(f"    Chunk records: {chunk_count}")

    if faiss_count != chunk_count:
        print("    FAIL: FAISS vector count does not match chunk count.")
        return False

    print("    PASS")
    return True


def check_metadata(chunks):
    print("\n[2] Checking chunk metadata...")

    required_fields = {
        "chunk_id",
        "title",
        "filename",
        "page",
        "section",
        "text",
    }

    failures = []

    for i, chunk in enumerate(chunks):
        missing = required_fields - set(chunk.keys())

        if missing:
            failures.append(
                f"chunk {i}: missing {sorted(missing)}"
            )

        if not chunk.get("text", "").strip():
            failures.append(
                f"chunk {i}: empty text"
            )

        if not chunk.get("filename"):
            failures.append(
                f"chunk {i}: missing filename"
            )

    if failures:
        print("    FAIL")
        for failure in failures[:10]:
            print(f"    - {failure}")

        if len(failures) > 10:
            print(f"    ... and {len(failures) - 10} more")

        return False

    print(f"    Checked {len(chunks)} chunks")
    print("    PASS")
    return True


def check_duplicate_chunk_ids(chunks):
    print("\n[3] Checking duplicate chunk IDs...")

    ids = [chunk["chunk_id"] for chunk in chunks]
    duplicates = sorted(
        {chunk_id for chunk_id in ids if ids.count(chunk_id) > 1}
    )

    if duplicates:
        print("    FAIL")
        for chunk_id in duplicates[:10]:
            print(f"    - {chunk_id}")

        return False

    print("    No duplicate chunk IDs found")
    print("    PASS")
    return True


def check_papers(chunks):
    print("\n[4] Checking indexed papers...")

    papers = sorted(
        {
            chunk["filename"]
            for chunk in chunks
            if chunk.get("filename")
        }
    )

    if not papers:
        print("    FAIL: No papers found.")
        return False

    print(f"    Papers indexed: {len(papers)}")

    for paper in papers:
        count = sum(
            1 for chunk in chunks
            if chunk["filename"] == paper
        )
        print(f"    - {paper}: {count} chunks")

    print("    PASS")
    return True


def check_retrieval(chunks):
    print("\n[5] Checking hybrid retrieval...")

    retriever = HybridRetriever()

    question = "What dataset was used?"

    results = retriever.retrieve(
        question,
        top_k=5,
    )

    if not results:
        print("    FAIL: Retrieval returned no results.")
        return False

    print(f"    Retrieved results: {len(results)}")

    failures = []

    for i, result in enumerate(results, start=1):
        required = [
            "chunk_id",
            "filename",
            "page",
            "section",
            "text",
        ]

        missing = [
            field for field in required
            if field not in result
        ]

        if missing:
            failures.append(
                f"result {i}: missing {missing}"
            )

        if not result.get("text", "").strip():
            failures.append(
                f"result {i}: empty text"
            )

    if failures:
        print("    FAIL")

        for failure in failures:
            print(f"    - {failure}")

        return False

    print("    PASS")
    return True


def check_paper_filtering(chunks):
    print("\n[6] Checking paper-level filtering...")

    papers = sorted(
        {
            chunk["filename"]
            for chunk in chunks
            if chunk.get("filename")
        }
    )

    if len(papers) < 2:
        print("    SKIP: At least two papers are required.")
        return True

    retriever = HybridRetriever()

    selected_paper = papers[0]

    results = retriever.retrieve(
        "What is the methodology?",
        top_k=5,
        filename=selected_paper,
    )

    if not results:
        print("    FAIL: No results returned.")
        return False

    wrong_papers = [
        result["filename"]
        for result in results
        if result["filename"] != selected_paper
    ]

    if wrong_papers:
        print("    FAIL: Results from another paper were returned.")

        for paper in wrong_papers:
            print(f"    - {paper}")

        return False

    print(f"    Selected paper: {selected_paper}")
    print("    All retrieved results belong to the selected paper")
    print("    PASS")

    return True


def main():
    print("=" * 60)
    print("ResearchPaper-RAG System Validation")
    print("=" * 60)

    if not CHUNKS_FILE.exists():
        print(f"\nERROR: Missing {CHUNKS_FILE}")
        return

    if not FAISS_FILE.exists():
        print(f"\nERROR: Missing {FAISS_FILE}")
        return

    chunks = load_chunks()

    checks = [
        check_index_consistency(chunks),
        check_metadata(chunks),
        check_duplicate_chunk_ids(chunks),
        check_papers(chunks),
        check_retrieval(chunks),
        check_paper_filtering(chunks),
    ]

    passed = sum(checks)
    total = len(checks)

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("Overall result: PASS")
    else:
        print("Overall result: FAIL")

    print("=" * 60)


if __name__ == "__main__":
    main()