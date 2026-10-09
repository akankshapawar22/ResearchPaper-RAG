import json
from pathlib import Path

from hybrid_retriever import HybridRetriever


ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "indexes"
CHUNKS_FILE = INDEX_DIR / "chunks.jsonl"


EVALUATION_SET = [
    {
        "question": "What dataset was used?",
        "relevant_chunks": [
            "2296_AnastasiaMoumtzidou_etal2020 (1)_00024",
        ],
    },
    {
        "question": "How many events are included in the dataset?",
        "relevant_chunks": [
            "2296_AnastasiaMoumtzidou_etal2020 (1)_00024",
        ],
    },
    {
        "question": "What image bands were used?",
        "relevant_chunks": [
            "2296_AnastasiaMoumtzidou_etal2020 (1)_00019",
            "2296_AnastasiaMoumtzidou_etal2020 (1)_00020",
            "2296_AnastasiaMoumtzidou_etal2020 (1)_00021",
        ],
    },
    {
        "question": "What is the main contribution of the Pre-Trained Image Processing Transformer?",
        "relevant_chunks": [
            "Chen_Pre-Trained_Image_Processing_Transformer_CVPR_2021_paper_00001",
            "Chen_Pre-Trained_Image_Processing_Transformer_CVPR_2021_paper_00004",
            "Chen_Pre-Trained_Image_Processing_Transformer_CVPR_2021_paper_00024",
        ],
    },
    {
        "question": "What is the role of spectral bands in satellite image classification?",
        "relevant_chunks": [
            "InfluenceofSpectralBandsinSatelliteImageClassification_00000",
            "InfluenceofSpectralBandsinSatelliteImageClassification_00009",
            "InfluenceofSpectralBandsinSatelliteImageClassification_00010",
            "InfluenceofSpectralBandsinSatelliteImageClassification_00011",
        ],
    },
]


def load_chunks():
    chunks = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                chunks.append(json.loads(line))

    return chunks


def reciprocal_rank(results, relevant_ids):
    for rank, result in enumerate(results, start=1):
        if result["chunk_id"] in relevant_ids:
            return 1.0 / rank

    return 0.0


def recall_at_k(results, relevant_ids, k):
    retrieved_ids = {
        result["chunk_id"]
        for result in results[:k]
    }

    relevant_found = retrieved_ids.intersection(relevant_ids)

    return len(relevant_found) / len(relevant_ids)


def hit_at_k(results, relevant_ids, k):
    retrieved_ids = {
        result["chunk_id"]
        for result in results[:k]
    }

    return (
        1.0
        if retrieved_ids.intersection(relevant_ids)
        else 0.0
    )


def main():
    print("=" * 70)
    print("ResearchPaper-RAG Retrieval Evaluation")
    print("=" * 70)

    if not CHUNKS_FILE.exists():
        print(f"\nERROR: Missing {CHUNKS_FILE}")
        return

    chunks = load_chunks()

    chunk_ids = {
        chunk["chunk_id"]
        for chunk in chunks
    }

    print("\nChecking evaluation set...")

    invalid_ids = []

    for item in EVALUATION_SET:
        for chunk_id in item["relevant_chunks"]:
            if chunk_id not in chunk_ids:
                invalid_ids.append(chunk_id)

    if invalid_ids:
        print("ERROR: The following chunk IDs do not exist:")

        for chunk_id in sorted(set(invalid_ids)):
            print(f"  - {chunk_id}")

        return

    print(f"Evaluation questions: {len(EVALUATION_SET)}")
    print("Evaluation set validation: PASS")

    print("\nLoading hybrid retriever...")
    retriever = HybridRetriever()

    recall5_scores = []
    recall10_scores = []
    mrr_scores = []
    hit5_scores = []
    hit10_scores = []

    print("\n" + "-" * 70)

    for i, item in enumerate(EVALUATION_SET, start=1):
        question = item["question"]
        relevant_ids = set(item["relevant_chunks"])

        print(f"\nQuestion {i}: {question}")

        results = retriever.retrieve(
            question,
            top_k=10,
        )

        r5 = recall_at_k(results, relevant_ids, 5)
        r10 = recall_at_k(results, relevant_ids, 10)
        mrr = reciprocal_rank(results, relevant_ids)
        h5 = hit_at_k(results, relevant_ids, 5)
        h10 = hit_at_k(results, relevant_ids, 10)

        recall5_scores.append(r5)
        recall10_scores.append(r10)
        mrr_scores.append(mrr)
        hit5_scores.append(h5)
        hit10_scores.append(h10)

        first_relevant_rank = None

        for rank, result in enumerate(results, start=1):
            if result["chunk_id"] in relevant_ids:
                first_relevant_rank = rank
                break

        print(f"  First relevant rank: {first_relevant_rank}")
        print(f"  Recall@5:  {r5:.3f}")
        print(f"  Recall@10: {r10:.3f}")
        print(f"  MRR:       {mrr:.3f}")
        print(f"  Hit@5:     {h5:.3f}")
        print(f"  Hit@10:    {h10:.3f}")

        print("  Retrieved chunks:")

        for rank, result in enumerate(results, start=1):
            marker = (
                " <-- RELEVANT"
                if result["chunk_id"] in relevant_ids
                else ""
            )

            print(
                f"    {rank}. "
                f"{result['chunk_id']} | "
                f"{result['section']}"
                f"{marker}"
            )

    mean_recall5 = sum(recall5_scores) / len(recall5_scores)
    mean_recall10 = sum(recall10_scores) / len(recall10_scores)
    mean_mrr = sum(mrr_scores) / len(mrr_scores)
    mean_hit5 = sum(hit5_scores) / len(hit5_scores)
    mean_hit10 = sum(hit10_scores) / len(hit10_scores)

    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Evaluated questions: {len(EVALUATION_SET)}")
    print(f"Recall@5:            {mean_recall5:.3f}")
    print(f"Recall@10:           {mean_recall10:.3f}")
    print(f"MRR:                 {mean_mrr:.3f}")
    print(f"Hit Rate@5:          {mean_hit5:.3f}")
    print(f"Hit Rate@10:         {mean_hit10:.3f}")

    print("=" * 70)


if __name__ == "__main__":
    main()