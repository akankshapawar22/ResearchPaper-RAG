from sentence_transformers import CrossEncoder

from hybrid_retriever import HybridRetriever


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self):
        print("Loading cross-encoder reranker...")
        self.model = CrossEncoder(MODEL_NAME, device="cpu")
        print("Reranker ready.\n")

    def rerank(
        self,
        question,
        candidates,
        top_k=5,
        diversify=True,
    ):
        """Reorder candidate chunks by relevance and optionally
        promote evidence from multiple papers."""

        if not question.strip() or not candidates:
            return []

        pairs = [
            (question, candidate["text"])
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs,
            batch_size=4,
            show_progress_bar=False,
        )

        reranked = []

        for candidate, score in zip(candidates, scores):
            item = candidate.copy()
            item["reranker_score"] = float(score)
            reranked.append(item)

        reranked.sort(
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

        if not diversify or top_k <= 1:
            return reranked[:top_k]

      

        selected = []
        selected_ids = set()
        selected_papers = set()

        for item in reranked:
            paper = item.get("filename") or item.get("title")

            if paper not in selected_papers:
                selected.append(item)
                selected_ids.add(item["chunk_id"])
                selected_papers.add(paper)

                if len(selected) >= top_k:
                    break

        # Second pass:
        # Fill remaining slots with highest-scoring results.
        if len(selected) < top_k:
            for item in reranked:
                if item["chunk_id"] in selected_ids:
                    continue

                selected.append(item)
                selected_ids.add(item["chunk_id"])

                if len(selected) >= top_k:
                    break

        return selected[:top_k]


def main():
    retriever = HybridRetriever()
    reranker = Reranker()

    print("Research Paper Search with Reranking")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Enter your question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Exiting search.")
            break

        candidates = retriever.retrieve(
            question,
            top_k=10,
            candidate_k=15,
        )

        results = reranker.rerank(
            question,
            candidates,
            top_k=5,
            diversify=True,
        )

        if not results:
            print("No matching chunks found.\n")
            continue

        print(f"\nTop {len(results)} reranked results:\n")

        for rank, result in enumerate(results, start=1):
            print(
                f"--- Result {rank} | "
                f"Reranker score: "
                f"{result['reranker_score']:.4f} ---"
            )

            print(f"Paper: {result['title']}")
            print(f"Page: {result['page']}")
            print(f"Section: {result['section']}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Text: {result['text'][:1000]}")
            print()


if __name__ == "__main__":
    main()