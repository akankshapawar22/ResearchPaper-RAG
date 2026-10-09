import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from dense_retriever import DenseRetriever


class HybridRetriever:
    def __init__(self):
        print("Loading hybrid retriever...")

        self.dense_retriever = DenseRetriever()

        self.index_dir = Path(__file__).resolve().parent.parent / "indexes"
        self.chunks_path = self.index_dir / "chunks.jsonl"

        self.chunks = []

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.chunks.append(json.loads(line))

        if len(self.chunks) != self.dense_retriever.index.ntotal:
            raise ValueError(
                f"Chunk/index mismatch: {len(self.chunks)} chunks "
                f"but FAISS contains {self.dense_retriever.index.ntotal} vectors."
            )

        self.tokenized_corpus = [
            self._tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

        self.rrf_constant = 60

        print(f"Loaded {len(self.chunks)} chunks.")
        print("BM25 index ready.")
        print("Hybrid retriever ready.\n")

    def _tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def _detect_table_reference(self, question):
        question_lower = question.lower()

        table_terms = [
            "table",
            "tab.",
            "comparison",
            "compare",
            "best result",
            "highest accuracy",
            "highest f1",
            "highest f-score",
            "lowest error",
            "performance",
        ]

        return any(term in question_lower for term in table_terms)

    def _table_match_score(self, chunk):
        text = chunk.get("text", "").lower()
        section = chunk.get("section", "").lower()

        score = 0

        if "table" in text:
            score += 1

        if re.search(r"\btable\s+\d+", text):
            score += 2

        if "results" in section:
            score += 1

        if "performance" in section:
            score += 1

        if "comparison" in section:
            score += 1

        return score

    def _detect_question_type(self, question):
        question_lower = question.lower()

        question_types = []

        dataset_terms = [
            "dataset",
            "data set",
            "data used",
            "dataset used",
            "training data",
            "test data",
            "training set",
            "test set",
        ]

        methodology_terms = [
            "method",
            "methodology",
            "approach",
            "architecture",
            "model",
            "framework",
            "how does",
            "how do they",
        ]

        results_terms = [
            "result",
            "results",
            "finding",
            "findings",
            "performance",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "f-score",
            "score",
        ]

        limitation_terms = [
            "limitation",
            "limitations",
            "drawback",
            "drawbacks",
            "future work",
            "future research",
        ]

        conclusion_terms = [
            "conclusion",
            "conclude",
            "main contribution",
            "contribution",
        ]

        if any(term in question_lower for term in dataset_terms):
            question_types.append("dataset")

        if any(term in question_lower for term in methodology_terms):
            question_types.append("methodology")

        if any(term in question_lower for term in results_terms):
            question_types.append("results")

        if any(term in question_lower for term in limitation_terms):
            question_types.append("limitations")

        if any(term in question_lower for term in conclusion_terms):
            question_types.append("conclusion")

        return question_types

    def _section_match_score(self, chunk, question_types):
        section = chunk.get("section", "").lower()
        text = chunk.get("text", "").lower()

        score = 0.0

        if "dataset" in question_types:
            dataset_sections = [
                "dataset",
                "data set",
                "data description",
                "dataset description",
                "data set description",
                "experiments",
                "methodology",
            ]

            if any(name in section for name in dataset_sections):
                score += 1.0

            dataset_text_terms = [
                "dataset consists",
                "data set consists",
                "dataset used",
                "data set used",
                "training set",
                "test set",
                "training data",
                "test data",
            ]

            if any(term in text[:1500] for term in dataset_text_terms):
                score += 0.5

        if "methodology" in question_types:
            methodology_sections = [
                "method",
                "methodology",
                "approach",
                "architecture",
                "model",
                "framework",
                "experimental setup",
            ]

            if any(name in section for name in methodology_sections):
                score += 1.0

        if "results" in question_types:
            results_sections = [
                "result",
                "results",
                "experimental results",
                "evaluation",
                "experiments",
                "performance",
            ]

            if any(name in section for name in results_sections):
                score += 1.0

        if "limitations" in question_types:
            limitation_sections = [
                "limitation",
                "limitations",
                "future work",
                "future research",
                "discussion",
                "conclusion",
            ]

            if any(name in section for name in limitation_sections):
                score += 1.0

        if "conclusion" in question_types:
            conclusion_sections = [
                "conclusion",
                "conclusions",
                "discussion",
                "future work",
            ]

            if any(name in section for name in conclusion_sections):
                score += 1.0

        return score

    def _section_priority(self, section, question_type):
       

        section = section.lower().strip()

        preferences = {
            "dataset": [
                "data set description",
                "dataset description",
                "data description",
                "dataset",
                "data set",
                "data",
                "experiments",
                "methodology",
            ],

            "methodology": [
                "methodology",
                "method",
                "approach",
                "architecture",
                "model",
                "framework",
                "experimental setup",
                "experiments",
            ],

            "results": [
                "results",
                "experimental results",
                "results and discussion",
                "evaluation",
                "performance",
                "experiments",
            ],

            "limitations": [
                "limitations",
                "limitation",
                "future work",
                "future research",
                "discussion",
                "conclusion",
            ],

            "conclusion": [
                "conclusion",
                "conclusions",
                "future work",
                "discussion",
            ],
        }

        phrases = preferences.get(question_type, [])

        for position, phrase in enumerate(phrases):
            if phrase in section:
                return len(phrases) - position

        return 0

    def _get_section_candidates(
        self,
        question_types,
        filename=None,
        max_candidates=5,
    ):
        

        if not question_types:
            return []

        candidates = []

        for chunk_index, chunk in enumerate(self.chunks):

            if filename is not None:
                if chunk.get("filename") != filename:
                    continue

            section = chunk.get("section", "")
            text = chunk.get("text", "").lower()

            best_priority = 0
            matched_type = None

            for question_type in question_types:
                priority = self._section_priority(
                    section,
                    question_type
                )

                if priority > best_priority:
                    best_priority = priority
                    matched_type = question_type

            if best_priority == 0:
                continue

            text_bonus = 0.0

            if matched_type == "dataset":
                dataset_terms = [
                    "dataset consists",
                    "data set consists",
                    "dataset used",
                    "data set used",
                    "training set",
                    "test set",
                    "training data",
                    "test data",
                    "mediaeval",
                ]

                if any(term in text for term in dataset_terms):
                    text_bonus = 1.0

            elif matched_type == "results":
                result_terms = [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "f-score",
                    "results",
                    "performance",
                ]

                if any(term in text for term in result_terms):
                    text_bonus = 0.5

            elif matched_type == "methodology":
                methodology_terms = [
                    "method",
                    "architecture",
                    "model",
                    "approach",
                    "training",
                ]

                if any(term in text for term in methodology_terms):
                    text_bonus = 0.5

            candidates.append(
                {
                    "priority": best_priority + text_bonus,
                    "chunk_index": chunk_index,
                    "chunk": chunk,
                }
            )

        candidates.sort(
            key=lambda item: (
                -item["priority"],
                item["chunk_index"],
            )
        )

        return [
            item["chunk"]
            for item in candidates[:max_candidates]
        ]

    def retrieve(
        self,
        question,
        top_k=10,
        filename=None,
    ):
        if not question.strip():
            return []

        question_types = self._detect_question_type(question)
        table_reference = self._detect_table_reference(question)


        if filename is None:
            dense_candidates = self.dense_retriever.retrieve(
                question,
                top_k=top_k,
                filename=None,
            )
        else:
            dense_candidates = self.dense_retriever.retrieve(
                question,
                top_k=len(self.chunks),
                filename=filename,
            )

            dense_candidates = dense_candidates[:top_k]


        query_tokens = self._tokenize(question)

        bm25_scores = self.bm25.get_scores(query_tokens)

        ranked_bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )

        bm25_candidates = []

        for index in ranked_bm25_indices:

            chunk = self.chunks[index]

            if filename is not None:
                if chunk.get("filename") != filename:
                    continue

            bm25_candidates.append(
                {
                    **chunk,
                    "bm25_score": float(bm25_scores[index]),
                }
            )

            if len(bm25_candidates) >= top_k:
                break

        section_candidates = self._get_section_candidates(
            question_types=question_types,
            filename=filename,
            max_candidates=5,
        )


        fused = {}

        for rank, item in enumerate(dense_candidates, start=1):

            chunk_id = item["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    **item,
                    "dense_score": item.get("score"),
                    "bm25_score": None,
                    "rrf_score": 0.0,
                    "section_candidate": False,
                }

            fused[chunk_id]["rrf_score"] += (
                1.0 / (self.rrf_constant + rank)
            )

        # BM25 results
        for rank, item in enumerate(bm25_candidates, start=1):

            chunk_id = item["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    **item,
                    "dense_score": None,
                    "bm25_score": item.get("bm25_score"),
                    "rrf_score": 0.0,
                    "section_candidate": False,
                }

            else:
                fused[chunk_id]["bm25_score"] = item.get(
                    "bm25_score"
                )

            fused[chunk_id]["rrf_score"] += (
                1.0 / (self.rrf_constant + rank)
            )


        section_candidate_ids = []

        for chunk in section_candidates:

            chunk_id = chunk["chunk_id"]

            section_candidate_ids.append(chunk_id)

            if chunk_id not in fused:
                fused[chunk_id] = {
                    **chunk,
                    "dense_score": None,
                    "bm25_score": None,
                    "rrf_score": 0.0,
                    "section_candidate": True,
                }
            else:
                fused[chunk_id]["section_candidate"] = True


        for item in fused.values():

            item["table_match"] = 0
            item["section_match"] = 0.0

            if table_reference:
                item["table_match"] = self._table_match_score(item)

                item["rrf_score"] += (
                    0.01 * item["table_match"]
                )

            if question_types:
                item["section_match"] = self._section_match_score(
                    item,
                    question_types,
                )

                item["rrf_score"] += (
                    0.005 * item["section_match"]
                )

        # ---------------------------------------------------------
        # 7. Sort normally by RRF score
        # ---------------------------------------------------------

        results = list(fused.values())

        results.sort(
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

        # ---------------------------------------------------------
        # 8. Guarantee at least one directly relevant section
        #
        # For example:
        #
        # "What dataset was used?"
        #
        # must include a "Data set description" chunk if one
        # exists, even if dense/BM25 ranked it poorly.
        # ---------------------------------------------------------

        if question_types and section_candidate_ids:

            mandatory_count = min(
                len(question_types),
                len(section_candidate_ids),
            )

            mandatory_ids = set(
                section_candidate_ids[:mandatory_count]
            )

            mandatory = [
                item
                for item in results
                if item["chunk_id"] in mandatory_ids
            ]

            remaining = [
                item
                for item in results
                if item["chunk_id"] not in mandatory_ids
            ]

            results = mandatory + remaining

        # ---------------------------------------------------------
        # 9. Return final candidate pool
        # ---------------------------------------------------------

        results = results[:top_k]

        return results


if __name__ == "__main__":

    retriever = HybridRetriever()

    question = input(
        "Enter your question: "
    ).strip()

    results = retriever.retrieve(
        question,
        top_k=10,
    )

    print("\nTop retrieved chunks:\n")

    for i, result in enumerate(results, start=1):

        print(
            f"{i}. "
            f"{result.get('title', '')} | "
            f"page {result.get('page', '')} | "
            f"section: {result.get('section', '')}"
        )

        print(
            f"   RRF: {result.get('rrf_score', 0):.6f} | "
            f"Section match: {result.get('section_match', 0):.2f} | "
            f"Section candidate: "
            f"{result.get('section_candidate', False)}"
        )

        print(
            f"   {result.get('text', '')[:400]}"
        )

        print()