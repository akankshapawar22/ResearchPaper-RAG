from hybrid_retriever import HybridRetriever
from reranker import Reranker
from llm import generate_answer


class RAGPipeline:
    def __init__(self):
        print("Initializing RAG pipeline...")

        self.retriever = HybridRetriever()
        self.reranker = Reranker()

        print("RAG pipeline ready.\n")

    def _get_papers(self):
        """
        Return unique paper filenames from the indexed chunks.
        """
        papers = []

        for chunk in self.retriever.chunks:
            filename = chunk.get("filename")

            if filename and filename not in papers:
                papers.append(filename)

        return papers

    def _is_dataset_question(self, question):
        """
        Detect questions that explicitly ask about datasets/data.
        """

        question_lower = question.lower()

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

        return any(
            term in question_lower
            for term in dataset_terms
        )

    def _is_dataset_section(self, result):
        """
        Identify evidence that comes directly from a dataset-related
        section.
        """

        section = (
            result.get("section", "")
            .lower()
            .strip()
        )

        dataset_sections = [
            "data set description",
            "dataset description",
            "data description",
            "dataset",
            "data set",
        ]

        return any(
            section_name in section
            for section_name in dataset_sections
        )

    def _select_dataset_evidence(
        self,
        question,
        candidates,
        reranked,
        sources_per_paper=2,
    ):
        """
        For dataset questions, make sure a directly relevant
        dataset-description chunk is not discarded by the
        CrossEncoder reranker.

        The reranker is still used for relevance ranking, but
        explicit dataset-section evidence receives priority.
        """

        if not self._is_dataset_question(question):
            return reranked[:sources_per_paper]

        dataset_candidates = [
            item
            for item in candidates
            if self._is_dataset_section(item)
        ]

        if not dataset_candidates:
            return reranked[:sources_per_paper]

        dataset_candidates.sort(
            key=lambda item: (
                1
                if "data set description"
                in item.get("section", "").lower()
                else 0,
                1
                if "dataset description"
                in item.get("section", "").lower()
                else 0,
                item.get("section_match", 0),
            ),
            reverse=True,
        )

        selected = []

        selected.append(dataset_candidates[0])

        selected_ids = {
            dataset_candidates[0]["chunk_id"]
        }

        for item in reranked:

            if item["chunk_id"] in selected_ids:
                continue

            selected.append(item)
            selected_ids.add(item["chunk_id"])

            if len(selected) >= sources_per_paper:
                break

        return selected[:sources_per_paper]

    def _retrieve_all_papers(
        self,
        question,
        candidate_k=10,
        sources_per_paper=2,
        final_top_k=6,
    ):
        """
        Retrieve evidence separately from each paper.

        This prevents one paper from dominating broad questions
        such as "What are the key results?"

        For dataset questions, directly relevant dataset-section
        evidence is preserved even if the CrossEncoder ranks
        another passage slightly higher.
        """

        all_results = []

        papers = self._get_papers()

        for filename in papers:

            candidates = self.retriever.retrieve(
                question,
                top_k=max(candidate_k, 15),
                filename=filename,
            )

            if not candidates:
                continue


            reranked = self.reranker.rerank(
                question,
                candidates,
                top_k=max(candidate_k, 15),
                diversify=False,
            )

            if not reranked:
                continue

            selected_for_paper = self._select_dataset_evidence(
                question=question,
                candidates=candidates,
                reranked=reranked,
                sources_per_paper=sources_per_paper,
            )

            all_results.extend(selected_for_paper)

        if not all_results:
            return []

        all_results.sort(
            key=lambda item: item.get(
                "reranker_score",
                float("-inf")
            ),
            reverse=True,
        )


        selected = []
        selected_ids = set()
        selected_papers = set()

        for item in all_results:

            paper = (
                item.get("filename")
                or item.get("title")
            )

            if paper in selected_papers:
                continue

            selected.append(item)
            selected_ids.add(item["chunk_id"])
            selected_papers.add(paper)

            if len(selected) >= final_top_k:
                break


        if len(selected) < final_top_k:

            for item in all_results:

                if item["chunk_id"] in selected_ids:
                    continue

                selected.append(item)
                selected_ids.add(item["chunk_id"])

                if len(selected) >= final_top_k:
                    break

        return selected[:final_top_k]

    def answer_question_with_sources(
        self,
        question,
        candidate_k=10,
        top_k=6,
        filename=None,
    ):
        """
        Retrieve evidence, generate an answer, and return
        both the answer and the exact sources used to generate it.

        This keeps [S1], [S2], etc. synchronized with the
        Supporting Evidence shown in the Streamlit interface.
        """

        if not question or not question.strip():
            return {
                "answer": "Please enter a question.",
                "sources": [],
            }


        if filename:

            candidates = self.retriever.retrieve(
                question,
                top_k=max(candidate_k, 15),
                filename=filename,
            )

            if not candidates:
                return {
                    "answer": (
                        "I could not find relevant evidence "
                        "in the selected paper."
                    ),
                    "sources": [],
                }

            reranked = self.reranker.rerank(
                question,
                candidates,
                top_k=max(candidate_k, 15),
                diversify=False,
            )

            if self._is_dataset_question(question):

                results = self._select_dataset_evidence(
                    question=question,
                    candidates=candidates,
                    reranked=reranked,
                    sources_per_paper=top_k,
                )

            else:

                results = reranked[:top_k]

        else:

            results = self._retrieve_all_papers(
                question=question,
                candidate_k=candidate_k,
                sources_per_paper=2,
                final_top_k=top_k,
            )

            if not results:
                return {
                    "answer": (
                        "I could not find relevant evidence "
                        "in the available research papers."
                    ),
                    "sources": [],
                }

        context_parts = []

        for index, result in enumerate(
            results,
            start=1
        ):

            source_id = f"S{index}"

            title = result.get(
                "title",
                "Unknown paper"
            )

            authors = result.get(
                "authors"
            ) or "Authors not available"

            page = result.get(
                "page",
                "Unknown"
            )

            section = result.get(
                "section"
            ) or "Section not available"

            chunk_id = result.get(
                "chunk_id",
                "Unknown"
            )

            text = result.get(
                "text",
                ""
            )

            context_parts.append(
                f"""
[{source_id}]
Paper: {title}
Authors: {authors}
Page: {page}
Section: {section}
Chunk ID: {chunk_id}

Passage:
{text}
""".strip()
            )

        context = "\n\n".join(context_parts)


        table_instruction = ""

        if "table" in question.lower():

            table_instruction = """
TABLE QUESTION INSTRUCTION:

The user is asking about a table.

Prefer evidence that explicitly refers to the requested
table number or contains the table's values.

Preserve numerical values exactly as reported.

Do not invent missing table values.
"""


        if filename:

            scope_instruction = """
PAPER SCOPE:

The user selected one specific paper.

Answer only using evidence from that paper.
Do not introduce information from other papers.
"""

        else:

            scope_instruction = """
PAPER SCOPE:

The user selected "All Papers".

The evidence may come from multiple research papers.

Keep the findings of different papers clearly separated.

Do NOT combine numerical results from different papers
into a single performance number.

Do NOT imply that metrics from different tasks, datasets,
or experimental settings are directly comparable.

If the papers address different research problems,
explain that their results should not be directly compared.
"""


        dataset_instruction = ""

        if self._is_dataset_question(question):

            dataset_instruction = """
DATASET QUESTION INSTRUCTION:

The user is asking which dataset or data was used.

Prefer explicit dataset-description evidence over general
mentions of images, experiments, or methodology.

For each paper, identify the dataset as specifically as
the supplied evidence allows.

If the evidence gives a formal dataset/task name, report it.

If the evidence provides dataset statistics such as the number
of events, training samples, test samples, classes, images,
or bands, include the most relevant statistics.

Do not infer a dataset name merely from the type of imagery.

If a paper's supplied evidence does not identify its dataset,
say so explicitly rather than guessing.
"""


        results_style_instruction = ""

        question_lower = question.lower()

        if any(
            phrase in question_lower
            for phrase in [
                "key result",
                "key results",
                "results",
                "findings",
                "performance",
                "experimental result",
                "experimental results",
                "outcome",
                "outcomes",
            ]
        ):

            results_style_instruction = """
RESULTS FORMAT:

The question asks about research results.

Start the answer with the heading:

Key Results

Organize the answer using a numbered list.

When evidence comes from multiple papers:

1. Give each relevant paper its own numbered item.
2. Use a short descriptive name for the paper or research topic.
3. If a paper contains multiple important results,
   use bullet points inside that numbered item.
4. Preserve exact reported numerical values.
5. Include citations such as [S1], [S2] immediately
   after the claims they support.
6. Do not merge results from unrelated papers.
7. If the papers use different tasks, datasets, or metrics,
   end with a short statement explaining that the results
   are not directly comparable.
"""

        elif self._is_dataset_question(question):

            results_style_instruction = """
DATASET ANSWER FORMAT:

The question asks about datasets.

Start with a short direct answer.

When multiple papers are involved, use a numbered list
with one item per paper.

For each paper:

- Give the dataset/task name when explicitly available.
- Give important dataset statistics when available.
- Clearly state when the supplied evidence does not identify
  the dataset.
- Do not confuse a dataset with a preprocessing method,
  image modality, spectral band combination, or augmentation
  technique.

Cite every factual statement using [S1], [S2], etc.
"""

        else:

            results_style_instruction = """
ANSWER STYLE:

Give a clear, concise scientific answer.

Use paragraphs or bullet points when useful.

Cite factual claims using [S1], [S2], etc.

Do not add information that is not supported by the
provided evidence.
"""


        prompt = f"""
You are a scientific research assistant.

Answer the user's question using ONLY the supplied research
paper passages.

Do not use outside knowledge.

If the evidence is insufficient, explicitly say so.

Every factual claim derived from the passages must include
a source citation such as [S1], [S2], etc.

Do not invent numbers, datasets, methods, authors, results,
or conclusions.

When exact metrics are present, preserve them accurately.

{scope_instruction}

{dataset_instruction}

{table_instruction}

{results_style_instruction}

USER QUESTION:
{question}

SUPPLIED EVIDENCE:
{context}
"""

        # --------------------------------------------------
        # GENERATE ANSWER
        # --------------------------------------------------

        answer = generate_answer(prompt)

        return {
            "answer": answer,
            "sources": results,
        }

    def answer_question(
        self,
        question,
        candidate_k=10,
        top_k=5,
        filename=None,
    ):
        """
        Backward-compatible method for CLI usage.
        """

        result = self.answer_question_with_sources(
            question=question,
            candidate_k=candidate_k,
            top_k=top_k,
            filename=filename,
        )

        return result["answer"]


def main():

    pipeline = RAGPipeline()

    print("Research Paper RAG")
    print("Type 'exit' to quit.\n")

    while True:

        question = input(
            "Enter your question: "
        ).strip()

        if question.lower() in {
            "exit",
            "quit",
        }:
            print("Exiting.")
            break

        result = pipeline.answer_question_with_sources(
            question=question,
            candidate_k=10,
            top_k=6,
        )

        print("\nAnswer:")
        print(result["answer"])

        print("\nSupporting Evidence:")

        for index, source in enumerate(
            result["sources"],
            start=1,
        ):

            print(
                f"\n[S{index}] "
                f"{source.get('title', 'Unknown paper')}"
            )

            print(
                f"Page: {source.get('page', 'Unknown')}"
            )

            print(
                f"Section: "
                f"{source.get('section', 'Unknown')}"
            )

            print(
                f"Chunk ID: "
                f"{source.get('chunk_id', 'Unknown')}"
            )

            print(
                source.get("text", "")[:1000]
            )

        print()


if __name__ == "__main__":
    main()