import re
from pathlib import Path

from rag_pipeline import RAGPipeline


EVALUATION_SET = [
    {
        "question": "What dataset was used?",
        "expected_facts": [
            "MediaEval 2019 Satellite Task 4",
            "City-centered satellite sequences",
            "335 events",
            "267 training",
            "68 test",
        ],
        "relevant_source_keywords": [
            "2296_AnastasiaMoumtzidou_etal2020"
        ],
    },
    {
        "question": "How many events are included in the dataset?",
        "expected_facts": [
            "335 events",
            "267 training",
            "68 test",
        ],
        "relevant_source_keywords": [
            "2296_AnastasiaMoumtzidou_etal2020"
        ],
    },
    {
        "question": "What image bands were used?",
        "expected_facts": [
            "B03",
            "B04",
            "B08",
            "B11",
        ],
        "relevant_source_keywords": [
            "2296_AnastasiaMoumtzidou_etal2020"
        ],
    },
    {
        "question": "What is the main contribution of the Pre-Trained Image Processing Transformer?",
        "expected_facts": [
            "pre-trained model",
            "image processing transformer",
            "multi-heads",
            "multi-tails",
            "fine-tuning",
        ],
        "relevant_source_keywords": [
            "Chen_Pre-Trained_Image_Processing_Transformer_CVPR_2021"
        ],
    },
    {
        "question": "What is the role of spectral bands in satellite image classification?",
        "expected_facts": [
            "RGB",
            "NIR",
            "13 bands",
            "improved performance",
        ],
        "relevant_source_keywords": [
            "InfluenceofSpectralBandsinSatelliteImageClassification"
        ],
    },
]


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def check_expected_facts(answer, expected_facts):
    normalized_answer = normalize_text(answer)

    found = []
    missing = []

    for fact in expected_facts:
        if normalize_text(fact) in normalized_answer:
            found.append(fact)
        else:
            missing.append(fact)

    if expected_facts:
        score = len(found) / len(expected_facts)
    else:
        score = 0.0

    return score, found, missing


def extract_citations(answer):
    return re.findall(r"\[S\d+\]", answer)


def check_citation_format(answer):
    citations = extract_citations(answer)

    if not citations:
        return False, []

    return True, citations


def check_citation_sources(answer, sources):
    citations = extract_citations(answer)

    if not citations:
        return False, []

    valid_citations = []

    for citation in citations:
        match = re.match(r"\[S(\d+)\]", citation)

        if not match:
            continue

        source_number = int(match.group(1))

        if 1 <= source_number <= len(sources):
            valid_citations.append(citation)

    return len(valid_citations) == len(citations), valid_citations


def main():
    print("=" * 70)
    print("ResearchPaper-RAG Answer-Level Evaluation")
    print("=" * 70)

    print("\nLoading RAG pipeline...")
    pipeline = RAGPipeline()

    print("\nEvaluation questions:", len(EVALUATION_SET))

    correctness_scores = []
    relevance_scores = []
    grounded_scores = []
    citation_scores = []

    for index, item in enumerate(EVALUATION_SET, start=1):

        question = item["question"]

        print("\n" + "-" * 70)
        print(f"Question {index}: {question}")
        print("-" * 70)

        try:
            result = pipeline.answer_question_with_sources(
                question,
                filename=None,
                candidate_k=5,
            )

            answer = result.get("answer", "")
            sources = result.get("sources", [])

            print("\nANSWER:")
            print(answer)

            print("\nSOURCES:")
            print(f"Sources returned: {len(sources)}")

            for source_index, source in enumerate(sources, start=1):
                print(
                    f"  [S{source_index}] "
                    f"{source.get('title', '')} | "
                    f"page {source.get('page', '')} | "
                    f"{source.get('section', '')}"
                )

            # ---------------------------------------------------------
            # 1. Correctness
            # ---------------------------------------------------------

            correctness, found, missing = check_expected_facts(
                answer,
                item["expected_facts"],
            )

            correctness_scores.append(correctness)

            print("\nCORRECTNESS:")
            print(f"  Score: {correctness:.3f}")

            if found:
                print("  Expected facts found:")
                for fact in found:
                    print(f"    - {fact}")

            if missing:
                print("  Expected facts missing:")
                for fact in missing:
                    print(f"    - {fact}")

            # ---------------------------------------------------------
            # 2. Relevance
            # ---------------------------------------------------------

            question_words = set(
                normalize_text(question).split()
            )

            answer_words = set(
                normalize_text(answer).split()
            )

            if question_words:
                overlap = len(
                    question_words.intersection(answer_words)
                ) / len(question_words)
            else:
                overlap = 0.0

            relevance = min(overlap * 1.5, 1.0)

            relevance_scores.append(relevance)

            print("\nRELEVANCE:")
            print(f"  Score: {relevance:.3f}")

            # ---------------------------------------------------------
            # 3. Groundedness
            # ---------------------------------------------------------

            combined_source_text = " ".join(
                source.get("text", "")
                for source in sources
            )

            normalized_source_text = normalize_text(
                combined_source_text
            )

            answer_sentences = [
                sentence.strip()
                for sentence in re.split(
                    r"[.!?]+",
                    answer
                )
                if sentence.strip()
            ]

            grounded_sentence_count = 0

            for sentence in answer_sentences:

                sentence_words = set(
                    normalize_text(sentence).split()
                )

                if not sentence_words:
                    continue

                overlap = len(
                    sentence_words.intersection(
                        set(normalized_source_text.split())
                    )
                ) / len(sentence_words)

                if overlap >= 0.25:
                    grounded_sentence_count += 1

            if answer_sentences:
                groundedness = (
                    grounded_sentence_count
                    / len(answer_sentences)
                )
            else:
                groundedness = 0.0

            grounded_scores.append(groundedness)

            print("\nGROUNDEDNESS:")
            print(f"  Score: {groundedness:.3f}")

            # ---------------------------------------------------------
            # 4. Citation correctness
            # ---------------------------------------------------------

            citation_format_ok, citations = check_citation_format(
                answer
            )

            citation_valid, valid_citations = check_citation_sources(
                answer,
                sources,
            )

            if citation_format_ok and citation_valid:
                citation_score = 1.0
            elif citation_format_ok:
                citation_score = 0.5
            else:
                citation_score = 0.0

            citation_scores.append(citation_score)

            print("\nCITATION CORRECTNESS:")
            print(f"  Score: {citation_score:.3f}")
            print(
                f"  Citations found: "
                f"{', '.join(citations) if citations else 'None'}"
            )

            if not citation_format_ok:
                print("  WARNING: No [S#] citations found.")

            if citation_format_ok and not citation_valid:
                print(
                    "  WARNING: One or more citations "
                    "refer to invalid source numbers."
                )

        except Exception as exc:
            print("\nERROR while evaluating question:")
            print(str(exc))

            correctness_scores.append(0.0)
            relevance_scores.append(0.0)
            grounded_scores.append(0.0)
            citation_scores.append(0.0)

    # -----------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------

    def mean(values):
        if not values:
            return 0.0

        return sum(values) / len(values)

    mean_correctness = mean(correctness_scores)
    mean_relevance = mean(relevance_scores)
    mean_groundedness = mean(grounded_scores)
    mean_citation = mean(citation_scores)

    print("\n" + "=" * 70)
    print("ANSWER-LEVEL EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Evaluated questions: {len(EVALUATION_SET)}")
    print(f"Correctness:         {mean_correctness:.3f}")
    print(f"Relevance:           {mean_relevance:.3f}")
    print(f"Groundedness:        {mean_groundedness:.3f}")
    print(f"Citation correctness:{mean_citation:.3f}")

    print("=" * 70)


if __name__ == "__main__":
    main()