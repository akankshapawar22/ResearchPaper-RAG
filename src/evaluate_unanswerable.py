import re

from rag_pipeline import RAGPipeline


EVALUATION_SET = [
    {
        "question": "What is the exact salary of the authors of the flood detection paper?",
    },
    {
        "question": "What GPU model was used to train the Image Processing Transformer?",
    },
    {
        "question": "What is the street address of the institution where the spectral bands study was conducted?",
    },
    {
        "question": "What was the exact publication acceptance date of the flood detection paper?",
    },
    {
        "question": "What is the authors' favorite programming language?",
    },
]


REFUSAL_PATTERNS = [
    r"not enough evidence",
    r"insufficient evidence",
    r"not provided",
    r"not mentioned",
    r"not specified",
    r"cannot determine",
    r"can't determine",
    r"cannot be determined",
    r"can't be determined",
    r"unable to determine",
    r"unable to answer",
    r"cannot answer",
    r"can't answer",
    r"not available",
    r"not contained",
    r"not included",
    r"not stated",
    r"the supplied evidence does not",
    r"the provided evidence does not",
    r"the papers do not",
    r"the paper does not",
    r"the documents do not",
    r"no information",
]


def is_refusal(answer):
    answer_lower = answer.lower()

    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, answer_lower):
            return True

    return False


def extract_citations(answer):
    return re.findall(r"\[S\d+\]", answer)


def main():
    print("=" * 70)
    print("ResearchPaper-RAG Unanswerable / Hallucination Evaluation")
    print("=" * 70)

    print("\nLoading RAG pipeline...")
    pipeline = RAGPipeline()

    refusal_scores = []

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

            refused = is_refusal(answer)
            citations = extract_citations(answer)

            if refused:
                score = 1.0
                print("\nHALLUCINATION CHECK: PASS")
                print(
                    "  The system explicitly indicates that "
                    "the requested information is unavailable."
                )
            else:
                score = 0.0
                print("\nHALLUCINATION CHECK: REVIEW")
                print(
                    "  The system did not clearly indicate that "
                    "the requested information is unavailable."
                )

            print(
                f"  Citations in answer: "
                f"{', '.join(citations) if citations else 'None'}"
            )

            refusal_scores.append(score)

        except Exception as exc:
            print("\nERROR while evaluating question:")
            print(str(exc))
            refusal_scores.append(0.0)

    mean_score = (
        sum(refusal_scores) / len(refusal_scores)
        if refusal_scores
        else 0.0
    )

    print("\n" + "=" * 70)
    print("UNANSWERABLE / HALLUCINATION EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Evaluated questions: {len(EVALUATION_SET)}")
    print(f"Safe refusal rate:   {mean_score:.3f}")
    print(
        f"Safe refusals:       "
        f"{sum(1 for score in refusal_scores if score == 1.0)}"
        f"/{len(refusal_scores)}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()