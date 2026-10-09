import statistics
import time

from rag_pipeline import RAGPipeline


EVALUATION_QUESTIONS = [
    "What dataset was used?",
    "What is the main contribution of the Pre-Trained Image Processing Transformer?",
    "What is the role of spectral bands in satellite image classification?",
    "What are the key results of the flood detection study?",
    "What are the limitations of the flood detection study?",
]


def measure_question(pipeline, question):
    print("\n" + "-" * 70)
    print(f"Question: {question}")
    print("-" * 70)

    # ---------------------------------------------------------
    # 1. Complete end-to-end latency
    # ---------------------------------------------------------

    start_total = time.perf_counter()

    result = pipeline.answer_question_with_sources(
        question,
        filename=None,
        candidate_k=5,
    )

    end_total = time.perf_counter()

    total_time = end_total - start_total

    # ---------------------------------------------------------
    # 2. Basic result information
    # ---------------------------------------------------------

    answer = result.get("answer", "")
    sources = result.get("sources", [])

    print(f"Sources returned: {len(sources)}")
    print(f"Answer length: {len(answer)} characters")

    print("\nEND-TO-END LATENCY:")
    print(f"  {total_time:.3f} seconds")

    return total_time


def main():
    print("=" * 70)
    print("ResearchPaper-RAG Latency Evaluation")
    print("=" * 70)

    print("\nLoading RAG pipeline...")
    pipeline = RAGPipeline()

    print("\nEvaluation questions:", len(EVALUATION_QUESTIONS))

    # ---------------------------------------------------------
    # Warm-up request
    # ---------------------------------------------------------

    print("\nRunning warm-up request...")
    print(
        "The first request may include one-time initialization "
        "effects such as model loading or connection setup."
    )

    try:
        pipeline.answer_question_with_sources(
            "What dataset was used?",
            filename=None,
            candidate_k=5,
        )

        print("Warm-up complete.")

    except Exception as exc:
        print("\nWarm-up failed:")
        print(str(exc))
        return

    # ---------------------------------------------------------
    # Timed evaluation
    # ---------------------------------------------------------

    latencies = []

    for question in EVALUATION_QUESTIONS:

        try:
            latency = measure_question(
                pipeline,
                question,
            )

            latencies.append(latency)

        except Exception as exc:
            print("\nERROR while evaluating question:")
            print(str(exc))

    if not latencies:
        print("\nNo successful latency measurements.")
        return

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    average_latency = statistics.mean(latencies)
    median_latency = statistics.median(latencies)
    minimum_latency = min(latencies)
    maximum_latency = max(latencies)

    print("\n" + "=" * 70)
    print("LATENCY EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Successful questions: {len(latencies)}")
    print(f"Average latency:      {average_latency:.3f} seconds")
    print(f"Median latency:       {median_latency:.3f} seconds")
    print(f"Minimum latency:      {minimum_latency:.3f} seconds")
    print(f"Maximum latency:      {maximum_latency:.3f} seconds")

    print("=" * 70)


if __name__ == "__main__":
    main()