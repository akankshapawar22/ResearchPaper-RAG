import json
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI


load_dotenv()



PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_DIR = PROJECT_ROOT / "indexes"



MODEL_NAME = os.getenv(
    "AZURE_OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-large"
)

API_KEY = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")

API_VERSION = "2024-12-01-preview"

BATCH_SIZE = 8



def load_chunks():
    """
    Load processed chunks and preserve their metadata.
    """

    chunks = []

    for file_path in sorted(PROCESSED_DIR.glob("*.jsonl")):

        with file_path.open("r", encoding="utf-8") as file:

            for line in file:

                if not line.strip():
                    continue

                chunk = json.loads(line)

                if chunk.get("text", "").strip():
                    chunks.append(chunk)

    if not chunks:
        raise ValueError(
            f"No text chunks found in {PROCESSED_DIR}. "
            "Run the PDF processor first."
        )

    return chunks



def create_client():
    """
    Create the Azure OpenAI client for embeddings.
    """

    if not API_KEY:
        raise ValueError(
            "AZURE_OPENAI_EMBEDDING_API_KEY not found. "
            "Check your .env file."
        )

    if not ENDPOINT:
        raise ValueError(
            "AZURE_OPENAI_EMBEDDING_ENDPOINT not found. "
            "Check your .env file."
        )

    return AzureOpenAI(
        api_key=API_KEY,
        azure_endpoint=ENDPOINT,
        api_version=API_VERSION,
    )


def generate_embeddings(client, texts):
    """
    Generate embeddings for all chunks in batches.
    """

    all_embeddings = []

    total_batches = (
        len(texts) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    for batch_number, start in enumerate(
        range(0, len(texts), BATCH_SIZE),
        start=1
    ):

        batch = texts[
            start:start + BATCH_SIZE
        ]

        print(
            f"Generating embeddings: "
            f"batch {batch_number}/{total_batches}"
        )

        response = client.embeddings.create(
            model=MODEL_NAME,
            input=batch
        )

        batch_embeddings = sorted(
            response.data,
            key=lambda item: item.index
        )

        vectors = [
            item.embedding
            for item in batch_embeddings
        ]

        all_embeddings.extend(vectors)

    embeddings = np.asarray(
        all_embeddings,
        dtype="float32"
    )

    faiss.normalize_L2(embeddings)

    return embeddings



def build_index():
    """
    Generate Azure embeddings and create
    the FAISS dense retrieval index.
    """

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    
    chunks = load_chunks()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        f"Loaded {len(chunks)} chunks."
    )

    print(
        f"Embedding model: {MODEL_NAME}"
    )

    print(
        f"Embedding API version: {API_VERSION}"
    )

    
    client = create_client()

    print(
        "\nGenerating Azure OpenAI embeddings..."
    )

    
    embeddings = generate_embeddings(
        client,
        texts
    )

    
    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)


    index_path = (
        INDEX_DIR / "dense.faiss"
    )

    metadata_path = (
        INDEX_DIR / "chunks.jsonl"
    )

    faiss.write_index(
        index,
        str(index_path)
    )

    with metadata_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        for chunk in chunks:

            file.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False
                ) + "\n"
            )


    print(
        "\nDense index created successfully."
    )

    print(
        f"Chunks indexed: {index.ntotal}"
    )

    print(
        f"Embedding dimensions: "
        f"{embeddings.shape[1]}"
    )

    print(
        f"FAISS index: {index_path}"
    )

    print(
        f"Chunk metadata: {metadata_path}"
    )



if __name__ == "__main__":
    build_index()