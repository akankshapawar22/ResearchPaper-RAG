import json
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI



load_dotenv()



PROJECT_ROOT = Path(__file__).resolve().parent.parent

INDEX_PATH = PROJECT_ROOT / "indexes" / "dense.faiss"
METADATA_PATH = PROJECT_ROOT / "indexes" / "chunks.jsonl"



MODEL_NAME = os.getenv(
    "AZURE_OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-large"
)

API_KEY = os.getenv(
    "AZURE_OPENAI_EMBEDDING_API_KEY"
)

ENDPOINT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_ENDPOINT"
)

API_VERSION = "2024-12-01-preview"



class DenseRetriever:

    def __init__(self):


        if not INDEX_PATH.exists() or not METADATA_PATH.exists():
            raise FileNotFoundError(
                "Dense index or metadata not found. "
                "Run python src\\embeddings.py first."
            )


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


        print("Loading FAISS index...")

        self.index = faiss.read_index(
            str(INDEX_PATH)
        )


        print("Loading chunk metadata...")

        with METADATA_PATH.open(
            "r",
            encoding="utf-8"
        ) as file:

            self.chunks = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        # -----------------------------------------------------
        # Validate index and metadata
        # -----------------------------------------------------

        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                f"Index contains {self.index.ntotal} vectors, "
                f"but metadata contains {len(self.chunks)} chunks."
            )


        if self.index.d != 3072:
            raise ValueError(
                f"Expected a 3072-dimensional FAISS index "
                f"for text-embedding-3-large, but found "
                f"{self.index.d} dimensions."
            )


        print("Connecting to Azure OpenAI...")

        self.client = AzureOpenAI(
            api_key=API_KEY,
            azure_endpoint=ENDPOINT,
            api_version=API_VERSION,
        )

        self.model_name = MODEL_NAME


        print(
            f"Embedding model: {self.model_name}"
        )

        print(
            f"Embedding dimensions: {self.index.d}"
        )

        print(
            f"Retriever ready: "
            f"{len(self.chunks)} chunks loaded.\n"
        )


    def embed_query(self, question):
        """
        Generate a query embedding using the same
        Azure OpenAI embedding model used to create
        the FAISS index.
        """

        response = self.client.embeddings.create(
            model=self.model_name,
            input=question,
        )

        embedding = np.asarray(
            response.data[0].embedding,
            dtype=np.float32
        )


        if embedding.shape[0] != self.index.d:
            raise ValueError(
                f"Query embedding has dimension "
                f"{embedding.shape[0]}, but FAISS index "
                f"expects {self.index.d}."
            )
# Normalize query embedding to match the normalized
# document vectors. Inner product then corresponds
# to cosine similarity.

        embedding = embedding.reshape(1, -1)

        faiss.normalize_L2(embedding)

        return embedding


    def retrieve(
        self,
        question,
        top_k=5,
        filename=None
    ):
        """
        Return the top matching chunks for a question.

        Parameters
        ----------
        question : str
            User's research question.

        top_k : int
            Number of results to return.

        filename : str or None
            If provided, restrict retrieval to the selected
            research paper.

            None -> search across all papers
            filename -> search only that paper
        """


        if not question or not question.strip():
            return []

        # -----------------------------------------------------
        # Validate index
        # -----------------------------------------------------

        if self.index.ntotal == 0:
            return []


        query_embedding = self.embed_query(
            question
        )


        if filename:

            search_k = self.index.ntotal

        else:

            search_k = min(
                top_k,
                self.index.ntotal
            )

        # -----------------------------------------------------
        # Search FAISS
        # -----------------------------------------------------

        scores, indices = self.index.search(
            query_embedding,
            search_k
        )


        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            if idx < 0:
                continue

            chunk = self.chunks[
                int(idx)
            ]


            if filename and chunk["filename"] != filename:
                continue

            results.append(
                {
                    "score": float(score),
                    "chunk_id": chunk["chunk_id"],
                    "title": chunk["title"],
                    "authors": chunk["authors"],
                    "filename": chunk["filename"],
                    "page": chunk["page"],
                    "section": chunk["section"],
                    "text": chunk["text"],
                }
            )

            # -------------------------------------------------
            # Stop once enough results have been collected
            # -------------------------------------------------

            if len(results) >= top_k:
                break

        return results



def main():

    retriever = DenseRetriever()

    print(
        "Research Paper Dense Search"
    )

    print(
        "Type 'exit' to quit.\n"
    )

    while True:

        question = input(
            "Enter your question: "
        ).strip()


        if question.lower() in {
            "exit",
            "quit"
        }:

            print(
                "Exiting search."
            )

            break


        results = retriever.retrieve(
            question,
            top_k=5
        )


        if not results:

            print(
                "No matching chunks found.\n"
            )

            continue


        print(
            f"\nTop {len(results)} "
            f"matching chunks:\n"
        )

        for rank, result in enumerate(
            results,
            start=1
        ):

            print(
                f"--- Result {rank} | "
                f"Score: {result['score']:.4f} ---"
            )

            print(
                f"Paper: {result['title']}"
            )

            print(
                f"Authors: "
                f"{result['authors'] or 'Not available'}"
            )

            print(
                f"Page: {result['page']}"
            )

            print(
                f"Section: {result['section']}"
            )

            print(
                f"Chunk ID: {result['chunk_id']}"
            )

            print(
                f"Text: {result['text'][:1000]}"
            )

            print()



if __name__ == "__main__":
    main()

