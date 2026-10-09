import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")

if not api_key:
    raise ValueError("AZURE_OPENAI_EMBEDDING_API_KEY not found.")

if not endpoint:
    raise ValueError("AZURE_OPENAI_EMBEDDING_ENDPOINT not found.")

if not model:
    raise ValueError("AZURE_OPENAI_EMBEDDING_MODEL not found.")

client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    api_version="2024-12-01-preview",
)

print("Connecting to Azure OpenAI Embeddings...")
print(f"Embedding deployment: {model}")

try:
    response = client.embeddings.create(
        model=model,
        input="Sentinel-2 satellite imagery is used for flood detection."
    )

    embedding = response.data[0].embedding

    print("\nAzure embedding connection successful.")
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")

except Exception as e:
    print("\nAzure embedding connection failed:")
    print(type(e).__name__)
    print(e)