import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
model = os.getenv("AZURE_OPENAI_MODEL")

if not api_key:
    raise ValueError("AZURE_OPENAI_API_KEY not found.")

if not endpoint:
    raise ValueError("AZURE_OPENAI_ENDPOINT not found.")

if not model:
    raise ValueError("AZURE_OPENAI_MODEL not found.")

base_url = endpoint.rstrip("/") + "/openai/v1/"

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

print("Connecting to Azure OpenAI...")
print(f"Model/deployment: {model}")

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: Azure connection successful."
            }
        ],
        
    )

    print("\nAzure response:")
    print(response.choices[0].message.content)

except Exception as e:
    print("\nAzure connection failed:")
    print(type(e).__name__)
    print(e)