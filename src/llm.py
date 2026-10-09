import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
model = os.getenv("AZURE_OPENAI_MODEL")

if not api_key:
    raise ValueError(
        "AZURE_OPENAI_API_KEY not found. "
        "Check your .env file."
    )

if not endpoint:
    raise ValueError(
        "AZURE_OPENAI_ENDPOINT not found. "
        "Check your .env file."
    )

if not model:
    raise ValueError(
        "AZURE_OPENAI_MODEL not found. "
        "Check your .env file."
    )


base_url = endpoint.rstrip("/") + "/openai/v1/"

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

print(f"Azure OpenAI model: {model}")



SYSTEM_INSTRUCTION = """
You are an evidence-based scientific research assistant.

Answer the user's question using only the research-paper
passages supplied in the prompt.

GENERAL RULES:

1. Answer the exact question asked.
2. Use only information supported by the retrieved passages.
3. Do not invent facts, results, or explanations.
4. Cite factual claims using the provided source IDs,
   such as [S1] or [S2].
5. If the retrieved passages do not contain the answer,
   clearly state that the available evidence is insufficient.
6. Use clear, concise academic language.
7. Do not include unrelated information or headings.

QUESTION-AWARE ANSWERING:

Adapt the answer format to the user's question.

- For questions about bands, datasets, models, optimizers,
  architectures, or parameters:
  Give the specific names, values, combinations, and roles
  supported by the passages. Use a table when it improves
  clarity.

- For methodology questions:
  Explain the method in logical steps, using only the
  details supported by the passages.

- For comparison questions:
  Compare the methods or results directly and cite the
  supporting passages.

- For results or performance questions:
  Report the exact metrics available in the passages.
  Do not infer metrics that are not reported.

- For limitations questions:
  Separate findings into:
  1. Explicit Limitations
  2. Future Work / Scope Boundaries
  3. Inferred Limitations

  Do not treat a future-work item or methodological
  choice as an explicit limitation.

  Clearly label any inference and explain its evidence.

  If no explicit limitations are found, state that.

- For definition or explanation questions:
  Explain the requested concept in the context of the
  paper, without adding unsupported claims.

EVIDENCE DISCIPLINE:

1. Do not apply a fixed answer template to every question.
2. Do not discuss limitations unless the user asks about
   limitations or they are directly relevant.
3. Do not describe a method as a limitation simply because
   it uses a particular dataset, band combination,
   pretrained model, or augmentation technique.
4. Do not infer cloud-related failures, generalization
   problems, or false detection rates without direct
   supporting evidence.
5. Distinguish the authors' statements from your own
   cautious inferences.
"""


# ---------------------------------------------------------
# GENERATE ANSWER
# ---------------------------------------------------------

def generate_answer(prompt):

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_INSTRUCTION,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
      
    )

    answer = response.choices[0].message.content

    if not answer:
        raise ValueError(
            "Azure OpenAI returned an empty response."
        )

    return answer