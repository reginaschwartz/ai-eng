import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(override=False)
client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

SYSTEM_PROMPT = """
Classify each support message into exactly one class:
- billing: charges, invoices, payments, or refunds
- technical: errors, broken behavior, or setup problems
- account: sign-in, profile, or account-access questions

Return only the class name: billing, technical, or account.
""".strip()


def classify_support_message(message: str) -> str:
    cleaned_message = message.strip()
    if not cleaned_message:
        raise ValueError("message must not be empty")

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": cleaned_message},
        ],
    )

    response_message = completion.choices[0].message
    if response_message.refusal:
        raise RuntimeError(f"The model refused classification: {response_message.refusal}")
    if response_message.content is None:
        raise RuntimeError("The model returned no class")

    return response_message.content.strip()


predicted_class = classify_support_message("I was charged twice this month.")
print(predicted_class)
