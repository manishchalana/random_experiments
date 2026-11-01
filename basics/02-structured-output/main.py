import argparse
import json
from typing import Literal

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

_ = load_dotenv()

MODEL_NAME = 'claude-haiku-5-5'
MAX_ATTEMPTS = 3

PRICE_PER_MTOK = {
    'input': 0.10,
    'output': 0.50,
}


class Expense(BaseModel):
    merchant: str | None = Field(
        description="Name of the merchant, or null if the note doesn't say"
    )
    product: str | None = Field(
        description="Name of the product, or null if the note doesn't say"
    )
    amount: float | None = Field(
        gt=0, description="Total amount, or null if the note doesn't say"
    )
    currency: str | None = Field(
        pattern=r'^[A-Z]{3}$',
        description="ISO code like USD, or null if the note doesn't say",
    )
    category: Literal['food', 'transport', 'housing', 'entertainment', 'other']
    date: str | None = Field(description="YYYY-MM-DD, or null if the note has no date")


# Built from the model so the prompt and the validator can't drift apart.
SYSTEM_PROMPT = (
    "You extract a single expense from a short note.\n"
    "Reply with only a JSON object that matches this JSON schema, no other text:\n"
    f"{json.dumps(Expense.model_json_schema(), indent=2)}\n"
    "If the note doesn't state something, don't guess."
)

DEMO_NOTES = [
    "uber home 23.40",
    "Paid rent 1,850 EUR on 2026-10-01",
    "Movie tickets for two, 18 bucks each",
    "lunch at Tacos El Gordo yesterday, $14.75 + tip",
    "coffee",
]


def get_text(response) -> str:
    return ''.join(block.text for block in response.content if block.type == 'text')


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1e6) * PRICE_PER_MTOK['input'] + (
        output_tokens / 1e6
    ) * PRICE_PER_MTOK['output']


def strip_code_fences(text: str) -> str:
    left_index = text.find('{')
    right_index = text.rfind('}')
    if left_index == -1 or right_index == -1 or right_index < left_index:
        return text
    return text[left_index : right_index + 1]


def parse_expense(text: str) -> Expense:
    return Expense.model_validate_json(strip_code_fences(text))


def extract_expense(client: Anthropic, note: str, max_tokens: int) -> Expense:
    messages = [{'role': 'user', 'content': note}]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
            thinking={'type': 'disabled'},
        )
        text = get_text(response)
        usage = response.usage
        cost = estimate_cost(usage.input_tokens, usage.output_tokens)
        print(
            f"  attempt {attempt}: stop_reason={response.stop_reason} | in={usage.input_tokens} out={usage.output_tokens} | ${cost:.6f}"
        )
        print(f"    raw: {text!r}")
        print(f"    blocks: {[block.type for block in response.content]}")
        if response.stop_reason == 'max_tokens':
            raise RuntimeError(
                f"Reply cut off at max_tokens={max_tokens}; try a higher --max-tokens"
            )
        try:
            return parse_expense(text)
        except ValidationError as e:
            print(f"    parse error: {e}")
            messages.append({'role': 'assistant', 'content': text})
            messages.append(
                {
                    'role': 'user',
                    'content': f"Error: {e}\nPlease reply with only the corrected JSON.",
                }
            )

    raise RuntimeError(f"No valid expense after {MAX_ATTEMPTS} attempts")


def main():
    parser = argparse.ArgumentParser(
        description="Pull a structured expense out of a free-text note."
    )
    parser.add_argument('note', nargs='?')
    parser.add_argument('--demo', action='store_true', help="Run the built-in notes")
    parser.add_argument('--max-tokens', type=int, default=300)
    args = parser.parse_args()

    if not args.demo and args.note is None:
        parser.error("give a note or use --demo")
    notes = DEMO_NOTES if args.demo else [args.note]

    client = Anthropic()

    for note in notes:
        print(f"--> {note!r}")
        try:
            expense = extract_expense(client, note, args.max_tokens)
            print(f"<-- {expense.model_dump()}")
        except RuntimeError as e:
            print(f"<-- FAILED: {e}")
        print()


if __name__ == '__main__':
    main()
