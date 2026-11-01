import argparse

from anthropic import Anthropic, transform_schema
from dotenv import load_dotenv
from pydantic import ValidationError

from main import DEMO_NOTES, MODEL_NAME, Expense, estimate_cost, get_text

_ = load_dotenv()


MAX_ATTEMPTS = 3

SYSTEM_PROMPT = "You extract a single expense from a short note."


def extract_expense(client: Anthropic, note: str, max_tokens: int) -> Expense:
    messages = [{'role': 'user', 'content': note}]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
            thinking={'type': 'disabled'},
            output_config={
                'format': {
                    'type': 'json_schema',
                    'schema': transform_schema(Expense),
                },
            },
        )

        text = get_text(response)
        usage = response.usage
        cost = estimate_cost(usage.input_tokens, usage.output_tokens)
        print(
            f"  attempt {attempt}: stop_reason={response.stop_reason} | in={usage.input_tokens} out={usage.output_tokens} | ${cost:.6f}"
        )

        print(f"    blocks: {[block.type for block in response.content]}")
        if response.stop_reason == 'max_tokens':
            raise RuntimeError(
                f"Reply cut off at max_tokens={max_tokens}; try a higher --max-tokens"
            )
        if response.stop_reason == 'refusal':
            raise RuntimeError("Reply refused")
        try:
            return Expense.model_validate_json(text)
        except ValidationError as e:
            print(f"  attempt {attempt}: validation error: {e}")
            messages.append({'role': 'assistant', 'content': text})
            messages.append(
                {
                    'role': 'user',
                    'content': f"Your previous reply failed validation: {e}\nReply with only the corrected JSON.",
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
