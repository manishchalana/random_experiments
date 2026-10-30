import argparse

from anthropic import Anthropic
from dotenv import load_dotenv

# Load ANTHROPIC_API_KEY from .env
_ = load_dotenv()

MODEL_NAME = 'claude-haiku-5-5'

# USD per million tokens (checked Oct 2026).
# This is the rate for prompts up to 100K tokens; Haiku 5.5 charges more above that.
PRICE_PER_MTOK = {
    'input': 0.10,
    'output': 0.50,
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a concise assistant. Answer in at most three sentences."
)


def ask(client: Anthropic, prompt: str, system: str, max_tokens: int):
    return client.messages.create(
        model=MODEL_NAME,
        max_tokens=max_tokens,
        system=system,
        messages=[{'role': 'user', 'content': prompt}],
    )


def get_text(response) -> str:
    return ''.join(block.text for block in response.content if block.type == 'text')


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1e6) * PRICE_PER_MTOK['input'] + (
        output_tokens / 1e6
    ) * PRICE_PER_MTOK['output']


def main():
    parser = argparse.ArgumentParser(
        description="Send one prompt to Claude and show what it cost."
    )
    parser.add_argument('prompt')
    parser.add_argument('--system', default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument('--max-tokens', type=int, default=500)
    args = parser.parse_args()

    client = Anthropic()

    print(f"--> {MODEL_NAME} | max_tokens={args.max_tokens}")
    print(f"    system: {args.system}")
    print(f"    user:   {args.prompt}")

    response = ask(client, args.prompt, args.system, args.max_tokens)

    usage = response.usage
    cost = estimate_cost(usage.input_tokens, usage.output_tokens)

    print(
        f"<-- stop_reason={response.stop_reason} | in={usage.input_tokens} out={usage.output_tokens} tokens | ${cost:.6f} | num_blocks={len(response.content)}"
    )
    print()
    print(get_text(response))


if __name__ == '__main__':
    main()
