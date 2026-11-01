import argparse

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

_ = load_dotenv()

MODEL_NAME = "claude-haiku-5-5"

PRICE_PER_MTOK = {
    "input": 0.10,
    "output": 0.50,
}

# Made-up rates so the model can't answer from memory: if a reply uses these
# numbers, it came from the tool.
USD_PER_UNIT = {
    "USD": 1.0,
    "EUR": 1.13,
    "GBP": 1.31,
    "JPY": 0.0071,
    "INR": 0.0117,
}

SYSTEM_PROMPT = (
    "You answer questions about money. "
    "For any currency conversion, use the convert_currency tool instead of your own rates. "
    "Keep answers to one or two sentences."
)

DEMO_QUESTIONS = [
    "How much is 1,850 EUR in USD?",
    "I spent 23.40 GBP and 3,200 JPY on a trip. What's that in USD?",
    "What's the capital of France?",
    "Convert 50 USD to BTC",
    "Convert 100 USD to EUR, then convert that result to JPY",
]


class ConvertInput(BaseModel):
    amount: float = Field(
        gt=0, description="The amount of money to convert, must be greater than 0."
    )
    from_currency: str = Field(
        pattern=r"^[A-Z]{3}$",
        description="The ISO currency code to convert from, e.g., 'USD'.",
    )
    to_currency: str = Field(
        pattern=r"^[A-Z]{3}$",
        description="The ISO currency code to convert to, e.g., 'EUR'.",
    )


CONVERT_TOOL = {
    "name": "convert_currency",
    "description": (
        "Convert a given amount of money from one currency to another using the fixed exchange rate. "
        f"Supported currencies are {', '.join(USD_PER_UNIT)}"
    ),
    "input_schema": ConvertInput.model_json_schema(),
}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
    for currency_type, currency in (('from', from_currency), ('to', to_currency)):
        if currency not in USD_PER_UNIT:
            raise ValueError(
                f"Unsupported {currency_type} currency code: {currency}. "
                f"Supported codes are: {', '.join(USD_PER_UNIT)}"
            )

    return round(amount * USD_PER_UNIT[from_currency] / USD_PER_UNIT[to_currency], 2)


def run_tool(block) -> dict:
    print(f"  tool call: {block.name}({block.input}) id={block.id}")
    if block.name != CONVERT_TOOL["name"]:
        return {
            'type': 'tool_result',
            'tool_use_id': block.id,
            'content': f"Unknown tool: {block.name}",
            'is_error': True,
        }

    try:
        args = ConvertInput.model_validate(block.input)
        tool_output = convert_currency(**args.model_dump())

    except (ValidationError, ValueError) as e:
        print(f"  tool error: {e}")
        return {
            'type': 'tool_result',
            'tool_use_id': block.id,
            'content': str(e),
            'is_error': True,
        }
    print(f"  tool result: {args.amount} {args.from_currency} = {tool_output} {args.to_currency}")
    return {
        'type': 'tool_result',
        'tool_use_id': block.id,
        'content': (
            f"{args.amount} {args.from_currency} = {tool_output} {args.to_currency}"
        ),
        'is_error': False,
    }


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1e6) * PRICE_PER_MTOK["input"] + (
        output_tokens / 1e6
    ) * PRICE_PER_MTOK["output"]


def call_model(client: Anthropic, messages: list, use_tools: bool):
    tools = [CONVERT_TOOL] if use_tools else []
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=messages,
        tools=tools,
        thinking={"type": "disabled"},
    )
    usage = response.usage
    cost = estimate_cost(usage.input_tokens, usage.output_tokens)
    print(
        f"  model call: stop_reason={response.stop_reason} | in={usage.input_tokens} out={usage.output_tokens} | ${cost:.6f}"
    )
    print(f"    blocks: {[block.type for block in response.content]}")
    return response


def get_text(response) -> str:
    return "".join(block.text for block in response.content if block.type == "text")


def ask(client: Anthropic, question: str, use_tools: bool) -> str:
    messages = [{"role": "user", "content": question}]
    response = call_model(client, messages, use_tools)

    if response.stop_reason != "tool_use":
        return get_text(response)

    messages.append({"role": "assistant", "content": response.content})
    all_tool_calls = [block for block in response.content if block.type == "tool_use"]
    tool_results = [run_tool(block) for block in all_tool_calls]
    messages.append({"role": "user", "content": tool_results})
    response = call_model(client, messages, use_tools)
    if response.stop_reason == 'tool_use':
        return "The model wanted to use a tool again, but we only allow one round of tool use."
    return get_text(response)


def main():
    parser = argparse.ArgumentParser(
        description="Answer money questions with a currency conversion tool."
    )
    parser.add_argument("question", nargs="?")
    parser.add_argument(
        "--demo", action="store_true", help="Run the built-in questions"
    )
    parser.add_argument(
        "--no-tools", action="store_true", help="Ask without giving the model the tool"
    )
    args = parser.parse_args()

    if not args.demo and args.question is None:
        parser.error("give a question or use --demo")
    questions = DEMO_QUESTIONS if args.demo else [args.question]

    client = Anthropic()

    for question in questions:
        print(f"--> {question!r}")
        answer = ask(client, question, use_tools=not args.no_tools)
        print(f"<-- {answer}")
        print()


if __name__ == "__main__":
    main()
