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

MAX_TURNS = 6
MAX_COST_USD = 0.01  # per question

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
    "Convert 100 USD to EUR, then convert that result to JPY",
    (
        "Start with 100 USD. Convert it to EUR, then that to GBP, then that to JPY, "
        "then that to INR, then back to USD. How much did I end up with?"
    ),
    "Convert 50 USD to BTC",
    "What's the capital of France?",
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
    print(
        f"  tool result: {args.amount} {args.from_currency} = {tool_output} {args.to_currency}"
    )
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


def call_model(client: Anthropic, messages: list, use_tools: bool) -> tuple:
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
    return response, cost


def get_text(response) -> str:
    return "".join(block.text for block in response.content if block.type == "text")


def ask(client: Anthropic, question: str, use_tools: bool) -> str:
    messages = [{"role": "user", "content": question}]
    total_cost = 0.0
    last_result = None
    for turn in range(1, MAX_TURNS + 1):
        print(f"  turn {turn}")

        # Check the total cost before the model call, so that we don't waste dollars unnecessarily
        if total_cost > MAX_COST_USD:
            return f"Cost limit exceeded: ${total_cost:.6f}. The last result was: {last_result}"

        response, cost = call_model(client, messages, use_tools)

        total_cost += cost
        print(f"  total cost so far: ${total_cost:.6f}")

        if response.stop_reason == 'max_tokens':
            return (
                f"Answer was cut off at max_tokens. The last result was: {last_result}"
            )

        if response.stop_reason == 'end_turn':
            return get_text(response)

        if response.stop_reason == 'tool_use':
            if turn == MAX_TURNS:
                return f"Maximum number of turns reached: {MAX_TURNS}. The last result was: {last_result}"
            tool_results_list = []
            for block in response.content:
                if block.type != 'tool_use':
                    continue
                result = run_tool(block)
                tool_results_list.append(result)
                if not result['is_error']:
                    last_result = result['content']
            messages.append({"role": "assistant", "content": response.content})
            messages.append({'role': 'user', 'content': tool_results_list})
            continue

        return f"Unexpected stop_reason: {response.stop_reason}"


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
