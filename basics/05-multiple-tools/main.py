import argparse
import json
import random
from typing import Literal

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

# The price "service" fails randomly so we can watch retries happen.
PRICE_FAILURE_RATE = 0.3
TOOL_RETRIES = 2
assert isinstance(TOOL_RETRIES, int) and TOOL_RETRIES >= 0, (
    "TOOL_RETRIES must be a non-negative integer"
)

# Made-up rates so the model can't answer from memory: if a reply uses these
# numbers, it came from the tool.
USD_PER_UNIT = {
    "USD": 1.0,
    "EUR": 1.13,
    "GBP": 1.31,
    "JPY": 0.0071,
    "INR": 0.0117,
}

# Made-up local prices, same trick as the rates.
PRICES = {
    "tokyo": {
        "coffee": (550, "JPY"),
        "metro ticket": (210, "JPY"),
        "beer": (700, "JPY"),
    },
    "london": {
        "coffee": (3.6, "GBP"),
        "metro ticket": (2.9, "GBP"),
        "beer": (6.5, "GBP"),
    },
    "paris": {
        "coffee": (3.2, "EUR"),
        "metro ticket": (2.15, "EUR"),
        "beer": (7.0, "EUR"),
    },
    "mumbai": {
        "coffee": (220, "INR"),
        "metro ticket": (40, "INR"),
        "beer": (350, "INR"),
    },
}

city_item_combinations_string = '\n'.join(
    [f"{city}: {', '.join(PRICES[city])}" for city in PRICES]
)

SYSTEM_PROMPT = (
    "You answer questions about money and prices. "
    "Use the tools for prices, currency conversion and arithmetic; never guess or compute these yourself. "
    "Keep answers to one or two sentences."
)

DEMO_QUESTIONS = [
    "How much is a coffee in Tokyo in USD?",
    "I'm buying a coffee in Tokyo and a coffee in London. What's the total in USD?",
    "Which is cheaper in USD, a beer in Paris or a beer in Mumbai, and by how much?",
    "What does a croissant cost in Paris?",
    "Split a 47.80 EUR bill between 3 people. How much is each share in GBP?",
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


class PriceInput(BaseModel):
    city: str = Field(description="City name, e.g. 'Tokyo'.")
    item: str = Field(description="Item to price, e.g. 'coffee'.")


class CalculateInput(BaseModel):
    operation: Literal['add', 'subtract', 'multiply', 'divide']
    numbers: list[float] = Field(
        min_length=2,
        description="Applied left to right, e.g. subtract [10, 3, 2] gives 5.",
    )


# Temporary failure: the same call might work if repeated.
class ToolUnavailable(Exception):
    pass


def lookup_price(city: str, item: str) -> dict:
    if random.random() < PRICE_FAILURE_RATE:
        raise ToolUnavailable("price service timed out")
    city_lowercase = city.lower().strip()
    item_lowercase = item.lower().strip()

    if city_lowercase not in PRICES:
        raise ValueError(
            f"Unknown city: {city_lowercase}. Known cities are: {', '.join(PRICES)}"
        )
    if item_lowercase not in PRICES[city_lowercase]:
        raise ValueError(
            f"Unknown item: {item_lowercase} in {city_lowercase}. Known items are: {', '.join(PRICES[city_lowercase])}"
        )
    price, currency = PRICES[city_lowercase][item_lowercase]
    return {"price": price, "currency": currency}


def calculate(operation: str, numbers: list[float]) -> float:
    if not numbers:
        raise ValueError("No numbers provided for calculation.")
    if operation not in ('add', 'subtract', 'multiply', 'divide'):
        raise ValueError(f"Unsupported operation: {operation}")
    result = numbers[0]
    for number in numbers[1:]:
        if operation == 'add':
            result += number
        elif operation == 'subtract':
            result -= number
        elif operation == 'multiply':
            result *= number
        elif operation == 'divide':
            if number == 0:
                raise ValueError("Cannot divide by zero.")
            result /= number
    return result


def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
    for currency_type, currency in (('from', from_currency), ('to', to_currency)):
        if currency not in USD_PER_UNIT:
            raise ValueError(
                f"Unsupported {currency_type} currency code: {currency}. "
                f"Supported codes are: {', '.join(USD_PER_UNIT)}"
            )

    return round(amount * USD_PER_UNIT[from_currency] / USD_PER_UNIT[to_currency], 2)


TOOLS = {
    "convert_currency": {
        "function": convert_currency,
        "input_model": ConvertInput,
        "description": (
            "Convert a given amount of money from one currency to another using the fixed exchange rate. "
            f"Supported currencies are {', '.join(USD_PER_UNIT)}"
        ),
    },
    "lookup_price": {
        "function": lookup_price,
        "input_model": PriceInput,
        "description": (
            "Look up the price of an item in a city. "
            "Returns price and corresponding local currency in which price is quoted. "
            "Supports following city and corresponding items combinations:\n"
            f"{city_item_combinations_string}"
        ),
    },
    "calculate": {
        "function": calculate,
        "input_model": CalculateInput,
        "description": (
            "Perform an operation, left-to-right, on the given sequence of numbers. "
            "Use for any arithmetic instead of calculating yourself."
        ),
    },
}

TOOL_DEFINITIONS = [
    {
        "name": name,
        "description": tool["description"],
        "input_schema": tool["input_model"].model_json_schema(),
    }
    for name, tool in TOOLS.items()
]


def tool_result(tool_use_id: str, content: str, is_error: bool = False) -> dict:
    return {
        'type': 'tool_result',
        'tool_use_id': tool_use_id,
        'content': content,
        'is_error': is_error,
    }


def run_tool(block) -> dict:
    print(f"  tool call: {block.name}({block.input}) id={block.id}")
    if block.name not in TOOLS:
        tool_error = f"Unknown tool: {block.name}. Available tools are: {', '.join(TOOLS.keys())}"
        print(f"  tool error: {tool_error}")
        return tool_result(block.id, tool_error, is_error=True)
    else:
        print(f"  tool {block.name} is available")

    tool = TOOLS[block.name]
    try:
        args = tool['input_model'].model_validate(block.input)
        print(f"  input validation succeeded for tool {block.name}")
    except ValidationError as e:
        validation_error = f"Input validation for the tool {block.name} failed with following error: {e}"
        print(f"  input validation error: {validation_error}")
        return tool_result(block.id, validation_error, is_error=True)

    for try_i in range(1, TOOL_RETRIES + 2):
        try:
            output = tool['function'](**args.model_dump())
            output_str = json.dumps(output)
            print(f"  '{block.name}' tool call succeeded on attempt {try_i}")
            print(f"  tool result: {output_str}")
            return tool_result(block.id, output_str, is_error=False)
        except ToolUnavailable as e:
            print(f"  tool unavailable on attempt {try_i}: {e}")
            continue
        except ValueError as e:
            print(f"  tool error: {e}")
            return tool_result(block.id, str(e), is_error=True)
    print(f"  all {TOOL_RETRIES + 1} attempts failed for tool {block.name}")
    return tool_result(
        block.id,
        f"Service is down, unable to run tool {block.name}. Don't retry.",
        is_error=True,
    )


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1e6) * PRICE_PER_MTOK["input"] + (
        output_tokens / 1e6
    ) * PRICE_PER_MTOK["output"]


def call_model(client: Anthropic, messages: list, use_tools: bool) -> tuple:
    tools = TOOL_DEFINITIONS if use_tools else []
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
        description="Answer money questions with price lookup, currency conversion and arithmetic tools."
    )
    parser.add_argument("question", nargs="?")
    parser.add_argument(
        "--demo", action="store_true", help="Run the built-in questions"
    )
    parser.add_argument(
        "--no-tools", action="store_true", help="Ask without giving the model any tools"
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
