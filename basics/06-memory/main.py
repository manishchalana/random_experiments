import argparse

from anthropic import Anthropic
from dotenv import load_dotenv

_ = load_dotenv()

MODEL_NAME = "claude-haiku-5-5"

PRICE_PER_MTOK = {
    "input": 0.10,
    "output": 0.50,
}

# Deliberately tiny so trimming kicks in after a few exchanges.
MAX_HISTORY_TOKENS = 600

# Trim down to this, not just under the max, so we don't trim on every turn.
TRIM_TARGET_TOKENS = 300
MAX_COST_USD = 0.05  # per session

SYSTEM_PROMPT = "You are a friendly assistant. Keep answers to two or three sentences."


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1e6) * PRICE_PER_MTOK["input"] + (
        output_tokens / 1e6
    ) * PRICE_PER_MTOK["output"]


def call_model(client: Anthropic, system: str, messages: list) -> tuple:
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=300,
        system=system,
        messages=messages,
        thinking={"type": "disabled"},
    )
    usage = response.usage
    cost = estimate_cost(usage.input_tokens, usage.output_tokens)
    print(
        f"    model call: stop_reason={response.stop_reason} | in={usage.input_tokens} out={usage.output_tokens} | ${cost:.6f}"
    )
    print(f"    blocks: {[block.type for block in response.content]}")
    return response, cost


def get_text(response) -> str:
    return "".join(block.text for block in response.content if block.type == "text")


def count_history_tokens(client: Anthropic, system: str, messages: list) -> int:
    return client.messages.count_tokens(
        system=system, messages=messages, model=MODEL_NAME
    ).input_tokens


def find_cut_index(client: Anthropic, system: str, messages: list) -> int:
    if messages:
        if messages[0]['role'] != 'user':
            raise ValueError("The first message in the history must be from the user.")
        if messages[-1]['role'] == 'user':
            raise ValueError("The last message in the history cannot be from the user.")

    for msg_i, msg in enumerate(messages):
        if msg['role'] != 'user':
            continue

        if count_history_tokens(client, system, messages[msg_i:]) <= TRIM_TARGET_TOKENS:
            return msg_i

    # In case of fallback, let's keep the last two messages
    return len(messages) - 2


def trim_history(
    client: Anthropic, system: str, messages: list, current_cost: float
) -> tuple:
    assert messages, "Cannot trim an empty message history."
    cut_index = find_cut_index(client, system, messages)
    messages_to_summarize = messages[:cut_index]
    messages_to_keep = messages[cut_index:]

    if messages_to_summarize:
        summary_prompt = "You are a helpful assistant. You will be given a previous system prompt and a conversation history. Keep facts about the user (name, numbers, preferences) exactly, as a short list. Then one line on topics discussed. If newer messages contradict the earlier summary, keep the newer fact. Under 100 words.\n\n"
        _temp_messages_string = '\n\n'.join(
            [f'{msg['role']}: {msg['content']}' for msg in messages_to_summarize]
        )
        messages_string_to_summarize = f"{system}\n\n{_temp_messages_string}"
        summary_response = call_model(
            client,
            summary_prompt,
            [{'role': 'user', 'content': messages_string_to_summarize}],
        )
        summary_text = get_text(summary_response[0])
        new_system_prompt = f"{SYSTEM_PROMPT}\n\nNotes from earlier in this conversation (written about the user, not you):\n{summary_text}"
        current_cost += summary_response[1]
        print(
            f"    trimmed {len(messages_to_summarize)} messages, keeping {len(messages_to_keep)} messages"
        )
        print(f"    summary of trimmed history: {summary_text}")
        return new_system_prompt, messages_to_keep, current_cost

    return system, messages_to_keep, current_cost


def chat(client: Anthropic, memory: str):
    messages = []
    total_cost = 0.0
    system_prompt = SYSTEM_PROMPT
    while True:
        user_input = input("you> ").strip()
        if user_input in ("/quit", "/exit"):
            break
        if not user_input:
            continue

        if total_cost > MAX_COST_USD:
            print(f"Cost limit reached: ${total_cost:.6f}")
            break

        this_user_message = {'role': 'user', 'content': user_input}
        if memory == "none":
            messages = [this_user_message]
        else:
            messages.append(this_user_message)
        response, cost = call_model(client, system_prompt, messages)
        total_cost += cost
        reply = get_text(response)
        print(f"assistant> {reply}")
        messages.append({'role': 'assistant', 'content': reply})

        print(f"   history so far: {len(messages)}, total cost: ${total_cost:.6f}")

        if memory == "trim":
            total_tokens = count_history_tokens(client, system_prompt, messages)
            if total_tokens > MAX_HISTORY_TOKENS:
                system_prompt, messages, total_cost = trim_history(
                    client, system_prompt, messages, total_cost
                )


def main():
    parser = argparse.ArgumentParser(
        description="Chat with the model, with different ways of keeping history."
    )
    parser.add_argument(
        "--memory",
        choices=["none", "full", "trim"],
        default="full",
        help="none: no history, full: resend everything, trim: drop oldest exchanges",
    )
    args = parser.parse_args()

    client = Anthropic()
    print(f"memory={args.memory}, /quit to exit")
    chat(client, args.memory)


if __name__ == "__main__":
    main()
