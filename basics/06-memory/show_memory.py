# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "rich>=15.0.0",
# ]
# ///
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

MEMORY_FILE = Path(__file__).parent / "memory.json"


def main() -> None:
    try:
        with open(MEMORY_FILE, "r") as f:
            memory_data = json.load(f)
    except FileNotFoundError:
        print("Memory file not found.")
        return
    except json.JSONDecodeError:
        print("Error decoding JSON from memory file.")
        return

    table = Table(title="Memory")
    table.add_column("Fact", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    for key, value in memory_data.items():
        table.add_row(key, str(value))

    console = Console()
    console.print(table)


if __name__ == "__main__":
    main()
