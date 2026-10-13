from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class MenuAction:
    number: str
    command: str
    title: str
    handler: Callable[[Path], None]


def run_menu(actions: list[MenuAction], root: Path) -> int:
    options = {action.number: action for action in actions}

    while True:
        print("\nPuns-LLM")

        for action in actions:
            print(f"{action.number}. {action.title}")

        print("0. Exit")

        try:
            selected = input("Select an option: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if selected == "0":
            return 0

        if selected not in options:
            print("Invalid option")
            continue

        try:
            options[selected].handler(root)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Error: {error}")
