import argparse
import json
import sys
from pathlib import Path

from puns_llm.data.prepare import prepare_corpus
from puns_llm.menu import MenuAction, run_menu


def prepare_action(root: Path) -> None:
    report = prepare_corpus(root)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Saved: data/processed/puns_only.json")
    print("Saved: results/data_preparation/summary.json")


def main() -> int:
    actions = [
        MenuAction("1", "prepare", "Prepare corpus", prepare_action),
    ]

    parser = argparse.ArgumentParser(prog="puns-llm")
    parser.add_argument("--action", choices=[action.command for action in actions])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])

    args = parser.parse_args()

    if args.action is None:
        return run_menu(actions, args.root)

    try:
        next(action.handler for action in actions if action.command == args.action)(args.root)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0
