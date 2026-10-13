import argparse
import json
import sys
from pathlib import Path

from puns_llm.data.prepare import prepare_corpus
from puns_llm.evaluation.score import evaluate_run
from puns_llm.inference.zero_shot import PROTOCOLS, run_zero_shot
from puns_llm.menu import MenuAction, run_menu


def prepare_action(root: Path) -> None:
    report = prepare_corpus(root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Saved: data/processed/puns_only.json")
    print("Saved: results/data_preparation/summary.json")


def zero_shot_action(root: Path) -> None:
    report = run_zero_shot(root)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def evaluate_action(root: Path) -> None:
    settings = json.loads((root / "configs/experiments/zero_shot.json").read_text())
    report = evaluate_run(
        root,
        settings["model"],
        settings["protocol"],
        settings["limit"],
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> int:
    actions = [
        MenuAction("1", "prepare", "Prepare corpus", prepare_action),
        MenuAction("2", "zero-shot", "Run linguistic inference", zero_shot_action),
        MenuAction("3", "evaluate", "Evaluate predictions", evaluate_action),
    ]

    parser = argparse.ArgumentParser(prog="puns-llm")
    parser.add_argument("--action", choices=[action.command for action in actions])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--model")
    parser.add_argument("--protocol", choices=[*PROTOCOLS, "zero_shot"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ollama-host")

    args = parser.parse_args()

    if args.action is None:
        return run_menu(actions, args.root)

    try:
        if args.action == "prepare":
            prepare_action(args.root)
        elif args.action == "zero-shot":
            report = run_zero_shot(
                args.root,
                model=args.model,
                protocol=args.protocol,
                limit=args.limit,
                host=args.ollama_host,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            settings = json.loads((args.root / "configs/experiments/zero_shot.json").read_text())
            report = evaluate_run(
                args.root,
                args.model or settings["model"],
                args.protocol or settings["protocol"],
                settings["limit"] if args.limit is None else args.limit,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0
