import argparse
import json
import sys
from pathlib import Path

from puns_llm.data.prepare import prepare_corpus
from puns_llm.evaluation.official import evaluate_official
from puns_llm.inference.official import run_official
from puns_llm.inference.ollama import OllamaClient
from puns_llm.inference.zero_shot import PROTOCOLS
from puns_llm.menu import MenuAction, run_menu
from puns_llm.utils.freeze import freeze_methodology, verify_freeze


def show(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def configured_client(root: Path) -> OllamaClient:
    settings = json.loads((root / "configs/experiments/zero_shot.json").read_text(encoding="utf-8"))
    return OllamaClient(settings["host"], settings["timeout_seconds"])


def prepare_action(root: Path) -> None:
    show(prepare_corpus(root))


def run_action(root: Path) -> None:
    verify_freeze(root, configured_client(root))
    show(run_official(root, "gemma3:27b", "zero_shot_v2"))


def evaluate_action(root: Path) -> None:
    show(evaluate_official(root, "gemma3:27b", "zero_shot_v2"))


def freeze_action(root: Path) -> None:
    show(freeze_methodology(root, configured_client(root)))


def verify_action(root: Path) -> None:
    show(verify_freeze(root, configured_client(root)))


def main() -> int:
    actions = [
        MenuAction("1", "prepare", "Prepare corpus", prepare_action),
        MenuAction("2", "run-official", "Run all puns", run_action),
        MenuAction("3", "evaluate-official", "Evaluate official results", evaluate_action),
        MenuAction("4", "freeze", "Freeze methodology", freeze_action),
        MenuAction("5", "verify-freeze", "Verify methodology", verify_action),
    ]
    parser = argparse.ArgumentParser(prog="puns-llm")
    parser.add_argument("--action", choices=[action.command for action in actions])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--model", choices=["gemma3:27b", "qwen3:32b"])
    parser.add_argument("--protocol", choices=list(PROTOCOLS))
    args = parser.parse_args()
    if args.action is None:
        return run_menu(actions, args.root)
    try:
        if args.action == "prepare":
            prepare_action(args.root)
        elif args.action == "freeze":
            freeze_action(args.root)
        elif args.action == "verify-freeze":
            verify_action(args.root)
        else:
            model = args.model or "gemma3:27b"
            protocol = args.protocol or "zero_shot_v2"
            if args.action == "run-official":
                verify_freeze(args.root, configured_client(args.root))
                show(run_official(args.root, model, protocol))
            else:
                show(evaluate_official(args.root, model, protocol))
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0
