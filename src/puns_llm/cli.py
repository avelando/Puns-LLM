import argparse
import json
import sys
from pathlib import Path

from puns_llm.data.cohorts import cohort_ids, make_cohorts
from puns_llm.data.prepare import prepare_corpus
from puns_llm.evaluation.score import evaluate_run
from puns_llm.evaluation.score_v2 import evaluate_v2
from puns_llm.evaluation.score_v3 import evaluate_v3
from puns_llm.inference.ollama import OllamaClient
from puns_llm.inference.zero_shot import PROTOCOLS, run_zero_shot
from puns_llm.menu import MenuAction, run_menu
from puns_llm.utils.freeze import freeze_methodology, verify_freeze


def show(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def settings_for(root: Path) -> dict:
    return json.loads((root / "configs/experiments/zero_shot.json").read_text(encoding="utf-8"))


def configured_client(root: Path, host: str | None = None) -> OllamaClient:
    settings = settings_for(root)
    return OllamaClient(host or settings["host"], settings["timeout_seconds"])


def prepare_action(root: Path) -> None:
    show(prepare_corpus(root))
    print("Saved: data/processed/puns_only.json")


def inference_action(root: Path) -> None:
    show(run_zero_shot(root))


def legacy_action(root: Path) -> None:
    config = settings_for(root)
    show(evaluate_run(root, config["model"], config["protocol"], config["limit"]))


def score_action(root: Path) -> None:
    config = settings_for(root)
    show(evaluate_v2(root, config["model"], config["protocol"], limit=config["limit"]))


def score_v3_action(root: Path) -> None:
    config = settings_for(root)
    show(evaluate_v3(root, config["model"], config["protocol"], limit=config["limit"]))


def report_cohorts(payload: dict) -> None:
    show(
        {
            "pilot_size": len(payload["pilot_ids"]),
            "official_size": len(payload["official_ids"]),
            "development_size": len(payload["development_ids"]),
            "pilot_sign_categories": payload["pilot_sign_categories"],
            "saved": "configs/experiments/cohorts.json",
        }
    )


def pilot_action(root: Path) -> None:
    report_cohorts(make_cohorts(root))


def freeze_action(root: Path) -> None:
    show(freeze_methodology(root, configured_client(root)))


def verify_action(root: Path) -> None:
    show(verify_freeze(root, configured_client(root)))


def main() -> int:
    actions = [
        MenuAction("1", "prepare", "Prepare corpus", prepare_action),
        MenuAction("2", "zero-shot", "Run linguistic inference", inference_action),
        MenuAction("3", "evaluate", "Evaluate legacy results", legacy_action),
        MenuAction("4", "evaluate-v2", "Evaluate with extended metrics", score_action),
        MenuAction("8", "evaluate-v3", "Evaluate with corrected token metrics", score_v3_action),
        MenuAction("5", "make-pilot", "Create pilot and official ID cohorts", pilot_action),
        MenuAction("6", "freeze", "Freeze methodology", freeze_action),
        MenuAction("7", "verify-freeze", "Verify frozen methodology", verify_action),
    ]
    parser = argparse.ArgumentParser(prog="puns-llm")
    parser.add_argument("--action", choices=[item.command for item in actions])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--model")
    parser.add_argument("--protocol", choices=[*PROTOCOLS, "zero_shot"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ollama-host")
    parser.add_argument("--cohort", choices=["pilot", "official"])
    parser.add_argument("--pilot-size", type=int, default=48)
    parser.add_argument("--retry-invalid", action="store_true")
    args = parser.parse_args()
    if args.action is None:
        return run_menu(actions, args.root)
    try:
        config = settings_for(args.root)
        model = args.model or config["model"]
        protocol = args.protocol or config["protocol"]
        if args.cohort and args.limit is not None:
            raise ValueError("Do not combine --cohort and --limit")
        ids = cohort_ids(args.root, args.cohort) if args.cohort else None
        if args.cohort == "official":
            verify_freeze(args.root, configured_client(args.root, args.ollama_host))
        if args.action == "prepare":
            prepare_action(args.root)
        elif args.action == "make-pilot":
            report_cohorts(make_cohorts(args.root, size=args.pilot_size))
        elif args.action == "freeze":
            show(freeze_methodology(args.root, configured_client(args.root, args.ollama_host)))
        elif args.action == "verify-freeze":
            show(verify_freeze(args.root, configured_client(args.root, args.ollama_host)))
        elif args.action == "zero-shot":
            if protocol == "zero_shot":
                raise ValueError("The original zero_shot run is read-only")
            show(
                run_zero_shot(
                    args.root,
                    model=model,
                    protocol=protocol,
                    ids=ids,
                    limit=args.limit,
                    host=args.ollama_host,
                    retry_invalid=args.retry_invalid,
                )
            )
        elif args.action in ("evaluate-v2", "evaluate-v3"):
            show(
                (evaluate_v2 if args.action == "evaluate-v2" else evaluate_v3)(
                    args.root,
                    model,
                    protocol,
                    ids=ids,
                    limit=0
                    if ids is not None
                    else config["limit"]
                    if args.limit is None
                    else args.limit,
                    require_complete=args.cohort == "official",
                )
            )
        else:
            show(
                evaluate_run(
                    args.root,
                    model,
                    protocol,
                    config["limit"] if args.limit is None else args.limit,
                )
            )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0
