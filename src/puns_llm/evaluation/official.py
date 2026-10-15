import json
from pathlib import Path

from puns_llm.evaluation.score_v3 import score_item, summarize
from puns_llm.inference.official import directory_for, load_predictions, read_json, sha


def evaluate_official(root: Path, model: str, protocol: str) -> dict:
    data_path = root / "data/processed/puns_only.json"
    dataset = read_json(data_path)
    ids = [item["id"] for item in dataset]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate corpus IDs")
    output = directory_for(root, model, protocol)
    metadata = read_json(output / "metadata.json")
    if (
        metadata.get("model") != model
        or metadata.get("protocol") != protocol
        or metadata.get("dataset_sha256") != sha(data_path.read_bytes())
        or metadata.get("expected_puns") != len(dataset)
    ):
        raise ValueError("Inference metadata mismatch")

    predictions = load_predictions(output / "predictions.jsonl", set(ids))
    details = [
        score_item(item, predictions.get(item["id"]), protocol.startswith("oracle_"))
        for item in dataset
    ]
    report = summarize(details, model, protocol)
    report["scope"] = "all_puns"
    report["recorded"] = len(predictions)
    report["processed_all_puns"] = len(predictions) == len(dataset)
    report["total_model_calls"] = sum(item.get("attempt_count", 1) for item in predictions.values())
    (output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "details.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in details),
        encoding="utf-8",
    )
    return report
