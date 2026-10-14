import hashlib
import json
from pathlib import Path

from puns_llm.inference.ollama import OllamaClient
from puns_llm.inference.schema import PREDICTION_SCHEMA, build_messages, parse_prediction
from puns_llm.inference.zero_shot import PROTOCOLS


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def directory_for(root: Path, model: str, protocol: str) -> Path:
    if model not in ("gemma3:27b", "qwen3:32b") or protocol not in PROTOCOLS:
        raise ValueError("Unknown model or protocol")
    return root / "results/official" / model.replace(":", "_") / protocol


def load_predictions(path: Path, eligible_ids: set[str]) -> dict[str, dict]:
    records = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            identifier = record.get("id")
            if identifier not in eligible_ids or identifier in records:
                raise ValueError(f"Duplicate or unexpected prediction ID: {identifier}")
            records[identifier] = record
    return records


def infer_with_retries(
    client: OllamaClient,
    model: str,
    messages: list[dict],
    options: dict,
    think: bool | None,
    max_invalid_retries: int,
) -> tuple[dict | None, list[dict]]:
    attempts = []
    prediction = None
    for attempt_number in range(1, max_invalid_retries + 2):
        response = client.chat(model, messages, PREDICTION_SCHEMA, options, think)
        content = response["message"]["content"]
        try:
            prediction = parse_prediction(content)
            error = None
        except ValueError as exception:
            error = str(exception)
        attempts.append(
            {
                "number": attempt_number,
                "error": error,
                "raw_response": content,
                "prompt_eval_count": response.get("prompt_eval_count"),
                "eval_count": response.get("eval_count"),
                "total_duration": response.get("total_duration"),
            }
        )
        if error is None:
            break
    return prediction, attempts


def run_official(
    root: Path,
    model: str,
    protocol: str,
    client: OllamaClient | None = None,
) -> dict:
    settings = read_json(root / "configs/experiments/zero_shot.json")
    config = read_json(root / "configs/experiments/official.json")
    models = read_json(root / "configs/models/ollama.json")["models"]
    if model not in models or protocol not in PROTOCOLS:
        raise ValueError("Unknown model or protocol")
    max_retries = config["max_invalid_retries"]
    if type(max_retries) is not int or not 0 <= max_retries <= 5:
        raise ValueError("max_invalid_retries must be an integer between 0 and 5")

    data_path = root / "data/processed/puns_only.json"
    examples = read_json(data_path)
    if not isinstance(examples, list) or not examples:
        raise ValueError("Corpus must be a nonempty list")
    ids = [item["id"] for item in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate corpus IDs")

    instructions = (root / "configs/prompts" / PROTOCOLS[protocol]).read_text(encoding="utf-8")
    options = {
        "temperature": settings["temperature"],
        "seed": settings["seed"],
        "num_ctx": settings["num_ctx"],
        "num_predict": settings["num_predict"],
    }
    client = client or OllamaClient(settings["host"], settings["timeout_seconds"])
    profile = {
        "format_version": 2,
        "scope": "all_puns",
        "model": model,
        "model_digest": client.model_digest(model),
        "protocol": protocol,
        "expected_puns": len(examples),
        "dataset_sha256": sha(data_path.read_bytes()),
        "prompt_sha256": sha(instructions.encode()),
        "schema_sha256": sha(json.dumps(PREDICTION_SCHEMA, sort_keys=True).encode()),
        "options": options,
        "think": models[model]["think"],
        "max_invalid_retries": max_retries,
    }

    output = directory_for(root, model, protocol)
    metadata_path = output / "metadata.json"
    predictions_path = output / "predictions.jsonl"
    if metadata_path.exists():
        if read_json(metadata_path) != profile:
            raise ValueError(f"Existing run metadata differs: {metadata_path}")
    elif predictions_path.exists():
        raise ValueError(f"Predictions exist without metadata: {predictions_path}")
    else:
        output.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    records = load_predictions(predictions_path, set(ids))
    cached = len(records)
    for index, sample in enumerate(examples, 1):
        identifier = sample["id"]
        if identifier in records:
            continue
        if protocol.startswith("oracle_"):
            content = instructions.replace("{text}", sample["text"]).replace(
                "{pun_signs}",
                json.dumps([sign["pun_sign"] for sign in sample["signs"]], ensure_ascii=False),
            )
            messages = [{"role": "user", "content": content}]
        else:
            messages = build_messages(sample["text"], instructions)

        prediction, attempts = infer_with_retries(
            client, model, messages, options, models[model]["think"], max_retries
        )

        final = attempts[-1]
        record = {
            "id": identifier,
            "model": model,
            "prediction": prediction,
            "raw_response": final["raw_response"],
            "error": final["error"],
            "attempt_count": len(attempts),
            "attempts": attempts,
        }
        with predictions_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
        records[identifier] = record
        print(
            f"{model} | {protocol} | {index}/{len(examples)} | {identifier} | "
            f"{'valid' if prediction is not None else 'invalid'} | attempts={len(attempts)}",
            flush=True,
        )

    report = {
        "model": model,
        "protocol": protocol,
        "selected": len(examples),
        "recorded": len(records),
        "valid_json": sum(item["prediction"] is not None for item in records.values()),
        "invalid_json": sum(item["prediction"] is None for item in records.values()),
        "cached_records": cached,
        "new_records": len(records) - cached,
        "total_model_calls": sum(item["attempt_count"] for item in records.values()),
        "complete_coverage": len(records) == len(examples),
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
