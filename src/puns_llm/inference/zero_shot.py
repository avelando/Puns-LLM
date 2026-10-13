import hashlib
import json
from pathlib import Path

from puns_llm.inference.ollama import OllamaClient
from puns_llm.inference.schema import (
    PREDICTION_SCHEMA,
    build_messages,
    parse_prediction,
)

SPLITS = {"train", "validation", "test"}


def fingerprint(value: object) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def read_predictions(path: Path) -> list[dict]:
    if not path.exists():
        return []

    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def run_zero_shot(
    root: Path,
    model: str | None = None,
    split: str | None = None,
    limit: int | None = None,
    host: str | None = None,
    client: OllamaClient | None = None,
) -> dict:
    settings = load_json(root / "configs/experiments/zero_shot.json")
    models = load_json(root / "configs/models/ollama.json")["models"]

    model = model or settings["model"]
    split = split or settings["split"]
    host = host or settings["host"]
    limit = settings["limit"] if limit is None else limit

    if model not in models:
        raise ValueError(f"Unconfigured model: {model}")

    if split not in SPLITS:
        raise ValueError(f"Invalid split: {split}")

    if type(limit) is not int or limit < 0:
        raise ValueError("Limit must be a nonnegative integer")

    data_path = root / "data" / "processed" / f"{split}_puns.json"
    dataset = load_json(data_path)

    if not isinstance(dataset, list) or any(
        not isinstance(item, dict) or item.get("split") != split for item in dataset
    ):
        raise ValueError(f"Invalid processed split: {split}")

    examples = dataset

    if not examples:
        raise ValueError(f"No examples in split: {split}")

    if limit:
        examples = examples[:limit]

    instructions = (root / "configs/prompts/zero_shot.txt").read_text(encoding="utf-8")

    options = {
        "temperature": settings["temperature"],
        "seed": settings["seed"],
        "num_ctx": settings["num_ctx"],
        "num_predict": settings["num_predict"],
    }

    think = models[model]["think"]
    client = client or OllamaClient(host, settings["timeout_seconds"])
    digest = client.model_digest(model)

    profile = {
        "backend": "ollama",
        "model": model,
        "model_digest": digest,
        "split": split,
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(instructions.encode("utf-8")).hexdigest(),
        "schema_sha256": fingerprint(PREDICTION_SCHEMA),
        "options": options,
        "think": think,
    }

    output_dir = root / "results/zero_shot" / model.replace(":", "_") / split

    metadata_path = output_dir / "metadata.json"
    predictions_path = output_dir / "predictions.jsonl"

    if metadata_path.exists():
        if load_json(metadata_path) != profile:
            raise ValueError(f"Run configuration changed: {metadata_path}")
    elif predictions_path.exists():
        raise ValueError(f"Missing metadata for existing predictions: {predictions_path}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    completed = {
        record["id"]
        for record in read_predictions(predictions_path)
        if record.get("prediction") is not None
    }

    with predictions_path.open("a", encoding="utf-8") as stream:
        for sample in examples:
            if sample["id"] in completed:
                continue

            messages = build_messages(sample["text"], instructions)
            response = client.chat(
                model,
                messages,
                PREDICTION_SCHEMA,
                options,
                think,
            )

            content = response["message"]["content"]

            try:
                prediction = parse_prediction(content)
                error = None
                completed.add(sample["id"])
            except ValueError as exception:
                prediction = None
                error = str(exception)

            record = {
                "id": sample["id"],
                "split": split,
                "model": model,
                "prediction": prediction,
                "raw_response": content,
                "error": error,
                "prompt_eval_count": response.get("prompt_eval_count"),
                "eval_count": response.get("eval_count"),
                "total_duration": response.get("total_duration"),
            }

            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()

            status = "ok" if error is None else "invalid"
            print(f"{model} | {split} | {sample['id']} | {status}")

    attempts = read_predictions(predictions_path)
    last_attempts = {item["id"]: item for item in attempts}

    selected_ids = {item["id"] for item in examples}

    successful_ids = {
        item["id"]
        for item in attempts
        if item["id"] in selected_ids and item["prediction"] is not None
    }

    invalid_count = sum(
        last_attempts[item_id]["prediction"] is None
        for item_id in selected_ids & last_attempts.keys()
        if item_id not in successful_ids
    )

    report = {
        "model": model,
        "split": split,
        "selected": len(examples),
        "successful": len(successful_ids),
        "invalid": invalid_count,
        "remaining": len(selected_ids - successful_ids),
        "total_attempts_in_file": len(attempts),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return report
