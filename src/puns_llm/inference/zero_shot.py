import hashlib
import json
from pathlib import Path

from puns_llm.inference.ollama import OllamaClient
from puns_llm.inference.schema import PREDICTION_SCHEMA, build_messages, parse_prediction

PROTOCOLS = {"zero_shot_v2": "zero_shot_v2.txt", "one_shot": "one_shot.txt"}


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
    limit: int | None = None,
    host: str | None = None,
    client: OllamaClient | None = None,
    protocol: str | None = None,
) -> dict:
    settings = load_json(root / "configs/experiments/zero_shot.json")
    models = load_json(root / "configs/models/ollama.json")["models"]

    model = model or settings["model"]
    host = host or settings["host"]
    limit = settings["limit"] if limit is None else limit
    protocol = protocol or settings["protocol"]

    if model not in models:
        raise ValueError(f"Unconfigured model: {model}")

    if protocol not in PROTOCOLS:
        raise ValueError(f"Unsupported inference protocol: {protocol}")

    if type(limit) is not int or limit < 0:
        raise ValueError("Limit must be a nonnegative integer")

    data_path = root / "data/processed/puns_only.json"
    dataset = load_json(data_path)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Processed dataset must be a nonempty list")

    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("id"), str)
        or not isinstance(item.get("text"), str)
        for item in dataset
    ):
        raise ValueError("Invalid processed example")

    identifiers = [item["id"] for item in dataset]

    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate IDs in processed dataset")

    examples = dataset[:limit] if limit else dataset
    instructions = (root / "configs/prompts" / PROTOCOLS[protocol]).read_text(encoding="utf-8")

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
        "protocol": protocol,
        "model": model,
        "model_digest": digest,
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(instructions.encode("utf-8")).hexdigest(),
        "schema_sha256": fingerprint(PREDICTION_SCHEMA),
        "options": options,
        "think": think,
    }

    output_dir = root / "results" / protocol / model.replace(":", "_")
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

    attempts = read_predictions(predictions_path)
    completed = {item["id"] for item in attempts if item.get("prediction") is not None}

    cached_valid = sum(item["id"] in completed for item in examples)
    new_attempts = 0

    print(f"{model} | {protocol} | selected={len(examples)} | cached={cached_valid}")

    with predictions_path.open("a", encoding="utf-8") as stream:
        for sample in examples:
            if sample["id"] in completed:
                continue

            response = client.chat(
                model,
                build_messages(sample["text"], instructions),
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
            new_attempts += 1

            status = "valid_json" if error is None else "invalid_json"
            print(f"{model} | {sample['id']} | {status}")

    attempts = read_predictions(predictions_path)
    selected_ids = {item["id"] for item in examples}

    successful_ids = {
        item["id"]
        for item in attempts
        if item["id"] in selected_ids and item.get("prediction") is not None
    }

    last_attempts = {item["id"]: item for item in attempts}

    invalid = sum(
        last_attempts[item_id].get("prediction") is None
        for item_id in selected_ids - successful_ids
        if item_id in last_attempts
    )

    report = {
        "model": model,
        "protocol": protocol,
        "selected": len(examples),
        "valid_json": len(successful_ids),
        "cached_valid": cached_valid,
        "new_attempts": new_attempts,
        "invalid_json": invalid,
        "remaining": len(selected_ids - successful_ids),
        "total_attempts_in_file": len(attempts),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return report
