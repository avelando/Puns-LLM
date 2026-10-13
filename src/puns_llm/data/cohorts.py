import hashlib
import json
from collections import Counter
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_cohorts(root: Path, size: int = 48, seed: int = 42) -> dict:
    if not 16 <= size <= 200:
        raise ValueError("Pilot size must be between 16 and 200")
    path = root / "configs/experiments/cohorts.json"
    if path.exists():
        raise FileExistsError(f"Selection already exists: {path}")
    dataset_path = root / "data/processed/puns_only.json"
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    development_ids = set()
    for protocol in ("zero_shot", "zero_shot_v2", "one_shot"):
        predictions = root / "results" / protocol / "gemma3_27b/predictions.jsonl"
        if predictions.exists():
            with predictions.open(encoding="utf-8") as stream:
                development_ids.update(json.loads(line)["id"] for line in stream if line.strip())
    by_id = {item["id"]: item for item in data}
    if len(by_id) != len(data):
        raise ValueError("Duplicate dataset identifiers")
    if not development_ids.issubset(by_id):
        raise ValueError("Development IDs do not belong to corpus")
    development_texts = {" ".join(by_id[key]["text"].casefold().split()) for key in development_ids}
    eligible = [
        item for item in data if " ".join(item["text"].casefold().split()) not in development_texts
    ]

    def order(item):
        return digest(f"{seed}:{item['id']}".encode("utf-8"))

    eligible.sort(key=order)
    selected = []
    chosen_texts = set()

    def choose(predicate, count):
        for item in eligible:
            key = " ".join(item["text"].casefold().split())
            if len(selected) >= size or count <= 0:
                break
            if item["id"] in selected or key in chosen_texts or not predicate(item):
                continue
            selected.append(item["id"])
            chosen_texts.add(key)
            count -= 1

    choose(
        lambda item: any(sign["homograph"] and not sign["homophone"] for sign in item["signs"]), 2
    )
    choose(lambda item: len(item["signs"]) > 1, 6)
    for homograph, homophone in ((False, False), (False, True), (True, True)):
        choose(
            lambda item, h=homograph, p=homophone: any(
                sign["homograph"] == h and sign["homophone"] == p for sign in item["signs"]
            ),
            max(3, size // 4),
        )
    choose(lambda item: True, size)
    if len(selected) != size:
        raise ValueError("Not enough eligible examples for the selected pilot size")
    official = [
        item["id"]
        for item in data
        if item["id"] not in set(selected) | development_ids
        and " ".join(item["text"].casefold().split()) not in chosen_texts | development_texts
    ]
    distribution = Counter()
    for identifier in selected:
        item = by_id[identifier]
        distribution["multiple_signs" if len(item["signs"]) > 1 else "single_sign"] += 1
        for sign in item["signs"]:
            distribution[f"{int(sign['homograph'])}_{int(sign['homophone'])}"] += 1
    result = {
        "version": 1,
        "seed": seed,
        "dataset_sha256": digest(dataset_path.read_bytes()),
        "development_ids": sorted(development_ids),
        "pilot_ids": selected,
        "official_ids": official,
        "pilot_sign_categories": dict(sorted(distribution.items())),
        "excluded_from_official": len(data) - len(official),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def cohort_ids(root: Path, cohort: str) -> list[str]:
    if cohort not in ("pilot", "official"):
        raise ValueError("Cohort must be pilot or official")
    file = root / "configs/experiments/cohorts.json"
    payload = json.loads(file.read_text(encoding="utf-8"))
    dataset = root / "data/processed/puns_only.json"
    if payload["dataset_sha256"] != digest(dataset.read_bytes()):
        raise ValueError("Cohort selection does not match the dataset")
    ids = payload[f"{cohort}_ids"]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Empty or duplicate cohort IDs")
    return ids
