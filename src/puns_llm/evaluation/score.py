import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def token_overlap(gold: str, predicted: str) -> float:
    gold_tokens = Counter(re.findall(r"\w+", normalize(gold)))
    predicted_tokens = Counter(re.findall(r"\w+", normalize(predicted)))

    if not gold_tokens or not predicted_tokens:
        return 0.0

    shared = sum((gold_tokens & predicted_tokens).values())
    gold_count = sum(gold_tokens.values())
    predicted_count = sum(predicted_tokens.values())
    return 2 * shared / (gold_count + predicted_count)


def exact_pairs(gold_signs: list[dict], predicted_signs: list[dict]) -> list[tuple[int, int]]:
    positions = defaultdict(list)

    for index, sign in enumerate(predicted_signs):
        positions[normalize(sign["pun_sign"])].append(index)

    matches = []

    for index, sign in enumerate(gold_signs):
        candidates = positions[normalize(sign["pun_sign"])]

        if candidates:
            matches.append((index, candidates.pop(0)))

    return matches


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def score_example(gold: dict, attempt: dict) -> dict:
    gold_signs = gold["signs"]
    prediction = attempt.get("prediction")
    predicted_signs = prediction["signs"] if prediction is not None else []
    matches = exact_pairs(gold_signs, predicted_signs)
    matches_detail = []

    for gold_index, predicted_index in matches:
        reference = gold_signs[gold_index]
        predicted = predicted_signs[predicted_index]
        references = {normalize(value) for value in reference["alternative_signs"]}
        alternatives = {normalize(value) for value in predicted["alternative_signs"]}

        matches_detail.append(
            {
                "pun_sign": reference["pun_sign"],
                "alternative_correct": bool(references & alternatives),
                "homograph_correct": reference["homograph"] == predicted["homograph"],
                "homophone_correct": reference["homophone"] == predicted["homophone"],
                "gold_category": f"{int(reference['homograph'])}_{int(reference['homophone'])}",
                "predicted_category": f"{int(predicted['homograph'])}_{int(predicted['homophone'])}",
            }
        )

    location_complete = len(matches) == len(gold_signs) == len(predicted_signs)
    end_to_end = location_complete and all(
        item["alternative_correct"] and item["homograph_correct"] and item["homophone_correct"]
        for item in matches_detail
    )

    overlaps = [
        max(
            (
                token_overlap(sign["pun_sign"], candidate["pun_sign"])
                for candidate in predicted_signs
            ),
            default=0.0,
        )
        for sign in gold_signs
    ]

    return {
        "id": gold["id"],
        "valid_json": prediction is not None,
        "error": attempt.get("error"),
        "gold_signs": len(gold_signs),
        "predicted_signs": len(predicted_signs),
        "exact_sign_matches": len(matches),
        "location_exact": location_complete,
        "token_overlap_gold_sum": sum(overlaps),
        "matched_signs": matches_detail,
        "end_to_end_correct": end_to_end,
    }


def evaluate_run(
    root: Path,
    model: str,
    protocol: str = "zero_shot_v2",
    limit: int = 0,
) -> dict:
    if type(limit) is not int or limit < 0:
        raise ValueError("Limit must be a nonnegative integer")

    if protocol not in ("zero_shot", "zero_shot_v2", "one_shot"):
        raise ValueError(f"Unknown protocol: {protocol}")

    dataset_path = root / "data/processed/puns_only.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    selected = dataset[:limit] if limit else dataset

    result_dir = root / "results" / protocol / model.replace(":", "_")
    metadata = json.loads((result_dir / "metadata.json").read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

    if metadata.get("dataset_sha256") != expected_hash:
        raise ValueError("The dataset does not match the recorded inference run")

    attempts = {}
    predictions_path = result_dir / "predictions.jsonl"

    with predictions_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue

            record = json.loads(line)
            identifier = record["id"]

            if identifier not in attempts or record.get("prediction") is not None:
                attempts[identifier] = record

    details = [
        score_example(item, attempts[item["id"]]) for item in selected if item["id"] in attempts
    ]

    matched_details = [match for detail in details for match in detail["matched_signs"]]

    gold_total = sum(item["gold_signs"] for item in details)
    predicted_total = sum(item["predicted_signs"] for item in details)
    matched_total = sum(item["exact_sign_matches"] for item in details)

    precision = matched_total / predicted_total if predicted_total else 0.0
    recall = matched_total / gold_total if gold_total else 0.0

    alt_correct = sum(item["alternative_correct"] for item in matched_details)
    homograph_correct = sum(item["homograph_correct"] for item in matched_details)
    homophone_correct = sum(item["homophone_correct"] for item in matched_details)

    joint_correct = sum(
        item["alternative_correct"] and item["homograph_correct"] and item["homophone_correct"]
        for item in matched_details
    )

    category_support = Counter(item["gold_category"] for item in matched_details)
    category_predictions = Counter(item["predicted_category"] for item in matched_details)

    category_correct = Counter(
        item["gold_category"]
        for item in matched_details
        if item["gold_category"] == item["predicted_category"]
    )

    categories = ("0_0", "0_1", "1_0", "1_1")
    category_f1 = {}

    for category in categories:
        correct = category_correct[category]
        class_precision = (
            correct / category_predictions[category] if category_predictions[category] else 0.0
        )
        class_recall = correct / category_support[category] if category_support[category] else 0.0
        category_f1[category] = f1(class_precision, class_recall)

    scored = len(details)
    matched = len(matched_details)

    report = {
        "model": model,
        "protocol": protocol,
        "selected": len(selected),
        "scored": scored,
        "missing": len(selected) - scored,
        "valid_json": sum(item["valid_json"] for item in details),
        "invalid_json": sum(not item["valid_json"] for item in details),
        "location": {
            "gold_signs": gold_total,
            "predicted_signs": predicted_total,
            "exact_matches": matched_total,
            "precision": precision,
            "recall": recall,
            "f1": f1(precision, recall),
            "example_exact_match_accuracy": (
                sum(item["location_exact"] for item in details) / scored if scored else 0.0
            ),
            "mean_best_token_overlap_per_gold_sign": (
                sum(item["token_overlap_gold_sum"] for item in details) / gold_total
                if gold_total
                else 0.0
            ),
        },
        "conditional_on_exact_location": {
            "matched_signs": matched,
            "alternative_accuracy": alt_correct / matched if matched else 0.0,
            "homograph_accuracy": homograph_correct / matched if matched else 0.0,
            "homophone_accuracy": homophone_correct / matched if matched else 0.0,
            "joint_accuracy": joint_correct / matched if matched else 0.0,
            "category_macro_f1_supported": (
                sum(category_f1[category] for category in categories if category_support[category])
                / sum(bool(category_support[category]) for category in categories)
                if matched
                else 0.0
            ),
            "category_f1": category_f1,
            "category_support": {category: category_support[category] for category in categories},
        },
        "end_to_end_example_accuracy": (
            sum(item["end_to_end_correct"] for item in details) / scored if scored else 0.0
        ),
    }

    output_dir = root / "results/evaluation" / protocol / model.replace(":", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    (output_dir / "details.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in details),
        encoding="utf-8",
    )

    return report
