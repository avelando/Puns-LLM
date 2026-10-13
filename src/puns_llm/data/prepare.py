import json
from collections import Counter, defaultdict
from pathlib import Path

SOURCES = ("train.jsonl", "validation.jsonl", "test.jsonl")


def load_puns(path: Path) -> dict[str, dict]:
    records = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(records, list):
        raise TypeError("puns.json must contain a list")

    indexed = {}

    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise TypeError("Invalid pun record")

        pun_id = record["id"]
        signs = record.get("signs")

        if pun_id in indexed or not isinstance(record.get("text"), str):
            raise ValueError(f"Duplicate ID or invalid text: {pun_id}")

        if not isinstance(signs, list) or not signs:
            raise ValueError(f"Missing signs: {pun_id}")

        for sign in signs:
            if not isinstance(sign, dict):
                raise TypeError(f"Invalid sign: {pun_id}")

            if not isinstance(sign.get("pun sign"), str):
                raise TypeError(f"Invalid pun sign: {pun_id}")

            if not isinstance(sign.get("alternative sign"), list):
                raise TypeError(f"Invalid alternative signs: {pun_id}")

            if not sign["alternative sign"] or not all(
                isinstance(value, str) and value.strip() for value in sign["alternative sign"]
            ):
                raise ValueError(f"Invalid alternative signs: {pun_id}")

            if type(sign.get("homograph")) is not bool or type(sign.get("homophone")) is not bool:
                raise ValueError(f"Invalid linguistic labels: {pun_id}")

        indexed[pun_id] = record

    return indexed


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                record = json.loads(line)

                if not isinstance(record, dict):
                    raise TypeError(f"Invalid record: {path}:{line_number}")

                yield record


def prepare_corpus(root: Path) -> dict:
    raw_dir = root / "data" / "raw"
    originals = load_puns(raw_dir / "puns.json")
    positives = {}

    for filename in SOURCES:
        for sample in load_jsonl(raw_dir / filename):
            source_id = sample.get("id")
            label = sample.get("label")

            if not isinstance(source_id, str) or type(label) is not int or label not in (0, 1):
                raise ValueError(f"Invalid example in {filename}")

            pun_id, separator, suffix = source_id.rpartition(".")

            if not separator or not pun_id or suffix not in ("H", "N"):
                raise ValueError(f"Invalid example ID: {source_id}")

            if (label == 1) != (suffix == "H"):
                raise ValueError(f"Inconsistent label: {source_id}")

            if label == 0:
                continue

            if pun_id in positives or pun_id not in originals:
                raise ValueError(f"Duplicate or unknown positive example: {pun_id}")

            if sample.get("text") != originals[pun_id]["text"]:
                raise ValueError(f"Text mismatch: {pun_id}")

            tokens = sample.get("tokens")
            labels = sample.get("labels")

            if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
                raise ValueError(f"Invalid tokens: {pun_id}")

            if not isinstance(labels, list) or len(tokens) != len(labels):
                raise ValueError(f"Invalid token labels: {pun_id}")

            if not all(type(value) is int and value in (0, 1) for value in labels):
                raise ValueError(f"Nonbinary token labels: {pun_id}")

            if not any(labels):
                raise ValueError(f"Missing positive token labels: {pun_id}")

            positives[pun_id] = {"tokens": tokens, "labels": labels}

    if positives.keys() != originals.keys():
        raise ValueError(f"Missing positive examples: {len(originals.keys() - positives.keys())}")

    processed = []
    categories = Counter()
    text_groups = defaultdict(list)

    for pun_id, original in originals.items():
        signs = []

        for sign in original["signs"]:
            signs.append(
                {
                    "pun_sign": sign["pun sign"],
                    "alternative_signs": sign["alternative sign"],
                    "homograph": sign["homograph"],
                    "homophone": sign["homophone"],
                }
            )

            categories[f"{int(sign['homograph'])}_{int(sign['homophone'])}"] += 1

        processed.append(
            {
                "id": pun_id,
                "text": original["text"],
                "tokens": positives[pun_id]["tokens"],
                "labels": positives[pun_id]["labels"],
                "signs": signs,
            }
        )

        text_groups[original["text"]].append(pun_id)

    report = {
        "total_puns": len(processed),
        "total_signs": sum(len(record["signs"]) for record in processed),
        "sign_categories": {key: categories[key] for key in ("0_0", "0_1", "1_0", "1_1")},
        "duplicate_text_groups": [ids for ids in text_groups.values() if len(ids) > 1],
    }

    output_dir = root / "data" / "processed"
    report_dir = root / "results" / "data_preparation"

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "puns_only.json").write_text(
        json.dumps(processed, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    (report_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return report
