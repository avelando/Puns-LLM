import json
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "validation", "test")


def load_puns(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(path)

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError("puns.json must contain a list")

    indexed = {}

    for item in data:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError("Invalid pun record")

        pun_id = item["id"]
        signs = item.get("signs")

        if pun_id in indexed or not isinstance(item.get("text"), str):
            raise ValueError(f"Duplicate ID or invalid text: {pun_id}")

        if not isinstance(signs, list) or not signs:
            raise ValueError(f"Missing signs: {pun_id}")

        for sign in signs:
            if not isinstance(sign, dict):
                raise ValueError(f"Invalid sign: {pun_id}")

            alternatives = sign.get("alternative sign")

            if not isinstance(sign.get("pun sign"), str):
                raise ValueError(f"Invalid pun sign: {pun_id}")

            if not isinstance(alternatives, list) or not alternatives:
                raise ValueError(f"Invalid alternative signs: {pun_id}")

            if not all(isinstance(value, str) for value in alternatives):
                raise ValueError(f"Invalid alternative sign value: {pun_id}")

            if type(sign.get("homograph")) is not bool or type(sign.get("homophone")) is not bool:
                raise ValueError(f"Invalid linguistic labels: {pun_id}")

        indexed[pun_id] = item

    return indexed


def load_jsonl(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                record = json.loads(line)

                if not isinstance(record, dict):
                    raise ValueError(f"Invalid record: {path}:{line_number}")

                yield record


def prepare_corpus(root: Path) -> dict:
    raw_dir = root / "data" / "raw"
    output_dir = root / "data" / "processed"
    report_dir = root / "results" / "data_preparation"

    originals = load_puns(raw_dir / "puns.json")
    processed = []
    seen = set()
    split_counts = {}
    categories = Counter()
    duplicate_texts = defaultdict(list)

    for split in SPLITS:
        counts = Counter()

        for sample in load_jsonl(raw_dir / f"{split}.jsonl"):
            source_id = sample.get("id")
            label = sample.get("label")

            if not isinstance(source_id, str) or type(label) is not int or label not in (0, 1):
                raise ValueError(f"Invalid example ID or label in {split}")

            pun_id, separator, suffix = source_id.rpartition(".")

            if not separator or not pun_id or suffix not in ("H", "N"):
                raise ValueError(f"Invalid example ID: {source_id}")

            if (label == 1) != (suffix == "H"):
                raise ValueError(f"Inconsistent label: {source_id}")

            counts["total"] += 1

            if label == 0:
                counts["non_puns"] += 1
                continue

            counts["puns"] += 1

            if pun_id in seen:
                raise ValueError(f"Duplicate positive example ID: {pun_id}")

            if pun_id not in originals:
                raise ValueError(f"Missing annotation: {pun_id}")

            original = originals[pun_id]

            if sample.get("text") != original["text"]:
                raise ValueError(f"Text mismatch: {pun_id}")

            tokens = sample.get("tokens")
            labels = sample.get("labels")

            if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
                raise ValueError(f"Invalid tokens: {pun_id}")

            if not isinstance(labels, list) or len(tokens) != len(labels):
                raise ValueError(f"Invalid token labels: {pun_id}")

            if not all(type(value) is int and value in (0, 1) for value in labels):
                raise ValueError(f"Nonbinary token labels: {pun_id}")

            if not any(labels):
                raise ValueError(f"Missing positive token labels: {pun_id}")

            signs = []

            for sign in original["signs"]:
                signs.append({
                    "pun_sign": sign["pun sign"],
                    "alternative_signs": sign["alternative sign"],
                    "homograph": sign["homograph"],
                    "homophone": sign["homophone"],
                })

                categories[f"{int(sign['homograph'])}_{int(sign['homophone'])}"] += 1

            record = {
                "id": pun_id,
                "source_id": source_id,
                "split": split,
                "text": original["text"],
                "tokens": tokens,
                "labels": labels,
                "signs": signs,
            }

            processed.append(record)
            duplicate_texts[record["text"]].append({"id": pun_id, "split": split})
            seen.add(pun_id)

        split_counts[split] = dict(counts)

    if seen != originals.keys():
        raise ValueError(f"Missing positive examples: {len(originals.keys() - seen)}")

    duplicates = [entries for entries in duplicate_texts.values() if len(entries) > 1]

    report = {
        "total_puns": len(processed),
        "total_signs": sum(len(item["signs"]) for item in processed),
        "splits": split_counts,
        "sign_categories": {
            key: categories.get(key, 0)
            for key in ("0_0", "0_1", "1_0", "1_1")
        },
        "duplicate_text_groups": duplicates,
    }

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
