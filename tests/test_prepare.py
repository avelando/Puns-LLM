import json

import pytest

from puns_llm.data.prepare import prepare_corpus


def create_fixture(root):
    raw = root / "data/raw"
    raw.mkdir(parents=True)

    originals = [
        {
            "id": "1.1",
            "text": "A word.",
            "signs": [
                {
                    "pun sign": "word",
                    "alternative sign": ["word", "other"],
                    "homograph": True,
                    "homophone": False,
                }
            ],
        },
        {
            "id": "1.2",
            "text": "Another pun.",
            "signs": [
                {
                    "pun sign": "pun",
                    "alternative sign": ["fun"],
                    "homograph": False,
                    "homophone": False,
                }
            ],
        },
    ]

    (raw / "puns.json").write_text(
        json.dumps(originals),
        encoding="utf-8",
    )

    positive = {
        "id": "1.1.H",
        "text": "A word.",
        "label": 1,
        "tokens": ["A", "word", "."],
        "labels": [0, 1, 0],
    }

    negative = {
        "id": "1.1.N",
        "text": "A sentence.",
        "label": 0,
        "tokens": ["A", "sentence", "."],
        "labels": [0, 0, 0],
    }

    second = {
        "id": "1.2.H",
        "text": "Another pun.",
        "label": 1,
        "tokens": ["Another", "pun", "."],
        "labels": [0, 1, 0],
    }

    (raw / "train.jsonl").write_text(
        "\n".join(json.dumps(item) for item in (positive, negative)) + "\n",
        encoding="utf-8",
    )

    (raw / "validation.jsonl").write_text(
        json.dumps(second) + "\n",
        encoding="utf-8",
    )

    (raw / "test.jsonl").write_text("", encoding="utf-8")

    return raw


def test_prepare_puns_only(tmp_path):
    create_fixture(tmp_path)

    report = prepare_corpus(tmp_path)
    output = tmp_path / "data/processed"
    records = json.loads((output / "puns_only.json").read_text())

    assert report["total_puns"] == 2
    assert report["total_signs"] == 2
    assert report["sign_categories"]["1_0"] == 1
    assert len(records) == 2
    assert records[0]["id"] == "1.1"
    assert records[1]["id"] == "1.2"
    assert records[0]["labels"] == [0, 1, 0]
    assert records[0]["signs"][0]["alternative_signs"] == ["word", "other"]
    assert all("split" not in item for item in records)
    assert sorted(path.name for path in output.glob("*.json")) == ["puns_only.json"]


def test_reject_mismatched_text(tmp_path):
    raw = create_fixture(tmp_path)
    path = raw / "train.jsonl"

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["text"] = "Different text."

    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises(ValueError, match="Text mismatch"):
        prepare_corpus(tmp_path)

    assert not (tmp_path / "data/processed/puns_only.json").exists()
