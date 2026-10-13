import json

import pytest

from puns_llm.data.prepare import prepare_corpus


def create_fixture(root):
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)

    originals = [{
        "id": "1.1",
        "text": "A word.",
        "signs": [{
            "pun sign": "word",
            "alternative sign": ["word", "other"],
            "homograph": True,
            "homophone": False,
        }],
    }]

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

    (raw / "train.jsonl").write_text(
        "\n".join(json.dumps(item) for item in [positive, negative]) + "\n",
        encoding="utf-8",
    )

    for split in ("validation", "test"):
        (raw / f"{split}.jsonl").write_text("", encoding="utf-8")

    return raw


def test_prepare_puns_only(tmp_path):
    create_fixture(tmp_path)

    report = prepare_corpus(tmp_path)
    records = json.loads(
        (tmp_path / "data/processed/puns_only.json").read_text()
    )

    assert report["total_puns"] == 1
    assert report["total_signs"] == 1
    assert report["splits"]["train"]["non_puns"] == 1
    assert report["sign_categories"]["1_0"] == 1
    assert records[0]["split"] == "train"
    assert records[0]["source_id"] == "1.1.H"
    assert records[0]["labels"] == [0, 1, 0]
    assert records[0]["signs"][0]["alternative_signs"] == ["word", "other"]


def test_reject_mismatched_text(tmp_path):
    raw = create_fixture(tmp_path)
    path = raw / "train.jsonl"

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["text"] = "Different text."

    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Text mismatch"):
        prepare_corpus(tmp_path)

    assert not (tmp_path / "data/processed/puns_only.json").exists()
