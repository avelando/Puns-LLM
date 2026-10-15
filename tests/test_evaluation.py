import json

import pytest
from test_zero_shot import FakeOllama, create_project

from puns_llm.evaluation.score import evaluate_run, score_example, token_overlap
from puns_llm.inference.zero_shot import run_zero_shot


def test_exact_location_with_wrong_alternative():
    gold = {
        "id": "1.1",
        "signs": [
            {
                "pun_sign": "nada",
                "alternative_signs": ["nada"],
                "homograph": True,
                "homophone": True,
            }
        ],
    }

    attempt = {
        "prediction": {
            "signs": [
                {
                    "pun_sign": "nada",
                    "alternative_signs": ["nothing"],
                    "homograph": True,
                    "homophone": False,
                }
            ]
        }
    }

    result = score_example(gold, attempt)

    assert result["location_exact"]
    assert result["matched_signs"][0]["alternative_correct"] is False
    assert result["matched_signs"][0]["homophone_correct"] is False
    assert result["end_to_end_correct"] is False


def test_extraneous_sign_penalizes_precision():
    gold = {
        "id": "1.1",
        "signs": [
            {
                "pun_sign": "nada",
                "alternative_signs": ["nada"],
                "homograph": True,
                "homophone": True,
            }
        ],
    }

    attempt = {
        "prediction": {
            "signs": [
                {
                    "pun_sign": "nada",
                    "alternative_signs": ["nada"],
                    "homograph": True,
                    "homophone": True,
                },
                {
                    "pun_sign": "ensina",
                    "alternative_signs": ["ensina"],
                    "homograph": True,
                    "homophone": True,
                },
            ]
        }
    }

    result = score_example(gold, attempt)

    assert result["exact_sign_matches"] == 1
    assert result["predicted_signs"] == 2
    assert result["location_exact"] is False
    assert result["end_to_end_correct"] is False


def test_partial_span_not_exact():
    assert token_overlap("nada", "o aluno nada") == pytest.approx(0.5)


def test_evaluation_writes_automatic_metrics(tmp_path):
    create_project(tmp_path)
    run_zero_shot(tmp_path, limit=2, client=FakeOllama())

    result = evaluate_run(
        tmp_path,
        "gemma3:27b",
        "zero_shot_v2",
        limit=2,
    )

    assert result["selected"] == 2
    assert result["scored"] == 2
    assert result["location"]["f1"] == pytest.approx(0.0)
    assert result["conditional_on_exact_location"]["alternative_accuracy"] == 0.0
    assert result["end_to_end_example_accuracy"] == 0.0

    assert (tmp_path / "results/evaluation/zero_shot_v2/gemma3_27b/details.jsonl").exists()


def test_evaluation_rejects_changed_dataset(tmp_path):
    create_project(tmp_path)
    run_zero_shot(tmp_path, client=FakeOllama())

    data_path = tmp_path / "data/processed/puns_only.json"
    data = json.loads(data_path.read_text())

    data[0]["text"] = "Changed text"
    data_path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="dataset does not match"):
        evaluate_run(tmp_path, "gemma3:27b", "zero_shot_v2")
