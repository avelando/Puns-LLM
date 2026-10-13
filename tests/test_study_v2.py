import json

import pytest
from test_zero_shot import FakeOllama, create_project

from puns_llm.data.cohorts import cohort_ids, make_cohorts
from puns_llm.evaluation.score_v2 import evaluate_v2, partial_pairs, score_item, summarize
from puns_llm.inference.zero_shot import run_zero_shot
from puns_llm.utils.freeze import freeze_methodology, verify_freeze


def make_gold():
    return {
        "id": "a",
        "text": "Ela socorre.",
        "signs": [
            {
                "pun_sign": "socorre",
                "alternative_signs": ["só corre"],
                "homograph": False,
                "homophone": True,
            }
        ],
    }


def test_alt_precision_penalizes_extras():
    reference = make_gold()
    predicted = {
        "prediction": {
            "signs": [
                {
                    "pun_sign": "socorre",
                    "alternative_signs": ["só corre", "SOCORRE", "ele corre"],
                    "homograph": False,
                    "homophone": True,
                }
            ]
        }
    }
    result = score_item(reference, predicted, False)
    report = summarize([result], "fake", "zero_shot_v2")
    alt = report["conditional_on_exact_location"]["alternative"]
    assert alt["precision"] == pytest.approx(1 / 3)
    assert alt["recall"] == 1
    assert alt["f1"] == pytest.approx(0.5)
    assert report["end_to_end"]["strict_set_accuracy"] == 0


def test_one_to_one_tokens():
    references = [
        {"pun_sign": "meio ambiente"},
        {"pun_sign": "meio"},
    ]
    predictions = [{"pun_sign": "meio ambiente"}]
    assert len(partial_pairs(references, predictions)) == 1


def test_missing_counted_as_false_negative():
    result = score_item(make_gold(), None, False)
    report = summarize([result], "fake", "one_shot")
    assert report["missing"] == 1
    assert report["location_exact"]["fn"] == 1
    assert report["complete"] is False


def test_oracle_sends_only_gold_signs(tmp_path):
    create_project(tmp_path)
    data_path = tmp_path / "data/processed/puns_only.json"
    samples = json.loads(data_path.read_text())
    samples[0]["signs"][0]["pun_sign"] = "socorre"
    data_path.write_text(json.dumps(samples))
    path = tmp_path / "configs/prompts/oracle_zero_shot_v2.txt"
    path.write_text("Text: {text} Gold: {pun_signs}")
    client = FakeOllama()
    run_zero_shot(tmp_path, client=client, protocol="oracle_zero_shot_v2")
    call = json.dumps(client.calls, ensure_ascii=False)
    assert "socorre" in call
    assert "gold secret" not in call
    assert "homophone" not in call
    assert "homograph" not in call


def test_old_runs_preserved_by_v2(tmp_path):
    create_project(tmp_path)
    run_zero_shot(tmp_path, client=FakeOllama())
    old = tmp_path / "results/zero_shot_v2/gemma3_27b/predictions.jsonl"
    old_bytes = old.read_bytes()
    report = evaluate_v2(tmp_path, "gemma3:27b", "zero_shot_v2", limit=2)
    assert report["complete"] is False
    assert report["missing"] == 1
    assert old.read_bytes() == old_bytes


def test_pilot_excludes_dev_and_includes_rare_cases(tmp_path):
    (tmp_path / "data/processed").mkdir(parents=True)
    rows = []
    for i in range(120):
        rare = i >= 110
        rows.append(
            {
                "id": f"a.{i}",
                "text": f"Text {i}",
                "signs": [
                    {
                        "pun_sign": "x",
                        "alternative_signs": ["x"],
                        "homograph": rare,
                        "homophone": False,
                    }
                ],
            }
        )
    path = tmp_path / "data/processed/puns_only.json"
    path.write_text(json.dumps(rows))
    selection = make_cohorts(tmp_path, size=20, seed=42)
    assert len(selection["pilot_ids"]) == 20
    assert len(selection["official_ids"]) == 100
    assert selection["pilot_sign_categories"]["1_0"] == 2
    assert cohort_ids(tmp_path, "pilot") == selection["pilot_ids"]
    with pytest.raises(FileExistsError):
        make_cohorts(tmp_path, size=20)


class FakeDigests:
    def model_digest(self, name):
        return f"sha-for-{name}"


def test_freeze_detects_prompt_mutation(tmp_path):
    from puns_llm.utils.freeze import FILES

    for filename in FILES:
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    (tmp_path / "configs/models/ollama.json").write_text(
        json.dumps({"models": {"gemma3:27b": {}, "qwen3:32b": {}}})
    )
    freeze_methodology(tmp_path, FakeDigests())
    verify_freeze(tmp_path, FakeDigests())
    (tmp_path / "configs/prompts/one_shot.txt").write_text("changed")
    with pytest.raises(ValueError, match="Frozen methodology differs"):
        verify_freeze(tmp_path, FakeDigests())


def test_token_labels_are_used():
    from puns_llm.evaluation.score_v2 import labeled_token_counts

    reference = {
        "text": "Você ensina, e o aluno nada.",
        "tokens": ["Você", "ensina", ",", "e", "o", "aluno", "nada", "."],
        "labels": [0, 0, 0, 0, 0, 0, 1, 0],
    }
    assert labeled_token_counts(reference, [{"pun_sign": "o aluno nada"}]) == (1, 2, 0)
    assert labeled_token_counts(reference, [{"pun_sign": "nada"}]) == (1, 0, 0)


class FailedOllama(FakeOllama):
    def chat(self, model, messages, schema, options, think):
        raise RuntimeError("Model disconnected")


def test_backend_failure_is_recorded_before_raising(tmp_path):
    create_project(tmp_path)
    with pytest.raises(RuntimeError, match="Backend failure saved"):
        run_zero_shot(tmp_path, client=FailedOllama())
    path = tmp_path / "results/zero_shot_v2/gemma3_27b/predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["prediction"] is None
    assert "Model disconnected" in rows[0]["error"]
