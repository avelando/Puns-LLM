import json

from test_zero_shot import create_project

from puns_llm.evaluation.official import evaluate_official
from puns_llm.inference.official import run_official


class RepeatingOllama:
    def __init__(self, failures):
        self.failures = failures
        self.calls = 0

    def model_digest(self, model):
        return "test-digest"

    def chat(self, model, messages, schema, options, think):
        self.calls += 1
        if self.calls <= self.failures:
            return {"message": {"content": '{"signs": ['}, "eval_count": 512}
        return {
            "message": {
                "content": json.dumps(
                    {
                        "signs": [
                            {
                                "pun_sign": "socorre",
                                "alternative_signs": ["só corre"],
                                "homograph": False,
                                "homophone": True,
                            }
                        ]
                    }
                )
            },
            "eval_count": 25,
        }


def fixture(tmp_path):
    create_project(tmp_path)
    (tmp_path / "configs/experiments/official.json").write_text(
        json.dumps({"max_invalid_retries": 2})
    )


def test_invalid_retried_without_duplicate_rows(tmp_path):
    fixture(tmp_path)
    client = RepeatingOllama(failures=2)

    first = run_official(tmp_path, "gemma3:27b", "zero_shot_v2", client=client)
    assert first["selected"] == 2
    assert first["recorded"] == 2
    assert first["valid_json"] == 2
    assert client.calls == 4

    path = tmp_path / "results/official/gemma3_27b/zero_shot_v2/predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == len({item["id"] for item in rows}) == 2
    assert rows[0]["attempt_count"] == 3
    assert [item["error"] is None for item in rows[0]["attempts"]] == [False, False, True]

    again = run_official(tmp_path, "gemma3:27b", "zero_shot_v2", client=client)
    assert again["cached_records"] == 2
    assert again["new_records"] == 0
    assert client.calls == 4

    report = evaluate_official(tmp_path, "gemma3:27b", "zero_shot_v2")
    assert report["recorded"] == 2
    assert report["processed_all_puns"] is True
    assert report["total_model_calls"] == 4


def test_exhausted_retries_counted_as_invalid(tmp_path):
    fixture(tmp_path)
    client = RepeatingOllama(failures=100)

    summary = run_official(tmp_path, "gemma3:27b", "zero_shot_v2", client=client)
    assert summary["selected"] == 2
    assert summary["recorded"] == 2
    assert summary["valid_json"] == 0
    assert summary["invalid_json"] == 2
    assert summary["total_model_calls"] == 6

    report = evaluate_official(tmp_path, "gemma3:27b", "zero_shot_v2")
    assert report["invalid_json"] == 2
    assert report["processed_all_puns"] is True
    assert report["complete"] is False


def test_model_protocol_separation(tmp_path):
    fixture(tmp_path)
    first = RepeatingOllama(failures=0)
    second = RepeatingOllama(failures=0)
    run_official(tmp_path, "gemma3:27b", "zero_shot_v2", client=first)
    (tmp_path / "configs/prompts/one_shot.txt").write_text("Example: token\n\n{text}")
    run_official(tmp_path, "gemma3:27b", "one_shot", client=second)
    assert (tmp_path / "results/official/gemma3_27b/zero_shot_v2/metadata.json").exists()

    assert (tmp_path / "results/official/gemma3_27b/one_shot/metadata.json").exists()
