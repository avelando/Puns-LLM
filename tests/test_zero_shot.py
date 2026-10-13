import json

import pytest

from puns_llm.inference.zero_shot import run_zero_shot


class FakeOllama:
    def __init__(self):
        self.calls = []

    def model_digest(self, model):
        return "test-digest"

    def chat(self, model, messages, schema, options, think):
        self.calls.append(messages)

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


def create_project(tmp_path):
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "configs/experiments").mkdir(parents=True)
    (tmp_path / "configs/models").mkdir(parents=True)
    (tmp_path / "configs/prompts").mkdir(parents=True)

    examples = [
        {
            "id": "1.1",
            "split": "validation",
            "text": "Porque ele socorre.",
            "signs": [
                {
                    "pun_sign": "secret",
                    "alternative_signs": ["gold secret"],
                }
            ],
        },
        {
            "id": "1.2",
            "split": "validation",
            "text": "Porque o outro socorre.",
            "signs": [
                {
                    "pun_sign": "secret",
                    "alternative_signs": ["gold secret"],
                }
            ],
        },
    ]

    (tmp_path / "data/processed/validation_puns.json").write_text(
        json.dumps(examples),
        encoding="utf-8",
    )

    settings = {
        "model": "gemma3:27b",
        "split": "validation",
        "limit": 1,
        "host": "http://localhost:11434",
        "temperature": 0,
        "seed": 42,
        "num_ctx": 4096,
        "num_predict": 512,
        "timeout_seconds": 300,
    }

    (tmp_path / "configs/experiments/zero_shot.json").write_text(
        json.dumps(settings),
        encoding="utf-8",
    )

    (tmp_path / "configs/models/ollama.json").write_text(
        json.dumps({"models": {"gemma3:27b": {"think": None}}}),
        encoding="utf-8",
    )

    (tmp_path / "configs/prompts/zero_shot.txt").write_text(
        "Identify the pun.\n\n{text}",
        encoding="utf-8",
    )


def test_zero_shot_resume_without_gold_leakage(tmp_path):
    create_project(tmp_path)
    client = FakeOllama()

    first = run_zero_shot(tmp_path, client=client)
    second = run_zero_shot(tmp_path, limit=2, client=client)
    third = run_zero_shot(tmp_path, limit=2, client=client)

    assert first["successful"] == 1
    assert second["successful"] == 2
    assert third["total_attempts_in_file"] == 2
    assert len(client.calls) == 2
    assert "gold secret" not in json.dumps(client.calls)


def test_refuse_changed_configuration(tmp_path):
    create_project(tmp_path)
    client = FakeOllama()

    run_zero_shot(tmp_path, client=client)

    settings = tmp_path / "configs/experiments/zero_shot.json"
    data = json.loads(settings.read_text())
    data["temperature"] = 0.5
    settings.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="Run configuration changed"):
        run_zero_shot(tmp_path, client=client)
