import json

import pytest

from puns_llm.inference.schema import build_messages, parse_prediction


def test_parse_multiple_signs():
    output = {
        "signs": [
            {
                "pun_sign": "socorre",
                "alternative_signs": ["só corre"],
                "homograph": False,
                "homophone": True,
            },
            {
                "pun_sign": "nada",
                "alternative_signs": ["nada"],
                "homograph": True,
                "homophone": True,
            },
        ]
    }

    assert parse_prediction(json.dumps(output)) == output


def test_reject_invalid_labels():
    invalid = {
        "signs": [
            {
                "pun_sign": "socorre",
                "alternative_signs": ["só corre"],
                "homograph": "false",
                "homophone": True,
            }
        ]
    }

    with pytest.raises(ValueError, match="linguistic labels"):
        parse_prediction(json.dumps(invalid))


def test_prompt_excludes_gold_annotations():
    messages = build_messages(
        "Porque ele socorre.",
        "Analyze Portuguese wordplay.\n\n{text}",
    )

    assert messages[0]["content"] == "Analyze Portuguese wordplay.\n\nPorque ele socorre."
    assert "só corre" not in json.dumps(messages, ensure_ascii=False)
    assert len(messages) == 1


def test_prompt_requires_placeholder():
    with pytest.raises(ValueError, match="placeholder"):
        build_messages("Pun text", "Identify the pun")
