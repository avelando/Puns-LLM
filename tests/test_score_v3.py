def test_repeated_gold_sign_marks_every_occurrence():
    from puns_llm.evaluation.score_v3 import labeled_token_counts

    example = {
        "text": "Ele é maduro, maduro, maduro.",
        "tokens": ["Ele", "é", "maduro", ",", "maduro", ",", "maduro", "."],
        "labels": [0, 0, 1, 0, 1, 0, 1, 0],
        "signs": [{"pun_sign": "maduro"}],
    }
    assert labeled_token_counts(example, example["signs"]) == (3, 0, 0)


def test_token_evaluation_skips_ambiguous_reference():
    from puns_llm.evaluation.score_v3 import labeled_token_counts

    example = {
        "text": "A pata olha para a pata.",
        "tokens": ["A", "pata", "olha", "para", "a", "pata", "."],
        "labels": [0, 0, 0, 0, 0, 1, 0],
        "signs": [{"pun_sign": "pata"}],
    }
    assert labeled_token_counts(example, example["signs"]) is None


def test_duplicate_pun_signs_are_matched_by_interpretation():
    from puns_llm.evaluation.score_v3 import score_item

    example = {
        "id": "4.681",
        "text": "O animal doméxico.",
        "tokens": ["O", "animal", "doméxico", "."],
        "labels": [0, 0, 1, 0],
        "signs": [
            {
                "pun_sign": "doméxico",
                "alternative_signs": ["do México"],
                "homograph": False,
                "homophone": True,
            },
            {
                "pun_sign": "doméxico",
                "alternative_signs": ["doméstico"],
                "homograph": False,
                "homophone": False,
            },
        ],
    }
    candidates = [
        {
            "pun_sign": "doméxico",
            "alternative_signs": ["doméstico"],
            "homograph": False,
            "homophone": False,
        },
        {
            "pun_sign": "doméxico",
            "alternative_signs": ["do México"],
            "homograph": False,
            "homophone": True,
        },
    ]
    result = score_item(example, {"prediction": {"signs": candidates}}, False)
    assert all(item["alternative_any_match"] for item in result["matched_signs"])
    assert result["end_to_end_strict"] is True
