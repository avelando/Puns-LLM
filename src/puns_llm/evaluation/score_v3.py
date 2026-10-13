import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from puns_llm.evaluation.score import f1, normalize

CATEGORIES = ("0_0", "0_1", "1_0", "1_1")
PROTOCOLS = ("zero_shot", "zero_shot_v2", "one_shot", "oracle_zero_shot_v2", "oracle_one_shot")


def tokens(value: str) -> Counter:
    return Counter(re.findall(r"\w+", normalize(value)))


def partial_pairs(gold: list[dict], predicted: list[dict]) -> list[tuple[int, int]]:
    states = {0: (0, ())}
    for predicted_index, item in enumerate(predicted):
        updated = dict(states)
        right = tokens(item["pun_sign"])
        for mask, (score, pairs) in states.items():
            for gold_index, reference in enumerate(gold):
                if mask & (1 << gold_index):
                    continue
                left = tokens(reference["pun_sign"])
                intersection = sum((left & right).values())
                if not intersection:
                    continue
                key = mask | (1 << gold_index)
                value = (score + intersection, pairs + ((gold_index, predicted_index),))
                if key not in updated or value[0] > updated[key][0]:
                    updated[key] = value
        states = updated
    return list(max(states.values(), key=lambda value: value[0])[1])


def annotation_pairs(gold: list[dict], predicted: list[dict]) -> list[tuple[int, int]]:
    reference_groups = defaultdict(list)
    prediction_groups = defaultdict(list)
    for index, item in enumerate(gold):
        reference_groups[normalize(item["pun_sign"])].append(index)
    for index, item in enumerate(predicted):
        prediction_groups[normalize(item["pun_sign"])].append(index)

    matches = []
    for sign, gold_indices in reference_groups.items():
        states = {0: ((0, 0, 0), ())}
        for predicted_index in prediction_groups.get(sign, []):
            updated = dict(states)
            candidate = predicted[predicted_index]
            for mask, (quality, pairs) in states.items():
                for local_index, gold_index in enumerate(gold_indices):
                    if mask & (1 << local_index):
                        continue
                    reference = gold[gold_index]
                    alternative_match = bool(
                        {normalize(value) for value in reference["alternative_signs"]}
                        & {normalize(value) for value in candidate["alternative_signs"]}
                    )
                    homograph_match = reference["homograph"] == candidate["homograph"]
                    homophone_match = reference["homophone"] == candidate["homophone"]
                    gain = (
                        int(alternative_match),
                        int(homograph_match and homophone_match),
                        int(homograph_match) + int(homophone_match),
                    )
                    next_quality = tuple(a + b for a, b in zip(quality, gain))
                    next_pairs = pairs + ((gold_index, predicted_index),)
                    key = mask | (1 << local_index)
                    if key not in updated or next_quality > updated[key][0]:
                        updated[key] = (next_quality, next_pairs)
            states = updated
        _, pairs = max(states.items(), key=lambda item: (item[0].bit_count(), item[1][0]))
        matches.extend(pairs[1])
    return sorted(matches)


def token_mask(text: str, spans: list[tuple[int, int]], signs: list[dict]) -> list[bool]:
    mask = [False] * len(spans)
    for sign in signs:
        for found in re.finditer(re.escape(sign["pun_sign"]), text, flags=re.IGNORECASE):
            for index, (start, end) in enumerate(spans):
                if start >= found.start() and end <= found.end():
                    mask[index] = True
    return mask


def labeled_token_counts(gold: dict, predicted: list[dict]) -> tuple[int, int, int] | None:
    text = gold.get("text")
    source_tokens = gold.get("tokens")
    labels = gold.get("labels")
    if (
        not isinstance(text, str)
        or not isinstance(source_tokens, list)
        or not isinstance(labels, list)
        or len(source_tokens) != len(labels)
    ):
        return None
    spans = []
    cursor = 0
    for token in source_tokens:
        position = text.find(token, cursor)
        if position < 0:
            return None
        spans.append((position, position + len(token)))
        cursor = position + len(token)
    annotated_mask = [bool(label) for label in labels]
    if "signs" in gold:
        reference_mask = token_mask(text, spans, gold["signs"])
        if reference_mask != annotated_mask:
            return None

    mask = token_mask(text, spans, predicted)
    tp = sum(label and value for label, value in zip(annotated_mask, mask))
    fp = sum(not label and value for label, value in zip(annotated_mask, mask))
    fn = sum(label and not value for label, value in zip(annotated_mask, mask))
    return tp, fp, fn


def binary_metrics(pairs: list[dict], key: str) -> dict:
    tp = sum(pair[f"gold_{key}"] and pair[f"pred_{key}"] for pair in pairs)
    fp = sum(not pair[f"gold_{key}"] and pair[f"pred_{key}"] for pair in pairs)
    fn = sum(pair[f"gold_{key}"] and not pair[f"pred_{key}"] for pair in pairs)
    tn = sum(not pair[f"gold_{key}"] and not pair[f"pred_{key}"] for pair in pairs)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": p,
        "recall": r,
        "f1": f1(p, r),
        "accuracy": (tp + tn) / len(pairs) if pairs else 0.0,
    }


def score_item(gold: dict, attempt: dict | None, oracle: bool) -> dict:
    references = gold["signs"]
    prediction = attempt.get("prediction") if attempt else None
    candidates = prediction["signs"] if isinstance(prediction, dict) else []
    exact = annotation_pairs(references, candidates)
    partial = partial_pairs(references, candidates)
    matched = []
    for gi, pi in exact:
        reference, candidate = references[gi], candidates[pi]
        expected = {normalize(value) for value in reference["alternative_signs"]}
        output = {normalize(value) for value in candidate["alternative_signs"]}
        both = expected & output
        category = f"{int(reference['homograph'])}_{int(reference['homophone'])}"
        identical = normalize(reference["pun_sign"]) in expected
        matched.append(
            {
                "gold_index": gi,
                "predicted_index": pi,
                "gold_category": category,
                "predicted_category": f"{int(candidate['homograph'])}_{int(candidate['homophone'])}",
                "gold_homograph": reference["homograph"],
                "pred_homograph": candidate["homograph"],
                "gold_homophone": reference["homophone"],
                "pred_homophone": candidate["homophone"],
                "alternative_identical": identical,
                "alternative_any_match": bool(both),
                "alternative_top1": normalize(candidate["alternative_signs"][0]) in expected,
                "alternative_tp": len(both),
                "alternative_fp": len(output - expected),
                "alternative_fn": len(expected - output),
                "alternative_set_exact": expected == output,
                "homograph_correct": reference["homograph"] == candidate["homograph"],
                "homophone_correct": reference["homophone"] == candidate["homophone"],
            }
        )
    token_counts = labeled_token_counts(gold, candidates)
    complete = len(exact) == len(references) == len(candidates)
    strict = complete and all(
        item["alternative_set_exact"] and item["homograph_correct"] and item["homophone_correct"]
        for item in matched
    )
    any_match = complete and all(
        item["alternative_any_match"] and item["homograph_correct"] and item["homophone_correct"]
        for item in matched
    )
    return {
        "id": gold["id"],
        "attempted": attempt is not None,
        "valid_json": prediction is not None,
        "gold_signs": len(references),
        "predicted_signs": len(candidates),
        "exact_matches": len(exact),
        "token_counts": token_counts,
        "partial_span_pairs": partial,
        "location_exact": complete,
        "matched_signs": matched,
        "end_to_end_any_match": any_match if not oracle else None,
        "end_to_end_strict": strict if not oracle else None,
        "oracle_all_interpretations_correct": strict if oracle else None,
    }


def ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def prf(tp: int, fp: int, fn: int) -> dict:
    p, r = ratio(tp, tp + fp), ratio(tp, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1(p, r)}


def summarize(details: list[dict], model: str, protocol: str) -> dict:
    oracle = protocol.startswith("oracle_")
    matches = [item for detail in details for item in detail["matched_signs"]]
    gold_total = sum(item["gold_signs"] for item in details)
    pred_total = sum(item["predicted_signs"] for item in details)
    exact = sum(item["exact_matches"] for item in details)
    alt = prf(
        sum(item["alternative_tp"] for item in matches),
        sum(item["alternative_fp"] for item in matches),
        sum(item["alternative_fn"] for item in matches),
    )
    categories = {}
    confusion = {gold: {pred: 0 for pred in CATEGORIES} for gold in CATEGORIES}
    for item in matches:
        confusion[item["gold_category"]][item["predicted_category"]] += 1
    for cat in CATEGORIES:
        tp = confusion[cat][cat]
        fp = sum(confusion[other][cat] for other in CATEGORIES if other != cat)
        fn = sum(confusion[cat][other] for other in CATEGORIES if other != cat)
        categories[cat] = {**prf(tp, fp, fn), "support": sum(confusion[cat].values())}
    support = [cat for cat in CATEGORIES if categories[cat]["support"]]
    alternative_groups = {}
    for name, identical in (("identical", True), ("different", False)):
        selected = [item for item in matches if item["alternative_identical"] is identical]
        alternative_groups[name] = {
            "support": len(selected),
            "any_match_accuracy": ratio(
                sum(item["alternative_any_match"] for item in selected), len(selected)
            ),
            "top1_accuracy": ratio(
                sum(item["alternative_top1"] for item in selected), len(selected)
            ),
            "set_f1": prf(
                sum(item["alternative_tp"] for item in selected),
                sum(item["alternative_fp"] for item in selected),
                sum(item["alternative_fn"] for item in selected),
            )["f1"],
        }
    result = {
        "evaluator_version": 3,
        "model": model,
        "protocol": protocol,
        "selected": len(details),
        "attempted": sum(item["attempted"] for item in details),
        "missing": sum(not item["attempted"] for item in details),
        "invalid_json": sum(item["attempted"] and not item["valid_json"] for item in details),
        "complete": all(item["valid_json"] for item in details),
        "conditional_on_exact_location": {
            "matched_signs": len(matches),
            "alternative": {
                **alt,
                "any_match_accuracy": ratio(
                    sum(item["alternative_any_match"] for item in matches), len(matches)
                ),
                "top1_accuracy": ratio(
                    sum(item["alternative_top1"] for item in matches), len(matches)
                ),
                "set_exact_accuracy": ratio(
                    sum(item["alternative_set_exact"] for item in matches), len(matches)
                ),
                "subgroups": alternative_groups,
            },
            "homograph": binary_metrics(matches, "homograph"),
            "homophone": binary_metrics(matches, "homophone"),
            "category_confusion": confusion,
            "category_metrics": categories,
            "macro_f1_all_categories": sum(categories[cat]["f1"] for cat in CATEGORIES) / 4,
            "macro_f1_supported_categories": ratio(
                sum(categories[cat]["f1"] for cat in support), len(support)
            ),
        },
    }
    if oracle:
        result["oracle"] = {
            "matched_gold_signs": len(matches),
            "gold_signs": gold_total,
            "sign_copy_recall": ratio(exact, gold_total),
            "all_interpretations_correct": ratio(
                sum(item["oracle_all_interpretations_correct"] for item in details), len(details)
            ),
        }
    else:
        result["location_exact"] = {
            **prf(exact, pred_total - exact, gold_total - exact),
            "example_accuracy": ratio(
                sum(item["location_exact"] for item in details), len(details)
            ),
        }
        available = [item["token_counts"] for item in details if item["token_counts"] is not None]
        ttp = sum(item[0] for item in available)
        tfp = sum(item[1] for item in available)
        tfn = sum(item[2] for item in available)
        result["location_tokens_labeled"] = {
            **prf(ttp, tfp, tfn),
            "aligned_examples": len(available),
            "unaligned_examples": len(details) - len(available),
            "matching_rule": "all_literal_occurrences_with_gold_consistency_check",
        }
        result["end_to_end"] = {
            "any_match_accuracy": ratio(
                sum(item["end_to_end_any_match"] for item in details), len(details)
            ),
            "strict_set_accuracy": ratio(
                sum(item["end_to_end_strict"] for item in details), len(details)
            ),
        }
    return result


def evaluate_v3(
    root: Path,
    model: str,
    protocol: str,
    ids: list[str] | None = None,
    limit: int = 0,
    require_complete: bool = False,
) -> dict:
    if protocol not in PROTOCOLS or type(limit) is not int or limit < 0:
        raise ValueError("Invalid evaluation protocol or limit")
    data_path = root / "data/processed/puns_only.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    index = {item["id"]: item for item in data}
    if ids is not None:
        if len(ids) != len(set(ids)) or any(identifier not in index for identifier in ids):
            raise ValueError("Invalid evaluation IDs")
        selected = [index[identifier] for identifier in ids]
    else:
        selected = data[:limit] if limit else data
    result_dir = root / "results" / protocol / model.replace(":", "_")
    metadata = json.loads((result_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("dataset_sha256") != hashlib.sha256(data_path.read_bytes()).hexdigest():
        raise ValueError("The dataset does not match the recorded inference run")
    if metadata.get("model") != model or metadata.get("protocol", "zero_shot") != protocol:
        raise ValueError("Inference metadata does not match the requested run")
    attempts = {}
    with (result_dir / "predictions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            key = record["id"]
            if record.get("prediction") is not None or key not in attempts:
                attempts[key] = record
    details = [
        score_item(item, attempts.get(item["id"]), protocol.startswith("oracle_"))
        for item in selected
    ]
    report = summarize(details, model, protocol)
    report["selection_sha256"] = hashlib.sha256(
        json.dumps([item["id"] for item in selected]).encode()
    ).hexdigest()
    if require_complete and not report["complete"]:
        raise ValueError(
            f"Incomplete evaluation: missing={report['missing']}, invalid={report['invalid_json']}"
        )
    out = (
        root
        / "results/evaluation_v3"
        / protocol
        / model.replace(":", "_")
        / report["selection_sha256"][:12]
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "details.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in details), encoding="utf-8"
    )
    return report
