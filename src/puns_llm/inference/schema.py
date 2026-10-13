import json


PREDICTION_SCHEMA = {
    "type": "object",
    "properties": {
        "signs": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "pun_sign": {"type": "string"},
                    "alternative_signs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "homograph": {"type": "boolean"},
                    "homophone": {"type": "boolean"},
                },
                "required": [
                    "pun_sign",
                    "alternative_signs",
                    "homograph",
                    "homophone",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["signs"],
    "additionalProperties": False,
}


def parse_prediction(content: str) -> dict:
    prediction = json.loads(content)

    if not isinstance(prediction, dict) or set(prediction) != {"signs"}:
        raise ValueError("Invalid prediction structure")

    signs = prediction["signs"]

    if not isinstance(signs, list) or not signs:
        raise ValueError("Prediction must contain at least one sign")

    for sign in signs:
        if not isinstance(sign, dict) or set(sign) != {
            "pun_sign",
            "alternative_signs",
            "homograph",
            "homophone",
        }:
            raise ValueError("Invalid sign structure")

        if not isinstance(sign["pun_sign"], str) or not sign["pun_sign"].strip():
            raise ValueError("Invalid pun sign")

        alternatives = sign["alternative_signs"]

        if (
            not isinstance(alternatives, list)
            or not alternatives
            or not all(
                isinstance(value, str) and value.strip()
                for value in alternatives
            )
        ):
            raise ValueError("Invalid alternative signs")

        if type(sign["homograph"]) is not bool or type(sign["homophone"]) is not bool:
            raise ValueError("Invalid linguistic labels")

    return prediction


def build_messages(text: str, instructions: str) -> list[dict]:
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": f"Portuguese pun:\n{text}"},
    ]