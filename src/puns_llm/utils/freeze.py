import hashlib
import json
from pathlib import Path

from puns_llm.inference.ollama import OllamaClient

FILES = (
    "data/processed/puns_only.json",
    "configs/experiments/cohorts.json",
    "configs/experiments/zero_shot.json",
    "configs/models/ollama.json",
    "configs/prompts/zero_shot_v2.txt",
    "configs/prompts/one_shot.txt",
    "configs/prompts/oracle_zero_shot_v2.txt",
    "configs/prompts/oracle_one_shot.txt",
    "src/puns_llm/inference/zero_shot.py",
    "src/puns_llm/inference/schema.py",
    "src/puns_llm/evaluation/score.py",
    "src/puns_llm/evaluation/score_v2.py",
    "src/puns_llm/evaluation/score_v3.py",
    "src/puns_llm/data/cohorts.py",
    "src/puns_llm/data/prepare.py",
    "src/puns_llm/inference/ollama.py",
    "src/puns_llm/cli.py",
    "src/puns_llm/utils/freeze.py",
    "pyproject.toml",
    "uv.lock",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(root: Path, client: OllamaClient) -> dict:
    configs = json.loads((root / "configs/models/ollama.json").read_text(encoding="utf-8"))
    return {
        "version": 1,
        "files_sha256": {file: sha(root / file) for file in FILES},
        "models": {name: client.model_digest(name) for name in configs["models"]},
    }


def freeze_methodology(root: Path, client: OllamaClient) -> dict:
    path = root / "configs/experiments/frozen_methodology.json"
    if path.exists():
        raise FileExistsError(f"Methodology already frozen: {path}")
    result = manifest(root, client)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def verify_freeze(root: Path, client: OllamaClient) -> dict:
    path = root / "configs/experiments/frozen_methodology.json"
    expected = json.loads(path.read_text(encoding="utf-8"))
    current = manifest(root, client)
    if current != expected:
        raise ValueError("Frozen methodology differs from the current files or model digests")
    return expected
