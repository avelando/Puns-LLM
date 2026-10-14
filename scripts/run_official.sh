#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
uv run puns-llm --action verify-freeze >/dev/null

for model in gemma3:27b qwen3:32b; do
    for protocol in zero_shot_v2 one_shot oracle_zero_shot_v2 oracle_one_shot; do
        uv run puns-llm --action run-official --model "$model" --protocol "$protocol"
        uv run puns-llm --action evaluate-official --model "$model" --protocol "$protocol"
    done
done
