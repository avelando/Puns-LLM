#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d results ]]; then
    echo "Results directory not found"
    exit 1
fi

if [[ -d results/official ]]; then
    echo "An official run already exists. Refusing to archive it."
    exit 1
fi

if [[ -e artifacts/archived_runs/previous_results ]]; then
    echo "Archive already exists. Refusing to overwrite it."
    exit 1
fi

mkdir -p artifacts/archived_runs/previous_results

shopt -s nullglob dotglob
for entry in results/*; do
    mv "$entry" artifacts/archived_runs/previous_results/
done
shopt -u nullglob dotglob

if [[ -f configs/experiments/frozen_methodology.json ]]; then
    mv configs/experiments/frozen_methodology.json \
        artifacts/archived_runs/frozen_methodology_previous.json
fi

mkdir -p results/official

echo "Previous results archived locally under artifacts/archived_runs/."
echo "Run a new methodology freeze before starting the clean official experiments."
