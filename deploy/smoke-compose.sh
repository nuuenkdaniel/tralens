#!/usr/bin/env bash
set -euo pipefail

source_dir=$(cd "$(dirname "$0")/.." && pwd)
work=$(mktemp -d)
cp "$source_dir/compose.yml" "$work/"
cd "$work"
compose=(docker compose --file "$work/compose.yml" --project-name "tralens-smoke-${RANDOM}${RANDOM}")

cleanup() {
    "${compose[@]}" logs --no-color reader || true
    "${compose[@]}" down || true
    rm -rf "$work"
}
trap cleanup EXIT

export TRALENS_UID="$(id -u)" TRALENS_GID="$(id -g)"
export TRALENS_FRONTIER_PROVIDER=disabled
export TRALENS_PORT="${TRALENS_SMOKE_PORT:-7871}"
export TRALENS_LIBRARY="$work/tralens-data/library"
export TRALENS_CONFIG="$work/tralens-data/config"
export TRALENS_STATE="$work/tralens-data/state"
export TRALENS_MODELS="$work/tralens-data/models"
export TRALENS_ASSETS="$work/tralens-data/assets"
unset COMPOSE_PROFILES TRALENS_ORIGIN TRALENS_FONTS TRALENS_OLLAMA TRALENS_OCR_ROOT
install -d -m 700 tralens-data/{library,config,state,models,assets}

export TRALENS_REGISTRATION=open
"${compose[@]}" up -d --wait --wait-timeout 90 --pull never
curl -fsS "http://127.0.0.1:$TRALENS_PORT/healthz"
"${compose[@]}" exec -T reader python - < "$source_dir/tests/compose_smoke.py"
export TRALENS_ORIGIN=https://tralens.example
"${compose[@]}" up -d --wait --wait-timeout 90 --pull never
"${compose[@]}" exec -T reader python - < "$source_dir/tests/compose_smoke.py" | tee smoke-result.json

"${compose[@]}" down
tar -cf backup.tar tralens-data
mv tralens-data original-data
tar -xf backup.tar
export TRALENS_REGISTRATION=closed
"${compose[@]}" up -d --wait --wait-timeout 90 --pull never
job=$(python3 -c 'import json; print(json.load(open("smoke-result.json"))["job"])')
"${compose[@]}" exec -T reader python - "$job" < "$source_dir/tests/compose_smoke.py"
