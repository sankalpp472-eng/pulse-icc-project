#!/usr/bin/env bash
# Download every published tau2-bench AIRLINE trajectory file (13 agent models).
#   9 leaderboard submissions -> Sierra's public S3 bucket (no AWS account needed: --no-sign-request)
#   4 models shipped in the tau2-bench GitHub repo (data/tau2/results/final)
#
# Needs: AWS CLI v2 and curl.   Run from the Pulse repo root (Git Bash works on Windows):
#   bash download_tau2_airline.sh                 # -> predictor/data/tau2/
#   bash download_tau2_airline.sh some/other/dir
#
# Files are named <model-id>__airline.json so `python predictor/tau_data.py --tau2` picks them up.
# Leaderboard files are large (tens to hundreds of MB each); files already present are skipped.

OUT="${1:-predictor/data/tau2}"
mkdir -p "$OUT"
BUCKET="s3://sierra-tau-bench-public/submissions"
RAW="https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/results/final"
ok=0; fail=0

get_s3 () {   # $1 = submission folder, $2 = trajectory file name
  local dest="$OUT/$1__airline.json"
  if [ -s "$dest" ]; then echo "skip   $1 (already there)"; return; fi
  echo "get    $1"
  if aws s3 cp "$BUCKET/$1/trajectories/$2" "$dest" --no-sign-request --only-show-errors; then
    ok=$((ok+1)); else echo "FAILED $1"; rm -f "$dest"; fail=$((fail+1)); fi
}

get_repo () {  # $1 = short model id, $2 = file name in the repo
  local dest="$OUT/repo-$1__airline.json"
  if [ -s "$dest" ]; then echo "skip   repo-$1 (already there)"; return; fi
  echo "get    repo-$1"
  if curl -sSfL "$RAW/$2" -o "$dest"; then
    ok=$((ok+1)); else echo "FAILED repo-$1"; rm -f "$dest"; fail=$((fail+1)); fi
}

# ---- leaderboard submissions (taubench.com)
get_s3 claude-opus-4-5_sierra_2026-02-26          claude-opus-4-5_high_airline_gpt-5.2_4trials.json
get_s3 claude-sonnet-4-5_sierra_2026-02-26        claude-sonnet-4-5_enabled_airline_gpt-5.2_4trials.json
get_s3 gemini-3-flash_sierra_2026-03-02           geminiflash-airline.json
get_s3 gemini-3-pro_sierra_2026-03-02             geminipro-airline.json
get_s3 glm-5-think_sierra_2026-03-02              glm-5_enabled_airline_gpt-5.2_4trials.json
get_s3 gpt-5-2-none_sierra_2026-02-26             gpt-5.2_none_airline_gpt-5.2_4trials.json
get_s3 gpt-5-2_sierra_2026-02-26                  gpt-5.2_high_airline_gpt-5.2_4trials.json
get_s3 qwen3-max_qwen_2025-10-30                  airline_llm_agent_qwen3-max-2025-10-30_user_simulator_gpt-4.1-2025-04-14.json
get_s3 qwen3.5-397b-a17b-think_sierra_2026-03-02  qwen3.5-397b-a17b_enabled_airline_gpt-5.2_4trials.json

# ---- models shipped in the tau2-bench repo
get_repo claude-3-7-sonnet-20250219  claude-3-7-sonnet-20250219_airline_default_gpt-4.1-2025-04-14_4trials.json
get_repo gpt-4.1-2025-04-14          gpt-4.1-2025-04-14_airline_default_gpt-4.1-2025-04-14_4trials.json
get_repo gpt-4.1-mini-2025-04-14     gpt-4.1-mini-2025-04-14_airline_base_gpt-4.1-2025-04-14_4trials.json
get_repo o4-mini-2025-04-16          o4-mini-2025-04-16_airline_default_gpt-4.1-2025-04-14_4trials.json

echo
echo "done: $ok downloaded, $fail failed -> $OUT"
ls -lh "$OUT"/*__airline.json 2>/dev/null