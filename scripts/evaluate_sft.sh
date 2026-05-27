#!/usr/bin/env bash
set -euo pipefail

heartbeat_path="${HEARTBEAT_PATH:-logs/evaluate_sft_heartbeat.log}"
heartbeat_interval="${HEARTBEAT_INTERVAL_SECONDS:-60}"
heartbeat_pid=""
load_in_4bit="${LOAD_IN_4BIT:-false}"
load_in_8bit="${LOAD_IN_8BIT:-false}"
if [[ "$load_in_4bit" == "true" ]]; then
  load_in_8bit="${LOAD_IN_8BIT:-false}"
fi

start_heartbeat() {
  local target_pid="$1"
  local label="$2"
  if [[ -z "$heartbeat_path" ]]; then
    return
  fi
  mkdir -p "$(dirname "$heartbeat_path")"
  (
    start_time="$(date +%s)"
    while kill -0 "$target_pid" 2>/dev/null; do
      now="$(date +%s)"
      elapsed=$((now - start_time))
      printf '%s label=%s pid=%s elapsed_seconds=%s event=alive\n' \
        "$(date --iso-8601=seconds)" "$label" "$target_pid" "$elapsed" >> "$heartbeat_path"
      sleep "$heartbeat_interval"
    done
    now="$(date +%s)"
    elapsed=$((now - start_time))
    printf '%s label=%s pid=%s elapsed_seconds=%s event=stopped\n' \
      "$(date --iso-8601=seconds)" "$label" "$target_pid" "$elapsed" >> "$heartbeat_path"
  ) &
  heartbeat_pid="$!"
}

stop_heartbeat() {
  if [[ -n "$heartbeat_pid" ]]; then
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
  fi
}
trap stop_heartbeat EXIT

args=(
  -m src.eval
  --model_name_or_path "${MODEL_NAME_OR_PATH:-outputs/sft_adapter}" \
  --eval_dir "${EVAL_DIR:-data/eval/complex_or_eval.jsonl}" \
  --output_path "${OUTPUT_PATH:-outputs/sft_eval.jsonl}" \
  --responses_path "${RESPONSES_PATH:-outputs/sft_eval.responses.jsonl}" \
  --code_timeout_seconds "${CODE_TIMEOUT_SECONDS:-120}" \
  --answer_tolerance "${ANSWER_TOLERANCE:-1e-6}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-8192}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --generation_log_path "${GENERATION_LOG_PATH:-logs/evaluate_sft_generations.jsonl}"
)
if [[ "$load_in_4bit" == "true" ]]; then args+=(--load_in_4bit); fi
if [[ "$load_in_8bit" == "true" ]]; then args+=(--load_in_8bit); fi
if [[ "${PRECOMPUTE_RESPONSES:-false}" == "true" ]]; then args+=(--precompute_responses); fi
if [[ "${PARTIAL_RUN:-false}" == "true" ]]; then args+=(--partial_run); fi
if [[ -n "${MAX_EVAL_EXAMPLES:-}" ]]; then args+=(--max_eval_examples "$MAX_EVAL_EXAMPLES"); fi

python "${args[@]}" &
eval_pid="$!"
start_heartbeat "$eval_pid" "evaluate_sft"
set +e
wait "$eval_pid"
status="$?"
set -e
if [[ -n "$heartbeat_path" ]]; then
  printf '%s label=%s pid=%s event=process_exit exit_code=%s\n' \
    "$(date --iso-8601=seconds)" "evaluate_sft" "$eval_pid" "$status" >> "$heartbeat_path"
fi
exit "$status"
