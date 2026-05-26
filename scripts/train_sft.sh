#!/usr/bin/env bash
set -euo pipefail

heartbeat_path="${HEARTBEAT_PATH:-logs/train_sft_heartbeat.log}"
heartbeat_interval="${HEARTBEAT_INTERVAL_SECONDS:-60}"
heartbeat_pid=""
use_qlora="${USE_QLORA:-true}"
load_in_4bit="${LOAD_IN_4BIT:-$use_qlora}"

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

python -m src.train_sft \
  --model_name_or_path "${MODEL_NAME_OR_PATH:-Qwen/Qwen3.5-2B}" \
  --train_path "${TRAIN_PATH:-data/train/complex_or_variations.jsonl}" \
  --output_dir "${OUTPUT_DIR:-outputs/sft_adapter}" \
  --use_lora "${USE_LORA:-true}" \
  --use_qlora "$use_qlora" \
  --load_in_4bit "$load_in_4bit" \
  --load_in_8bit "${LOAD_IN_8BIT:-false}" \
  --max_train_examples "${MAX_TRAIN_EXAMPLES:-200}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-3}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --max_seq_length "${MAX_SEQ_LENGTH:-4096}" &

train_pid="$!"
start_heartbeat "$train_pid" "train_sft"
set +e
wait "$train_pid"
status="$?"
set -e
if [[ -n "$heartbeat_path" ]]; then
  printf '%s label=%s pid=%s event=process_exit exit_code=%s\n' \
    "$(date --iso-8601=seconds)" "train_sft" "$train_pid" "$status" >> "$heartbeat_path"
fi
exit "$status"
