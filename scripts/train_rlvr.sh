#!/usr/bin/env bash
set -euo pipefail

heartbeat_path="${HEARTBEAT_PATH:-logs/train_rlvr_heartbeat.log}"
heartbeat_interval="${HEARTBEAT_INTERVAL_SECONDS:-60}"
heartbeat_pid=""
train_log_path="${TRAIN_LOG_PATH:-logs/train_rlvr_$(date +%Y%m%d_%H%M%S).log}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-true}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

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

mkdir -p "$(dirname "$train_log_path")"

if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  printf '%s resume_from_checkpoint=%s\n' "$(date --iso-8601=seconds)" "${RESUME_FROM_CHECKPOINT}" | tee -a "$train_log_path"
fi

python -m src.train_rlvr \
  --model_name_or_path "${MODEL_NAME_OR_PATH:-outputs/sft_adapter}" \
  --train_path "${TRAIN_PATH:-data/train/complex_or_variations.jsonl}" \
  --output_dir "${OUTPUT_DIR:-outputs/rlvr_adapter}" \
  --load_in_4bit "${LOAD_IN_4BIT:-true}" \
  --load_in_8bit "${LOAD_IN_8BIT:-false}" \
  --code_timeout_seconds "${CODE_TIMEOUT_SECONDS:-15}" \
  --answer_tolerance "${ANSWER_TOLERANCE:-1e-3}" \
  --reward_answer_weight "${REWARD_ANSWER_WEIGHT:-10.0}" \
  --reward_exec_weight "${REWARD_EXEC_WEIGHT:-1.0}" \
  --reward_ortools_weight "${REWARD_ORTOOLS_WEIGHT:-0.1}" \
  --reward_format_weight "${REWARD_FORMAT_WEIGHT:-0.25}" \
  --reward_script_validation_weight "${REWARD_SCRIPT_VALIDATION_WEIGHT:-0.25}" \
  --reward_syntax_weight "${REWARD_SYNTAX_WEIGHT:-0.25}" \
  --syntax_error_penalty "${SYNTAX_ERROR_PENALTY:--10.0}" \
  --execution_timeout_penalty "${EXECUTION_TIMEOUT_PENALTY:--5.0}" \
  --execution_error_penalty "${EXECUTION_ERROR_PENALTY:--4.0}" \
  --generation_log_path "${GENERATION_LOG_PATH:-logs/train_rlvr_generations.jsonl}" \
  --generation_preview_chars "${GENERATION_PREVIEW_CHARS:-500}" \
  --max_train_examples "${MAX_TRAIN_EXAMPLES:-1800}" \
  --resume_step "${RESUME_STEP:-0}" \
  --save_steps "${SAVE_STEPS:-20}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-5}" \
  --resume_from_checkpoint "${RESUME_FROM_CHECKPOINT:-}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-2}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-2}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-2}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH:-1920}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH:-768}" \
  --num_generations "${NUM_GENERATIONS:-4}" 2>&1 | tee -a "$train_log_path" &

train_pid="$!"
start_heartbeat "$train_pid" "train_rlvr"
set +e
wait "$train_pid"
status="$?"
set -e
if [[ -n "$heartbeat_path" ]]; then
  printf '%s label=%s pid=%s event=process_exit exit_code=%s\n' \
    "$(date --iso-8601=seconds)" "train_rlvr" "$train_pid" "$status" >> "$heartbeat_path"
fi
exit "$status"
