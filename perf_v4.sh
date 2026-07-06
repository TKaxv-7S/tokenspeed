#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ROOT=${ROOT:-$SCRIPT_DIR}
SOURCE_ROOT=${SOURCE_ROOT:-$ROOT}
OUT_BASE=${OUT_BASE:-$ROOT/perf_v4_out}
HARNESS_DIR=${HARNESS_DIR:-$SOURCE_ROOT/test/agentic_benchmark/evalscope_trie}
EVALSCOPE_RUNNER=${EVALSCOPE_RUNNER:-$HARNESS_DIR/run_evalscope_perf.py}
HARNESS_HELPER=$HARNESS_DIR/benchmark_harness.py
COLLECTOR=$HARNESS_DIR/collect_outputs.py

MODEL=${MODEL:-}
MODEL_NAME=${MODEL_NAME:-DeepSeek-V4-Pro}
TOKENIZER_PATH=${TOKENIZER_PATH:-$MODEL}
DATASET=${DATASET:-swe_smith}
DATASET_PATH=${DATASET_PATH:-}

PORT=${PORT:-18023}
CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3,4,5,6,7}
NUM_GPUS=${NUM_GPUS:-8}
TP_SIZE=${TP_SIZE:-8}
MOE_TOPOLOGY=${MOE_TOPOLOGY:-ep8}
MOE_BACKEND=${MOE_BACKEND:-flashinfer_trtllm}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-96000}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-2560000}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
SPEC_STEPS=${SPEC_STEPS:-3}
DISABLE_OVERLAP_SCHEDULE=${DISABLE_OVERLAP_SCHEDULE:-0}
ENABLE_TRTLLM_ALLREDUCE=${ENABLE_TRTLLM_ALLREDUCE:-0}

WARMUP1_DATASET_OFFSET=${WARMUP1_DATASET_OFFSET:-30}
WARMUP2_DATASET_OFFSET=${WARMUP2_DATASET_OFFSET:-31}
FORMAL_DATASET_OFFSET=${FORMAL_DATASET_OFFSET:-0}
MAX_TOKENS=${MAX_TOKENS:-500}
EXTRA_ARGS=${EXTRA_ARGS:-'{"ignore_eos": true, "temperature": 0.0}'}
API_KEY=${API_KEY:-EMPTY}
SEED=${SEED:-0}
HEALTH_TIMEOUT_SEC=${HEALTH_TIMEOUT_SEC:-3600}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}

SITECUSTOMIZE_DIR=${SITECUSTOMIZE_DIR:-}
RUNTIME_PYTHONPATH=$SOURCE_ROOT/python:$SOURCE_ROOT/tokenspeed-kernel/python:$SOURCE_ROOT/tokenspeed-scheduler/python:$SOURCE_ROOT/tokenspeed-mla/python
if [[ -d "$SITECUSTOMIZE_DIR" ]]; then
  RUNTIME_PYTHONPATH=$SITECUSTOMIZE_DIR:$RUNTIME_PYTHONPATH
fi
TOKENSPEED_BIN=${TOKENSPEED_BIN:-$(command -v tokenspeed)}

RUN_ID=${RUN_ID:-tokenspeed_v4pro_${MOE_TOPOLOGY}_${MOE_BACKEND}_$(date -u +%Y%m%d_%H%M%S)}
RUN_DIR=$OUT_BASE/runs/$RUN_ID
SERVER_LOG=
SERVER_PID_FILE=
SERVER_RUNNING=0

if [[ -e "$RUN_DIR" ]]; then
  echo "RUN_ID output directory already exists: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*" \
    | tee -a "$RUN_DIR/run.log" >&2
}

stop_server() {
  if [[ "$SERVER_RUNNING" != "1" || ! -s "$SERVER_PID_FILE" ]]; then
    return
  fi
  local pid attempt
  pid=$(cat "$SERVER_PID_FILE")
  kill -- -"$pid" >/dev/null 2>&1 || kill "$pid" >/dev/null 2>&1 || true
  for attempt in $(seq 1 60); do
    if ! kill -0 -- -"$pid" >/dev/null 2>&1; then
      SERVER_RUNNING=0
      return
    fi
    sleep 1
  done
  log "server process group $pid did not exit; sending SIGKILL"
  kill -KILL -- -"$pid" >/dev/null 2>&1 || kill -KILL "$pid" >/dev/null 2>&1 || true
  SERVER_RUNNING=0
}

trap 'stop_server || true' EXIT

wait_port_free() {
  local port=$1 timeout=${2:-120} start=$SECONDS
  while ! python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', $port)); s.close()" 2>/dev/null; do
    if (( SECONDS - start > timeout )); then
      echo "port $port still in use after ${timeout}s" >&2
      return 1
    fi
    sleep 1
  done
}

validate_config() {
  if [[ -z "$MODEL" ]]; then
    echo "MODEL must point to the model snapshot" >&2
    return 2
  fi
  if [[ -z "$DATASET_PATH" ]]; then
    echo "DATASET_PATH must point to the SWE-smith dataset" >&2
    return 2
  fi
  if [[ "$TP_SIZE" != "8" || ( "$MOE_TOPOLOGY" != "ep8" && "$MOE_TOPOLOGY" != "tp8" ) ]]; then
    echo "strict V4 perf requires TP_SIZE=8 and MOE_TOPOLOGY=ep8 or tp8" >&2
    return 2
  fi
  if [[ "$MOE_BACKEND" != "flashinfer_trtllm" && "$MOE_BACKEND" != "mega_moe" ]]; then
    echo "V4 perf requires MOE_BACKEND=flashinfer_trtllm or mega_moe" >&2
    return 2
  fi
  if [[ "$DISABLE_OVERLAP_SCHEDULE" != "0" && "$DISABLE_OVERLAP_SCHEDULE" != "1" ]]; then
    echo "DISABLE_OVERLAP_SCHEDULE must be 0 or 1" >&2
    return 2
  fi
  if [[ "$ENABLE_TRTLLM_ALLREDUCE" != "0" && "$ENABLE_TRTLLM_ALLREDUCE" != "1" ]]; then
    echo "ENABLE_TRTLLM_ALLREDUCE must be 0 or 1" >&2
    return 2
  fi
  for path in "$MODEL" "$DATASET_PATH" "$EVALSCOPE_RUNNER" "$HARNESS_HELPER" "$COLLECTOR"; do
    if [[ ! -e "$path" ]]; then
      echo "required path is missing: $path" >&2
      return 2
    fi
  done
  if [[ -z "$TOKENSPEED_BIN" || ! -x "$TOKENSPEED_BIN" ]]; then
    echo "tokenspeed executable is missing: $TOKENSPEED_BIN" >&2
    return 2
  fi
}

record_partition() {
  python3 "$HARNESS_HELPER" partition \
    --dataset "$DATASET_PATH" \
    --output "$RUN_DIR/request_partition.json" \
    --warmup1-offset "$WARMUP1_DATASET_OFFSET" \
    --warmup2-offset "$WARMUP2_DATASET_OFFSET" \
    --formal-offset "$FORMAL_DATASET_OFFSET"
}

assert_source_binding() {
  env PYTHONPATH="$RUNTIME_PYTHONPATH" python3 - "$SOURCE_ROOT" <<'PY' \
    > "$RUN_DIR/runtime-binding.txt"
import hashlib
import inspect
import sys
from pathlib import Path

import tokenspeed_scheduler
import tokenspeed_scheduler.tokenspeed_scheduler_ext as scheduler_ext
import tokenspeed
import tokenspeed_kernel
from tokenspeed.runtime.layers.attention import deepseek_v4_ops

root = Path(sys.argv[1]).resolve()
modules = {
    "tokenspeed": (tokenspeed, root / "python"),
    "deepseek_v4_ops": (deepseek_v4_ops, root / "python"),
    "tokenspeed_kernel": (tokenspeed_kernel, root / "tokenspeed-kernel/python"),
    "tokenspeed_scheduler": (
        tokenspeed_scheduler,
        root / "tokenspeed-scheduler/python",
    ),
    "scheduler_ext": (scheduler_ext, root / "tokenspeed-scheduler/python"),
}
for name, (module, expected_root) in modules.items():
    path = Path(inspect.getfile(module)).resolve()
    if not path.is_relative_to(expected_root):
        raise SystemExit(f"{name} source mismatch: {path} not under {expected_root}")
    line = f"{name}: {path}"
    if path.suffix == ".so":
        line += f" sha256={hashlib.sha256(path.read_bytes()).hexdigest()}"
    print(line)
PY
  cat "$RUN_DIR/runtime-binding.txt"
}

record_manifest() {
  local partition_sha dataset_sha harness_sha
  partition_sha=$(sha256sum "$RUN_DIR/request_partition.json" | awk '{print $1}')
  dataset_sha=$(sha256sum "$DATASET_PATH" | awk '{print $1}')
  harness_sha=$(sha256sum "$0" "$EVALSCOPE_RUNNER" "$HARNESS_HELPER" | sha256sum | awk '{print $1}')
  {
    echo "run_id: $RUN_ID"
    echo "hostname: $(hostname)"
    echo "date_utc: $(date -u --iso-8601=seconds)"
    echo "engine: tokenspeed"
    echo "source_root: $SOURCE_ROOT"
    echo "git_head: $(git -C "$SOURCE_ROOT" rev-parse HEAD)"
    echo "git_diff_sha256: $(git -C "$SOURCE_ROOT" diff --binary HEAD | sha256sum | awk '{print $1}')"
    echo "model: $MODEL"
    echo "dataset: $DATASET_PATH"
    echo "dataset_sha256: $dataset_sha"
    echo "request_partition_sha256: $partition_sha"
    echo "harness_sha256: $harness_sha"
    echo "api: openai_plain_completion"
    echo "formal_selection: disjoint"
    echo "warmup_offsets: $WARMUP1_DATASET_OFFSET $WARMUP2_DATASET_OFFSET"
    echo "formal_offset: $FORMAL_DATASET_OFFSET"
    echo "cuda_devices: $CUDA_DEVICES"
    echo "tp_size: $TP_SIZE"
    echo "moe_topology: $MOE_TOPOLOGY"
    echo "moe_backend: $MOE_BACKEND"
    echo "max_model_len: $MAX_MODEL_LEN"
    echo "max_total_tokens: $MAX_TOTAL_TOKENS"
    echo "chunked_prefill_size: $CHUNKED_PREFILL_SIZE"
    echo "gpu_memory_utilization: $GPU_MEMORY_UTILIZATION"
    echo "spec_steps: $SPEC_STEPS"
    echo "disable_overlap_schedule: $DISABLE_OVERLAP_SCHEDULE"
    echo "enable_trtllm_allreduce: $ENABLE_TRTLLM_ALLREDUCE"
    echo "max_tokens: $MAX_TOKENS"
    echo "seed: $SEED"
    echo "extra_args: $EXTRA_ARGS"
  } > "$RUN_DIR/manifest.txt"
}

start_server() {
  local moe_flags
  local overlap_flags=()
  local allreduce_flags=()
  if [[ "$MOE_TOPOLOGY" == "ep8" ]]; then
    moe_flags="--enable-expert-parallel --dense-tp-size 1 --moe-backend $MOE_BACKEND"
  else
    moe_flags="--moe-backend $MOE_BACKEND"
  fi
  if [[ "$DISABLE_OVERLAP_SCHEDULE" == "1" ]]; then
    overlap_flags+=(--disable-overlap-schedule)
  fi
  if [[ "$ENABLE_TRTLLM_ALLREDUCE" == "1" ]]; then
    allreduce_flags+=(--enable-trtllm-allreduce)
  fi

  log "starting TokenSpeed $MOE_TOPOLOGY server; log=$SERVER_LOG"
  env \
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    PYTHONPATH="$RUNTIME_PYTHONPATH" \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    setsid "$TOKENSPEED_BIN" serve "$MODEL" \
      --served-model-name "$MODEL_NAME" \
      --trust-remote-code \
      --tensor-parallel-size "$TP_SIZE" \
      $moe_flags \
      --kv-cache-dtype fp8_e4m3 \
      --attention-use-fp4-indexer-cache \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-total-tokens "$MAX_TOTAL_TOKENS" \
      --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --disable-kvstore \
      --speculative-algorithm MTP \
      --speculative-num-steps "$SPEC_STEPS" \
      "${overlap_flags[@]}" \
      "${allreduce_flags[@]}" \
      --seed "$SEED" \
      --enable-metrics \
      --enable-cache-report \
      --health-check-timeout-secs 120 \
      --health-failure-threshold 100 \
      --health-check-interval-secs 86400 \
      --host 0.0.0.0 \
      --port "$PORT" \
      > "$SERVER_LOG" 2>&1 < /dev/null &
  echo $! > "$SERVER_PID_FILE"
  SERVER_RUNNING=1
}

wait_health() {
  local deadline=$((SECONDS + HEALTH_TIMEOUT_SEC))
  log "waiting for TokenSpeed readiness"
  until curl --max-time 10 -fsS "http://127.0.0.1:$PORT/readiness" >/dev/null 2>&1 \
    || curl --max-time 10 -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    if ! kill -0 "$(cat "$SERVER_PID_FILE")" >/dev/null 2>&1; then
      tail -240 "$SERVER_LOG" > "$RUN_DIR/server.tail" || true
      return 1
    fi
    if grep -Eiq "CUDA out of memory|OutOfMemory|Scheduler hit an exception|Traceback|Killed" "$SERVER_LOG"; then
      tail -240 "$SERVER_LOG" > "$RUN_DIR/server.tail" || true
      return 1
    fi
    if (( SECONDS > deadline )); then
      log "health timeout after ${HEALTH_TIMEOUT_SEC}s"
      return 1
    fi
    sleep 10
  done
  log "server is ready"
  sleep 5
}

run_evalscope() {
  local label=$1 parallel=$2 number=$3 offset=$4 phase=$5
  local output_root=$RUN_DIR/$phase/p${parallel}
  log "evalscope $label p=$parallel n=$number offset=$offset"
  python3 "$EVALSCOPE_RUNNER" perf \
    --model "$MODEL_NAME" \
    --url "http://127.0.0.1:$PORT/v1/completions" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --api openai_plain_completion \
    --api-key "$API_KEY" \
    --dataset "$DATASET" \
    --dataset-path "$DATASET_PATH" \
    --dataset-offset "$offset" \
    --max-tokens "$MAX_TOKENS" \
    --multi-turn \
    --number "$number" \
    --parallel "$parallel" \
    --stream \
    --no-apply-chat-template \
    --extra-args "$EXTRA_ARGS" \
    --seed "$SEED" \
    --no-test-connection \
    --no-timestamp \
    --outputs-dir "$output_root" \
    > "$RUN_DIR/${label}.log" 2>&1
}

validate_result() {
  local section=$1 parallel=$2 number=$3 phase=$4
  python3 "$HARNESS_HELPER" validate-result \
    --result-dir "$RUN_DIR/$phase/p${parallel}/$MODEL_NAME/parallel_${parallel}_number_${number}" \
    --partition "$RUN_DIR/request_partition.json" \
    --section "$section" \
    --parallel "$parallel" \
    --number "$number" \
    --seed "$SEED" \
    --max-tokens "$MAX_TOKENS"
}

run_curve() {
  SERVER_LOG=$RUN_DIR/server.log
  SERVER_PID_FILE=$RUN_DIR/server.pid
  wait_port_free "$PORT"
  start_server
  wait_health

  run_evalscope warmup_p1 1 1 "$WARMUP1_DATASET_OFFSET" warmup_p1
  validate_result warmup 1 1 warmup_p1
  run_evalscope warmup_p2 2 2 "$WARMUP2_DATASET_OFFSET" warmup_p2
  validate_result warmup 2 2 warmup_p2

  local spec parallel number offset
  for spec in "1:2:0" "2:4:2" "4:8:6" "8:16:14"; do
    IFS=: read -r parallel number offset <<< "$spec"
    offset=$((FORMAL_DATASET_OFFSET + offset))
    run_evalscope "formal_p${parallel}" "$parallel" "$number" "$offset" sweep
    validate_result formal "$parallel" "$number" sweep
    if grep -Eiq "CUDA out of memory|OutOfMemory|Scheduler hit an exception|Traceback|Killed" "$SERVER_LOG"; then
      log "server fatal error during formal p${parallel}"
      return 1
    fi
  done
  stop_server
  wait_port_free "$PORT"
}

main() {
  validate_config
  record_partition
  if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    log "preflight passed"
    return
  fi
  assert_source_binding
  record_manifest
  run_curve
  python3 "$COLLECTOR" "$RUN_DIR" \
    --num-gpus "$NUM_GPUS" \
    --csv "$RUN_DIR/sweep.csv" \
    --svg "$RUN_DIR/sweep.svg" \
    --title "TokenSpeed V4-Pro SWE-smith $MOE_TOPOLOGY"
  log "done: $RUN_DIR"
}

main "$@"
