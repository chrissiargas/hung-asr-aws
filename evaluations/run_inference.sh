#!/usr/bin/env bash
# Fair Whisper vs. Dual-Fusion evaluation for every Hungarian test set.
# Usage:  bash run_fair_eval.sh                 (all datasets below)
#         bash run_fair_eval.sh fleurs yodas    (only the ones you name)
set -euo pipefail

LANG_TAG=hungarian
EXP=2                          # exp of the trained model, so the test split matches its training config
MODEL_DECODING=legacy           # plain | penalized | legacy
WHISPER_DECODING=whisper_standard         # plain | penalized | legacy | whisper_standard
DATETIME=Aug31_12-08
MACHINE=hu-asr-medium-3-0
TURN=37500
if (( $# )); then DATASETS=("$@"); else DATASETS=(common_voice fleurs massive voxpopuli yodas); fi

for ds in "${DATASETS[@]}"; do
    manifest="manifests/${LANG_TAG}_${ds}_test.jsonl"
    runs="runs/${LANG_TAG}_${ds}_test"
    whisper_out="${runs}/whisper_${WHISPER_DECODING}"
    model_out="${runs}/dual_fusion_${MODEL_DECODING}"
    mkdir -p "$runs" "$(dirname "$manifest")"
    echo "=================== ${ds} ==================="

    # Skip runs that already finished, so the loop can be restarted after a crash.
    [[ -f "${whisper_out}/predictions.jsonl" ]] || torchrun --standalone --nproc_per_node=4 \
        evaluations/fair_inference.py \
        --manifest "$manifest" --out "$whisper_out" --decoding "$WHISPER_DECODING"

done
