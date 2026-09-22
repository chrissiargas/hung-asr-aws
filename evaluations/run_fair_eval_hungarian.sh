#!/usr/bin/env bash
# Fair Whisper vs. Dual-Fusion evaluation for every Hungarian test set.
# Usage:  bash run_fair_eval_hungarian.sh                 (all datasets below)
#         bash run_fair_eval_hungarian.sh fleurs yodas    (only the ones you name)
set -euo pipefail

LANG_TAG=greek
EXP=1                          # exp of the trained model, so the test split matches its training config
MODEL_DECODING=legacy           # plain | penalized | legacy
WHISPER_DECODING=whisper_standard         # plain | penalized | legacy | whisper_standard
if (( $# )); then DATASETS=("$@"); else DATASETS=(common_voice fleurs massive voxpopuli yodas); fi

for ds in "${DATASETS[@]}"; do
    manifest="manifests/${LANG_TAG}_${ds}_test.jsonl"
    runs="runs/${LANG_TAG}_${ds}_test"
    whisper_out="${runs}/whisper_${WHISPER_DECODING}"
    model_out="${runs}/dual_fusion_${MODEL_DECODING}"
    mkdir -p "$runs" "$(dirname "$manifest")"
    echo "=================== ${ds} ==================="

    # Build the manifest once and never overwrite it: it is the frozen test set both systems share.
    [[ -f "$manifest" ]] || python evaluations/fair_eval.py build-manifest \
        --exp "$EXP" --datasets "$ds" --split test --out "$manifest"

    # Skip runs that already finished, so the loop can be restarted after a crash.
    [[ -f "${whisper_out}/predictions.jsonl" ]] || torchrun --standalone --nproc_per_node=4 \
        evaluations/fair_inference.py \
        --manifest "$manifest" --out "$whisper_out" --decoding "$WHISPER_DECODING"

    [[ -f "${model_out}/predictions.jsonl" ]] || torchrun --standalone --nproc_per_node=4 \
        evaluations/fair_speechLM.py \
        --manifest "$manifest" --out "$model_out" --decoding "$MODEL_DECODING" \
        --exp "$EXP" --machine greekasr-0 --datetime Sep10_13-36 --name full \
         --train_datasets common_voice fleurs hparl tedx logotypographia stoma

    python evaluations/fair_eval.py compare "$model_out" "$whisper_out" \
        | tee "${runs}/compare_${MODEL_DECODING}_vs_${WHISPER_DECODING}.txt"
done
