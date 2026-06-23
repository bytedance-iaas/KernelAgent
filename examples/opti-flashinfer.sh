#!/usr/bin/env bash
# Optimize all 5 FlashInfer contest kernels sequentially.
# Run from the repo root: bash examples/opti-flashinfer.sh
set -euo pipefail

export OPENAI_API_KEY=<your key>
export OPENAI_BASE_URL=<your base URL>
export CUDA_VISIBLE_DEVICES=7

STRATEGY=flashinfer_beam_search
# Change the ROUNDS if needed
ROUNDS=1
SCRIPT="python examples/run_flashinfer.py"

KERNELS=(
    "gdn_decode_qk4_v8_d128_k_last"
    "gdn_prefill_qk4_v8_d128_k_last"
    "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64"
    "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64"
    "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
)

TOTAL=${#KERNELS[@]}
PASSED=0
FAILED=()

for i in "${!KERNELS[@]}"; do
    DEF="${KERNELS[$i]}"
    NUM=$((i + 1))
    echo ""
    echo "######################################################################"
    echo "# [$NUM/$TOTAL] $DEF"
    echo "######################################################################"

    if $SCRIPT \
        --definition "$DEF" \
        --strategy "$STRATEGY" \
        --max-rounds "$ROUNDS" \
        --contest-fast; then
        PASSED=$((PASSED + 1))
    else
        echo "!!! FAILED: $DEF"
        FAILED+=("$DEF")
    fi
done

echo ""
echo "======================================================================"
echo "SUMMARY: $PASSED/$TOTAL passed"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed:"
    for f in "${FAILED[@]}"; do echo "  - $f"; done
    exit 1
fi
