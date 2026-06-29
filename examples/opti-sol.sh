export OPENAI_API_KEY="your key"
export OPENAI_BASE_URL="your openai url, optional"
export CUDA_VISIBLE_DEVICES=7

python run_opt_manager.py \
    --kernel-dir ../sol_execbench_solutions/workdir/011_fp8_moe_gate_routing/ \
    --strategy beam_search \
    --max-rounds 1
