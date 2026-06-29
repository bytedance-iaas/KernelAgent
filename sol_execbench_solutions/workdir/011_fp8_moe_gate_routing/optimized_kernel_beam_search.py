import torch
import triton
import triton.language as tl


@triton.jit
def _quant_x_kernel(x_ptr, sx_ptr, qx_ptr, T: tl.constexpr,
                    BLOCK_M: tl.constexpr, GROUP_BLOCKS: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    rb = tl.arange(0, BLOCK_M * GROUP_BLOCKS)
    kk = tl.arange(0, BK)

    rows = pid_m * BLOCK_M + rb // GROUP_BLOCKS
    blocks = pid_g * GROUP_BLOCKS + rb % GROUP_BLOCKS
    cols = blocks[:, None] * BK + kk[None, :]

    mask = rows[:, None] < T
    vals = tl.load(x_ptr + rows[:, None] * 7168 + cols, mask=mask, other=0.0).to(tl.float32)
    scales = tl.load(sx_ptr + rows * 56 + blocks, mask=rows < T, other=1.0).to(tl.float32)
    inv_scales = 1.0 / scales

    q = vals * inv_scales[:, None]
    q = tl.minimum(tl.maximum(q, -448.0), 448.0)
    tl.store(qx_ptr + rows[:, None] * 7168 + cols, q.to(tl.float8e4nv), mask=mask)


@triton.jit
def _quant_w_kernel(w_ptr, sw_ptr, qwt_ptr,
                    BE: tl.constexpr, BK: tl.constexpr):
    pid_kb = tl.program_id(0)
    pid_eb = tl.program_id(1)

    offs_k = pid_kb * BK + tl.arange(0, BK)
    offs_e = pid_eb * BE + tl.arange(0, BE)

    vals = tl.load(w_ptr + offs_e[None, :] * 7168 + offs_k[:, None]).to(tl.float32)
    scale = tl.load(sw_ptr + pid_kb * 2 + ((pid_eb * BE) // 128)).to(tl.float32)
    inv_scale = 1.0 / scale

    q = vals * inv_scale
    q = tl.minimum(tl.maximum(q, -448.0), 448.0)
    tl.store(qwt_ptr + offs_k[:, None] * 256 + offs_e[None, :], q.to(tl.float8e4nv))


@triton.jit
def _scaled_gemm_kernel(qx_ptr, qwt_ptr, sx_ptr, sw_ptr, logits_ptr, T,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for kb in tl.range(0, 56):
        a = tl.load(
            qx_ptr + offs_m[:, None] * 7168 + (kb * BK + offs_k)[None, :],
            mask=offs_m[:, None] < T,
            other=0.0,
        )
        b = tl.load(qwt_ptr + (kb * BK + offs_k)[:, None] * 256 + offs_n[None, :])

        d = tl.dot(a, b, out_dtype=tl.float32)

        sx = tl.load(sx_ptr + offs_m * 56 + kb, mask=offs_m < T, other=0.0).to(tl.float32)
        sw = tl.load(sw_ptr + kb * 2 + (pid_n // 2)).to(tl.float32)
        acc += d * (sx[:, None] * sw)

    tl.store(
        logits_ptr + offs_m[:, None] * 256 + offs_n[None, :],
        acc.to(tl.bfloat16),
        mask=offs_m[:, None] < T,
    )


@triton.jit
def _routing_kernel(logits_ptr, bias_ptr, topk_idx_ptr, topk_weight_ptr,
                    routed_scaling_factor, T: tl.constexpr):
    row = tl.program_id(0)

    offs = tl.arange(0, 256)
    gids = tl.arange(0, 8)
    ks = tl.arange(0, 8)

    neg_inf = -3.4028234663852886e38

    logits = tl.load(logits_ptr + row * 256 + offs).to(tl.float32)
    scores = tl.sigmoid(logits)
    bias = tl.load(bias_ptr + offs).to(tl.float32)
    choice = scores + bias

    group_scores = tl.full((8,), neg_inf, dtype=tl.float32)

    for g in tl.static_range(0, 8):
        in_g = (offs >= g * 32) & (offs < (g + 1) * 32)
        vals_g = tl.where(in_g, choice, neg_inf)

        m1 = tl.max(vals_g, axis=0)
        idx1 = tl.min(tl.where(vals_g == m1, offs, 2147483647), axis=0)
        m2 = tl.max(tl.where(offs == idx1, neg_inf, vals_g), axis=0)

        group_scores = tl.where(gids == g, m1 + m2, group_scores)

    work_g = group_scores
    selected_g = gids < 0

    for _ in tl.static_range(0, 4):
        mv = tl.max(work_g, axis=0)
        gid = tl.min(tl.where(work_g == mv, gids, 2147483647), axis=0)
        selected_g = selected_g | (gids == gid)
        work_g = tl.where(gids == gid, neg_inf, work_g)

    expert_group = offs // 32
    keep = offs < 0

    for g in tl.static_range(0, 8):
        sg = tl.max(tl.where((gids == g) & selected_g, 1, 0), axis=0)
        keep = keep | ((expert_group == g) & (sg == 1))

    vals = tl.where(keep, choice, neg_inf)

    idx_vec = tl.zeros((8,), dtype=tl.int64)
    w_vec = tl.zeros((8,), dtype=tl.float32)
    denom = tl.zeros((), dtype=tl.float32)

    for j in tl.static_range(0, 8):
        mv = tl.max(vals, axis=0)
        idx = tl.min(tl.where(vals == mv, offs, 2147483647), axis=0)
        wt = tl.sum(tl.where(offs == idx, scores, 0.0), axis=0)

        idx_vec = tl.where(ks == j, idx.to(tl.int64), idx_vec)
        w_vec = tl.where(ks == j, wt, w_vec)
        denom += wt

        vals = tl.where(offs == idx, neg_inf, vals)

    inv_denom = routed_scaling_factor / (denom + 1.0e-20)
    out_w = w_vec * inv_denom

    tl.store(topk_idx_ptr + row * 8 + ks, idx_vec)
    tl.store(topk_weight_ptr + row * 8 + ks, out_w)


def kernel_function(hidden_states, weight, e_score_correction_bias,
                    scale_x, scale_w, routed_scaling_factor):
    T = hidden_states.shape[0]

    qx = torch.empty((T, 7168), device=hidden_states.device, dtype=torch.float8_e4m3fn)
    qwt = torch.empty((7168, 256), device=hidden_states.device, dtype=torch.float8_e4m3fn)
    logits = torch.empty((T, 256), device=hidden_states.device, dtype=torch.bfloat16)

    topk_idx = torch.empty((T, 8), device=hidden_states.device, dtype=torch.int64)
    topk_weight = torch.empty((T, 8), device=hidden_states.device, dtype=torch.float32)

    _quant_x_kernel[(triton.cdiv(T, 2), 7)](
        hidden_states, scale_x, qx, T,
        BLOCK_M=2, GROUP_BLOCKS=8, BK=128,
        num_warps=4,
    )

    _quant_w_kernel[(56, 16)](
        weight, scale_w, qwt,
        BE=16, BK=128,
        num_warps=4,
    )

    _scaled_gemm_kernel[(triton.cdiv(T, 16), 4)](
        qx, qwt, scale_x, scale_w, logits, T,
        BM=16, BN=64, BK=128,
        num_warps=4, num_stages=3,
    )

    _routing_kernel[(T,)](
        logits, e_score_correction_bias, topk_idx, topk_weight,
        routed_scaling_factor, T,
        num_warps=8,
    )

    return topk_idx, topk_weight