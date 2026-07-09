"""Grouped fp8 groupwise GEMM on Blackwell tcgen05, built on tc_groupwise_gemm.

Each group g computes out[start:end] = dequant(a[start:end]) @ dequant(b[g]).T with
per-128-block groupwise scales, where [start,end) = [m_indptr[g], m_indptr[g+1]).
Implemented as one tensor-core launch per group (the single-GEMM groupwise kernel),
compile-cached per (Mg, N, K). scale_major_mode='K' only.
"""

from __future__ import annotations

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda

from tc_groupwise_gemm import DenseGemmKernel
from tc_persist_gemm import PersistentDenseGemmKernel
from tc_persist_t64 import PersistentDenseGemmT64

_MMA_TILER = (128, 128)
_CLUSTER = (1, 1)
_COMPILE_CACHE: dict = {}
_MAX_CLUSTERS_1 = None


def _max_clusters_1():
    """Persistent-wave capacity for cluster (1,1) (lazy: needs CUDA)."""
    global _MAX_CLUSTERS_1
    if _MAX_CLUSTERS_1 is None:
        _MAX_CLUSTERS_1 = cutlass_utils.HardwareInfo().get_max_active_clusters(1)
    return _MAX_CLUSTERS_1
def _stream():
    # Return the CURRENT stream each call (not cached) so the kernel launches on
    # torch's active stream — required for CUDA-graph capture (the capture stream)
    # and for correctness under stream contexts.
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def _fp8_dl(t: torch.Tensor, ld: int) -> cute.Tensor:
    """DLPack-safe cute view of a torch float8 tensor: some torch builds
    refuse fp8 in __dlpack__, so export the bytes as uint8 and set the cute
    element type explicitly (a same-itemsize dtype view works on any strides)."""
    cu = from_dlpack(t.view(torch.uint8), assumed_align=16)
    cu.element_type = cutlass.Float8E4M3FN
    return cu.mark_layout_dynamic(leading_dim=ld)


def _san(t: torch.Tensor, ld: int) -> torch.Tensor:
    """Force stride 1 on the leading dim when its size is 1. torch reports
    arbitrary strides for size-1 dims — even through .contiguous(): (1,4).t()
    .contiguous() keeps strides (1,4) — which trips mark_layout_dynamic(ld).
    A size-1 stride never affects addressing, so rewriting it is free. A size>1
    leading dim with stride != 1 is a real layout bug and is left to raise."""
    if t.size(ld) == 1 and t.stride(ld) != 1:
        st = list(t.stride()); st[ld] = 1
        t = t.as_strided(tuple(t.size()), tuple(st), t.storage_offset())
    return t


def _compiled(key, mk_args, st):
    """Return the compiled kernel for `key`; call mk_args() (fresh cute tensors)
    only on a cache miss (compile consumes them)."""
    c = _COMPILE_CACHE.get(key)
    if c is None:
        gemm = DenseGemmKernel(cutlass.Float32, False, _MMA_TILER, (1, 1), False)
        c = cute.compile(gemm, *mk_args(), st)
        _COMPILE_CACHE[key] = c
    return c


# Warp-specialized persistent TMA-store kernel: beats flashinfer on the batched
# (equal-groups) path. Its __call__ takes an extra Constexpr `max_active_clusters`
# baked at compile (passed to cute.compile, NOT to the runtime call).
# Two variants: 1-CTA (128,128) and 2-CTA (256,128) cluster (2,1). 2-CTA wins on
# large problems (2294 vs 2153 TF @4096x8192x8192 g=8; breaks the 8192^3 g=1 tie);
# 1-CTA wins when the grid would underfill (1144 vs 1095 @512x512x4096 g=8).
# NOTE (256,256) tiler is INCOMPATIBLE with the groupwise epilogue: the b_scale
# hoist assumes one N-tile == one 128-wide scale block.
def _compiled_persist(key, mk_args, st, use_2cta, tiler64=False):
    c = _COMPILE_CACHE.get(key)
    if c is None:
        if tiler64:
            # (64,128) tiles: the 2-row-epilogue variant (bit-identical to
            # the (128,128) kernel; ~2.1us faster when the 64-tile grid
            # fits one persistent wave - see tc_persist_t64 docstring).
            gemm = PersistentDenseGemmT64(
                cutlass.Float32, False, (64, 128), _CLUSTER, True
            )
            c = cute.compile(gemm, *mk_args(), _max_clusters_1(), st)
            _COMPILE_CACHE[key] = c
            return c
        tiler = (256, 128) if use_2cta else _MMA_TILER
        cluster = (2, 1) if use_2cta else _CLUSTER
        mac = cutlass_utils.HardwareInfo().get_max_active_clusters(
            cluster[0] * cluster[1]
        )
        gemm = PersistentDenseGemmKernel(
            cutlass.Float32, use_2cta, tiler, cluster, True  # use_tma_store=True
        )
        c = cute.compile(gemm, *mk_args(), mac, st)
        _COMPILE_CACHE[key] = c
    return c


def grouped_fp8_gemm(
    a: torch.Tensor,          # (M, K) fp8_e4m3
    a_scale: torch.Tensor,    # K: (M, K_blocks) | MN: (K_blocks, M) fp32
    b: torch.Tensor,          # (G, N, K) fp8_e4m3
    b_scale: torch.Tensor,    # K: (G, N_blocks, K_blocks) | MN: (G, K_blocks, N_blocks) fp32
    m_indptr: torch.Tensor,   # (G+1,) int32
    out_dtype: torch.dtype = torch.bfloat16,
    scale_major_mode: str = "K",
) -> torch.Tensor:
    M, K = a.shape
    G, N, K_b = b.shape
    assert K_b == K
    assert scale_major_mode in ("K", "MN")
    mn = scale_major_mode == "MN"
    if a_scale.dtype != torch.float32:
        a_scale = a_scale.float()
    if b_scale.dtype != torch.float32:
        b_scale = b_scale.float()
    out = torch.empty(M, N, device=a.device, dtype=out_dtype)
    st = _stream()

    # Fast path: equal groups (M divisible by G) -> one BATCHED launch (L = G).
    # G == 1 is the trivial equal-groups case (L = 1) and MUST take this path too:
    # the per-group fallback below host-syncs (m_indptr.cpu()) and uses the slower
    # non-persistent kernel. We key off M % G == 0 only (NO host sync) so the call
    # is CUDA-graph capturable; this is exact for equal-sized groups (the common
    # MoE case). Callers with unequal groups use the per-group path.
    if M % G == 0:
        m = M // G
        # (G*m,K)->(m,K,G); (G,N,K)->(N,K,G); (G*m,N)->(m,N,G); scales likewise.
        a_b = a.reshape(G, m, K).permute(1, 2, 0)
        b_b = b.permute(1, 2, 0)
        out_b = out.reshape(G, m, N).permute(1, 2, 0)
        # Scales: the kernel indexes a logical (rows, K_blocks, L) tensor with
        # scalar loads, so MN support is ZERO-COPY — permute the transposed
        # views into the same logical shape; only the strides (and the stride-1
        # mode marked dynamic below) differ. NOTE: MN views are slower than
        # K-major (each k_tile gather touches a fresh cache line vs K-major's
        # per-thread line reuse; L2 hit 92%->77%, 2227->1033 TF @big shape).
        # Prefer converting MN scales to K-major upstream (composed_kernel does).
        if mn:
            a_sb = a_scale.reshape(-1, G, m).permute(2, 0, 1)   # (m, KB, G)
            b_sb = b_scale.permute(2, 1, 0)                     # (NB, KB, G)
        else:
            a_sb = a_scale.reshape(G, m, -1).permute(1, 2, 0)   # (m, KB, G)
            b_sb = b_scale.permute(1, 2, 0)                     # (NB, KB, G)
        s_ld = 0 if mn else 1  # stride-1 mode of the scale views
        a_sb = _san(a_sb, s_ld); b_sb = _san(b_sb, s_ld)

        def _bargs():
            return (
                _fp8_dl(a_b, 1),
                _fp8_dl(b_b, 1),
                from_dlpack(out_b, assumed_align=16).mark_layout_dynamic(1),
                from_dlpack(a_sb, assumed_align=16).mark_layout_dynamic(s_ld),
                from_dlpack(b_sb, assumed_align=16).mark_layout_dynamic(s_ld),
            )

        # Winning path: warp-specialized persistent TMA-store kernel (beats
        # flashinfer). Requires N, K multiples of 128 (128-wide b_scale blocks);
        # any m works — the TMA store clamps OOB-padded M-tiles and the epilogue
        # clamps the a_scale row gather. Fall back only for non-128 N/K.
        if N % 128 == 0 and K % 128 == 0:
            # 2-CTA (256-M tiler) when m tiles by 256 and the problem is large
            # enough to fill the grid; measured crossover ~512 128x128 tiles.
            n_tiles_128 = (m // 128) * (N // 128) * G
            use_2cta = m % 256 == 0 and n_tiles_128 >= 512
            # (64,128) tiler when its tile grid fits ONE persistent wave:
            # smaller tiles fill more SMs at small/medium M (-2.1us measured
            # at m<=4352-class shapes, bit-identical output); beyond one
            # wave the quantization penalty makes it slower than (128,128).
            n_tiles_64 = ((m + 63) // 64) * (N // 128) * G
            tiler64 = (not use_2cta) and n_tiles_64 <= _max_clusters_1()
            _compiled_persist(
                ("persist_batched", m, N, K, G, use_2cta, tiler64, mn),
                _bargs, st, use_2cta, tiler64,
            )(*_bargs(), st)
        else:
            _compiled(("batched", m, N, K, G, mn), _bargs, st)(*_bargs(), st)
        return out

    # Variable-sized groups -> per-group launch (L = 1 batch each).
    a = a.contiguous(); b = b.contiguous()
    a_scale = a_scale.contiguous(); b_scale = b_scale.contiguous()
    offsets = m_indptr.detach().cpu().tolist()
    for g in range(G):
        s, e = int(offsets[g]), int(offsets[g + 1])
        if e <= s:
            continue
        Mg = e - s

        def _args(s=s, e=e, g=g, Mg=Mg):
            # Scale slices: logical (rows, K_blocks, 1) either way; MN uses the
            # zero-copy transposed view (stride-1 mode 0). assumed_align=4 because
            # a row/column slice base is only guaranteed fp32-aligned; scales are
            # scalar-loaded so alignment doesn't matter for performance.
            if mn:
                a_sv = a_scale[:, s:e].t().unsqueeze(-1)          # (Mg, KB, 1)
                b_sv = b_scale[g].t().unsqueeze(-1)               # (NB, KB, 1)
            else:
                a_sv = a_scale[s:e].unsqueeze(-1)                 # (Mg, KB, 1)
                b_sv = b_scale[g].unsqueeze(-1)                   # (NB, KB, 1)
            s_ld = 0 if mn else 1
            a_sv = _san(a_sv, s_ld); b_sv = _san(b_sv, s_ld)
            return (
                _fp8_dl(a[s:e].reshape(Mg, K, 1), 1),
                _fp8_dl(b[g].reshape(N, K, 1), 1),
                from_dlpack(out[s:e].reshape(Mg, N, 1), assumed_align=16).mark_layout_dynamic(1),
                from_dlpack(a_sv, assumed_align=4).mark_layout_dynamic(s_ld),
                from_dlpack(b_sv, assumed_align=4).mark_layout_dynamic(s_ld),
            )

        # Persistent TMA-store kernel per group (any Mg: OOB M-tiles are clamped
        # by the TMA store + a_scale gather clamp); same 2-CTA heuristic as the
        # batched path. Old non-persistent kernel only for non-128 N/K.
        if N % 128 == 0 and K % 128 == 0:
            n_tiles_128 = ((Mg + 127) // 128) * (N // 128)
            use_2cta = Mg % 256 == 0 and n_tiles_128 >= 512
            _compiled_persist(
                ("persist_group", Mg, N, K, use_2cta, mn), _args, st, use_2cta
            )(*_args(), st)
        else:
            _compiled((Mg, N, K, mn), _args, st)(*_args(), st)

    return out
