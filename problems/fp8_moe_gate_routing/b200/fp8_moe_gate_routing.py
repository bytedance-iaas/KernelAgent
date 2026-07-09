"""FP8 MoE gate routing (noaux_tc) in CuTe DSL — TC pipeline + CUDA graphs.

Round 3: per-stage graph timings showed the routing epilogue dominating
(47 us at 256 tokens, 108 us at 8192) because top-8 selection ran ~16K
serial unrolled ops on 8 threads per CTA. Rewritten warp-per-token: each
of the CTA's 8 warps owns one token — group top-2 via 4-lane merges,
top-8 via warp shuffle argmax (lowest-index tie-break), and the
torch-topk output order rebuilt with a shuffle prefix sum. No block-wide
syncs after the bias load.

Stages inside the per-shape CUDA graph (static I/O, copy-in per call):
  1. `_quant_kernel`   — vectorized bf16 -> fp8 quantization (activations:
                         1x128 scales; weight: 128x128), exact reference
                         round trip (clamp(x/s) -> e4m3 rn-satfinite).
  2. tcgen05 GEMM      — `tc_grouped.grouped_fp8_gemm` (G=1 dense,
                         sync-free, capture-safe) -> bf16 logits.
  3. `_routing_kernel` — sigmoid + bias, group top-2 sums, top-4 group
                         mask, top-8 experts in torch-topk order, normalize.
"""

import ctypes
import sys
from pathlib import Path

import torch

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.cute.arch as arch
from cutlass.cute.runtime import from_dlpack

# tcgen05 fp8 groupwise GEMM: same-dir copy first (b200 layout), else the
# fp8_group_gemm reference location.
_dir = Path(__file__).resolve().parent
if str(_dir) not in sys.path:
    sys.path.insert(0, str(_dir))
if not (_dir / "tc_grouped.py").exists():
    for _anc in _dir.parents:
        _tc = _anc / "fp8_group_gemm" / "b200" / "cutedsl"
        if (_tc / "tc_grouped.py").exists():
            sys.path.insert(0, str(_tc))
            break
from tc_grouped import grouped_fp8_gemm

_NE = 256    # n_routed_experts
_NG = 8      # expert groups
_GSZ = 32    # experts per group
_TOPG = 4    # groups kept
_TOPK = 8    # experts kept per token
_QBLK = 128  # quantization block
_BM = 8      # tokens per routing CTA (one warp per token)
_VEC = 8     # fp8 conversion vector width
_QV = 16     # quant kernel elements per thread (16B fp8 store)


def _stream():
    return cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)


def _fp8_cute(t: torch.Tensor, ld: int, assumed_align: int = 16):
    """DLPack-safe cute view of a torch float8 tensor. Some torch builds
    refuse fp8 in __dlpack__ ("float8 types are not supported by dlpack"),
    so export the bytes as uint8 and set the cute element type explicitly
    (same trick as cutlass.utils.create_cute_tensor_for_fp8)."""
    cu = from_dlpack(t.view(torch.uint8), assumed_align=assumed_align)
    cu.element_type = cutlass.Float8E4M3FN
    return cu.mark_layout_dynamic(leading_dim=ld)


# --------------------------------------------------------------------------- #
# Stage 1: quantize bf16 -> fp8 with the reference's clamp + rn-satfinite
# --------------------------------------------------------------------------- #
# Two specialized kernels instead of one with a Constexpr mode flag:
# `cutlass.const_expr(<kernel param>)` is rejected as dynamic by some
# nvidia-cutlass-dsl versions (e.g. the SOL-ExecBench venv), and scalar
# div/clamp avoids TensorSSA-arithmetic version drift. The true IEEE
# per-element div keeps the fp8 round trip bit-exact with the reference.
@cute.kernel
def _quant_a_kernel(
    src: cute.Tensor,   # (R, K) bf16, contiguous
    scale: cute.Tensor,  # (R, K/128) f32 activation scales (BlockWise1x128)
    dst: cute.Tensor,   # (R, K) fp8_e4m3, contiguous
    R: cutlass.Int32,
    K: cutlass.Int32,
):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    row = bx
    c0 = tx * _QV

    atom_ld = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16,
                                  num_bits_per_copy=128)
    atom_st = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float8E4M3FN,
                                  num_bits_per_copy=128)
    fsrc = cute.make_rmem_tensor(_QV, cutlass.BFloat16)
    f32 = cute.make_rmem_tensor(_QV, cutlass.Float32)
    f8 = cute.make_rmem_tensor(_QV, cutlass.Float8E4M3FN)
    if True:   # exact-fit grid: K // _QV threads per row
        # 16B-aligned vector views (c0 multiple of _QV, K multiple of 128)
        off = row * K + c0
        sp = cute.make_ptr(cutlass.BFloat16, src.iterator.toint() + off * 2,
                           src.memspace, assumed_align=16)
        dp = cute.make_ptr(cutlass.Float8E4M3FN, dst.iterator.toint() + off,
                           dst.memspace, assumed_align=16)
        gsrc = cute.make_tensor(sp, cute.make_layout(_QV))
        gdst = cute.make_tensor(dp, cute.make_layout(_QV))
        s = scale[row, c0 // _QBLK]
        cute.copy(atom_ld, gsrc, fsrc)
        for t in range(_QV):
            f32[t] = cutlass.max(cutlass.min(fsrc[t].to(cutlass.Float32) / s, 448.0), -448.0)
        f8.store(f32.load().to(cutlass.Float8E4M3FN))
        cute.copy(atom_st, f8, gdst)


@cute.kernel
def _quant_w_kernel(
    src: cute.Tensor,   # (R, K) bf16, contiguous
    scale: cute.Tensor,  # (K/128, R/128) f32 weight scales (BlockWise128x128)
    dst: cute.Tensor,   # (R, K) fp8_e4m3, contiguous
    R: cutlass.Int32,
    K: cutlass.Int32,
):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    row = bx
    c0 = tx * _QV

    atom_ld = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16,
                                  num_bits_per_copy=128)
    atom_st = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float8E4M3FN,
                                  num_bits_per_copy=128)
    fsrc = cute.make_rmem_tensor(_QV, cutlass.BFloat16)
    f32 = cute.make_rmem_tensor(_QV, cutlass.Float32)
    f8 = cute.make_rmem_tensor(_QV, cutlass.Float8E4M3FN)
    if True:   # exact-fit grid: K // _QV threads per row
        off = row * K + c0
        sp = cute.make_ptr(cutlass.BFloat16, src.iterator.toint() + off * 2,
                           src.memspace, assumed_align=16)
        dp = cute.make_ptr(cutlass.Float8E4M3FN, dst.iterator.toint() + off,
                           dst.memspace, assumed_align=16)
        gsrc = cute.make_tensor(sp, cute.make_layout(_QV))
        gdst = cute.make_tensor(dp, cute.make_layout(_QV))
        s = scale[c0 // _QBLK, row // _QBLK]
        cute.copy(atom_ld, gsrc, fsrc)
        for t in range(_QV):
            f32[t] = cutlass.max(cutlass.min(fsrc[t].to(cutlass.Float32) / s, 448.0), -448.0)
        f8.store(f32.load().to(cutlass.Float8E4M3FN))
        cute.copy(atom_st, f8, gdst)


# --------------------------------------------------------------------------- #
# Stage 3: routing epilogue, one warp per token
# --------------------------------------------------------------------------- #
@cute.kernel
def _routing_kernel(
    logits: cute.Tensor,   # (M, NE) bf16
    bias: cute.Tensor,     # (NE,)   bf16
    rsf_t: cute.Tensor,    # (1,)    f32 routed_scaling_factor
    out_idx: cute.Tensor,  # (M, TOPK) int64
    out_w: cute.Tensor,    # (M, TOPK) f32
    M: cutlass.Int32,
):
    tx, ty, _ = cute.arch.thread_idx()   # block = [32, 8]: lane tx, warp/token ty
    bx, _, _ = cute.arch.block_idx()
    lane = tx
    tid = tx + ty * 32
    tok = ty
    grow = bx * _BM + ty

    smem = cutlass.utils.SmemAllocator()
    sScores = smem.allocate_tensor(cutlass.Float32, cute.make_layout((_BM, _NE)), byte_alignment=16)
    sBias = smem.allocate_tensor(cutlass.Float32, cute.make_layout(_NE), byte_alignment=16)
    sGroup = smem.allocate_tensor(cutlass.Float32, cute.make_layout((_BM, _NG)), byte_alignment=16)
    sGmask = smem.allocate_tensor(cutlass.Int32, cute.make_layout((_BM, _NG)), byte_alignment=16)
    sTop = smem.allocate_tensor(cutlass.Int32, cute.make_layout((_BM, _TOPK)), byte_alignment=16)
    sDenom = smem.allocate_tensor(cutlass.Float32, cute.make_layout(_BM), byte_alignment=16)

    sBias[tid] = bias[tid].to(cutlass.Float32)
    if tid < _BM * _TOPK:
        # defensive: NaN scores (e.g. capture warmup over garbage) make every
        # comparison false and leave slots unwritten; a preset in-range index
        # keeps the final gather in bounds no matter what.
        sTop[tid // _TOPK, tid % _TOPK] = cutlass.Int32(0)
    cute.arch.sync_threads()

    if grow < M:
        # scores_for_choice = sigmoid(f32(bf16 logit)) + bias; lane owns
        # experts [lane*8, lane*8+8) so ascending index == (lane, j) order.
        # Hot phases run on the lane's own 8 registers — smem is written once
        # and read only in the final gather (the strided [tok, lane*8+j]
        # pattern is 8-way bank-conflicted, so keep it off the hot loops).
        rsc = cute.make_rmem_tensor(8, cutlass.Float32)
        for j in range(8):
            e = lane * 8 + j
            lb = logits[grow, e].to(cutlass.Float32)
            # fast sigmoid: hardware exp2 + approx rcp (~1e-6 rel vs torch's
            # true-div sigmoid; deterministic, so exact ties still tie and
            # the matched-ratio tolerance absorbs the ulp-level shifts)
            score = arch.rcp_approx(1.0 + arch.exp2(lb * -1.4426950408889634))
            rsc[j] = score + sBias[e]
            sScores[tok, e] = rsc[j]

        # group top-2 sums: the lane's 8 experts ARE quarter lane%4 of group
        # lane//4, so the local pass reads only registers.
        g = lane // 4
        m1 = cutlass.Float32(-3.0e38)
        m2 = cutlass.Float32(-3.0e38)
        for j in range(8):
            v = rsc[j]
            if v > m1:
                m2 = m1
                m1 = v
            elif v > m2:
                m2 = v
        for off in (2, 1):   # merge (m1, m2) pairs across the 4-lane group
            o1 = arch.shuffle_sync_down(m1, off)
            o2 = arch.shuffle_sync_down(m2, off)
            lo = cutlass.min(m1, o1)
            m1 = cutlass.max(m1, o1)
            m2 = cutlass.max(lo, cutlass.max(m2, o2))
        if lane % 4 == 0:
            sGroup[tok, g] = m1 + m2
            sGmask[tok, g] = cutlass.Int32(0)
        cute.arch.sync_warp()

        # top-4 groups (tiny): lane 0 serial, strict > keeps lowest index on ties
        if lane == 0:
            for it in range(_TOPG):
                best = cutlass.Float32(-3.0e38)
                bi = cutlass.Int32(0)
                for gg in range(_NG):
                    v = sGroup[tok, gg]
                    if sGmask[tok, gg] == 0 and v > best:
                        best = v
                        bi = gg
                sGmask[tok, bi] = cutlass.Int32(1)
        cute.arch.sync_warp()

        # top-8 experts: 8 rounds of hardware warp redux (MIO-throttle was the
        # top stall with the 5-step shuffle reduce: 2 redux ops/round vs 10).
        # Tie-break matches torch: max value, then lowest expert index.
        mflag = sGmask[tok, lane // 4]          # lane's experts share one group
        taken = cutlass.Int32(0)                # bit j = lane's expert j taken
        vkth = cutlass.Float32(-3.0e38)
        for it in range(_TOPK):
            lv = cutlass.Float32(-3.0e38)
            li = cutlass.Int32(_NE)
            for j in range(8):
                v = rsc[j]
                if mflag == 1 and ((taken >> j) & 1) == 0 and v > lv:
                    lv = v
                    li = lane * 8 + j
            bmax = arch.warp_redux_sync(lv, "fmax")
            cand = li if lv == bmax else cutlass.Int32(_NE)
            bi = arch.warp_redux_sync(cand, "min")
            if bi // 8 == lane:                 # winner marks its slot taken
                taken = taken | (1 << (bi % 8))
            vkth = bmax

        # torch CUDA topk(sorted=False) layout: strictly-greater indices
        # ascending, then k-th-value ties ascending until k slots filled.
        # Slot = warp exclusive prefix sum of per-lane match counts.
        c1 = cutlass.Int32(0)
        for j in range(8):
            if mflag == 1 and rsc[j] > vkth:
                c1 = c1 + 1
        incl = c1
        for off in (1, 2, 4, 8, 16):
            n = arch.shuffle_sync_up(incl, off, mask_and_clamp=0)
            if lane >= off:
                incl = incl + n
        total1 = arch.shuffle_sync(incl, 31)
        base = incl - c1                        # exclusive prefix
        k = cutlass.Int32(0)
        for j in range(8):
            if mflag == 1 and rsc[j] > vkth:
                if base + k < _TOPK:
                    sTop[tok, base + k] = lane * 8 + j
                k = k + 1
        c2 = cutlass.Int32(0)
        for j in range(8):
            if mflag == 1 and rsc[j] == vkth:
                c2 = c2 + 1
        incl2 = c2
        for off in (1, 2, 4, 8, 16):
            n = arch.shuffle_sync_up(incl2, off, mask_and_clamp=0)
            if lane >= off:
                incl2 = incl2 + n
        base2 = total1 + incl2 - c2
        k = cutlass.Int32(0)
        for j in range(8):
            if mflag == 1 and rsc[j] == vkth:
                if base2 + k < _TOPK:
                    sTop[tok, base2 + k] = lane * 8 + j
                k = k + 1
        cute.arch.sync_warp()

        # normalize gathered sigmoid scores (undo bias), write outputs
        if lane == 0:
            d = cutlass.Float32(0.0)
            for j in range(_TOPK):
                ej = sTop[tok, j]
                d = d + (sScores[tok, ej] - sBias[ej])
            sDenom[tok] = d + 1e-20
        cute.arch.sync_warp()
        if lane < _TOPK:
            ej = sTop[tok, lane]
            sc = sScores[tok, ej] - sBias[ej]
            out_idx[grow, lane] = ej.to(cutlass.Int64)
            out_w[grow, lane] = sc / sDenom[tok] * rsf_t[0]


@cute.jit
def _quant_a_launch(
    src: cute.Tensor, scale: cute.Tensor, dst: cute.Tensor,
    R: cutlass.Int32, K: cutlass.Int32, stream: cuda_driver.CUstream,
    GRID_R: cutlass.Constexpr, GRID_C: cutlass.Constexpr,
):
    _quant_a_kernel(src, scale, dst, R, K).launch(
        grid=[GRID_R, 1, 1], block=[GRID_C, 1, 1], stream=stream)


@cute.jit
def _quant_w_launch(
    src: cute.Tensor, scale: cute.Tensor, dst: cute.Tensor,
    R: cutlass.Int32, K: cutlass.Int32, stream: cuda_driver.CUstream,
    GRID_R: cutlass.Constexpr, GRID_C: cutlass.Constexpr,
):
    _quant_w_kernel(src, scale, dst, R, K).launch(
        grid=[GRID_R, 1, 1], block=[GRID_C, 1, 1], stream=stream)


@cute.jit
def _routing_launch(
    logits: cute.Tensor, bias: cute.Tensor, rsf_t: cute.Tensor,
    out_idx: cute.Tensor, out_w: cute.Tensor,
    M: cutlass.Int32, stream: cuda_driver.CUstream, GRID_M: cutlass.Constexpr,
):
    _routing_kernel(logits, bias, rsf_t, out_idx, out_w, M).launch(
        grid=[GRID_M, 1, 1], block=[32, 8, 1], stream=stream)


_COMPILE_CACHE: dict = {}


def _quantize(src, scale, dst):
    R, K = src.shape
    amode = scale.shape[0] == R
    launch = _quant_a_launch if amode else _quant_w_launch
    assert K % _QV == 0
    grid_c = K // _QV          # = threads per block (one block per row)
    st = _stream()

    def _dyn():
        return (from_dlpack(src, assumed_align=16).mark_layout_dynamic(1),
                from_dlpack(scale).mark_layout_dynamic(1),
                _fp8_cute(dst, 1),
                cutlass.Int32(R), cutlass.Int32(K), st)

    key = ("quant", R, K, amode)
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(launch, *_dyn(), R, grid_c)
        _COMPILE_CACHE[key] = compiled
    compiled(*_dyn())


def _routing(logits, bias, rsf_t, out_idx, out_w):
    M = logits.shape[0]
    grid_m = (M + _BM - 1) // _BM
    st = _stream()

    def _dyn():
        return (from_dlpack(logits).mark_layout_dynamic(1),
                from_dlpack(bias).mark_layout_dynamic(0),
                from_dlpack(rsf_t).mark_layout_dynamic(0),
                from_dlpack(out_idx).mark_layout_dynamic(1),
                from_dlpack(out_w).mark_layout_dynamic(1),
                cutlass.Int32(M), st)

    key = ("routing", M)
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(_routing_launch, *_dyn(), grid_m)
        _COMPILE_CACHE[key] = compiled
    compiled(*_dyn())


class _Workspaces:
    """Per-(M,K) static intermediates + outputs, shared by all graphs of that shape."""

    def __init__(self, M: int, K: int, dev: torch.device):
        self.qx = torch.empty(M, K, device=dev, dtype=torch.float8_e4m3fn)
        self.qw = torch.empty(_NE, K, device=dev, dtype=torch.float8_e4m3fn)
        self.sw_nk = torch.empty(1, _NE // _QBLK, K // _QBLK, device=dev, dtype=torch.float32)
        self.indptr = torch.tensor([0, M], device=dev, dtype=torch.int32)
        self.rsf = torch.empty(1, device=dev, dtype=torch.float32)
        self.rsf_val = None    # host cache: skip the fill_ when unchanged
        self.out_idx = torch.empty(M, _TOPK, device=dev, dtype=torch.int64)
        self.out_w = torch.empty(M, _TOPK, device=dev, dtype=torch.float32)
        # Staging buffers for the copy-in path. Benign-initialized because
        # the copy-in graph's capture warmup traces the pipeline over these
        # buffers BEFORE any real data is staged: garbage scales would make
        # NaN scores in the routing warmup (zeros would too, via x/0).
        self.x = torch.zeros(M, K, device=dev, dtype=torch.bfloat16)
        self.w = torch.zeros(_NE, K, device=dev, dtype=torch.bfloat16)
        self.b = torch.zeros(_NE, device=dev, dtype=torch.bfloat16)
        self.sx = torch.ones(M, K // _QBLK, device=dev, dtype=torch.float32)
        self.sw = torch.ones(K // _QBLK, _NE // _QBLK, device=dev, dtype=torch.float32)


def _pipeline(x, w, b, sx, sw, ws):
    # quant_w + scale transpose are off the critical path (qx -> gemm ->
    # routing): fork them onto a side stream; graph capture records the DAG.
    main = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    side.wait_stream(main)
    with torch.cuda.stream(side):
        _quantize(w, sw, ws.qw)
        ws.sw_nk.copy_(sw.t().unsqueeze(0))
    _quantize(x, sx, ws.qx)
    main.wait_stream(side)
    logits = grouped_fp8_gemm(ws.qx, sx, ws.qw.unsqueeze(0), ws.sw_nk,
                              ws.indptr, out_dtype=torch.bfloat16)
    _routing(logits, b, ws.rsf, ws.out_idx, ws.out_w)


def _capture(fn, keep_graph=False):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):   # warmup: also JITs shape-specialized kernels
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=keep_graph)
    with torch.cuda.graph(g):
        fn()
    if keep_graph:
        g.instantiate()
    return g


# --------------------------------------------------------------------------- #
# Graph-exec pointer patching: after capturing the pipeline ONCE per shape
# (over the staging buffers), each call updates the captured kernel nodes'
# input-pointer slots to the caller's CURRENT buffers via
# cuGraphExecKernelNodeSetParams and replays — true zero-copy on fresh
# tensors with no re-capture. Slots are found EMPIRICALLY: scan every kernel
# node's parameter blobs (cuFuncGetParamInfo gives offset/size) for 8-byte
# words equal to a staging-buffer base pointer. The pipeline is exactly 5
# kernel nodes (quant_w, sw_nk copy, quant_x, gemm, routing); all five
# caller inputs (x, w, b, sx, sw) resolve to 7 slots. Patch cost ~10us,
# and it is skipped entirely when the caller's pointer set is unchanged
# (the allocator recycles one dominant set on fresh-per-call harnesses).
# --------------------------------------------------------------------------- #

def _drv_ok(ret):
    return int(ret[0]) == 0


class _GraphEntry:
    __slots__ = ("g", "exec_h", "table", "node_params", "cur_ptrs", "ok")

    def __init__(self, g):
        self.g = g
        self.exec_h = None
        self.table = None        # [(node, param_host_addr, byte_off, input_idx)]
        self.node_params = None  # {int(node): (node, CUDA_KERNEL_NODE_PARAMS)}
        self.cur_ptrs = None
        self.ok = False


def _scan_ptr_table(entry, staged_ptrs):
    """Locate the input-pointer slots inside the captured graph's kernel
    nodes. Returns False (entry.ok stays False) on any surprise; the copy-in
    path then serves every call."""
    cu = cuda_driver
    try:
        graph = cu.CUgraph(entry.g.raw_cuda_graph())
        r = cu.cuGraphGetNodes(graph, 32)
        if not _drv_ok(r):
            return False
        nodes, n = r[1], int(r[2])
        want = {p: i for i, p in enumerate(staged_ptrs)}
        table, node_params, found = [], {}, set()
        for node in nodes[:n]:
            r = cu.cuGraphNodeGetType(node)
            if not _drv_ok(r) or 'KERNEL' not in str(r[1]):
                continue
            r = cu.cuGraphKernelNodeGetParams(node)
            if not _drv_ok(r):
                return False
            params = r[1]
            kp = int(params.kernelParams)
            if not kp or int(params.extra):
                return False
            for i in range(64):
                r = cu.cuFuncGetParamInfo(params.func, i)
                if not _drv_ok(r):
                    break
                size = int(r[2])
                pa = ctypes.cast(kp, ctypes.POINTER(ctypes.c_void_p))[i]
                if not pa or size < 8:
                    continue
                words = ctypes.cast(pa, ctypes.POINTER(ctypes.c_uint64))
                for w in range(size // 8):
                    idx = want.get(words[w])
                    if idx is not None:
                        table.append((node, int(pa), w * 8, idx))
                        found.add(idx)
                        if int(node) not in node_params:
                            node_params[int(node)] = (node, params)
        if found != {0, 1, 2, 3, 4}:
            return False
        entry.exec_h = cu.CUgraphExec(entry.g.raw_cuda_graph_exec())
        entry.table = table
        entry.node_params = node_params
        entry.cur_ptrs = tuple(staged_ptrs)
        entry.ok = True
        return True
    except Exception:
        return False


def _patch_ptrs(entry, ptrs):
    """Point the exec's input slots at `ptrs`; True on success."""
    cu = cuda_driver
    touched = set()
    for node, pa, off, idx in entry.table:
        ctypes.cast(pa + off, ctypes.POINTER(ctypes.c_uint64))[0] = ptrs[idx]
        touched.add(int(node))
    for key, (node, params) in entry.node_params.items():
        if key in touched:
            if not _drv_ok(cu.cuGraphExecKernelNodeSetParams(
                    entry.exec_h, node, params)):
                # partial patches leave the exec in an unknown state: make
                # sure the fallback recaptures rather than replaying it.
                entry.cur_ptrs = ("broken",)
                return False
    entry.cur_ptrs = tuple(ptrs)
    return True


_WORKSPACES: dict = {}
_GRAPHS: dict = {}        # (M, K) -> _GraphEntry (patched-exec dispatch)
# vestigial names kept for external probes / compat:
_PTR_GRAPHS: dict = {}
_COPY_GRAPHS = _GRAPHS


def _workspaces(M, K, dev):
    ws = _WORKSPACES.get((M, K))
    if ws is None:
        ws = _Workspaces(M, K, dev)
        _WORKSPACES[(M, K)] = ws
    return ws


def kernel_function(
    hidden_states: torch.Tensor,           # (M, 7168) bf16
    weight: torch.Tensor,                  # (256, 7168) bf16
    e_score_correction_bias: torch.Tensor,  # (256,) bf16
    scale_x: torch.Tensor,                 # (M, 56) f32
    scale_w: torch.Tensor,                 # (56, 2) f32
    routed_scaling_factor: float,
):
    M, K = hidden_states.shape
    assert weight.shape[0] == _NE and K % _QBLK == 0
    dev = hidden_states.device
    ws = _workspaces(M, K, dev)
    inputs = (hidden_states, weight, e_score_correction_bias, scale_x, scale_w)

    entry = _GRAPHS.get((M, K))
    if entry is None:
        # One capture per shape, over the benign-initialized staging buffers
        # (keep_graph=True so the raw graph/exec handles stay accessible),
        # then locate the input-pointer slots. If scanning fails, entry.ok
        # stays False and the copy-in path serves every call.
        entry = _GraphEntry(_capture(
            lambda: _pipeline(ws.x, ws.w, ws.b, ws.sx, ws.sw, ws),
            keep_graph=True,
        ))
        staged = (ws.x.data_ptr(), ws.w.data_ptr(), ws.b.data_ptr(),
                  ws.sx.data_ptr(), ws.sw.data_ptr())
        _scan_ptr_table(entry, staged)
        _GRAPHS[(M, K)] = entry

    rv = float(routed_scaling_factor)
    if ws.rsf_val != rv:
        ws.rsf.fill_(rv)
        ws.rsf_val = rv

    if entry.ok and all(t.is_contiguous() for t in inputs):
        # Zero-copy: point the captured exec's kernel nodes at the caller's
        # CURRENT buffers (~10us, and only when the pointer set changed —
        # allocator recycling makes fresh-per-call harnesses reuse one set),
        # then replay.
        ptrs = tuple(t.data_ptr() for t in inputs)
        if ptrs == entry.cur_ptrs or _patch_ptrs(entry, ptrs):
            entry.g.replay()
            return ws.out_idx, ws.out_w
        entry.ok = False  # patching failed: fall through to copy-in forever

    # Copy-in fallback (non-contiguous inputs or patch machinery
    # unavailable): retarget the exec at the staging buffers if needed,
    # stage the inputs, replay.
    staged = (ws.x.data_ptr(), ws.w.data_ptr(), ws.b.data_ptr(),
              ws.sx.data_ptr(), ws.sw.data_ptr())
    if entry.ok and entry.cur_ptrs != staged and not _patch_ptrs(entry, staged):
        entry.ok = False
    if not entry.ok and entry.cur_ptrs is not None and entry.cur_ptrs != staged:
        # exec may be stuck pointing at stale caller buffers: recapture a
        # plain copy-in graph once and retire the patched one.
        entry.g = _capture(lambda: _pipeline(ws.x, ws.w, ws.b, ws.sx, ws.sw, ws))
        entry.cur_ptrs = None
    ws.x.copy_(hidden_states, non_blocking=True)
    ws.w.copy_(weight, non_blocking=True)
    ws.b.copy_(e_score_correction_bias, non_blocking=True)
    ws.sx.copy_(scale_x, non_blocking=True)
    ws.sw.copy_(scale_w, non_blocking=True)
    entry.g.replay()
    return ws.out_idx, ws.out_w


# SOL-ExecBench solution entrypoint convention: expose the callable as `run`.
run = kernel_function
