"""tile64 (64,128) variant of the persistent tcgen05 groupwise GEMM.

Subclass of tc_persist_gemm.PersistentDenseGemmKernel overriding ONLY the
device kernel: the groupwise epilogue's a_scale gather is generalized from
one to TWO M-rows per thread. With a 64-row CTA tile the t2r value modes are
((2,2,8),1,1) and exactly one extent-2 mode carries the M-row delta; the fix
computes row0/row1 and per-mode deltas once per tile from the coordinate
tensor, then per k_tile selects among four folded sa*sb scalars via
INTEGER-exact row indices (an fp blend `s0 + w*(s1-s0)` is NOT bit-exact).
The per-element pick is fully static. For 128-row tiles all deltas are zero
and the kernel degenerates bit-exactly to the original single-row path.

Verified BIT-IDENTICAL to the (128,128) original kernel on random fp8 inputs
at M=256/250/4352/5888/8192 (same 128-K-block promotion order). Perf (N=256,
K=7168, G=1, graph-bench): 16.4/18.4us @M=256/4352 vs 18.5/20.5 for tile128;
slower past 148 64-tiles (wave quantization) - dispatch accordingly
(tc_grouped.grouped_fp8_gemm does).
"""

from typing import Optional, Union

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, tcgen05

from tc_persist_gemm import PersistentDenseGemmKernel


class PersistentDenseGemmT64(PersistentDenseGemmKernel):
    """PersistentDenseGemmKernel with the 2-row groupwise epilogue; use with
    mma_tiler_mn=(64,128) (also valid, bit-identical, for (128,128))."""

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        a_scale: cute.Tensor,
        b_scale: cute.Tensor,
        epilogue_op: cutlass.Constexpr,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(self.use_tma_store):
                cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_acc_stage * 2
            ]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * (
            2 if use_2cta_instrs else 1
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_dealloc_sync_bar_id,
            num_threads=32 * len(self.epilogue_warp_id),
        )
        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        #
        # Setup smem tensor A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        #
        # Compute multicast mask for A/B buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)
        # Identity coordinate tensor over C, tiled + partitioned like tCgC, so the
        # epilogue can map each accumulator element to its (m, n, l) and gather the
        # groupwise scales a_scale[m, kb, l] and b_scale[n // BLK, kb, l].
        idC = cute.make_identity_tensor(mC_mnl.shape)
        cC_mnl = cute.local_tile(
            idC, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        tCcC = thr_mma.partition_C(cC_mnl)

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        #
        # Construct the scheduler
        #
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
        )
        work_tile = tile_sched.initial_work_tile_info()

        #
        # Specialized TMA load warp
        #

        if warp_idx == self.tma_warp_id:
            #
            # Persistent tile scheduling loop
            #

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), RestK)
                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), RestK)
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                ab_producer.reset()
                peek_ab_empty_status = ab_producer.try_acquire()

                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Conditionally wait for AB buffer empty
                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)

                    # TMA load A/B
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, handle.count)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, handle.count)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=b_full_mcast_mask,
                    )

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait A/B buffer empty
            #
            ab_producer.tail()

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Persistent tile scheduling loop
            #

            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer.reset()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()

                #
                # Groupwise Mma mainloop: each k_tile is ONE 128-wide scale block.
                # Commit its partial to a fresh acc bank so the epilogue can promote
                # (scale + accumulate) it. num_acc_stage=2 => MMA of block k+1 fills
                # one bank while the epilogue drains block k from the other.
                #
                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        acc_pipeline.producer_acquire(acc_producer_state)
                        # Fresh accumulation into this block's bank.
                        tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblk_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblk_crd = (None, None, kblk_idx, handle.index)
                            cute.gemm(
                                tiled_mma, tCtAcc, tCrA[kblk_crd], tCrB[kblk_crd], tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        handle.release()
                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()
                        acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)

        sC = None
        if cutlass.const_expr(self.use_tma_store):
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC = smem.allocate_tensor(
                element_type=self.c_dtype,
                layout=c_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=c_smem_layout_staged.inner,
            )

        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Persistent tile scheduling loop for epilogue (custom groupwise
            # per-block promotion; SIMT store).
            #
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            # t2r + coordinate partitions over the STAGED accumulator (set up once),
            # mirroring cutlass.utils.gemm.sm100.epilogue.
            tCtAcc_t = utils.gemm.sm100.transform_partitioned_tensor_layout(tCtAcc_base)
            tCgC_t = utils.gemm.sm100.transform_partitioned_tensor_layout(tCgC)
            tCcC_t = utils.gemm.sm100.transform_partitioned_tensor_layout(tCcC)
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = (
                utils.gemm.sm100.epilogue_tmem_copy_and_partition(
                    self, tidx, tCtAcc_t, tCgC_t, epi_tile, self.use_2cta_instrs
                )
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            gC_epi = cute.flat_divide(tCgC_t, epi_tile)
            cC_epi = cute.flat_divide(tCcC_t, epi_tile)
            tTR_gC_full = thr_copy_t2r.partition_D(gC_epi)
            tTR_cC_full = thr_copy_t2r.partition_D(cC_epi)
            tTR_rC = cute.make_rmem_tensor(
                tTR_gC_full[(None, None, None, 0, 0, 0, 0, 0)].shape, self.c_dtype
            )
            simt_atom = None
            if cutlass.const_expr(not self.use_tma_store):
                mclD = cute.max_common_layout(
                    tTR_rC.layout, tTR_gC_full[(None, None, None, 0, 0, 0, 0, 0)].layout
                )
                num_bits_per_copy = min(
                    tTR_gC_full.iterator.alignment * 8, cute.size(mclD) * self.c_dtype.width
                )
                simt_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyR2GOp(), self.c_dtype, num_bits_per_copy=num_bits_per_copy
                )
            _bank0 = cute.group_modes(
                tTR_tAcc_base[(None, None, None, None, None, 0)], 3,
                cute.rank(tTR_tAcc_base[(None, None, None, None, None, 0)]),
            )
            subtile_cnt = cute.size(_bank0.shape, mode=[3])
            tTR_rMaster = cute.make_rmem_tensor(
                (*tTR_rAcc.shape, subtile_cnt), self.acc_dtype
            )

            # TMA-store epilogue setup: master (rmem) -> r2s (smem sC) -> TMA -> C.
            # Replaces the SIMT store; matches the fast base's use_tma_store path.
            c_pipeline = None
            tiled_copy_r2s = tRS_rC = tRS_sC = None
            bSG_sC = bSG_gC_partitioned = None
            epilog_sync_barrier = None
            if cutlass.const_expr(self.use_tma_store):
                tiled_copy_r2s, tRS_rC, tRS_sC = (
                    utils.gemm.sm100.epilogue_smem_copy_and_partition(
                        self, tiled_copy_t2r, tTR_rC, tidx, sC
                    )
                )
                bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sC, 0, 2),
                    cute.group_modes(gC_epi, 0, 2),
                )
                epilog_sync_barrier = pipeline.NamedBarrier(
                    barrier_id=self.epilog_sync_bar_id,
                    num_threads=32 * len(self.epilogue_warp_id),
                )
                c_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.num_c_stage,
                    producer_group=pipeline.CooperativeGroup(
                        pipeline.Agent.Thread,
                        32 * len(self.epilogue_warp_id),
                        32 * len(self.epilogue_warp_id),
                    ),
                )

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                ll = mma_tile_coord_mnl[2]
                tTR_gC = cute.group_modes(
                    tTR_gC_full[(None, None, None, None, None, *mma_tile_coord_mnl)],
                    3, cute.rank(tTR_gC_full[(None, None, None, None, None, *mma_tile_coord_mnl)]),
                )
                tTR_cC = cute.group_modes(
                    tTR_cC_full[(None, None, None, None, None, *mma_tile_coord_mnl)],
                    3, cute.rank(tTR_cC_full[(None, None, None, None, None, *mma_tile_coord_mnl)]),
                )
                for s in range(subtile_cnt):
                    for i in range(cute.size(tTR_rAcc)):
                        tTR_rMaster[(None, None, None, s)][i] = cutlass.Float32(0.0)

                # Promote each 128-K block: t2r -> scale -> add to master.
                bm = mma_tile_coord_mnl[0] * self.cta_tile_shape_mnk[0]
                # tile64 fix (2-row a_scale gather): with a 64-row CTA tile a
                # thread's epi elements span TWO M-rows. The t2r value modes
                # are ((2,2,8),1,1); exactly one extent-2 mode carries the
                # M-delta. Compute per-tile: row0, row1 and Float32 weights
                # w0/w1/w01 (0 or 1) telling whether flipping value mode 0 /
                # mode 1 / either changes the row. Per k_tile this folds to 3
                # extra FMAs; the per-element pick is fully static. For
                # 128-row tiles all deltas are 0 and everything degenerates
                # to the single-row path bit-exactly.
                mmax = cute.size(mC_mnl, mode=[0]) - 1
                cC0 = tTR_cC[(None, None, None, 0)]
                row0_raw = cC0[0][0]
                d0 = cC0[1][0] - row0_raw
                d1 = cC0[2][0] - row0_raw
                row1_raw = row0_raw + d0 + d1
                a0 = cutlass.max(d0, -d0)
                a1 = cutlass.max(d1, -d1)
                g_row = cutlass.min(row0_raw, mmax)
                g_row1 = cutlass.min(row1_raw, mmax)
                # Exact integer row select per value-mode combo (bit-exact:
                # each variant loads the true row's scale; no fp blending).
                dg = g_row1 - g_row
                r_m0 = g_row + cutlass.min(a0, 1) * dg
                r_m1 = g_row + cutlass.min(a1, 1) * dg
                r_m01 = g_row + cutlass.min(a0 + a1, 1) * dg
                for k_tile in range(k_tile_cnt):
                    tTR_tAcc = cute.group_modes(
                        tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)],
                        3, cute.rank(tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]),
                    )
                    acc_pipeline.consumer_wait(acc_consumer_state)
                    # b_scale constant across a 128-wide N-tile -> one load/block.
                    sb = b_scale[mma_tile_coord_mnl[1], k_tile, ll]
                    # Two per-thread rows (row0/row1) -> two folded scales; the
                    # three selected variants cover every value-mode combo.
                    sasb = a_scale[g_row, k_tile, ll] * sb
                    sasb_m0 = a_scale[r_m0, k_tile, ll] * sb
                    sasb_m1 = a_scale[r_m1, k_tile, ll] * sb
                    sasb_m01 = a_scale[r_m01, k_tile, ll] * sb
                    for s in range(subtile_cnt):
                        cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, s)], tTR_rAcc)
                        if s == subtile_cnt - 1:
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                acc_pipeline.consumer_release(acc_consumer_state)
                            acc_consumer_state.advance()
                        tTR_rMaster_mn = tTR_rMaster[(None, None, None, s)]
                        for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                            i0, i1 = i % 2, (i // 2) % 2
                            m_i = (sasb, sasb_m0, sasb_m1, sasb_m01)[i0 + 2 * i1]
                            tTR_rMaster_mn[i] = tTR_rMaster_mn[i] + tTR_rAcc[i] * m_i

                # Store master accumulator to C.
                if cutlass.const_expr(self.use_tma_store):
                    # ((ATOM_V, REST_V), EPI_M, EPI_N) -> ((ATOM_V,REST_V), (EPI_M,EPI_N))
                    bSG_gC = cute.group_modes(
                        bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)],
                        1, cute.rank(bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]),
                    )
                    for s in range(subtile_cnt):
                        # master (rmem) -> convert -> r2s to smem sC stage
                        tTR_rC.store(
                            epilogue_op(
                                tTR_rMaster[(None, None, None, s)].load().to(self.c_dtype)
                            )
                        )
                        c_buffer = s % self.num_c_stage
                        cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
                        # smem visible to TMA proxy, then all epilogue warps sync
                        cute.arch.fence_proxy("async.shared", space="cta")
                        epilog_sync_barrier.arrive_and_wait()
                        # one warp issues the TMA store for this subtile
                        if warp_idx == self.epilogue_warp_id[0]:
                            cute.copy(
                                tma_atom_c, bSG_sC[(None, c_buffer)], bSG_gC[(None, s)]
                            )
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        epilog_sync_barrier.arrive_and_wait()
                else:
                    for s in range(subtile_cnt):
                        acc_vec = epilogue_op(tTR_rMaster[(None, None, None, s)].load().to(self.c_dtype))
                        tTR_rC.store(acc_vec)
                        cute.copy(simt_atom, tTR_rC, tTR_gC[(None, None, None, s)])

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Drain outstanding TMA stores before reclaiming resources.
            if cutlass.const_expr(self.use_tma_store):
                c_pipeline.producer_tail()

            # Synchronize before TMEM dealloc.
            tmem_dealloc_barrier.arrive_and_wait()

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
