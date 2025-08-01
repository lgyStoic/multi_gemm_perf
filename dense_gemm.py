# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
from typing import Tuple, Type
import math
import cuda.bindings.driver as cuda

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.hopper_helpers as sm90_utils

"""
A high-performance batched dense GEMM (C = A * B) example for the NVIDIA Hopper architecture
using CUTE DSL.
- Matrix A is MxKxL, L is batch dimension, A can be row-major("K") or column-major("M")
- Matrix B is NxKxL, L is batch dimension, B can be row-major("N") or column-major("K")
- Matrix C is MxNxL, L is batch dimension, C can be row-major("N") or column-major("M")

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Hopper's WGMMA for matrix multiply-accumulate (MMA) operations
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Supports multi-stage pipeline to overlap computation and memory access

This GEMM works as follows:
1. Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. Perform matrix multiply-accumulate (MMA) operations using WGMMA instruction.
3. Store results from registers (RMEM) to shared memory (SMEM), then to global memory (GMEM) with TMA operations.

Hopper WGMMA instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Perform MMA operation and store the result in Accumulator(register)

To run this example:

.. code-block:: bash

    python examples/hopper/dense_gemm.py                                   \
      --mnkl 8192,8192,8192,1 --tile_shape_mnk 128,256,64                  \
      --cluster_shape_mn 1,1 --a_dtype Float16 --b_dtype Float16           \
      --c_dtype Float16 --acc_dtype Float32                                \
      --a_major k --b_major k --c_major n

The above example command compute batched gemm with M=8192, N=8192, K=8192,
batch_count=1. The Hopper WGMMA tile shape is 128x256x64 and the cluster shape
is (1,1). The input, mma accumulator and output data type are set as fp16, fp32
and fp16, respectively.

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/hopper/dense_gemm.py                               \
      --mnkl 8192,8192,8192,1 --tile_shape_mnk 128,256,64                  \
      --cluster_shape_mn 1,1 --a_dtype Float16 --b_dtype Float16           \
      --c_dtype Float16 --acc_dtype Float32                                \
      --a_major k --b_major k --c_major n

Constraints:
* Supported input data types: fp16, fp8 (e4m3fn, e5m2)
* For fp16 types, A and B must have the same data type
* For fp8 types, A and B can have different types (e4m3fn or e5m2) but both must be 8-bit
* Fp8 types only support k-major layout
* Only fp32 accumulation is supported in this example
* CTA tile shape M must be 64/128
* CTA tile shape N must be 64/128/256
* CTA tile shape K must be 64
* Cluster shape M/N must be positive and power of 2, total cluster size <= 4
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 8, 16 for Float16, and Float8, respectively.
* OOB tiles are not allowed when TMA store is disabled
"""


# /////////////////////////////////////////////////////////////////////////////
#  Helpers to parse args
# /////////////////////////////////////////////////////////////////////////////
def parse_comma_separated_ints(s: str):
    try:
        return tuple([int(x.strip()) for x in s.split(",")])
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Invalid format. Expected comma-separated integers."
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Example of MxNxKxL GEMM on Hopper.")

    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(4096, 4096, 4096, 1),
        help="mnkl dimensions (comma-separated)",
    )
    parser.add_argument(
        "--tile_shape_mnk",
        type=parse_comma_separated_ints,
        choices=[(128, 128, 64), (128, 256, 64), (128, 64, 64), (64, 64, 64)],
        default=(128, 128, 64),
        help="Cta tile shape (comma-separated)",
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=parse_comma_separated_ints,
        choices=[(1, 1), (2, 1), (1, 2), (2, 2)],
        default=(1, 1),
        help="Cluster shape (comma-separated)",
    )
    parser.add_argument(
        "--a_dtype",
        type=cutlass.dtype,
        default=cutlass.Float16,
    )
    parser.add_argument(
        "--b_dtype",
        type=cutlass.dtype,
        default=cutlass.Float16,
    )
    parser.add_argument(
        "--c_dtype",
        type=cutlass.dtype,
        default=cutlass.Float16,
    )
    parser.add_argument(
        "--acc_dtype",
        type=cutlass.dtype,
        default=cutlass.Float32,
    )
    parser.add_argument("--a_major", choices=["k", "m"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k", "n"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument(
        "--tolerance", type=float, default=1e-01, help="Tolerance for validation"
    )

    args = parser.parse_args()

    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")
    if len(args.tile_shape_mnk) != 3:
        parser.error("--tile_shape_mnk must contain exactly 3 values")
    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    return args


# /////////////////////////////////////////////////////////////////////////////
#  Host setup and device kernel launch
# /////////////////////////////////////////////////////////////////////////////


class HopperWgmmaGemmKernel:
    """
    This class implements batched matrix multiplication (C = A x B) with support for various data types
    and architectural features specific to Hopper GPUs.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param tile_shape_mnk: Shape of the CTA tile (M,N,K)
    :type tile_shape_mnk: Tuple[int, int, int]
    :param cluster_shape_mnk: Cluster dimensions (M,N,K) for parallel processing
    :type cluster_shape_mnk: Tuple[int, int, int]

    :note: Data type requirements:
        - For 16-bit types: A and B must have the same data type
        - For 8-bit types: A and B can have different types (Float8E4M3FN/Float8E5M2) as long as both are 8-bit
        - Float8 types only support k-major layout

    :note: Supported data types:
        - Float16
        - Float8E4M3FN/Float8E5M2

    :note: Supported accumulation types:
        - Float32 (for all floating point inputs)

    :note: Constraints:
        - CTA tile M must be 64/128
        - CTA tile N must be 64/128/256
        - CTA tile K must be 64
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 4

    Example:
        >>> gemm = HopperWgmmaGemmKernel(
        ...     acc_dtype=cutlass.Float32,
        ...     tile_shape_mnk=(128, 256, 64),
        ...     cluster_shape_mnk=(1, 1, 1)
        ... )
        >>> gemm(a_tensor, b_tensor, c_tensor, stream)
    """

    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric],
        tile_shape_mnk: tuple[int, int, int],
        cluster_shape_mnk: tuple[int, int, int],
    ):
        """
        Initializes the configuration for a Hopper dense GEMM kernel.

        This configuration includes data types for operands, tile shape, cluster configuration,
        and thread layout.

        :param acc_dtype: Data type for accumulation during computation
        :type acc_dtype: type[cutlass.Numeric]
        :param tile_shape_mnk: Shape of the CTA tile (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param cluster_shape_mnk: Cluster dimensions (M,N,K) for parallel processing
        :type cluster_shape_mnk: Tuple[int, int, int]
        """

        self.acc_dtype = acc_dtype

        self.cluster_shape_mnk = cluster_shape_mnk
        self.mma_inst_shape_mn = None
        self.tile_shape_mnk = tuple(tile_shape_mnk)
        # For large tile size, using two warp groups is preferred because using only one warp
        # group may result in register spill
        self.atom_layout_mnk = (
            (2, 1, 1)
            if tile_shape_mnk[0] > 64 and tile_shape_mnk[1] > 128
            else (1, 1, 1)
        )
        self.num_mcast_ctas_a = None
        self.num_mcast_ctas_b = None
        self.is_a_mcast = False
        self.is_b_mcast = False

        self.occupancy = 1
        self.mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = self.mma_warp_groups * self.num_threads_per_warp_group
        self.smem_capacity = sm90_utils.SMEM_CAPACITY["sm90"]

        self.ab_stage = None
        self.epi_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None

        self.shared_storage = None
        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        """

        # check the cta tile shape
        if self.tile_shape_mnk[0] not in [64, 128]:
            raise ValueError("CTA tile shape M must be 64/128")
        if self.tile_shape_mnk[1] not in [64, 128, 256]:
            raise ValueError("CTA tile shape N must be 64/128/256")
        if self.tile_shape_mnk[2] not in [64]:
            raise ValueError("CTA tile shape K must be 64")

        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        self.num_mcast_ctas_a = self.cluster_shape_mnk[1]
        self.num_mcast_ctas_b = self.cluster_shape_mnk[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        is_cooperative = self.atom_layout_mnk == (2, 1, 1)
        self.epi_tile = self._sm90_compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=is_cooperative
        )

        # Compute stage before compute smem layout
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.smem_capacity,
            self.occupancy,
        )

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes
        - Setup TMA load/store atoms and tensors
        - Compute grid size
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a: Input tensor A
        :type a: cute.Tensor
        :param b: Input tensor B
        :type b: cute.Tensor
        :param c: Output tensor C
        :type c: cute.Tensor
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        """

        # setup static attributes before smem/grid/tma computation
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(
            self.a_dtype.width == 16 and self.a_dtype != self.b_dtype
        ):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(self.a_dtype.width != self.b_dtype.width):
            raise TypeError(
                f"Type width mismatch: {self.a_dtype.width} != {self.b_dtype.width}"
            )
        if cutlass.const_expr(self.a_dtype.width != 16 and self.a_dtype.width != 8):
            raise TypeError(f"a_dtype should be float16 or float8")

        self._setup_attributes()

        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
        )

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            self.cluster_shape_mnk[1],
        )

        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            self.cluster_shape_mnk[0],
        )

        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        grid = self._compute_grid(c, self.tile_shape_mnk, self.cluster_shape_mnk)

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            sa: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sb: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )
        return

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
    ):
        """
        GPU device kernel performing the batched GEMM computation.

        :param tma_atom_a: TMA copy atom for A tensor
        :type tma_atom_a: cute.CopyAtom
        :param mA_mkl: Input tensor A
        :type mA_mkl: cute.Tensor
        :param tma_atom_b: TMA copy atom for B tensor
        :type tma_atom_b: cute.CopyAtom
        :param mB_nkl: Input tensor B
        :type mB_nkl: cute.Tensor
        :param tma_atom_c: TMA copy atom for C tensor
        :type tma_atom_c: cute.CopyAtom
        :param mC_mnl: Output tensor C
        :type mC_mnl: cute.Tensor
        :param tiled_mma: Tiled MMA object
        :type tiled_mma: cute.TiledMma
        :param cta_layout_mnk: CTA layout
        :type cta_layout_mnk: cute.Layout
        :param a_smem_layout_staged: Shared memory layout for A
        :type a_smem_layout_staged: cute.ComposedLayout
        :param b_smem_layout_staged: Shared memory layout for B
        :type b_smem_layout_staged: cute.ComposedLayout
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        """

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch Tma desc
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Get cta/warp/thread idx
        # ///////////////////////////////////////////////////////////////////////////////
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        cidx, cidy, _ = cute.arch.cluster_idx()
        cdimx, cdimy, _ = cute.arch.cluster_dim()
        cluster_id = cidx + cdimx * cidy

        # CTA Swizzle to promote L2 data reuse
        group_size_m = 8
        s_shape = (
            (group_size_m, cdimx // group_size_m),
            cdimy,
        )
        s_stride = ((1, cdimy * group_size_m), group_size_m)
        s_layout = cute.make_layout(s_shape, stride=s_stride)
        num_reg_cids = cute.size(s_shape)
        cid_m, cid_n = s_layout.get_flat_coord(cluster_id % num_reg_cids)

        # Deal with the tail part
        if cluster_id >= num_reg_cids:
            tail_size_m = cdimx % group_size_m
            tail_layout = cute.make_layout(
                (tail_size_m, cdimy), stride=(1, tail_size_m)
            )
            tail_cid = cluster_id - num_reg_cids
            tail_cid_m, tail_cid_n = tail_layout.get_flat_coord(tail_cid)
            cid_m = cute.size(s_shape, mode=[0]) + tail_cid_m
            cid_n = tail_cid_n

        # Get the pid from cluster id
        bidx_in_cluster = cute.arch.block_in_cluster_idx()
        pid_m = cid_m * self.cluster_shape_mnk[0] + bidx_in_cluster[0]
        pid_n = cid_n * self.cluster_shape_mnk[1] + bidx_in_cluster[1]

        tile_coord_mnkl = (pid_m, pid_n, None, bidz)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get mcast mask
        # ///////////////////////////////////////////////////////////////////////////////
        a_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=1
        )
        b_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=0
        )

        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        # /////////////////////////////////////////////////////////////////////////////
        #  Alloc and init AB full/empty + ACC full mbar (pipeline)
        # /////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # mbar arrays
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        # Threads/warps participating in this pipeline
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        # Each warp will constribute to the arrive count with the number of mcast size
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        num_warps = self.threads_per_cta // 32
        consumer_arrive_cnt = mcast_size * num_warps
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )

        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        #  Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_arrive_relaxed()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Generate smem tensor A/B
        # ///////////////////////////////////////////////////////////////////////////////
        sa = storage.sa.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sb = storage.sb.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sc_ptr = cute.recast_ptr(
            sa.iterator, epi_smem_layout_staged.inner, dtype=self.c_dtype
        )
        sc = cute.make_tensor(sc_ptr, epi_smem_layout_staged.outer)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Local_tile partition global tensors
        # ///////////////////////////////////////////////////////////////////////////////
        # (bM, bK, RestK)
        gA_mkl = cute.local_tile(
            mA_mkl, self.tile_shape_mnk, tile_coord_mnkl, proj=(1, None, 1)
        )
        # (bN, bK, RestK)
        gB_nkl = cute.local_tile(
            mB_nkl, self.tile_shape_mnk, tile_coord_mnkl, proj=(None, 1, 1)
        )
        # (bM, bN)
        gC_mnl = cute.local_tile(
            mC_mnl, self.tile_shape_mnk, tile_coord_mnkl, proj=(1, 1, None)
        )

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition global tensor for TiledMMA_A/B/C
        # //////////////////////////////////////////////////////////////////////////////
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        warp_group_thread_layout = cute.make_layout(
            self.mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))

        tCgC = thr_mma.partition_C(gC_mnl)

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition shared tensor for TMA load A/B
        # //////////////////////////////////////////////////////////////////////////////
        #  TMA load A partition_S/D
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        sa_for_tma_partition = cute.group_modes(sa, 0, 2)
        gA_for_tma_partition = cute.group_modes(gA_mkl, 0, 2)
        tAsA, tAgA_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            sa_for_tma_partition,
            gA_for_tma_partition,
        )

        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        sb_for_tma_partition = cute.group_modes(sb, 0, 2)
        gB_for_tma_partition = cute.group_modes(gB_nkl, 0, 2)
        tBsB, tBgB_nkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            sb_for_tma_partition,
            gB_for_tma_partition,
        )

        # //////////////////////////////////////////////////////////////////////////////
        #  Make frangments
        # //////////////////////////////////////////////////////////////////////////////
        tCsA = thr_mma.partition_A(sa)
        tCsB = thr_mma.partition_B(sb)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        acc_shape = tCgC.shape
        accumulators = cute.make_fragment(acc_shape, self.acc_dtype)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Cluster wait
        # ///////////////////////////////////////////////////////////////////////////////
        # cluster wait for barrier init
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()
        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch
        # /////////////////////////////////////////////////////////////////////////////
        k_tile_cnt = cute.size(gA_mkl, mode=[2])
        prefetch_k_tile_cnt = cutlass.max(cutlass.min(self.ab_stage, k_tile_cnt), 0)

        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        if warp_idx == 0:
            # /////////////////////////////////////////////////////////////////////////////
            # Prefetch TMA load
            # /////////////////////////////////////////////////////////////////////////////
            for prefetch_idx in cutlass.range(prefetch_k_tile_cnt, unroll=1):
                # /////////////////////////////////////////////////////////////////////////////
                #  Wait for A/B buffers to be empty before loading into them
                #  Also sets the transaction barrier for the A/B buffers
                # /////////////////////////////////////////////////////////////////////////////
                mainloop_pipeline.producer_acquire(mainloop_producer_state)
                # /////////////////////////////////////////////////////////////////////////////
                #  Slice to global/shared memref to current k_tile
                # /////////////////////////////////////////////////////////////////////////////
                tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                # /////////////////////////////////////////////////////////////////////////////
                #  TMA load A/B
                # /////////////////////////////////////////////////////////////////////////////
                cute.copy(
                    tma_atom_a,
                    tAgA_k,
                    tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=a_mcast_mask,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_k,
                    tBsB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=b_mcast_mask,
                )
                # Mainloop pipeline's producer commit is a NOP
                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()

        # /////////////////////////////////////////////////////////////////////////////
        #  Prologue MMAs
        # /////////////////////////////////////////////////////////////////////////////
        k_pipe_mmas = 1

        mainloop_consumer_read_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        mainloop_consumer_release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        peek_ab_full_status = cutlass.Boolean(1)
        if mainloop_consumer_read_state.count < k_tile_cnt:
            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                mainloop_consumer_read_state
            )

        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
        num_k_blocks = cute.size(tCrA, mode=[2])
        for k_tile in range(k_pipe_mmas):
            # Wait for A/B buffer to be ready
            mainloop_pipeline.consumer_wait(
                mainloop_consumer_read_state, peek_ab_full_status
            )

            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (
                    None,
                    None,
                    k_block_idx,
                    mainloop_consumer_read_state.index,
                )
                tCrA_1phase = tCrA[k_block_coord]
                tCrB_1phase = tCrB[k_block_coord]

                cute.gemm(
                    tiled_mma,
                    accumulators,
                    tCrA_1phase,
                    tCrB_1phase,
                    accumulators,
                )
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

            cute.nvgpu.warpgroup.commit_group()
            mainloop_consumer_read_state.advance()
            peek_ab_full_status = cutlass.Boolean(1)
            if mainloop_consumer_read_state.count < k_tile_cnt:
                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                    mainloop_consumer_read_state
                )

        # /////////////////////////////////////////////////////////////////////////////
        #  MAINLOOP
        # /////////////////////////////////////////////////////////////////////////////
        for k_tile in cutlass.range(k_pipe_mmas, k_tile_cnt, 1, unroll=1):
            # /////////////////////////////////////////////////////////////////////////////
            #  Wait for TMA copies to complete
            # /////////////////////////////////////////////////////////////////////////////
            mainloop_pipeline.consumer_wait(
                mainloop_consumer_read_state, peek_ab_full_status
            )
            # /////////////////////////////////////////////////////////////////////////////
            #  WGMMA
            # /////////////////////////////////////////////////////////////////////////////
            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (
                    None,
                    None,
                    k_block_idx,
                    mainloop_consumer_read_state.index,
                )
                tCrA_1phase = tCrA[k_block_coord]
                tCrB_1phase = tCrB[k_block_coord]

                cute.gemm(
                    tiled_mma,
                    accumulators,
                    tCrA_1phase,
                    tCrB_1phase,
                    accumulators,
                )

            cute.nvgpu.warpgroup.commit_group()
            # Wait on the wgmma barrier for previous k_pipe_mmas wgmmas to complete
            cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)

            mainloop_pipeline.consumer_release(mainloop_consumer_release_state)

            mainloop_consumer_read_state.advance()
            mainloop_consumer_release_state.advance()

            peek_ab_full_status = cutlass.Boolean(1)
            if mainloop_consumer_read_state.count < k_tile_cnt:
                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                    mainloop_consumer_read_state
                )
            # /////////////////////////////////////////////////////////////////////////////
            #  TMA load
            # /////////////////////////////////////////////////////////////////////////////
            if warp_idx == 0 and mainloop_producer_state.count < k_tile_cnt:
                # /////////////////////////////////////////////////////////////////////////////
                #  Wait for A/B buffers to be empty before loading into them
                #  Also sets the transaction barrier for the A/B buffers
                # /////////////////////////////////////////////////////////////////////////////
                mainloop_pipeline.producer_acquire(mainloop_producer_state)

                # /////////////////////////////////////////////////////////////////////////////
                #  Slice to global/shared memref to current k_tile
                # /////////////////////////////////////////////////////////////////////////////
                tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                # /////////////////////////////////////////////////////////////////////////////
                #  TMA load A/B
                # /////////////////////////////////////////////////////////////////////////////
                cute.copy(
                    tma_atom_a,
                    tAgA_k,
                    tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=a_mcast_mask,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_k,
                    tBsB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=b_mcast_mask,
                )
                # Mainloop pipeline's producer commit is a NOP
                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()

        # /////////////////////////////////////////////////////////////////////////////
        #  EPILOG
        # /////////////////////////////////////////////////////////////////////////////
        cute.nvgpu.warpgroup.wait_group(0)

        if cute.size(self.cluster_shape_mnk) > 1:
            # Wait for all threads in the cluster to finish, avoid early release of smem
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        else:
            # For cluster that has a single thread block, it might have more than one warp groups.
            # Wait for all warp groups in the thread block to finish, because smem for tensor A in
            # the mainloop is reused in the epilogue.
            cute.arch.sync_threads()

        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            self.c_layout,
            elem_ty_d=self.c_dtype,
            elem_ty_acc=self.acc_dtype,
        )

        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                self.c_layout.is_m_major_c(),
                4,
            ),
            self.c_dtype,
        )

        tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)

        tiled_copy_r2s = cute.make_tiled_copy_S(
            copy_atom_r2s,
            tiled_copy_C_Atom,
        )

        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sc)
        # (R2S, R2S_M, R2S_N)
        tRS_rAcc = tiled_copy_r2s.retile(accumulators)

        # Allocate D registers.
        rD_shape = cute.shape(thr_copy_r2s.partition_S(sc))
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_fragment_like(tRS_rD_layout, self.acc_dtype)
        size_tRS_rD = cute.size(tRS_rD)

        sepi_for_tma_partition = cute.group_modes(sc, 0, 2)
        tcgc_for_tma_partition = cute.zipped_divide(gC_mnl, self.epi_tile)

        bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sepi_for_tma_partition,
            tcgc_for_tma_partition,
        )

        epi_tile_num = cute.size(tcgc_for_tma_partition, mode=[1])
        epi_tile_shape = tcgc_for_tma_partition.shape[1]

        for epi_idx in cutlass.range(epi_tile_num, unroll=epi_tile_num):
            # Copy from accumulators to D registers
            for epi_v in range(size_tRS_rD):
                tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

            # Type conversion
            tRS_rD_out = cute.make_fragment_like(tRS_rD_layout, self.c_dtype)
            acc_vec = tRS_rD.load()
            tRS_rD_out.store(acc_vec.to(self.c_dtype))

            # Copy from D registers to shared memory
            epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
            cute.copy(
                tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buffer)]
            )

            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            # barrier for sync
            cute.arch.barrier()

            # Get the global memory coordinate for the current epi tile.
            epi_tile_layout = cute.make_layout(
                epi_tile_shape, stride=(epi_tile_shape[1], 1)
            )
            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
            # Copy from shared memory to global memory
            if warp_idx == 0:
                cute.copy(
                    tma_atom_c,
                    bSG_sD[(None, epi_buffer)],
                    bSG_gD[(None, gmem_coord)],
                )
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(self.epi_stage - 1, read=True)

            cute.arch.barrier()

        return

    @staticmethod
    # 计算A/B操作数和epilogue阶段数的静态方法。
    # 根据CTA tile形状、A/B数据类型、共享内存容量和目标occupancy，返回A/B操作数的stage数和epilogue的stage数。
    # ab_stage决定流水线深度，epi_stage通常固定为4。
    def _compute_stages(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int]:
    
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (A/B operand stages, epilogue stages)
        :rtype: tuple[int, int]
        """
        epi_stage = 4  # 设置epilogue阶段数为固定值4，负责将计算结果写回全局内存
        # epi_smem will reuse smem ab.
        epi_bytes = 0  # epilogue共享内存使用量为0，因为它会重用AB操作数的共享内存空间

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))  # 从tile形状(M,N,K)提取A矩阵维度(M,K)
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))  # 从tile形状(M,N,K)提取B矩阵维度(N,K)
        ab_bytes_per_stage = (  # 计算每个流水线阶段A和B矩阵块占用的总字节数
            cute.size(a_shape) * a_dtype.width // 8  # A矩阵块字节数：元素数量×位宽÷8
            + cute.size(b_shape) * b_dtype.width // 8  # B矩阵块字节数：元素数量×位宽÷8
        )
        mbar_helpers_bytes = 1024  # memory barrier辅助结构占用1024字节，用于同步流水线各阶段

        ab_stage = (  # 计算AB操作数的流水线阶段数
            (smem_capacity - occupancy * 1024) // occupancy  # 每个CTA可用的共享内存容量
            - mbar_helpers_bytes  # 减去memory barrier开销
            - epi_bytes  # 减去epilogue内存开销(这里为0)
        ) // ab_bytes_per_stage  # 除以每阶段内存需求，得到可支持的阶段数
        return ab_stage, epi_stage  # 返回(AB操作数阶段数, epilogue阶段数)

    @staticmethod
    def _sm90_compute_tile_shape_or_override(
        tile_shape_mnk: tuple[int, int, int],
        element_type: type[cutlass.Numeric],
        is_cooperative: bool = False,
        epi_tile_override: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        """Compute the epilogue tile shape or use override if provided.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param element_type: Data type of elements
        :type element_type: type[cutlass.Numeric]
        :param is_cooperative: Whether to use cooperative approach
        :type is_cooperative: bool
        :param epi_tile_override: Optional override for epilogue tile shape
        :type epi_tile_override: Tuple[int, int] or None

        :return: Computed epilogue tile shape
        :rtype: Tuple[int, int]
        """
        if epi_tile_override is not None:
            return epi_tile_override
        if is_cooperative:
            tile_m = min(128, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = min(32, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)
        else:
            n_perf = 64 if element_type.width == 8 else 32
            tile_m = min(64, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = min(n_perf, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple[int, int, int],
        epi_tile: tuple[int, int],
        a_dtype: type[cutlass.Numeric],
        a_layout: utils.LayoutEnum,
        b_dtype: type[cutlass.Numeric],
        b_layout: utils.LayoutEnum,
        ab_stage: int,
        c_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        epi_stage: int,
    ) -> tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]:
        """Create shared memory layouts for A, B, and C tensors.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]
        :param a_dtype: Data type for matrix A
        :type a_dtype: type[cutlass.Numeric]
        :param a_layout: Layout enum for matrix A
        :type a_layout: utils.LayoutEnum
        :param b_dtype: Data type for matrix B
        :type b_dtype: type[cutlass.Numeric]
        :param b_layout: Layout enum for matrix B
        :type b_layout: utils.LayoutEnum
        :param ab_stage: Number of stages for A/B tensors
        :type ab_stage: int
        :param c_dtype: Data type for output matrix C
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout enum for the output matrix C
        :type c_layout: utils.LayoutEnum
        :param epi_stage: Number of epilogue stages
        :type epi_stage: int

        :return: Tuple of shared memory layouts for A, B, and C
        :rtype: Tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]
        """
        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))  # 从tile形状(M,N,K)提取A矩阵形状(M,K)

        a_is_k_major = (  # 判断矩阵A是否为K-major存储模式
            a_layout.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K  # 检查A布局的主要存储模式是否为K维度
        )
        b_is_k_major = (  # 判断矩阵B是否为K-major存储模式
            b_layout.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K  # 检查B布局的主要存储模式是否为K维度
        )
        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]  # 获取A矩阵主要模式维度大小：K维度(索引2)或M维度(索引0)
        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(  # 创建A矩阵的共享内存原子布局
            sm90_utils.get_smem_layout_atom(  # 获取适合SM90的共享内存布局原子
                a_layout,  # A矩阵的数据布局
                a_dtype,   # A矩阵的数据类型
                a_major_mode_size,  # A矩阵主要模式的维度大小
            ),
            a_dtype,  # A矩阵数据类型
        )
        a_smem_layout_staged = cute.tile_to_shape(  # 将A矩阵原子布局扩展为分阶段布局
            a_smem_layout_atom,  # A矩阵的原子布局
            cute.append(a_smem_shape, ab_stage),  # 在A矩阵形状后附加流水线阶段维度
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),  # 维度排序：K-major用(0,1,2)，否则用(1,0,2)
        )

        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))  # 从tile形状(M,N,K)提取B矩阵形状(N,K)

        b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]  # 获取B矩阵主要模式维度大小：K维度(索引2)或N维度(索引1)
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(  # 创建B矩阵的共享内存原子布局
            sm90_utils.get_smem_layout_atom(  # 获取适合SM90的共享内存布局原子
                b_layout,  # B矩阵的数据布局
                b_dtype,   # B矩阵的数据类型
                b_major_mode_size,  # B矩阵主要模式的维度大小
            ),
            b_dtype,  # B矩阵数据类型
        )
        b_smem_layout_staged = cute.tile_to_shape(  # 将B矩阵原子布局扩展为分阶段布局
            b_smem_layout_atom,  # B矩阵的原子布局
            cute.append(b_smem_shape, ab_stage),  # 在B矩阵形状后附加流水线阶段维度
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),  # 维度排序：K-major用(0,1,2)，否则用(1,0,2)
        )

        c_smem_shape = epi_tile  # C矩阵的共享内存形状直接使用epilogue tile形状
        c_major_mode_size = epi_tile[1] if c_layout.is_n_major_c() else epi_tile[0]  # C矩阵主要模式大小：N-major用N维度，否则用M维度
        c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(  # 创建C矩阵的共享内存原子布局
            sm90_utils.get_smem_layout_atom(  # 获取适合SM90的共享内存布局原子
                c_layout,  # C矩阵的数据布局
                c_dtype,   # C矩阵的数据类型
                c_major_mode_size,  # C矩阵主要模式的维度大小
            ),
            c_dtype,  # C矩阵数据类型
        )
        epi_smem_layout_staged = cute.tile_to_shape(  # 将C矩阵原子布局扩展为epilogue分阶段布局
            c_smem_layout_atom,  # C矩阵的原子布局
            cute.append(c_smem_shape, epi_stage),  # 在C矩阵形状后附加epilogue阶段维度
            order=(1, 0, 2) if c_layout.is_m_major_c() else (0, 1, 2),  # 维度排序：M-major用(1,0,2)，否则用(0,1,2)
        )

        return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged  # 返回A、B、C矩阵的分阶段共享内存布局

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        tile_shape_mnk: tuple[int, int, int],
        cluster_shape_mnk: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        """Compute grid shape for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mnk: Shape of each cluster in M, N, K dimensions.
        :type cluster_shape_mnk: tuple[int, int, int]

        :return: Grid shape for kernel launch.
        :rtype: tuple[int, int, int]
        """

        # 提取CTA tile的M和N维度，忽略K维度（因为输出矩阵C不涉及K维度）
        c_shape = (tile_shape_mnk[0], tile_shape_mnk[1])
        
        # 将输出张量C按照CTA tile形状进行分块划分
        # zipped_divide将张量划分为两部分：tile内部的布局和tile之间的布局
        # 返回的gc包含了如何将C矩阵分割成多个tile的信息
        gc = cute.zipped_divide(c, tiler=c_shape)
        
        # 计算需要的cluster数量
        # mode=[1]表示获取tile间布局的形状（即有多少个tile）
        # ceil_div执行向上取整除法，计算每个维度需要多少个cluster来覆盖所有tile
        clusters = cute.ceil_div(cute.get(gc.layout, mode=[1]).shape, cluster_shape_mnk)
        
        # 计算最终的CUDA网格大小
        # 将cluster数量乘以每个cluster的大小，得到总的thread block数量
        # 这就是CUDA kernel启动时需要的网格维度
        grid = tuple(x * y for x, y in zip(clusters, cluster_shape_mnk))
        return grid

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: tuple[int, int],
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for C tensor storage.

        :param tensor_c: Output tensor C
        :type tensor_c: cute.Tensor
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]

        :return: TMA atom and tensor for C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        c_cta_v_layout = cute.composition(
            cute.make_identity_layout(tensor_c.shape), epi_tile
        )
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            c_cta_v_layout,
        )

        return tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
        mcast_dim: int,
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """创建输入张量的TMA原子操作和张量。
        
        TMA (Tensor Memory Access) 是Hopper架构中用于高效内存操作的硬件单元，
        可以在全局内存和共享内存之间进行块级数据传输。此函数根据多播需求
        创建相应的TMA操作类型。

        :param tensor: 输入张量 (A 或 B矩阵)
        :type tensor: cute.Tensor
        :param smem_layout_staged: 张量的分阶段共享内存布局，包含流水线维度
        :type smem_layout_staged: cute.ComposedLayout
        :param smem_tile: 共享内存tile的形状 (M,K) 或 (N,K)
        :type smem_tile: Tuple[int, int]
        :param mcast_dim: 多播维度，决定多少个CTA参与多播操作
        :type mcast_dim: int

        :return: TMA原子操作和对应的张量
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        # 根据多播维度选择TMA操作类型
        # mcast_dim == 1: 单播模式，只有一个目标CTA
        # mcast_dim > 1: 多播模式，多个CTA同时接收数据，减少L2缓存流量
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()  # 全局内存到共享内存的单播操作
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()  # 全局内存到共享内存的多播操作
        )

        # 从分阶段布局中提取第0阶段的共享内存布局
        # (None, None, 0) 表示保留前两个维度，选择第0个流水线阶段
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        
        # 创建tiled TMA原子操作
        # 该函数将全局张量、共享内存布局和tile形状组合，
        # 生成用于数据传输的TMA原子操作和相应的张量视图
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,              # TMA操作类型（单播或多播）
            tensor,          # 源张量（全局内存中的A或B矩阵）
            smem_layout,     # 目标共享内存布局
            smem_tile,       # tile形状，定义每次传输的数据块大小
            num_multicast=mcast_dim,  # 多播的CTA数量
        )
        return tma_atom, tma_tensor

    @staticmethod
    def is_valid_dtypes(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
    ) -> bool:
        """
        Check if the dtypes are valid

        :param a_dtype: The data type of tensor A
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of tensor B
        :type b_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: major mode of tensor A
        :type a_major: str
        :param b_major: major mode of tensor B
        :type b_major: str

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        # tested a_dtype
        if a_dtype not in {
            cutlass.Float16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            is_valid = False
        # tested b_dtype
        if b_dtype not in {
            cutlass.Float16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            is_valid = False
        # tested acc_dtype
        if acc_dtype != cutlass.Float32:
            is_valid = False
        # tested c_dtype
        if c_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            is_valid = False
        # make sure a_dtype == b_dtype for Float16
        if a_dtype.width == 16 and a_dtype != b_dtype:
            is_valid = False
        # make sure a_dtype.width == b_dtype.width (i.e, Float8E4M3FN or Float8E5M2)
        if a_dtype.width != b_dtype.width:
            is_valid = False

        # for Float8 types, this implementation only supports k-major layout
        if (a_dtype.width == 8 and a_major != "k") or (
            b_dtype.width == 8 and b_major != "k"
        ):
            is_valid = False

        return is_valid


def run_dense_gemm(
    mnkl: Tuple[int, int, int, int],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    tile_shape_mnk: Tuple[int, int, int],
    cluster_shape_mn: Tuple[int, int],
    tolerance: float,
):
    """
    Prepare A/B/C tensors, launch GPU kernel, and reference checking.
    """

    print(f"Running Hopper Dense GEMM with:")
    print(f"mnkl: {mnkl}")
    print(
        f"A dtype: {a_dtype}, B dtype: {b_dtype}, C dtype: {c_dtype}, Acc dtype: {acc_dtype}"
    )
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Tile Shape: {tile_shape_mnk}, Cluster Shape: {cluster_shape_mn}")
    print(f"Tolerance: {tolerance}")

    # Unpack parameters
    m, n, k, l = mnkl
    cluster_shape_mnk = (*cluster_shape_mn, 1)

    # Skip unsupported types
    if not HopperWgmmaGemmKernel.is_valid_dtypes(
        a_dtype, b_dtype, acc_dtype, c_dtype, a_major, b_major
    ):
        raise TypeError(
            f"Skipping due to unsupported combination of types and majors: {a_dtype}, {b_dtype}, {acc_dtype}, {c_dtype}, {a_major=}, {b_major=}"
        )

    # Prepare pytorch tensors: A, B (random from 0 to 2) and C (all zero)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(1111)

    # Create and permute tensor A/B/C
    def create_and_permute_tensor(
        l, mode0, mode1, is_mode0_major, dtype, is_dynamic_layout=True
    ):
        # is_mode0_major: (l, mode1, mode0) -> (mode0, mode1, l)
        # else : (l, mode0, mode1) -> (mode0, mode1, l)
        shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
        permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
        is_unsigned = dtype in {cutlass.Uint8}
        # Temporarily use uint8 as torch does not support fp8 type
        torch_dtype = (
            cutlass_torch.dtype(dtype)
            if dtype not in {cutlass.Float8E5M2, cutlass.Float8E4M3FN}
            else torch.uint8
        )

        # Create dtype torch tensor (cpu)
        torch_tensor_cpu = cutlass.torch.create_and_permute_torch_tensor(
            shape,
            torch_dtype,
            permute_order=permute_order,
            init_type=cutlass.torch.TensorInitType.RANDOM,
            init_config=cutlass.torch.RandomInitConfig(
                min_val=0 if is_unsigned else -2, max_val=4 if is_unsigned else 2
            ),
        )
        # Create dtype torch tensor (gpu)
        torch_tensor = torch_tensor_cpu.cuda()

        # Create f32 torch tensor (cpu)
        f32_torch_tensor = torch_tensor_cpu.to(dtype=torch.float32)

        # Create dtype cute tensor (gpu)
        cute_tensor = from_dlpack(torch_tensor, assumed_align=16)
        cute_tensor.element_type = dtype
        if is_dynamic_layout:
            cute_tensor = cute_tensor.mark_layout_dynamic(
                leading_dim=(0 if is_mode0_major else 1)
            )
        cute_tensor = cutlass.torch.convert_cute_tensor(
            f32_torch_tensor,
            cute_tensor,
            dtype,
            is_dynamic_layout=is_dynamic_layout,
        )

        return f32_torch_tensor, cute_tensor, torch_tensor

    a, mA, a_torch = create_and_permute_tensor(l, m, k, a_major == "m", a_dtype)
    b, mB, b_torch = create_and_permute_tensor(l, n, k, b_major == "n", b_dtype)
    c, mC, c_torch = create_and_permute_tensor(l, m, n, c_major == "m", c_dtype)

    gemm = HopperWgmmaGemmKernel(acc_dtype, tile_shape_mnk, cluster_shape_mnk)

    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    # compile gemm kernel
    compiled_gemm = cute.compile(gemm, mA, mB, mC, stream)
    # warm up
    compiled_gemm(mA, mB, mC, stream)
    runs = 10
    # execution
    import time
    ta = time.time()
    for _ in range(0, runs):
        compiled_gemm(mA, mB, mC, stream)
        torch.cuda.synchronize()
    tb = time.time()
    print(f"Time taken: {tb - ta} seconds")
    print(f"Time per run: {(tb - ta) / runs} seconds")
    print(f"cute TLOPS: {2 * m * n * k * runs / (tb - ta) / 1e12} TFLOPS")

    # Ref check
    ta = time.time()
    for _ in range(0, runs):
        ref = (torch.einsum("mkl,nkl->mnl", a, b))
        torch.cuda.synchronize()
    tb = time.time()
    print(f"Time taken: {tb - ta} seconds")
    print(f"Time per run: {(tb - ta) / runs} seconds")
    print(f"torch TLOPS: {2 * m * n * k * runs / (tb - ta) / 1e12} TFLOPS")
    ref = (torch.einsum("mkl,nkl->mnl", a, b)).cpu()
    if c_dtype in (cutlass.Float8E4M3FN, cutlass.Float8E5M2):
        # m major: (l, n, m) -> (m, n, l)
        # k major: (l, m, n) -> (m, n, l)
        permute_order = (1, 2, 0) if c_major == "n" else (2, 1, 0)
        shape = (l, m, n) if c_major == "n" else (l, n, m)
        f8_torch_tensor = cutlass_torch.create_and_permute_torch_tensor(
            shape,
            torch.uint8,
            permute_order=permute_order,
            init_type=cutlass_torch.TensorInitType.SKIP,
        ).cuda()
        # Create dtype cute tensor (gpu)
        ref_c_tensor = from_dlpack(
            f8_torch_tensor, assumed_align=16
        ).mark_layout_dynamic(leading_dim=(1 if c_major == "n" else 0))
        ref_c_tensor.element_type = c_dtype
        ref_c_tensor = cutlass_torch.convert_cute_tensor(
            ref,
            ref_c_tensor,
            c_dtype,
            is_dynamic_layout=True,
        )
        ref_c = f8_torch_tensor.cpu()
    else:
        ref_c = ref.to(cutlass_torch.dtype(c_dtype))

    torch.testing.assert_close(c_torch.cpu(), ref_c, atol=tolerance, rtol=1e-03)


if __name__ == "__main__":
    args = parse_arguments()
    run_dense_gemm(
        args.mnkl,
        args.a_dtype,
        args.b_dtype,
        args.c_dtype,
        args.acc_dtype,
        args.a_major,
        args.b_major,
        args.c_major,
        args.tile_shape_mnk,
        args.cluster_shape_mn,
        args.tolerance,
    )
    print("PASS")
