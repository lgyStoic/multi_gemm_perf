import torch

import triton
import triton.language as tl
import numpy as np
from typing import Optional

def get_triton_dtype(torch_dtype):
    if torch_dtype == torch.float32:
        return tl.float32
    elif torch_dtype == torch.bfloat16:
        return tl.bfloat16
    elif torch_dtype == torch.float16:
        return tl.float16
    else:
        raise ValueError(f"Unsupported dtype {torch_dtype}")

@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n

def get_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                     num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
    ]

@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_persistent_left_part_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M, N, K,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):

    """
    Hopper-compatible persistent TMA matmul kernel.
    """
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    # Create tensor descriptors on device
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[K, N],
        strides=[2 * N, 1],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    # tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, warp_specialize=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_k, offs_bn])
            accumulator = tl.dot(a, b, accumulator)

        c = accumulator.to(tl.float16)
        c_desc.store([offs_am, offs_bn], c)

@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_persistent_right_part_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,  
    M, N, K,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    # Create tensor descriptors on device
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr + N,
        shape=[K, N],
        strides=[2 * N, 1],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    d_desc = tl.make_tensor_descriptor(
        d_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    # tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, warp_specialize=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        c_base = c_desc.load([offs_am, offs_bn])
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_k, offs_bn])
            accumulator = tl.dot(a, b, accumulator)
        c = tl.sigmoid(accumulator) * accumulator
        c_res = c_base * c
        d_desc.store([offs_am, offs_bn], c_res.to(tl.float16))

@torch.library.custom_op("meshylearning::_fused_linear_swiglu", mutates_args=())
def _fused_linear_swiglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Check constraints.
    assert (
        a.shape[-1] == b.shape[0]
    ), f"Incompatible dimensions, {a.shape} and {b.shape}"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert a.ndim == 2 or a.ndim == 3, "Matrix A must have 2 or 3 dimensions"
    outer_shape = a.shape[:-1]
    if a.ndim == 3:
        a = a.view(-1, a.shape[-1])
    M, K = a.shape
    K, N = b.shape
    assert N % 2 == 0, "Matrix B must have an even number of columns"

    # Allocates output. b is the weight so we follow its device and dtype.
    c = torch.ones(*outer_shape, N // 2, device=a.device, dtype=a.dtype).view(
        M, N // 2
    )

    d = torch.zeros(*outer_shape, N // 2, device=a.device, dtype=a.dtype).view(
        M, N // 2
    )

    # 1D launch kernel where each block gets its own program.
    
    def grid(META):
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)), )

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # Set up memory allocator for TMA descriptors
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    _matmul_persistent_left_part_kernel[grid](
        a,
        b,
        c,
        M,
        N // 2,
        K,
        NUM_SMS
    )
    # Second launch: compute the gate (right) part of the matrix multiplication,
    # and apply the SwiGLU activation against the results of left part GEMM.
    _matmul_persistent_right_part_kernel[grid](
        a,
        b,
        c,
        d,
        M,
        N // 2,
        K,
        NUM_SMS
    )

    return d
    
def apply_fused_linear_swiglu(
    x: torch.Tensor, 
    linear: torch.nn.Linear | torch.Tensor, 
) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()
    if isinstance(linear, torch.Tensor):
        weight_t = linear
    else:
        weight_t = linear.weight.t()
    y = _fused_linear_swiglu(x, weight_t)
    assert isinstance(y, torch.Tensor)
    y = y.view(*x.shape[:-1], -1)
    return y

def perf_swiglu(x: torch.Tensor,
                w: torch.Tensor,
                warmup_iterations: int = 10,
                iterations: int = 100):
    # warmup
    for _ in range(warmup_iterations):
        _fused_linear_swiglu(x, w)
    # measure
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        _fused_linear_swiglu(x, w)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations

def validate_swiglu(x: torch.Tensor, w: torch.Tensor):
    y = apply_fused_linear_swiglu(x, w)
    w1, w2 = w.chunk(2, dim=-1)
    y1 = torch.matmul(x, w1)
    y2 = torch.matmul(x, w2)
    p2 = y2 * torch.sigmoid(y2)
    y_ref = y1 * p2
    atol = 1e-2
    rtol = 1e-1
    is_close = torch.isclose(y, y_ref, atol=atol, rtol=rtol)
    total = y.numel()
    num_close = is_close.sum().item()
    percent_close = num_close / total * 100
    print(f"allclose 通过的元素数: {num_close}/{total} ({percent_close:.4f}%)")
    diff = (y - y_ref).abs()
    max_diff = diff.max()
    max_ref = y_ref.abs().max()
    percent = (max_diff / max_ref * 100).item() if max_ref > 0 else float('nan')
    print(f"最大差距: {max_diff.item()}，百分比: {percent:.4f}%")

if __name__ == "__main__":
    num = 4096
    torch.manual_seed(0)
    x = torch.randn(num, num, device="cuda", dtype=torch.float16)
    w = torch.randn(num, num* 6, device="cuda", dtype=torch.float16)
    validate_swiglu(x, w)