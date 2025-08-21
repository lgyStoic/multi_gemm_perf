"""
Hopper TMA Matmul
=====================
This script provides a Hopper-compatible TMA (Tensor Memory Accelerator) implementation of matrix multiplication.
It uses device-side tensor descriptors to avoid compatibility issues with warp specialization on Hopper GPUs.
"""

import torch
import triton
import triton.language as tl
from typing import Optional


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_hopper():
    return torch.cuda.get_device_capability()[0] == 9


def supports_ws():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


HAS_TENSOR_DESC = supports_tma() and hasattr(tl, "make_tensor_descriptor")
HAS_WARP_SPECIALIZE = supports_ws() and HAS_TENSOR_DESC


def matmul_get_configs():
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
    configs=matmul_get_configs(),
    key=["M", "N", "K", "WARP_SPECIALIZE"],
)
@triton.jit
def matmul_kernel_tma_hopper(a_ptr, b_ptr, c_ptr,  #
                            M, N, K,  #
                            BLOCK_SIZE_M: tl.constexpr,  #
                            BLOCK_SIZE_N: tl.constexpr,  #
                            BLOCK_SIZE_K: tl.constexpr,  #
                            GROUP_SIZE_M: tl.constexpr,  #
                            WARP_SPECIALIZE: tl.constexpr,  #
                            ):
    """
    Hopper-compatible TMA matmul kernel using device-side tensor descriptors.
    This avoids the host-side tensor descriptor issues with warp specialization on Hopper.
    """
    dtype = tl.float16

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
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Use warp specialization if supported and enabled
    for k in tl.range(k_tiles, warp_specialize=WARP_SPECIALIZE):
        offs_k = k * BLOCK_SIZE_K
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_k, offs_bn])
        accumulator = tl.dot(a, b, accumulator)

    c = accumulator.to(dtype)

    offs_cm = pid_m * BLOCK_SIZE_M
    offs_cn = pid_n * BLOCK_SIZE_N
    c_desc.store([offs_cm, offs_cn], c)


def matmul_tma_hopper(a, b, c, warp_specialize: bool = False):
    """
    Hopper-compatible TMA matrix multiplication.
    
    Args:
        a: Input tensor A of shape (M, K)
        b: Input tensor B of shape (K, N)
        warp_specialize: Whether to use warp specialization (only works on Hopper+)
    
    Returns:
        Output tensor C of shape (M, N)
    """
    # Check constraints
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"  
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert HAS_TENSOR_DESC, "TMA not supported on this device"

    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype

    # Set up memory allocator for TMA descriptors
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    def grid(META):
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

    matmul_kernel_tma_hopper[grid](
        a, b, c,  #
        M, N, K,  #
        WARP_SPECIALIZE=warp_specialize,  #
    )


def matmul_tma_hopper_persistent(a, b, c, warp_specialize: bool = False):
    """
    Hopper-compatible persistent TMA matrix multiplication.
    This version uses persistent kernels for better performance on large matrices.
    """
    # Check constraints
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert HAS_TENSOR_DESC, "TMA not supported on this device"

    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # Set up memory allocator for TMA descriptors
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    # Hopper warpspec doesn't work with flatten
    flatten = False if (warp_specialize and is_hopper()) else True

    def grid(META):
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)), )

    matmul_kernel_tma_hopper_persistent[grid](
        a, b, c,  #
        M, N, K,  #
        NUM_SMS=NUM_SMS,  #
        WARP_SPECIALIZE=warp_specialize,  #
        FLATTEN=flatten,
    )

@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.autotune(
    configs=matmul_get_configs(),
    key=["M", "N", "K", "WARP_SPECIALIZE", "FLATTEN"],
)
@triton.jit
def matmul_kernel_tma_hopper_persistent(a_ptr, b_ptr, c_ptr,  #
                                       M, N, K,  #
                                       BLOCK_SIZE_M: tl.constexpr,  #
                                       BLOCK_SIZE_N: tl.constexpr,  #
                                       BLOCK_SIZE_K: tl.constexpr,  #
                                       GROUP_SIZE_M: tl.constexpr,  #
                                       NUM_SMS: tl.constexpr,  #
                                       WARP_SPECIALIZE: tl.constexpr,  #
                                       FLATTEN: tl.constexpr,  #
                                       ):
    """
    Hopper-compatible persistent TMA matmul kernel.
    """
    dtype = tl.float16
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
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=FLATTEN, warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_k, offs_bn])
            accumulator = tl.dot(a, b, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M
        offs_cn = pid_n * BLOCK_SIZE_N

        c = accumulator.to(dtype)
        c_desc.store([offs_cm, offs_cn], c)


# Example usage and testing functions
def test_hopper_tma():
    """Test the Hopper TMA implementation"""
    if not is_cuda():
        print("CUDA not available")
        return
    
    if not supports_tma():
        print("TMA not supported on this device")
        return
    
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Compute Capability: {torch.cuda.get_device_capability()}")
    print(f"Is Hopper: {is_hopper()}")
    print(f"TMA Supported: {supports_tma()}")
    print(f"Warp Specialize Supported: {supports_ws()}")
    
    # Test with small matrices first
    M, N, K = 512, 512, 512
    dtype = torch.float16
    
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((N, K), device="cuda", dtype=dtype)
    
    print(f"\nTesting with {M}x{K} @ {K}x{N} matrices")
    
    # Reference result
    ref = torch.matmul(a, b)
    
    # Test non-persistent version
    try:
        c = torch.empty((M, N), device=a.device, dtype=dtype)
        matmul_tma_hopper(a, b, c, warp_specialize=False)
        error = torch.abs(ref - c).max().item()
        print(f"TMA Hopper (no warp_spec): max error = {error:.2e}")
        
        if is_hopper() and supports_ws():
            matmul_tma_hopper(a, b, c, warp_specialize=True)
            error_ws = torch.abs(ref - c).max().item()
            print(f"TMA Hopper (warp_spec): max error = {error_ws:.2e}")
    except Exception as e:
        print(f"TMA Hopper failed: {e}")
    
    # Test persistent version
    try:
        c = torch.empty((M, N), device=a.device, dtype=dtype)
        matmul_tma_hopper_persistent(a, b, c, warp_specialize=False)
        error_persistent = torch.abs(ref - c).max().item()
        print(f"TMA Hopper Persistent (no warp_spec): max error = {error_persistent:.2e}")
        
        if is_hopper() and supports_ws():
            matmul_tma_hopper_persistent(a, b, c, warp_specialize=True)
            error_persistent_ws = torch.abs(ref - c).max().item()
            print(f"TMA Hopper Persistent (warp_spec): max error = {error_persistent_ws:.2e}")
    except Exception as e:
        print(f"TMA Hopper Persistent failed: {e}")


if __name__ == "__main__":
    test_hopper_tma() 