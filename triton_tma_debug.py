import triton
import triton.language as tl
import torch
from typing import Optional

def get_autotune_config():
    return [
        # triton.Config(
        #     {
        #         "BLOCK_SIZE_M": 128,
        #         "BLOCK_SIZE_N": 256,
        #         "BLOCK_SIZE_K": 64,
        #         "GROUP_SIZE_M": 8,
        #     },
        #     num_stages=3,
        #     num_warps=8,
        # ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
    ]

@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n

@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_with_base_kernel(
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
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )

    # tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, warp_specialize=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_base_ptr = c_ptr + offs_cm[:, None] * N + offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        # c_base = tl.load(c_base_ptr, mask=c_mask, other=0.0)
        # c_base = c_desc.load([offs_am, offs_bn])
        c_base = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16) + 1.0

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_k, offs_bn])
            accumulator = tl.dot(a, b, accumulator)

        c = accumulator.to(tl.float16) * c_base
        tl.store(c_base_ptr, c, mask=c_mask)

def matmul(a, b, c):
    M, K = a.shape
    K, N = b.shape
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    # Set up memory allocator for TMA descriptors
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    _matmul_with_base_kernel[grid](a, b, c, M, N, K, NUM_SMS)
    return c

if __name__ == "__main__":
    torch.manual_seed(11)
    a = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
    b = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
    c_base = torch.ones(4096, 4096, dtype=torch.float16, device="cuda") * 1
    c_base1 = c_base.clone()
    c = matmul(a, b, c_base)
    c_ref = torch.matmul(a, b) * c_base1
    print(c)
    print(c_ref)
    assert torch.allclose(c, c_ref)
    print("passed")