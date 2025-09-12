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
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n

def get_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": 8, "WARP_SPECIALIZE": WS}, num_stages=s,
                      num_warps=w)
        for BM in [128, 256]
        for BN in [128, 256]
        for BK in [64, 128]
        for s in ([3, 4])
        for w in [8]
        for WS in [True, False]
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
    WARP_SPECIALIZE: tl.constexpr,
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

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
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
    WARP_SPECIALIZE: tl.constexpr,
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

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
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

# @torch.library.register_fake("meshylearning::_fused_linear_swiglu")
@_fused_linear_swiglu.register_fake
def _(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
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
    c = torch.empty(*outer_shape, N // 2, device=a.device, dtype=a.dtype).view(
        M, N // 2
    )

    return c
    
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

@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def calc_gemm(x_ptr, w_ptr, t_ptr,
            M, N, K, mode: tl.constexpr,
            NUM_SMS: tl.constexpr,
            WARP_SPECIALIZE: tl.constexpr,
            BLOCK_SIZE_M: tl.constexpr,
            BLOCK_SIZE_N: tl.constexpr,
            BLOCK_SIZE_K: tl.constexpr,
            GROUP_SIZE_M: tl.constexpr):
    """
    t = x @ w, 
    tensor is row major,
    if NT, assume N is normal, T is transposed; else if TN, assume T is normal, N is transposed
    if NT, x is N, w is T; so x is M, K, w is N, K
    if TN, x is T, w is N; so x is K, M, w is K, N
    if NN, x is N, w is N; so x is M, K, w is K, N
    """
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    # Create tensor descriptors on device
    if mode == "NN":
        x_desc = tl.make_tensor_descriptor(
            x_ptr,
            shape=[M, K],
            strides=[K, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )
        w_desc = tl.make_tensor_descriptor(
            w_ptr,
            shape=[K, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
        )
        t_desc = tl.make_tensor_descriptor(
            t_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )
    if mode == "TN":
        x_desc = tl.make_tensor_descriptor(
            x_ptr,
            shape=[K, M],
            strides=[M, 1],
            block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_M],
        )
        w_desc = tl.make_tensor_descriptor(
            w_ptr,
            shape=[K, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
        )
        t_desc = tl.make_tensor_descriptor(
            t_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )
    if mode == "NT":
        x_desc = tl.make_tensor_descriptor(
            x_ptr,
            shape=[M, K],
            strides=[K, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )
        w_desc = tl.make_tensor_descriptor(
            w_ptr,
            shape=[N, K],
            strides=[K, 1],
            block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
        )
        t_desc = tl.make_tensor_descriptor(
            t_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            if mode == "NN":
                x = x_desc.load([offs_am, offs_k])
                w = w_desc.load([offs_k, offs_bn])
                accumulator = tl.dot(x, w, accumulator)
            elif mode == "NT":
                x = x_desc.load([offs_am, offs_k])
                w = w_desc.load([offs_bn, offs_k])
                accumulator = tl.dot(x, tl.trans(w), accumulator)
            elif mode == "TN":
                x = x_desc.load([offs_k, offs_am])
                w = w_desc.load([offs_k, offs_bn])
                accumulator = tl.dot(tl.trans(x), w, accumulator)

        t = accumulator.to(tl.float16)
        t_desc.store([offs_am, offs_bn], t)

@triton.autotune(
    configs=[triton.Config({'GROUP_SIZE_M': g}) for g in [1, 2, 4, 8, 16, 32, 64]],
    key=["M", "N", "N_2"],
)
@triton.jit
def calc_t_swiglu_partial(t_ptr, grad_y_ptr, grad_swiglu_ptr,
                                    M: tl.constexpr, 
                                    N: tl.constexpr, 
                                    N_2: tl.constexpr, 
                                    POW_N_2: tl.constexpr,
                                    GROUP_SIZE_M: tl.constexpr):
    """
    calc t swiglu partial
    """
    pid = tl.program_id(axis=0)
        
    start_row = pid * GROUP_SIZE_M
    end_row = min(M, start_row + GROUP_SIZE_M)
    group_size = end_row - start_row
    y1_ptr = t_ptr + start_row * N
    y2_ptr = t_ptr + start_row * N + N_2
    dy_dy1_ptr = grad_y_ptr + start_row * N
    dy_dy2_ptr = grad_y_ptr + start_row * N + N_2
    grad_swiglu_tmp_ptr = grad_swiglu_ptr + start_row * N_2
    offset_n2 = tl.arange(0, POW_N_2)
    mask_n2 = offset_n2 < N_2

    for i in range(group_size):
        y1 = tl.load(y1_ptr + offset_n2, mask=mask_n2)
        y2 = tl.load(y2_ptr + offset_n2, mask=mask_n2)
        grad_swiglu = tl.load(grad_swiglu_tmp_ptr + offset_n2, mask=mask_n2)
        y2_sigmoid = tl.sigmoid(y2.to(tl.float32)).to(tl.float16)
        p2 = y2 * y2_sigmoid
        # Correct SwiGLU gradient computation:
        # For y1: grad_swiglu * p2 (where p2 = y2 * sigmoid(y2))
        # For y2: grad_swiglu * y1 * (sigmoid(y2) + y2 * sigmoid(y2) * (1 - sigmoid(y2)))
        grad_y1 = p2 * grad_swiglu
        tl.store(dy_dy1_ptr + offset_n2, grad_y1, mask=mask_n2)
        grad_y2 = y1 * (y2_sigmoid + p2 * (1.0 - y2_sigmoid)) * grad_swiglu
        tl.store(dy_dy2_ptr + offset_n2, grad_y2, mask=mask_n2)
        y2_ptr += N
        y1_ptr += N
        dy_dy2_ptr += N
        dy_dy1_ptr += N
        grad_swiglu_tmp_ptr += N_2

@torch.library.custom_op("meshylearning::fused_linear_swiglu_grad", mutates_args=())
def fused_linear_swiglu_grad(x: torch.Tensor,
                                w: torch.Tensor,
                                grad_swiglu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate grad_t, to compute grad_t, need y1, y2, and sigma(y2)
    """
    M, K = x.shape
    _, N = w.shape
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    device = x.device
    t = torch.zeros(M, N, device=device, dtype=torch.float16)
    grad_x = torch.zeros(M, K, device=device, dtype=torch.float16)
    grad_w = torch.zeros(K, N, device=device, dtype=torch.float16)
    grad_dt = torch.zeros(M, N, device=device, dtype=torch.float16)
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device=device, dtype=torch.int8)
    triton.set_allocator(alloc_fn)
    def grid(META):
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)), )
    calc_gemm[grid](x, w, t, M, N, K, "NN", NUM_SMS)
    def grid2(META):
        GROUP_SIZE_M = META["GROUP_SIZE_M"]
        return (triton.cdiv(M, GROUP_SIZE_M), )
    assert N % 2 == 0, "Matrix B must have an even number of columns"
    N_2 = N // 2
    POW_N_2 = triton.next_power_of_2(N_2)
    calc_t_swiglu_partial[grid2](t, grad_dt, grad_swiglu.contiguous(), M, N, N_2, POW_N_2);
    # Use PyTorch's built-in matrix multiplication for simplicity
    # grad_w = x.t() @ grad_dt  (K x M) @ (M x N) = (K x N)
    calc_gemm[grid](x, grad_dt, grad_w, K, N, M, "TN", NUM_SMS)
    # grad_x = grad_dt @ w.t()  (M x N) @ (N x K) = (M x K)
    calc_gemm[grid](grad_dt, w, grad_x, M, K, N, "NT", NUM_SMS)
    return grad_x, grad_w

def _fused_linear_swiglu_backward(ctx, grad_swiglu):
    x, w = ctx.saved_tensors
    grad_x, grad_w = fused_linear_swiglu_grad(x, w, grad_swiglu)

    return grad_x, grad_w

@torch.library.register_fake("meshylearning::fused_linear_swiglu_grad")
def _(x: torch.Tensor, w: torch.Tensor, grad_swiglu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim == 3:
        x = x.view(-1, x.shape[-1])
    M, K = x.shape
    N = w.shape[1]
    grad_w = torch.empty(K, N, dtype=w.dtype, device=w.device)
    grad_x = torch.empty(M, K, dtype=w.dtype, device=w.device)
    return grad_x, grad_w 

def _setup_context(ctx, inputs, output):
    a, b = inputs
    ctx.a_shape = a.shape
    ctx.save_for_backward(a, b)

# Implements fused MLP + SwiGLU activation function.
torch.library.register_autograd(
    "meshylearning::_fused_linear_swiglu",
    _fused_linear_swiglu_backward,
    setup_context=_setup_context,
)

def perf_swiglu(x: torch.Tensor,
                w: torch.Tensor,
                warmup_iterations: int = 10,
                iterations: int = 100):
    # warmup
    for _ in range(warmup_iterations):
        _fused_linear_swiglu(x, w)
    # measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iterations):
        _fused_linear_swiglu(x, w)
    end.record()
    torch.cuda.synchronize()
    fwd_time = start.elapsed_time(end) / iterations
    y = _fused_linear_swiglu(x, w)
    for _ in range(warmup_iterations):
        fused_linear_swiglu_grad(x, w, torch.ones_like(y))
    torch.cuda.synchronize()
    start.record()
    for _ in range(iterations):
        fused_linear_swiglu_grad(x, w, torch.ones_like(y))
    end.record()
    torch.cuda.synchronize()
    bwd_time = start.elapsed_time(end) / iterations
    return fwd_time, bwd_time

def validate_swiglu(x: torch.Tensor, w: torch.Tensor):
    w_autograd = w.clone().requires_grad_()
    x_autograd = x.clone().requires_grad_()
    w_autograd_1 = w.clone().requires_grad_()
    x_autograd_1 = x.clone().requires_grad_()
    y = apply_fused_linear_swiglu(x_autograd_1, w_autograd_1)
    w1, w2 = w_autograd.chunk(2, dim=-1)
    y1 = torch.matmul(x_autograd, w1)
    y2 = torch.matmul(x_autograd, w2)
    p2 = y2 * torch.sigmoid(y2)
    y_ref = y1 * p2
    assert torch.allclose(y, y_ref, atol=1e-2, rtol=1e-2)
    y.sum().backward(retain_graph=True)

    # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
    #     y.sum().backward(retain_graph=True)
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    y_ref.sum().backward()
    assert torch.allclose(x_autograd.grad, x_autograd_1.grad, atol=1.0, rtol=1e-1)
    assert torch.allclose(w_autograd.grad, w_autograd_1.grad, atol=1.0, rtol=1e-1)

if __name__ == "__main__":
    num = 512 
    # torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(0)
    x = torch.randn(num, num, device="cuda", dtype=torch.float16)
    w = torch.randn(num, num* 6, device="cuda", dtype=torch.float16)
    validate_swiglu(x, w)
    print("forward and backward passed")
    # t = perf_swiglu(x, w)
    # print(f"Triton persistent TMA: {t:.3f} ms")
    #t_ref = torch.matmul(x, w)

    #M, K = x.shape
    #_, N = w.shape
    #grad_x, grad_w = fused_linear_swiglu_grad(x, w, torch.ones(M, N, device="cuda", dtype=torch.float16))

    #y1_ref, y2_ref = t_ref.chunk(2, dim=-1)
    #sig_y2_ref = torch.sigmoid(y2_ref)
    #one_minus_sigmoid_ref = 1.0 - sig_y2_ref
    #p2_ref = y2_ref * sig_y2_ref
    #product_ref = p2_ref * one_minus_sigmoid_ref
    #o_ref = y1_ref * (sig_y2_ref + product_ref)
    #dt_ref = torch.concat([p2_ref, o_ref], dim=-1)

    #grad_x_ref = torch.matmul(dt_ref, w.t())
    #grad_w_ref = torch.matmul(x.t(), dt_ref)
    
    ## Check if they're close (allow for small numerical differences)
    ##print(grad_x_ref)
    ##print(grad_x)
    #assert torch.allclose(grad_w, grad_w_ref, atol=2.0, rtol=1e-2)
    #assert torch.allclose(grad_x, grad_x_ref, atol=2.0, rtol=1e-2)
    #print("backward passed")

    ## assert torch.allclose(grad_x, grad_x_ref, atol=1e-2, rtol=1e-1)
    ## assert torch.allclose(grad_w, grad_w_ref, atol=1e-2, rtol=1e-1)
