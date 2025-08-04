import torch

import triton
import triton.language as tl
import numpy as np



def get_triton_dtype(torch_dtype):
    if torch_dtype == torch.float32:
        return tl.float32
    elif torch_dtype == torch.bfloat16:
        return tl.bfloat16
    elif torch_dtype == torch.float16:
        return tl.float16
    else:
        raise ValueError(f"Unsupported dtype {torch_dtype}")


def get_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
    ]


@triton.jit
def _matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    grad_swiglu_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_swiglum,
    stride_swiglun,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMPUTE_TARGET: tl.constexpr,
    COMPUTE_PRECISION: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    # Load the right hand
    if COMPUTE_TARGET == "GRAD_E" or COMPUTE_TARGET == "FORWARD_SWIGLU":
        offs_bn += N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        b = b.to(COMPUTE_PRECISION)
        accumulator = tl.dot(a, b, accumulator, input_precision='ieee')
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if COMPUTE_TARGET == "GRAD_E":
        sigmoid = tl.sigmoid(accumulator)
        grad_c = sigmoid * accumulator
        grad_d = sigmoid + grad_c * (1 - sigmoid)
        c = grad_c.to(OUTPUT_DTYPE)
    elif COMPUTE_TARGET == "FORWARD_SWIGLU":
        accumulator = tl.sigmoid(accumulator) * accumulator
        c = accumulator.to(OUTPUT_DTYPE)
    else:
        c = accumulator.to(OUTPUT_DTYPE)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Storing to the right hand side.
    if COMPUTE_TARGET == "GRAD_D":
        offs_cn += N
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    if COMPUTE_TARGET == "GRAD_D":
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N * 2)
    else:
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if COMPUTE_TARGET == "GRAD_D" or COMPUTE_TARGET == "GRAD_E":
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        grad_swiglu_ptrs = (
            grad_swiglu_ptr
            + stride_swiglum * offs_cm[:, None]
            + stride_swiglun * offs_cn[None, :]
        )
        grad_swiglu_mask = c_mask
        grad_swiglu = tl.load(grad_swiglu_ptrs, mask=grad_swiglu_mask, other=0.0)
        c = c * grad_swiglu
    elif COMPUTE_TARGET == "FORWARD_SWIGLU":
        c_base = tl.load(c_ptrs, mask=c_mask, other=0.0)
        c = c_base * c
    tl.store(c_ptrs, c, mask=c_mask)

    if COMPUTE_TARGET == "GRAD_E":
        # recover the offsets for grad_d and grad_swiglu
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) + N
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N * 2)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        d_val = tl.load(c_ptrs, mask=c_mask, other=0.0)
        grad_d = d_val * grad_d
        grad_d = grad_d.to(OUTPUT_DTYPE)
        tl.store(c_ptrs, grad_d, mask=c_mask)


SWIGLU_MATMUL = _matmul_kernel


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
    c = torch.empty(*outer_shape, N // 2, device=a.device, dtype=a.dtype).view(
        M, N // 2
    )
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(N // 2, META["BLOCK_SIZE_N"]),
    )

    # First launch: compute the left part of the matrix multiplication.
    SWIGLU_MATMUL[grid](
        a,
        b,
        c,
        None,
        M,
        N // 2,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        0,
        0,
        COMPUTE_TARGET="NONE",
        COMPUTE_PRECISION=get_triton_dtype(a.dtype),
        OUTPUT_DTYPE=get_triton_dtype(c.dtype),
    )
    # Second launch: compute the gate (right) part of the matrix multiplication,
    # and apply the SwiGLU activation against the results of left part GEMM.
    SWIGLU_MATMUL[grid](
        a,
        b,
        c,
        None,
        M,
        N // 2,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        0,
        0,
        COMPUTE_TARGET="FORWARD_SWIGLU",
        COMPUTE_PRECISION=get_triton_dtype(a.dtype),
        OUTPUT_DTYPE=get_triton_dtype(c.dtype),
    )
    return c


@torch.library.register_fake("meshylearning::_fused_linear_swiglu")
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


@torch.library.custom_op("meshylearning::_mlp_swiglu_grad_de", mutates_args=())
def _mlp_swiglu_grad_de(
    a: torch.Tensor, b: torch.Tensor, grad_swiglu: torch.Tensor
) -> torch.Tensor:
    outer_shape = a.shape[:-1]
    if a.ndim == 3:
        a = a.view(-1, a.shape[-1])
    M, K = a.shape
    N = b.shape[1]
    # import pdb; pdb.set_trace()

    grad_de = torch.empty(*outer_shape, N, dtype=b.dtype, device=b.device).view(M, N)
    # print(f"grad de shape {grad_de.shape}")
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(N // 2, META["BLOCK_SIZE_N"]),
    )

    # First launch: compute the left part of the matrix multiplication.
    SWIGLU_MATMUL[grid](
        a,
        b,
        grad_de,
        grad_swiglu,
        M,
        N // 2,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        grad_de.stride(0),
        grad_de.stride(1),
        grad_swiglu.stride(0),
        grad_swiglu.stride(1),
        COMPUTE_TARGET="GRAD_D",
        COMPUTE_PRECISION=get_triton_dtype(a.dtype),
        OUTPUT_DTYPE=get_triton_dtype(grad_de.dtype),
    )

    # Second launch: compute grad d and grad e
    # 1. Update grad_d part, compute grad_d = grad_d * (e.sigmoid() + e * e.sigmoid() * (1 - e.sigmoid()))
    # 2. Update grad_e part, compute grad_e = grad_swiglu * grad_e * e.sigmoid()
    SWIGLU_MATMUL[grid](
        a,
        b,
        grad_de,
        grad_swiglu,
        M,
        N // 2,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        grad_de.stride(0),
        grad_de.stride(1),
        grad_swiglu.stride(0),
        grad_swiglu.stride(1),
        COMPUTE_TARGET="GRAD_E",
        COMPUTE_PRECISION=get_triton_dtype(a.dtype),
        OUTPUT_DTYPE=get_triton_dtype(grad_de.dtype),
    )
    return grad_de


@torch.library.register_fake("meshylearning::_mlp_swiglu_grad_de")
def _(a: torch.Tensor, b: torch.Tensor, grad_swiglu: torch.Tensor) -> torch.Tensor:
    if a.ndim == 3:
        a = a.view(-1, a.shape[-1])
    M, _ = a.shape
    N = b.shape[1]

    grad_de = torch.empty(M, N, dtype=b.dtype, device=b.device)

    return grad_de


def _fused_linear_swiglu_backward(ctx, grad_swiglu):
    a, b = ctx.saved_tensors

    # The fused mlp swiglu backward kernel implements the following:
    # grad_d = grad_swiglu * d * (e.sigmoid() + e * e.sigmoid() * (1 - e.sigmoid()))
    # grad_e = grad_swiglu * e * e.sigmoid()
    # grad_de = torch.cat([grad_d, grad_e], dim=-1)
    # It only allocates grad_de, and perform every computation inplace.
    a_shape = ctx.a_shape
    grad_de = _mlp_swiglu_grad_de(a, b, grad_swiglu)
    grad_a = torch.matmul(grad_de, b.t())
    if grad_a.shape != a_shape:
        grad_a = grad_a.view(*a_shape)
    grad_b = torch.matmul(a.view(-1, a.shape[-1]).t().to(b.dtype), grad_de)
    return grad_a, grad_b


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


def apply_fused_linear_swiglu(
    x: torch.Tensor, linear: torch.nn.Linear | torch.Tensor
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


def setup_autotune():
    global SWIGLU_MATMUL
    SWIGLU_MATMUL = triton.autotune(
        configs=get_autotune_config(),
        key=["M", "N", "K"],
    )(_matmul_kernel)

def perf_swiglu(x: torch.Tensor,
                w: torch.Tensor,
                warmup_iterations: int = 1,
                iterations: int = 100):
    # warmup
    for _ in range(warmup_iterations):
        apply_fused_linear_swiglu(x, w)

    # run
    times = []
    import time
    for _ in range(iterations):
        start_time = time.perf_counter()
        apply_fused_linear_swiglu(x, w)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)
    avg_time = np.mean(times)
    return avg_time