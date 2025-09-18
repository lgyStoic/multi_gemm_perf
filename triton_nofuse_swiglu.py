import torch
import triton
import triton.language as tl


from torch.library import triton_op, wrap_triton 



# Forward kernel: computes out = x * silu(gate)
@triton.jit
def swiglu_fwd_kernel(
    inp_ptr,  # pointer to input tensor, shape [B*L, 2*D]
    out_ptr,  # pointer to output tensor, shape [B*L, D]
    D: tl.constexpr,  # hidden dimension
    BLOCK_SIZE: tl.constexpr,
    PTR_TYPE: tl.constexpr = tl.int32,
):
    pid = tl.cast(tl.program_id(0), PTR_TYPE)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    # Each row has 2*D elements.
    base = pid * (2 * D)

    # Load the two halves: first D elements for x, next D for gate.
    x = tl.load(inp_ptr + base + offs, mask=mask)
    gate = tl.load(inp_ptr + base + D + offs, mask=mask)

    # Compute silu(gate) = gate * sigmoid(gate).
    # Upcast gate to fp32 for the sigmoid (tl.sigmoid requires fp32).
    gate_fp32 = tl.cast(gate, tl.float32)
    sig = tl.sigmoid(gate_fp32)
    silu_fp32 = gate_fp32 * sig
    silu = tl.cast(silu_fp32, gate.dtype)

    # Compute the output.
    out = x * silu
    tl.store(out_ptr + pid * D + offs, out, mask=mask)


# Backward kernel: computes gradients w.r.t. x and gate given grad_out.
@triton.jit
def swiglu_bwd_kernel(
    grad_out_ptr,  # pointer to grad output tensor, shape [B*L, D]
    inp_ptr,  # pointer to input tensor, shape [B*L, 2*D]
    grad_in_ptr,  # pointer to grad input tensor, shape [B*L, 2*D]
    D: tl.constexpr,  # hidden dimension
    BLOCK_SIZE: tl.constexpr,
    PTR_TYPE: tl.constexpr = tl.int32,
):
    pid = tl.cast(tl.program_id(0), PTR_TYPE)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    base = pid * (2 * D)

    # Load forward inputs.
    x = tl.load(inp_ptr + base + offs, mask=mask)
    gate = tl.cast(tl.load(inp_ptr + base + D + offs, mask=mask), tl.float32)
    # Load grad_out.
    grad_out = tl.cast(tl.load(grad_out_ptr + pid * D + offs, mask=mask), tl.float32)

    # Upcast gate to fp32 for computing sigmoid and its derivative.
    sig = tl.sigmoid(gate)
    silu = gate * sig

    # Derivative of silu(gate) with respect to gate:
    # d/dg silu(g) = sigmoid(g) + gate * sigmoid(g) * (1 - sigmoid(g))
    dsilu = sig + silu * (1.0 - sig)

    # Compute gradients.
    grad_x = grad_out * silu
    grad_gate = grad_out * tl.cast(x, tl.float32) * dsilu

    # Write gradients back.
    tl.store(grad_in_ptr + base + offs, tl.cast(grad_x, x.dtype), mask=mask)
    tl.store(grad_in_ptr + base + D + offs, tl.cast(grad_gate, x.dtype), mask=mask)


@triton_op("meshylearning::_swiglu", mutates_args=())
def _swiglu(x: torch.Tensor) -> torch.Tensor:
    torch._check(x.is_contiguous(), "Input to triton SwiGLU must be contiguous")
    L, D2 = x.shape
    torch._check(
        D2 % 2 == 0, "Triton SwiGLU should have a hidden dimension divisible by 2"
    )
    D = D2 // 2

    out = torch.empty((L, D), dtype=x.dtype, device=x.device)

    ptr_type = tl.int32 if x.nelement() * x.element_size() < 2**31 else tl.int64

    wrap_triton(swiglu_fwd_kernel)[(L,)](x, out, D, triton.next_power_of_2(D), ptr_type)

    return out

@triton_op("meshylearning::_swiglu_bwd", mutates_args=())
def _swiglu_bwd(grad_Y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    L, D = grad_Y.shape
    grad_x = torch.empty((L, D * 2), dtype=x.dtype, device=x.device)

    grad_Y = grad_Y.contiguous()

    ptr_type = tl.int32 if x.nelement() * x.element_size() < 2**31 else tl.int64

    wrap_triton(swiglu_bwd_kernel)[(L,)](
        grad_Y, x, grad_x, D, triton.next_power_of_2(D), ptr_type
    )

    return grad_x

def swiglu(x: torch.Tensor) -> torch.Tensor:
    x_shape = x.shape
    return _swiglu(x.flatten(0, -2).contiguous()).view(*x_shape[:-1], -1)


def _swiglu_setup_context(ctx, inputs, output):
    (x,) = inputs
    ctx.save_for_backward(x)


def _swiglu_backward(ctx, grad_Y):
    (x,) = ctx.saved_tensors
    return _swiglu_bwd(grad_Y, x)

_swiglu.register_autograd(
    _swiglu_backward,
    setup_context=_swiglu_setup_context,
)
    
def perf_swiglu(x, iterations=100):
    avg_fwd_time = 0
    avg_bwd_time = 0
    warmup_iterations = 10
    
    # Forward pass warmup
    for _ in range(warmup_iterations):
        x_autograd = x.clone().requires_grad_()
        y = _swiglu(x_autograd)
    
    # Forward pass performance test
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    x_autograd = x.clone()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iterations):
        y = _swiglu(x_autograd)
    end.record()
    end.synchronize()
    avg_fwd_time = start.elapsed_time(end) / iterations
    
    grad_y = torch.ones_like(y)

    
    # Backward pass warmup
    for _ in range(warmup_iterations):
        _swiglu_bwd(grad_y, x_autograd)
    
    # Backward pass performance test
    torch.cuda.synchronize()
    start.record()
    for _ in range(iterations):
        _swiglu_bwd(grad_y, x_autograd)
    end.record()
    end.synchronize()
    avg_bwd_time = start.elapsed_time(end) / iterations
    return avg_fwd_time, avg_bwd_time