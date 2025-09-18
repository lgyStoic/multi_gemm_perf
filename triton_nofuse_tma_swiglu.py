import torch
import triton
import triton.language as tl
from torch.library import triton_op
from torch.library import wrap_triton
from typing import Optional

# Forward kernel: computes out = x * silu(gate)
@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n}) 
        for m in [16,32,64,128] for n in [16,32,64,128]],
    key=["M", "N"],
)
@triton.jit
def swiglu_fwd_kernel(
    inp_ptr,  # pointer to input tensor, shape [B*L, 2*D]
    out_ptr,  # pointer to output tensor, shape [B*L, D]
    M: tl.constexpr,
    N2: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    N = N2 // 2
    x_desc = tl.make_tensor_descriptor(
        inp_ptr,
        shape=[M, N],
        strides=[N2, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    gate_desc = tl.make_tensor_descriptor(
        inp_ptr + N,
        shape=[M, N],
        strides=[N2, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N

    x = x_desc.load([offs_am, offs_bn])
    gate = gate_desc.load([offs_am, offs_bn])
    gate = gate * tl.sigmoid(gate.to(tl.float32)).to(tl.float16)
    out = x * gate
    out_desc.store([offs_am, offs_bn], out)

# Backward kernel: computes gradients w.r.t. x and gate given grad_out.
@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n}) 
        for m in [8, 16] for n in [64, 128, 256, 512]],
    key=["M", "N"],
)
@triton.jit
def swiglu_bwd_kernel(
    grad_out_ptr,  # pointer to grad output tensor, shape [B*L, D]
    inp_ptr,  # pointer to input tensor, shape [B*L, 2*D]
    grad_in_ptr,  # pointer to grad input tensor, shape [B*L, 2*D]
    M: tl.constexpr,
    N2: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    N = N2 // 2
    y1_desc = tl.make_tensor_descriptor(
        inp_ptr,
        shape=[M, N],
        strides=[N2, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    y2_desc = tl.make_tensor_descriptor(
        inp_ptr + N,
        shape=[M, N],
        strides=[N2, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    grad_out_desc = tl.make_tensor_descriptor(
        grad_out_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    grad_in_left_desc = tl.make_tensor_descriptor(
        grad_in_ptr,
        shape=[M, N],
        strides=[N2, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    grad_in_right_desc = tl.make_tensor_descriptor(
        grad_in_ptr + N,
        shape=[M, N],
        strides=[N2, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N

    y1 = y1_desc.load([offs_am, offs_bn])
    y2 = y2_desc.load([offs_am, offs_bn])

    grad_out = grad_out_desc.load([offs_am, offs_bn])

    y2_sigmoid = tl.sigmoid(y2.to(tl.float32)).to(tl.float16)
    silu = y2 * y2_sigmoid
    d_silu = y2_sigmoid + silu * (1.0 - y2_sigmoid)
    grad_in_right = grad_out * y1 * d_silu
    grad_in_left = grad_out * silu 
    grad_in_left_desc.store([offs_am, offs_bn], grad_in_left)
    grad_in_right_desc.store([offs_am, offs_bn], grad_in_right)

@triton_op("meshylearning::_swiglu", mutates_args=())
def _swiglu(x: torch.Tensor) -> torch.Tensor:
    torch._check(x.is_contiguous(), "Input to triton SwiGLU must be contiguous")
    L, D2 = x.shape
    torch._check(
        D2 % 2 == 0, "Triton SwiGLU should have a hidden dimension divisible by 2"
    )
    D = D2 // 2

    out = torch.empty((L, D), dtype=x.dtype, device=x.device)
    def grid(META):
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (triton.cdiv(L, BLOCK_M), triton.cdiv(D2, BLOCK_N), )
    
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device=x.device, dtype=torch.int8)
    triton.set_allocator(alloc_fn)

    wrap_triton(swiglu_fwd_kernel)[grid](x, out, L, D2)

    return out

@triton_op("meshylearning::_swiglu_bwd", mutates_args=())
def _swiglu_bwd(grad_Y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    L, D = grad_Y.shape
    D2 = D * 2
    grad_x = torch.empty((L, D2), dtype=x.dtype, device=x.device)

    grad_Y = grad_Y.contiguous()

    def grid(META):
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (triton.cdiv(L, BLOCK_M), triton.cdiv(D, BLOCK_N), )

    wrap_triton(swiglu_bwd_kernel)[grid](
        grad_Y, x, grad_x, L, D2
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
# Reference implementation of SwiGLU.
def ref_swiglu(x):
    x_chunk, gate = x.chunk(2, dim=-1)
    return x_chunk * torch.nn.functional.silu(gate.to(torch.float32)).type_as(x)
    
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

if __name__ == "__main__":
    torch.manual_seed(0)
    # Create the same random tensor for both implementations
    x_data = torch.randn(1, 1024, 1024, dtype=torch.float16, device="cuda")
    x = x_data.clone().requires_grad_()
    x_ref = x_data.clone().requires_grad_()
    y_ref = ref_swiglu(x_ref)
    y_triton = swiglu(x)
    target = torch.ones_like(y_triton)
    torch.nn.functional.mse_loss(y_triton, target).backward()
    torch.nn.functional.mse_loss(y_ref, target).backward()
    print(y_triton)
    print("==========")
    print(y_ref)
    assert torch.allclose(y_triton, y_ref, atol=1e-1)
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-1)
    print("passed")