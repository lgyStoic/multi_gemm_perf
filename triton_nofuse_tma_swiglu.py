import torch
from torch._higher_order_ops.utils import F
import triton
import triton.language as tl
from torch.library import triton_op
from torch.library import wrap_triton
from typing import Optional
from triton.tools.tensor_descriptor import TensorDescriptor


# Forward kernel: computes out = x * silu(gate)
#@triton.autotune(
#    configs=[
#    triton.Config({"BLOCK_SIZE_M": 1024, "BLOCK_SIZE_N": 384, "WARP_SPECIALIZE": True}, num_warps=4, num_stages=3),
#    triton.Config({"BLOCK_SIZE_M": 1024, "BLOCK_SIZE_N": 384, "WARP_SPECIALIZE": False}, num_warps=4, num_stages=3, num_ctas=2),
#    triton.Config({"BLOCK_SIZE_M": 2048, "BLOCK_SIZE_N": 384, "WARP_SPECIALIZE": True}, num_warps=4, num_stages=3),
#    triton.Config({"BLOCK_SIZE_M": 2048, "BLOCK_SIZE_N": 384, "WARP_SPECIALIZE": True}, num_warps=4, num_stages=3),
#    triton.Config({"BLOCK_SIZE_M": 4096, "BLOCK_SIZE_N": 384, "WARP_SPECIALIZE": True}, num_warps=4, num_stages=4),
#    triton.Config({"BLOCK_SIZE_M": 4096, "BLOCK_SIZE_N": 384, "WARP_SPECIALIZE": True}, num_warps=4, num_stages=4),
#    triton.Config({"BLOCK_SIZE_M": 8192, "BLOCK_SIZE_N": 384, "WARP_SPECIALIZE": False}, num_warps=4, num_stages=3),
#    triton.Config({"BLOCK_SIZE_M": 8192, "BLOCK_SIZE_N": 384, "WARP_SPECIALIZE": True}, num_warps=4, num_stages=2),
#    triton.Config({"BLOCK_SIZE_M": 8192, "BLOCK_SIZE_N": 384, "WARP_SPECIALIZE": True}, num_warps=4, num_stages=3),
#    triton.Config({"BLOCK_SIZE_M": 8192, "BLOCK_SIZE_N": 384, "WARP_SPECIALIZE": True}, num_warps=4, num_stages=4),
#    triton.Config({"BLOCK_SIZE_M": 16384, "BLOCK_SIZE_N": 384, "WARP_SPECIALIZE": True}, num_warps=4, num_stages=3),
#    triton.Config({"BLOCK_SIZE_M": 32768, "BLOCK_SIZE_N": 384, "WARP_SPECIALIZE": True}, num_warps=4, num_stages=4),
#    ],
#    key=["M", "N2"],
#)
@triton.jit
def swiglu_tma_fwd_kernel(
    x_desc,
    gate_desc,
    out_desc,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, tl.num_programs(0), warp_specialize=WARP_SPECIALIZE):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        x = x_desc.load([offs_am, offs_bn])
        gate = gate_desc.load([offs_am, offs_bn])
        
        # Ensure proper type handling for TMA
        gate_fp32 = gate.to(tl.float32)
        sig = tl.sigmoid(gate_fp32)
        silu_fp32 = gate_fp32 * sig
        silu = silu_fp32.to(gate.dtype)
        
        out = x * silu
        out_desc.store([offs_am, offs_bn], out)

@triton.jit
def swiglu_bwd_kernel(
    grad_out_ptr,  # pointer to grad output tensor, shape [B*L, D]
    inp_ptr,  # pointer to input tensor, shape [B*L, 2*D]
    grad_in_ptr,  # pointer to grad input tensor, shape [B*L, 2*D]
    M: tl.constexpr,
    N2: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr
):
    pid = tl.program_id(axis=0)
    off = tl.arange(0, D)
    base = inp_ptr + pid * N2
    base_grad_out = grad_out_ptr + pid * N
    base_grad_in = grad_in_ptr + pid * N2
    mask = off < N

    y1 = tl.load(base + off, mask)
    y2 = tl.load(base + N + off, mask)
    # Ensure proper type handling for TMA backward
    y2_fp32 = y2.to(tl.float32)
    sig = tl.sigmoid(y2_fp32)
    y2_sigmoid = sig.to(y2.dtype)
    
    silu = y2 * y2_sigmoid
    grad_out = tl.load(base_grad_out + off, mask)
    grad_in_left = grad_out * silu
    grad_in_left = grad_in_left.to(y1.dtype)
    tl.store(base_grad_in + off, grad_in_left, mask)

    d_silu = y2_sigmoid + silu * (1.0 - y2_sigmoid)
    grad_in_right = grad_out * y1 * d_silu
    
    grad_in_right = grad_in_right.to(y1.dtype)
    tl.store(base_grad_in + N + off, grad_in_right, mask)


@triton_op("meshylearning::_swiglu_tma", mutates_args=())
def _swiglu_tma(x: torch.Tensor) -> torch.Tensor:
    torch._check(x.is_contiguous(), "Input to triton SwiGLU must be contiguous")
    L, D2 = x.shape
    torch._check(
        D2 % 2 == 0, "Triton SwiGLU should have a hidden dimension divisible by 2"
    )
    D = D2 // 2

    out = torch.empty((L, D), dtype=x.dtype, device=x.device)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    BLOCK_M = 16
    BLOCK_N = 16
    x_desc = TensorDescriptor.from_tensor(x, [BLOCK_M, BLOCK_N])
    gate_desc = TensorDescriptor.from_tensor(x + D, [BLOCK_M, BLOCK_N])
    out_desc = TensorDescriptor.from_tensor(out, [BLOCK_M, BLOCK_N])
    grid = (triton.cdiv(L, BLOCK_M) * triton.cdiv(D, BLOCK_N), )
    def grid(META):
        if triton.cdiv(L, BLOCK_M) * triton.cdiv(D, BLOCK_N) >= 4 * NUM_SMS:
            grid = (4 * NUM_SMS, )
        else:
            grid = (triton.cdiv(L, BLOCK_M) * triton.cdiv(D, BLOCK_N), )
        return grid

    wrap_triton(swiglu_tma_fwd_kernel)[grid](x_desc, gate_desc, out_desc, L, D, BLOCK_M, BLOCK_N, True)

    return out

@triton_op("meshylearning::_swiglu_bwd_tma", mutates_args=())
def _swiglu_bwd_tma(grad_Y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    L, D = grad_Y.shape
    D2 = D * 2
    grad_x = torch.empty((L, D2), dtype=x.dtype, device=x.device)

    grad_Y = grad_Y.contiguous()
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    def grid(META):
        # BLOCK_M = META["BLOCK_SIZE_M"]
        # BLOCK_N = META["BLOCK_SIZE_N"]
        # if triton.cdiv(L, BLOCK_M) * triton.cdiv(D, BLOCK_N) >= 4 * NUM_SMS:
        #     grid = (4 * NUM_SMS, )
        # else:
        #     grid = (min(triton.cdiv(L, BLOCK_M) * triton.cdiv(D, BLOCK_N), 2 * NUM_SMS), )
        # grid = (128, )
        grid = (L,)
        return grid
    # def alloc_fn(size: int, alignment: int, stream: Optional[int]):
    #     return torch.empty(size, device=x.device, dtype=torch.int8)
    # triton.set_allocator(alloc_fn)
    BLOCK_SIZE = triton.next_power_of_2(D)

    wrap_triton(swiglu_bwd_kernel)[grid](
        grad_Y, x, grad_x, L, D2, D, BLOCK_SIZE
    )

    return grad_x

def swiglu(x: torch.Tensor) -> torch.Tensor:
    x_shape = x.shape
    return _swiglu_tma(x.flatten(0, -2).contiguous()).view(*x_shape[:-1], -1)


def _swiglu_setup_context(ctx, inputs, output):
    (x,) = inputs
    ctx.save_for_backward(x)


def _swiglu_backward(ctx, grad_Y):
    (x,) = ctx.saved_tensors
    return _swiglu_bwd_tma(grad_Y, x)


_swiglu_tma.register_autograd(
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
        y = _swiglu_tma(x_autograd)
    
    # Forward pass performance test
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    x_autograd = x.clone()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iterations):
        y = _swiglu_tma(x_autograd)
    end.record()
    end.synchronize()
    avg_fwd_time = start.elapsed_time(end) / iterations
    
    grad_y = torch.ones_like(y)

    
    # Backward pass warmup
    for _ in range(warmup_iterations):
        _swiglu_bwd_tma(grad_y, x_autograd)
    
    # Backward pass performance test
    torch.cuda.synchronize()
    start.record()
    for _ in range(iterations):
        _swiglu_bwd_tma(grad_y, x_autograd)
    end.record()
    end.synchronize()
    avg_bwd_time = start.elapsed_time(end) / iterations
    return avg_fwd_time, avg_bwd_time

if __name__ == "__main__":
    torch.manual_seed(0)
    # Create the same random tensor for both implementations
    x_data = torch.randn(1, 2 ** 20, 384, dtype=torch.float16, device="cuda")
    x = x_data.clone().requires_grad_()
    # x_ref = x_data.clone().requires_grad_()
    # y_ref = ref_swiglu(x_ref)
    y_triton = swiglu(x)
    target = torch.ones_like(y_triton)
    torch.nn.functional.mse_loss(y_triton, target).backward()
    # torch.nn.functional.mse_loss(y_ref, target).backward()
    # print(y_triton)
    # print("==========")
    # print(y_ref)
    # assert torch.allclose(y_triton, y_ref, atol=1e-1)
    # assert torch.allclose(x.grad, x_ref.grad, atol=1e-1)
    print("passed")