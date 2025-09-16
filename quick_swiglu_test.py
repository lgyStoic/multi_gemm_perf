import torch
import time
import numpy as np

def torch_swiglu_ref(x: torch.Tensor, w: torch.Tensor):
    w1, w2 = w.chunk(2, dim=-1)
    y1 = torch.matmul(x, w1)
    y2 = torch.matmul(x, w2)
    p2 = y2 * torch.sigmoid(y2)
    y = y1 * p2
    return y

# Try to import triton_swiglu and cute_gemm if available
try:
    from triton_swiglu import apply_fused_linear_swiglu
    TRITON_SWIGLU_AVAILABLE = True
except ImportError:
    TRITON_SWIGLU_AVAILABLE = False

try:
    import sys
    if sys.version_info[:2] != (3, 12):
        CUTE_AVAILABLE = False
    else:
        import cutlass
        from cute_swiglu import run_swiglu
        CUTE_AVAILABLE = True
except ImportError:
    CUTE_AVAILABLE = False

TORCH_SWIGLU_AVAILABLE = True

def benchmark_swiglu_cute(x, weight, iterations=100):
    # x: [B, in_features], weight: [in_features, 2*out_features]
    # Use cute_gemm to do matmul, then apply swiglu
    # cute_gemm expects mnkl, dtypes, etc.
    # We'll use the run() function from cute_gemm.py
    # This is a simplification: in practice, you'd want to use the CUTE kernel for the fused op.
    # Here, we just use it for matmul, then apply swiglu in torch.
    if not CUTE_AVAILABLE:
        raise RuntimeError("CUTE not available")
    from cute_swiglu import perf_swiglu
    avg_time = perf_swiglu(x, weight, iterations=iterations)
    return avg_time

def benchmark_swiglu_triton(x, weight, iterations=100):
    # x: [B, in_features], weight: [in_features, 2*out_features]
    if not TRITON_SWIGLU_AVAILABLE:
        raise RuntimeError("Triton swiglu not available")
    from triton_swiglu import perf_swiglu
    avg_fwd_time, avg_bwd_time = perf_swiglu(x, weight, iterations=iterations)
    return avg_fwd_time, avg_bwd_time

def benchmark_swiglu_tma_triton(x, weight, iterations=100):
    # x: [B, in_features], weight: [in_features, 2*out_features]
    if not TRITON_SWIGLU_AVAILABLE:
        raise RuntimeError("Triton swiglu not available")
    from triton_tma_swiglu import perf_swiglu
    avg_fwd_time, avg_bwd_time = perf_swiglu(x, weight, iterations=iterations)
    return avg_fwd_time, avg_bwd_time

def benchmark_swiglu_torch(x, weight, iterations=100):
    # x: [B, in_features], weight: [in_features, 2*out_features]
    if not TORCH_SWIGLU_AVAILABLE:
        raise RuntimeError("Torch swiglu not available")
    # warmup
    x_autograd = x.clone().requires_grad_()
    w_autograd = weight.clone().requires_grad_()
    for _ in range(10):
        if x_autograd.grad is not None:
            x_autograd.grad.zero_()
        if w_autograd.grad is not None:
            w_autograd.grad.zero_()
        y = torch_swiglu_ref(x_autograd, w_autograd)
        y.sum().backward()
    # run
    start = torch.cuda.Event(enable_timing=True)
    mid = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    fwd_cost = []
    bwd_cost = []
    for _ in range(iterations):
        x_autograd.grad.zero_()
        w_autograd.grad.zero_()
        torch.cuda.synchronize()
        start.record()
        y = torch_swiglu_ref(x_autograd, w_autograd)
        mid.record()
        y.sum().backward()
        end.record()
        torch.cuda.synchronize()
        fwd_cost.append(start.elapsed_time(mid))
        bwd_cost.append(mid.elapsed_time(end))
    fwd_time = sum(fwd_cost) / iterations
    bwd_time = sum(bwd_cost) / iterations
    return fwd_time, bwd_time

def benchmark(x, weight, iterations=100):
    triton_time = None
    print("-"*100)
    print(f"B={x.shape[0]}, in={x.shape[1]}, out={weight.shape[1]}, dtype={x.dtype}")
    if TRITON_SWIGLU_AVAILABLE:
        triton_fwd_time, triton_bwd_time = benchmark_swiglu_triton(x, weight, iterations=iterations)
        print(f"Triton normal forward: {triton_fwd_time:.3f} ms, backward: {triton_bwd_time:.3f} ms")
        triton_tma_fwd_time, triton_tma_bwd_time = benchmark_swiglu_tma_triton(x, weight, iterations=iterations)
        print(f"Triton persistent TMA forward: {triton_tma_fwd_time:.3f} ms, backward: {triton_tma_bwd_time:.3f} ms")
    # if CUTE_AVAILABLE:
    #     cute_time = benchmark_swiglu_cute(x, weight.t(), iterations=iterations)
    #     print(f"CUTE: {cute_time:.3f} ms")
    torch_fwd_time, torch_bwd_time = benchmark_swiglu_torch(x, weight, iterations=iterations)
    print(f"Torch: {torch_fwd_time:.3f} ms, {torch_bwd_time:.3f} ms")

def profile_memory():
    B = 4096
    in_features = 4096
    out_features = 4096 * 3
    dtype = torch.float16

    x = torch.randn(B, in_features, device="cuda", dtype=dtype)
    weight = torch.randn(in_features, 2*out_features, device="cuda", dtype=dtype)

    torch.cuda.reset_peak_memory_stats()
    benchmark_swiglu_torch(x, weight, iterations=1)
    peak_memory = torch.cuda.max_memory_allocated()
    print(f"Torch swiglu peak memory: {peak_memory / 1024 / 1024:.3f} MB")

    torch.cuda.reset_peak_memory_stats()
    benchmark_swiglu_triton(x, weight, iterations=1)
    peak_memory = torch.cuda.max_memory_allocated()
    print(f"Triton swiglu peak memory: {peak_memory / 1024 / 1024:.3f} MB")

    torch.cuda.reset_peak_memory_stats()
    benchmark_swiglu_tma_triton(x, weight, iterations=1)
    peak_memory = torch.cuda.max_memory_allocated()
    print(f"Triton persistent TMA swiglu peak memory: {peak_memory / 1024 / 1024:.3f} MB")

def speed_profile():
    torch.manual_seed(0)
    B = 4096
    in_features = 4096
    out_features = 4096 * 3
    dtype = torch.float16

    x = torch.randn(B, in_features, device="cuda", dtype=dtype)
    weight = torch.randn(in_features, 2*out_features, device="cuda", dtype=dtype)

    print(f"Testing SWIGLU performance: B={B}, in={in_features}, out={out_features}, dtype={dtype}")

    # Benchmark multiple shapes
    shapes = [
        (4096, 4096, 4096 * 3),
        (2048, 4096, 4096 * 3),
        (4096, 2048, 2048 * 3),
        (1024, 1024, 1024 * 3),
        (512, 4096, 4096 * 3),
        (4096, 512, 512 * 3),
        (8192, 4096, 4096 * 3),
        (4096, 8192, 8192 * 3),
    ]
    dtype = torch.float16
    for B, in_features, out_features in shapes:
        x = torch.randn(B, in_features, device="cuda", dtype=dtype)
        weight = torch.randn(in_features, 2*out_features, device="cuda", dtype=dtype)
        benchmark(x, weight)
    
    profile_memory(x, weight)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.profile:
        profile_memory()
    else:
        speed_profile()
