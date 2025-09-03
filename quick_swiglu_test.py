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
    avg_time = perf_swiglu(x, weight, iterations=iterations)
    return avg_time

def benchmark_swiglu_tma_triton(x, weight, iterations=100):
    # x: [B, in_features], weight: [in_features, 2*out_features]
    if not TRITON_SWIGLU_AVAILABLE:
        raise RuntimeError("Triton swiglu not available")
    from triton_tma_swiglu import perf_swiglu
    avg_time = perf_swiglu(x, weight, iterations=iterations)
    return avg_time

def benchmark_swiglu_torch(x, weight, iterations=100):
    # x: [B, in_features], weight: [in_features, 2*out_features]
    if not TORCH_SWIGLU_AVAILABLE:
        raise RuntimeError("Torch swiglu not available")
    # warmup
    for _ in range(10):
        torch_swiglu_ref(x, weight)
    # run
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iterations):
        torch_swiglu_ref(x, weight)
    end.record()
    torch.cuda.synchronize()
    cost = start.elapsed_time(end)
    return cost / iterations

def benchmark(x, weight, iterations=100):
    torch.set_grad_enabled(False)
    cute_time = None
    triton_time = None
    print("-"*100)
    print(f"B={x.shape[0]}, in={x.shape[1]}, out={weight.shape[1]}, dtype={x.dtype}")
    if TRITON_SWIGLU_AVAILABLE:
        triton_time = benchmark_swiglu_triton(x, weight, iterations=iterations)
        print(f"Triton normal: {triton_time:.3f} ms")
        triton_tma_time = benchmark_swiglu_tma_triton(x, weight, iterations=iterations)
        print(f"Triton persistent TMA: {triton_tma_time:.3f} ms")
    # if CUTE_AVAILABLE:
    #     cute_time = benchmark_swiglu_cute(x, weight.t(), iterations=iterations)
    #     print(f"CUTE: {cute_time:.3f} ms")
    torch_time = benchmark_swiglu_torch(x, weight, iterations=iterations)
    print(f"Torch: {torch_time:.3f} ms")

def main():
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


if __name__ == "__main__":
    main()
