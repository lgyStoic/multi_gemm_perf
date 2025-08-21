#!/usr/bin/env python3
"""
Quick GEMM Performance Test
===========================

A simplified script for quick GEMM performance comparison.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple

# Check availability
TRITON_AVAILABLE = False
try:
    from triton_gemm import matmul as triton_matmul
    from hopper_tma_gemm import matmul_tma_hopper as triton_matmul_tma_hopper
    from hopper_tma_gemm import matmul_tma_hopper_persistent as triton_matmul_tma_hopper_persistent
    TRITON_AVAILABLE = True
except ImportError:
    pass

CUTE_AVAILABLE = False
try:
    from cute_gemm import run_gemm
    CUTE_AVAILABLE = True
except ImportError:
    pass

def create_matrices(M: int, N: int, K: int, dtype: torch.dtype, device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create matrices for GEMM operations with consistent initialization.
    
    Args:
        M: Number of rows in matrix A and result matrix C
        N: Number of columns in matrix B and result matrix C  
        K: Number of columns in matrix A and rows in matrix B
        dtype: Data type for all matrices
        device: Device to create tensors on
        
    Returns:
        Tuple of (A, B, C) where:
        - A: Matrix of shape (M, K)
        - B: Matrix of shape (K, N) 
        - C: Output matrix of shape (M, N), initialized to zeros
    """
    # Set random seed for reproducible results
    torch.manual_seed(42)
    
    # Create matrices
    a = torch.randn(M, K, dtype=dtype, device=device)
    b = torch.randn(K, N, dtype=dtype, device=device)
    c = torch.zeros(M, N, dtype=dtype, device=device)
    
    return a, b, c

def calculate_tflops(M: int, N: int, K: int, time_ms: float) -> float:
    """
    Calculate TFLOPS for GEMM operation.
    
    Args:
        M, N, K: Matrix dimensions
        time_ms: Execution time in milliseconds
        
    Returns:
        TFLOPS value
    """
    flops = 2 * M * N * K  # Each output element requires K multiplications and K-1 additions
    time_seconds = time_ms / 1000
    return flops / time_seconds / 1e12

def benchmark_pytorch(M, N, K, dtype: torch.dtype, iterations: int = 100) -> Dict:
    """Benchmark PyTorch GEMM"""
    print(f"Testing PyTorch {M}x{N}x{K} {dtype}")
    
    # Create matrices using unified function
    a, b, c = create_matrices(M, N, K, dtype)
    
    # Warmup
    for _ in range(10):
        torch.mm(a, b, out=c)
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        torch.mm(a, b, out=c)
    end.record()
    torch.cuda.synchronize()
    cost = start.elapsed_time(end)
    avg_time = cost / iterations
    tflops = calculate_tflops(M, N, K, avg_time)
    
    return {
        "name": "PyTorch",
        "avg_time_ms": avg_time,
        "tflops": tflops 
    }

def benchmark_triton(M, N, K, dtype: torch.dtype, 
        warp_specialize: bool = False, 
        using_tma: bool = False, 
        using_persistent: bool = False,
        iterations: int = 100) -> Dict:
    """Benchmark Triton GEMM"""
    import sys
    if not TRITON_AVAILABLE:
        return {"name": "Triton", "error": "Not available"}
    
    print(f"Testing Triton {M}x{N}x{K} {dtype}")
    
    # Create matrices using unified function
    a, b, c = create_matrices(M, N, K, dtype)
    from functools import partial
    from hopper_tma_gemm import is_hopper

    if is_hopper():
        if using_persistent:
            matmul = partial(triton_matmul_tma_hopper_persistent, a, b, c, warp_specialize=warp_specialize)
        elif using_tma:
            matmul = partial(triton_matmul_tma_hopper, a, b, c, warp_specialize=warp_specialize)
        else:
            matmul = partial(triton_matmul, a, b, c)
    else:
        matmul = partial(triton_matmul, a, b, c, activation="")
    
    # Warmup
    for _ in range(10):
        matmul()
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Benchmark
    start.record()  
    for _ in range(iterations):
        matmul()
    
    end.record()
    torch.cuda.synchronize()
    cost = start.elapsed_time(end)
    avg_time = cost / iterations
    tflops = calculate_tflops(M, N, K, avg_time)
    
    base_name = "Triton" if not using_tma else "Triton TMA"
    full_name = f"{base_name} {'persistent' if using_persistent else ''}"
    return {
        "name": full_name,
        "avg_time_ms": avg_time,
        "tflops": tflops
    }

def benchmark_cute(M, N, K, dtype: torch.dtype, iterations: int = 100) -> Dict:
    """Benchmark CUTE GEMM"""
    if not CUTE_AVAILABLE:
        return {"name": "CUTE", "error": "Not available"}
    
    print(f"Testing CUTE {M}x{N}x{K} {dtype}")
    
    # Import cutlass for proper dtype handling
    try:
        import cutlass
    except ImportError:
        return {"name": "CUTE", "error": "CUTLASS not available"}
    
    # Create matrices using unified function
    a, b, c = create_matrices(M, N, K, dtype)

    result = run_gemm(a, b.t(), c, iterations=iterations, warmup_iterations=10, dtype=dtype, check_ref=False)
   
    # Convert result to our format
    return {
        "name": "CUTE",
        "avg_time_ms": result,  # Convert microseconds to milliseconds
        "tflops": calculate_tflops(M, N, K, result)  # TFLOPS
    }

def benchmark_cute_cpp(M, N, K, dtype: torch.dtype, iterations: int = 100) -> Dict:
    """Benchmark CUTE C++ GEMM"""
    if not CUTE_AVAILABLE:
        return {"name": "CUTE", "error": "Not available"}
    import subprocess

    # Call the CUTE C++ executable with M, N, K as command-line arguments
    exe = "./cute_tutorial_wgmma_tma_sm90"
    try:
        result = subprocess.run(
            [exe, str(M), str(N), str(K)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        # Try to parse output for timing and tflops
        avg_time_ms = None
        tflops = None
        for line in result.stdout.splitlines():
            if "CUTE_GEMM" in line:
                # CUTE_GEMM:     [15006.2]GFlop/s  (0.0179)ms
                import re
                m = re.search(r"CUTE_GEMM:\s*\[([0-9.]+)\]\s*GFlop/s\s*\(([0-9.]+)\)ms", line)
                if m:
                    tflops = float(m.group(1)) / 1000.0
                    avg_time_ms = float(m.group(2))
                    break
        if avg_time_ms is not None and tflops is not None:
            print(f"CUTE_CPP: {M}x{N}x{K} {dtype} {avg_time_ms}ms {tflops} TFLOPS")
            return {
                "name": "CUTE_CPP",
                "avg_time_ms": avg_time_ms,
                "tflops": tflops
            }
        else:
            return {
                "name": "CUTE_CPP",
                "error": "Could not parse output",
                "stdout": result.stdout,
                "stderr": result.stderr
            }
    except FileNotFoundError:
        print(f"Executable '{exe}' not found")
        return {"name": "CUTE_CPP", "error": f"Executable '{exe}' not found"}
    except subprocess.CalledProcessError as e:
        print(f"Execution failed: {e}")
        return {
            "name": "CUTE_CPP",
            "error": "Execution failed",
            "stdout": e.stdout,
            "stderr": e.stderr
        }
    
    print(f"Testing CUTE C++ {M}x{N}x{K} {dtype}")
    

def run_quick_test(sizes: List[int], dtype: torch.dtype = torch.float16, iterations: int = 50):
    """Run quick GEMM performance test"""
    print("=" * 60)
    print("QUICK GEMM PERFORMANCE TEST")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Data type: {dtype}")
    print(f"Iterations: {iterations}")
    print()
    
    all_results = []
    
    for (M, N, K) in sizes:
        print(f"\n--- Matrix Size: {M}x{N}x{K} ---")
        
        # Test PyTorch
        pytorch_result = benchmark_pytorch(M, N, K, dtype, iterations)
        all_results.append(pytorch_result)
        
        # Test Triton
        triton_result = benchmark_triton(M, N, K, dtype, iterations=iterations)
        all_results.append(triton_result)
        
        # Test Triton TMA
        triton_tma_result = benchmark_triton(M, N, K, dtype, warp_specialize=False, using_tma=True, iterations=iterations)
        all_results.append(triton_tma_result)

        triton_tma_persistent_result = benchmark_triton(M, N, K, dtype, 
                warp_specialize=False, 
                using_tma=True, 
                using_persistent=True, 
                iterations=iterations)
        all_results.append(triton_tma_persistent_result)

        # Test CUTE DSL
        cute_result = benchmark_cute(M, N, K, dtype, iterations)
        all_results.append(cute_result)

        # Test cute c++
        cute_cpp_result = benchmark_cute_cpp(M, N, K, dtype, iterations)
        all_results.append(cute_cpp_result)

        # Print results for this size
        print(f"\nResults for {M}x{N}x{K}:")
        print("-" * 40)
        
        results_for_size = [r for r in [pytorch_result, triton_result, triton_tma_result, triton_tma_persistent_result, cute_result, cute_cpp_result] if "error" not in r]
        results_for_size.sort(key=lambda x: x["tflops"], reverse=True)
        
        for i, result in enumerate(results_for_size):
            rank = f"{i+1}." if i < 3 else "  "
            print(f"{rank} {result['name']:10s}: {result['avg_time_ms']:8.2f} ms, "
                  f"{result['tflops']:6.2f} TFLOPS")
        
        # Print errors
        for result in [pytorch_result, triton_result, triton_tma_result, triton_tma_persistent_result, cute_result, cute_cpp_result]:
            if "error" in result:
                print(f"  {result['name']:10s}: {result['error']}")
    
    return all_results

def main():
    """Main function"""
    # Test sizes
    sizes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (1536, 1536, 1536),
        (2048, 2048, 2048),
        (3072, 3072, 3072),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        (16384, 16384, 16384),
        (384, 384, 3072),
        (64,64, 5120),
        (64,64,8192),
        (256,256, 5120),
        (256,256, 8192),
        (512,512, 5120),
        (512,512, 8192),
        (3968000, 64, 192),
        (1280000, 64, 384),
        (4096, 4096, 4096 * 3)
        ]
            
    # Test with float16
    print("Testing with float16...")
    results_float16 = run_quick_test(sizes, torch.float16, iterations=100)
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main() 