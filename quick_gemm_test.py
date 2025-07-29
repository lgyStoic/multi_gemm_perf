#!/usr/bin/env python3
"""
Quick GEMM Performance Test
===========================

A simplified script for quick GEMM performance comparison.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Tuple

# Import available GEMM implementations
TRITON_AVAILABLE = False
CUTE_AVAILABLE = False

try:
    from triton_gemm import matmul as triton_matmul
    TRITON_AVAILABLE = True
    print("Triton GEMM loaded successfully")
except Exception as e:
    print(f"Triton GEMM not available: {e}")

try:
    from cute_gemm import run
    CUTE_AVAILABLE = True
    print("CUTE GEMM loaded successfully")
except Exception as e:
    print(f"CUTE GEMM not available: {e}")

def benchmark_pytorch(M, N, K, dtype: torch.dtype, iterations: int = 100) -> Dict:
    """Benchmark PyTorch GEMM"""
    print(f"Testing PyTorch {M}x{N}x{K} {dtype}")
    
    # Create matrices
    a = torch.randn(M, K, dtype=dtype, device="cuda")
    b = torch.randn(K, N, dtype=dtype, device="cuda")
    c = torch.zeros(M, N, dtype=dtype, device="cuda")
    
    # Warmup
    for _ in range(10):
        torch.mm(a, b, out=c)
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        torch.mm(a, b, out=c)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    avg_time = np.mean(times)
    tflops = (2 * M * N * K) / (avg_time / 1000) / 1e12  # TFLOPS
    
    return {
        "name": "PyTorch",
        "avg_time_ms": avg_time,
        "tflops": tflops 
    }

def benchmark_triton(M, N, K, dtype: torch.dtype, iterations: int = 100) -> Dict:
    """Benchmark Triton GEMM"""
    import sys
    if not TRITON_AVAILABLE or sys.version_info[:2] == (3, 12):
        return {"name": "Triton", "error": "Not available"}
    
    print(f"Testing Triton {M}x{N}x{K} {dtype}")
    
    # Create matrices
    a = torch.randn(M, K, dtype=dtype, device="cuda")
    b = torch.randn(K, N, dtype=dtype, device="cuda")
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    # Warmup
    for _ in range(10):
        triton_matmul(a, b, c, activation="")
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        triton_matmul(a, b, c, activation="")
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    avg_time = np.mean(times)
    tflops = (2 * M * N * K) / (avg_time / 1000) / 1e12  # TFLOPS
    
    return {
        "name": "Triton",
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
    
    # CUTE parameters
    mnkl = (M, N, K, 1)
    tile_shape_mnk = (128, 256, 64)
    cluster_shape_mn = (1, 1)
    
    # Convert dtype to CUTLASS types
    a_dtype = b_dtype = c_dtype = cutlass.Float16
    acc_dtype = cutlass.Float32
   
    
    result = run(
        mnkl=mnkl,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        a_major="k",
        b_major="k",
        c_major="n",
        tile_shape_mnk=tile_shape_mnk,
        cluster_shape_mn=cluster_shape_mn,
        tolerance=1e-3,
        warmup_iterations=10,
        iterations=iterations,
        skip_ref_check=True
    )
    
    # Convert result to our format
    return {
        "name": "CUTE",
        "avg_time_ms": result / 1000.0,  # Convert microseconds to milliseconds
        "tflops": (2 * M * N * K) / (result / 1e3) / 1e12  # TFLOPS
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
        triton_result = benchmark_triton(M, N, K, dtype, iterations)
        all_results.append(triton_result)
        
        # Test CUTE DSL
        cute_result = benchmark_cute(M, N, K, dtype, iterations)
        all_results.append(cute_result)

        # Test cute c++
        cute_cpp_result = benchmark_cute_cpp(M, N, K, dtype, iterations)
        all_results.append(cute_cpp_result)

        # Print results for this size
        print(f"\nResults for {M}x{N}x{K}:")
        print("-" * 40)
        
        results_for_size = [r for r in [pytorch_result, triton_result, cute_result, cute_cpp_result] if "error" not in r]
        results_for_size.sort(key=lambda x: x["tflops"], reverse=True)
        
        for i, result in enumerate(results_for_size):
            rank = f"{i+1}." if i < 3 else "  "
            print(f"{rank} {result['name']:10s}: {result['avg_time_ms']:8.2f} ms, "
                  f"{result['tflops']:6.2f} TFLOPS")
        
        # Print errors
        for result in [pytorch_result, triton_result, cute_result, cute_cpp_result]:
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
        (512,512, 8192)
        ]
            
    # Test with float16
    print("Testing with float16...")
    results_fp16 = run_quick_test(sizes, torch.float16, iterations=100)
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main() 