import triton.testing as tt
import torch

@tt.perf_report(
    tt.Benchmark(
        x_names=["M", "N"],
        x_vals= [2**10 + i*128 for i in range(60, 80)],
        line_arg="provider",
        line_vals=["triton", "triton_tma"],
        line_names=["Triton", "Triton TMA"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="GB/s",
        plot_name="nofuse-swiglu-performance",
        args={},
    )
)
def benchmark(M, N, provider):
    x = torch.randn((M, N*2), device='cuda', dtype=torch.float16)
    if provider == "triton":
        from triton_nofuse_swiglu import _swiglu
        ms = tt.do_bench(lambda: _swiglu(x))
    if provider == "triton_tma":
        from triton_nofuse_tma_swiglu import _swiglu
        ms = tt.do_bench(lambda: _swiglu(x))
    gbps = lambda ms: (2 * x.numel() * x.element_size()*1e-9) / (ms*1e-3)
    
    return gbps(ms)
benchmark.run(show_plots=True, print_data=True)
