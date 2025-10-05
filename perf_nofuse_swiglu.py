import triton.testing as tt
import torch

@tt.perf_report(
    tt.Benchmark(
        x_names=["M"],
        x_vals= [2**i for i in range(15, 22)],
        line_arg="provider",
        line_vals=["triton", "triton_tma"],
        line_names=["Triton", "Triton TMA"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="GB/s",
        plot_name="nofuse-swiglu-fwd-performance",
        args={"N": 384},
    )
)
def benchmark_fwd(M, N, provider):
    x = torch.randn((M, N), device='cuda', dtype=torch.float16)
    if provider == "triton":
        from triton_nofuse_swiglu import _swiglu
        ms = tt.do_bench(lambda: _swiglu(x))
    if provider == "triton_tma":
        from triton_nofuse_tma_swiglu import _swiglu_tma
        ms = tt.do_bench(lambda: _swiglu_tma(x))
    gbps = lambda ms: (1.5 * x.numel() * x.element_size()*1e-9) / (ms*1e-3)
    
    return gbps(ms)
benchmark_fwd.run(show_plots=True, print_data=True)


@tt.perf_report(
    tt.Benchmark(
        x_names=["M"],
        x_vals= [2**i for i in range(15, 20)],
        line_arg="provider",
        line_vals=["triton", "triton_tma"],
        line_names=["Triton", "Triton TMA"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="GB/s",
        plot_name="nofuse-swiglu-bwd-performance",
        args={"N": 768 * 4},
    )
)
def benchmark_bwd(M, N, provider):
    x = torch.randn((M, N), device='cuda', dtype=torch.float16)
    y_target = torch.randn((M, N // 2), device='cuda', dtype=torch.float16)
    from triton_nofuse_swiglu import _swiglu
    diff = _swiglu(x) - y_target
    if provider == "triton":
        from triton_nofuse_swiglu import _swiglu_bwd
        ms = tt.do_bench(lambda: _swiglu_bwd(diff, x))
    if provider == "triton_tma":
        from triton_nofuse_tma_swiglu import _swiglu_bwd_tma
        ms = tt.do_bench(lambda: _swiglu_bwd_tma(diff, x))
    gbps = lambda ms: (2.5 * x.numel() * x.element_size()*1e-9) / (ms*1e-3)
    return gbps(ms)
benchmark_bwd.run(show_plots=True, print_data=True)