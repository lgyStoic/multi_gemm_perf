import pytest
import torch
from torch.library import opcheck

# Import the SwiGLU op; adjust the import path as needed.
from ..triton_nofuse_swiglu import _swiglu
from ..triton_nofuse_tma_swiglu import _swiglu_tma


# Fixture to create a contiguous input tensor with shape (L, 2*hidden_dim)
@pytest.fixture
def get_input(request):
    L, hidden_dim, dtype = request.param
    # Create an input tensor of shape (L, 2 * hidden_dim)
    x = torch.randn(L, 2 * hidden_dim, dtype=dtype, device="cuda")
    x.requires_grad = True
    return x


# Define a list of parameters: (L, hidden_dim, dtype)
simple_test_params = [
    (16, 512, torch.float16),
    (16, 1536, torch.float32),
]
full_test_params = [
    (16, 512, torch.float16),
    (16, 512, torch.bfloat16),
    (16, 512, torch.float32),
    (16, 1536, torch.bfloat16),
    (16, 1536, torch.float32),
    (16, 32768, torch.bfloat16),
]


@pytest.mark.parametrize("get_input", simple_test_params, indirect=True)
def test_schema(get_input):
    # Test that the op's schema is correct.
    x = get_input
    opcheck(_swiglu, (x,), test_utils="test_schema")


@pytest.mark.parametrize("get_input", simple_test_params, indirect=True)
def test_autograd_registration(get_input):
    # Test that the op has correct autograd registration.
    x = get_input
    opcheck(_swiglu, (x,), test_utils="test_autograd_registration")


@pytest.mark.parametrize("get_input", simple_test_params, indirect=True)
def test_faketensor(get_input):
    # Test that the op works with FakeTensor inputs.
    x = get_input
    opcheck(_swiglu, (x,), test_utils="test_faketensor")


@pytest.mark.parametrize("get_input", simple_test_params, indirect=True)
def test_aot_dispatch_static(get_input):
    # Test ahead-of-time dispatch with static shapes.
    x = get_input
    opcheck(_swiglu, (x,), test_utils="test_aot_dispatch_static")


@pytest.mark.parametrize("get_input", simple_test_params, indirect=True)
def test_aot_dispatch_dynamic(get_input):
    # Test ahead-of-time dispatch with dynamic shapes.
    x = get_input
    opcheck(_swiglu, (x,), test_utils="test_aot_dispatch_dynamic")


# Reference implementation of SwiGLU.
def ref_swiglu(x):
    x_chunk, gate = x.chunk(2, dim=-1)
    return x_chunk * torch.nn.functional.silu(gate.to(torch.float32)).type_as(x)


# Tolerance settings based on dtype.
def get_tol(dtype):
    if dtype == torch.float16:
        return 1e-3, 1e-5
    elif dtype == torch.bfloat16:
        return 1e-3, 1e-3
    else:
        return 1e-5, 1e-8


def test_large_tensor():
    x = torch.randn(8 * 38912, 4608 * 2, dtype=torch.bfloat16, device="cuda")
    r = _swiglu(x)

    torch.cuda.synchronize()
    assert r.shape == (8 * 38912, 4608)


@pytest.mark.parametrize("get_input", full_test_params, indirect=True)
def test_forward_backward_reference(get_input):
    # Clone the input for separate forward/backward passes.
    x_custom = get_input
    # Ensure a separate tensor for the reference so that gradients do not interfere.
    x_ref = x_custom.clone().detach().requires_grad_()

    # Get tolerance based on dtype.
    rtol, atol = get_tol(x_custom.dtype)

    # Forward pass: compute outputs with both implementations.
    y_custom = _swiglu_tma(x_custom)
    y_ref = ref_swiglu(x_ref)

    # Check that the outputs match.
    assert torch.allclose(y_custom, y_ref, rtol=rtol, atol=atol)

    target = torch.randn_like(y_ref)

    torch.nn.functional.mse_loss(y_custom, target).backward()
    torch.nn.functional.mse_loss(y_ref, target).backward()

    # Check that the input gradients match.
    D = x_custom.size(-1) // 2
    assert torch.allclose(x_custom.grad[:, :D], x_ref.grad[:, :D], rtol=rtol, atol=atol)
    assert torch.allclose(x_custom.grad[:, D:], x_ref.grad[:, D:], rtol=rtol, atol=atol)