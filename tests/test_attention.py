import pytest
import torch
from long_net import DilatedAttention


@pytest.fixture
def dilated_attention():
    return DilatedAttention(
        dim=512,
        heads=8,
        dilation_rate=2,
        segment_size=64,
        dropout=0.1,
        causal=False,
        use_xpos=True,
        use_rel_pos_bias=True,
        qk_norm=False,
        dtype=torch.float32,
        device="cpu",
    )


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 10, 512),
        (2, 20, 512),
        (5, 50, 512),
        (10, 100, 512),
        (20, 200, 512),
    ],
)
def test_forward_shape(dilated_attention, input_shape):
    x = torch.rand(input_shape)
    output = dilated_attention(x)
    assert (
        output.shape == input_shape
    ), f"Expected output shape {input_shape}, but got {output.shape}"


@pytest.mark.parametrize(
    "input_val",
    [
        torch.zeros((1, 10, 512)),
        torch.ones((1, 10, 512)),
        torch.full((1, 10, 512), fill_value=0.5),
    ],
)
def test_forward_values(dilated_attention, input_val):
    output = dilated_attention(input_val)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float64])
def test_forward_dtype(dilated_attention, input_dtype):
    x = torch.rand((1, 10, 512), dtype=input_dtype)
    output = dilated_attention(x)
    assert (
        output.dtype == input_dtype
    ), f"Expected output dtype {input_dtype}, but got {output.dtype}"


# Add more tests as needed
@pytest.mark.parametrize("causal", [True, False])
def test_forward_causal(dilated_attention, causal):
    dilated_attention.causal = causal
    x = torch.rand((1, 10, 512))
    output = dilated_attention(x)
    assert output is not None, "Output is None"


@pytest.mark.parametrize("use_xpos", [True, False])
def test_forward_use_xpos(dilated_attention, use_xpos):
    dilated_attention.use_xpos = use_xpos
    x = torch.rand((1, 10, 512))
    output = dilated_attention(x)
    assert output is not None, "Output is None"


@pytest.mark.parametrize("use_rel_pos_bias", [True, False])
def test_forward_use_rel_pos_bias(dilated_attention, use_rel_pos_bias):
    dilated_attention.use_rel_pos_bias = use_rel_pos_bias
    x = torch.rand((1, 10, 512))
    output = dilated_attention(x)
    assert output is not None, "Output is None"


@pytest.mark.parametrize("qk_norm", [True, False])
def test_forward_qk_norm(dilated_attention, qk_norm):
    dilated_attention.qk_norm = qk_norm
    x = torch.rand((1, 10, 512))
    output = dilated_attention(x)
    assert output is not None, "Output is None"


def test_forward_with_mask(dilated_attention):
    x = torch.rand((1, 10, 512))
    mask = torch.ones((1, 10, 10))
    output = dilated_attention(x, mask=mask)
    assert output is not None, "Output is None"
