"""Native segmented XOR kernel equals the PyTorch fallback bit for bit."""
import pytest
import torch

from opguard.tracer import xor_signature as xs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or xs._xor_reduce_cuda_ext is None
    or not hasattr(xs._xor_reduce_cuda_ext, "xor_reduce_segments_cuda"),
    reason="needs CUDA and the native xor_reduce_cuda extension with segments")

CASES = [  # (shape, dtype, seg_bytes); seg_bytes None = last-dim bytes
    ((38, 32, 128), torch.bfloat16, None),          # head vectors, 256 B (warp path)
    ((38, 4096), torch.bfloat16, None),             # hidden rows, 8 KB (block path)
    ((256, 152064), torch.bfloat16, None),          # logits rows, ~297 KB
    ((1000, 2), torch.float32, None),               # 8 B segments
    ((7, 4), torch.uint8, None),                    # 4 B segments
    ((1000,), torch.uint8, 64),                     # last segment partial
    ((1001,), torch.uint8, 8),                      # nbytes not a multiple of 4
    ((3, 511), torch.float32, None),                # 2044 B, just under the block path
    ((3, 512), torch.float32, None),                # 2048 B, first block-path size
]


def u32(t):
    return [v & 0xFFFFFFFF for v in t.cpu().tolist()]


@pytest.mark.parametrize("shape,dtype,seg", CASES)
def test_native_equals_fallback(shape, dtype, seg):
    torch.manual_seed(0)
    x = (torch.randn(shape, device="cuda") * 100).to(dtype) if dtype.is_floating_point \
        else torch.randint(0, 255, shape, device="cuda", dtype=dtype)
    seg = seg or x.shape[-1] * x.element_size()
    a = u32(xs.xor_chunk_signature(x, seg, native=True))
    b = u32(xs.xor_chunk_signature(x, seg, native=False))
    c = u32(xs.xor_chunk_signature(x.cpu(), seg))
    assert a == b == c


def test_unaligned_buffer():
    base = torch.randint(0, 255, (4 * 1001 + 1,), device="cuda", dtype=torch.uint8)
    x = base[1:].view(1001, 4)                      # data pointer off by one byte
    assert x.data_ptr() % 4
    assert u32(xs.xor_chunk_signature(x, 4, native=True)) == u32(xs.xor_chunk_signature(x.cpu(), 4))


def test_segments_merge_to_regional_signature():
    x = torch.randn(64, 4096, device="cuda").to(torch.bfloat16)
    segs = u32(xs.xor_chunk_signature(x, 8192))
    whole = u32(xs.xor_signature(x, regional=4, by="words"))
    for r in range(4):                              # each region is 16 rows here
        acc = 0
        for v in segs[16 * r:16 * (r + 1)]:
            acc ^= v
        assert acc == whole[r]
