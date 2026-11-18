"""Vector fingerprints (xor_chunk_signature with chunk = last-dim bytes)."""
import pytest
import torch

from opguard.tracer.xor_signature import xor_chunk_signature, xor_signature

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def u32(t):
    return [v & 0xFFFFFFFF for v in t.cpu().tolist()]


def merge(chunks):
    acc = 0
    for v in chunks:
        acc ^= v
    return acc


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(38, 32, 128), (7, 33), (1,)])
def test_chunks_merge_to_whole_tensor_xor(device, shape):
    x = torch.randn(shape, dtype=torch.bfloat16, device=device)
    chunks = u32(xor_chunk_signature(x, 256))
    whole = xor_signature(x, regional=1, by="words")
    assert merge(chunks) == u32(whole)[0]
    assert len(chunks) == -(-x.numel() * x.element_size() // 256)


@pytest.mark.parametrize("device", DEVICES)
def test_single_flip_changes_exactly_one_chunk(device):
    x = torch.randn(38, 32, 128, dtype=torch.bfloat16, device=device)
    y = x.clone()
    y[12, 17, 27] = y[12, 17, 27] + 0.0078125
    a, b = u32(xor_chunk_signature(x, 256)), u32(xor_chunk_signature(y, 256))
    diff = [i for i, (p, q) in enumerate(zip(a, b)) if p != q]
    flat_byte = ((12 * 32 + 17) * 128 + 27) * 2
    assert diff == [flat_byte // 256]


@pytest.mark.parametrize("device", DEVICES)
def test_rows_survive_concat_and_batching(device):
    a = torch.randn(5, 4096, dtype=torch.bfloat16, device=device)
    b = torch.randn(3, 4096, dtype=torch.bfloat16, device=device)
    packed = torch.cat([a, b])
    ca, cb, cp = (set(u32(xor_chunk_signature(t, 256))) for t in (a, b, packed))
    assert ca | cb == cp


@pytest.mark.parametrize("device", DEVICES)
def test_head_vectors_merge_to_token_rows(device):
    q = torch.randn(38, 4096, dtype=torch.bfloat16, device=device)
    split = q.view(38, 32, 128)
    rows = u32(xor_chunk_signature(q, 4096 * 2))
    heads = u32(xor_chunk_signature(split, 128 * 2))
    assert [merge(heads[t * 32:(t + 1) * 32]) for t in range(38)] == rows
