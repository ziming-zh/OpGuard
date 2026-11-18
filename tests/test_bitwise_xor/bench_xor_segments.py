"""Time and extra GPU memory of per-vector XOR: native kernel vs PyTorch fallback,
next to the existing whole-tensor (4-region) kernel. Manual: python bench_xor_segments.py"""
import torch

from opguard.tracer import xor_signature as xs

SHAPES = [((8192, 4096), "hidden rows, 8 KB"),
          ((8192, 32, 128), "head vectors, 256 B"),
          ((8192, 11008), "MLP rows, 21.5 KB"),
          ((2048, 152064), "logits rows, 297 KB")]


def measure(fn, iters=20):
    fn(); torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn(); torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - base
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters, extra


if __name__ == "__main__":
    print(f"{'tensor':<34}{'MB':>8}{'whole ms':>10}{'native ms':>11}{'torch ms':>10}"
          f"{'native GB/s':>13}{'native +MB':>12}{'torch +MB':>11}")
    for shape, label in SHAPES:
        x = torch.randn(shape, device="cuda").to(torch.bfloat16)
        seg = x.shape[-1] * x.element_size()
        mb = x.numel() * 2 / 2**20
        tw, _ = measure(lambda: xs.xor_signature(x, regional=4, by="words"))
        tn, mn = measure(lambda: xs.xor_chunk_signature(x, seg, native=True))
        tt, mt = measure(lambda: xs.xor_chunk_signature(x, seg, native=False))
        print(f"{label + ' ' + str(list(shape)):<34}{mb:>8.0f}{tw:>10.3f}{tn:>11.3f}{tt:>10.3f}"
              f"{mb / 1024 / (tn / 1000):>13.0f}{mn / 2**20:>12.1f}{mt / 2**20:>11.1f}")
        del x
        torch.cuda.empty_cache()
