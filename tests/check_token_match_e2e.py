"""Manual GPU check: real capture (summarize_tensor, OPGUARD_XOR_ROWS=1) of two
differently written toy stacks, then token_match.

rollout: varlen [T, H], prefill 4 prompt tokens of sequences A and B together,
then decode one token of each per step; trainer: sequence A at once, [1, S, H],
q kept head-split. Both compute row by row so they are batch-invariant.
Expected: clean -> no pivot; eps / one element / missing q_norm weight ->
q_norm on both sides, on the right token of A.

    PYTHONPATH=. python tests/check_token_match_e2e.py /tmp/out
"""
import json, math, os, shutil, sys
os.environ["OPGUARD_XOR_ROWS"] = "1"
import torch
from opguard.tracer.tensor_dump import summarize_tensor
from opguard.tracer.xor_signature import VECTOR_SCHEME, encode_xor_vectors
from opguard.trace_export.perfetto.token_match import main as token_main

torch.manual_seed(0)
dev = "cuda"
V, H, NH = 100, 256, 4
D = H // NH
bf = torch.bfloat16
E = torch.randn(V, H, device=dev, dtype=bf)
Wq, Wk, Wv, Wo = (torch.randn(H, H, device=dev, dtype=bf) / 16 for _ in range(4))
gq = torch.randn(D, device=dev, dtype=bf)
A = torch.randint(0, V, (8,), device=dev)
B = torch.randint(0, V, (8,), device=dev)
A[6] = A[2]                       # a repeated token id inside sequence A


def lin(x, W):                    # row by row: batch-invariant by construction
    return torch.stack([x[i] @ W for i in range(x.shape[0])])


def rope(x, pos):                 # x [N, NH, D]
    half = D // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=dev) / half)
    ang = pos[:, None].float() * f[None]
    c, s = ang.cos()[:, None, :], ang.sin()[:, None, :]
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1).to(bf)


def rms(x, eps, use_w=True):      # x [N, NH, D]
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * gq.float() if use_w else y).to(bf)


def attn(qn, kc, vc):             # one query row [NH, D] over its keys [L, NH, D]
    s = torch.einsum("hd,lhd->hl", qn.float(), kc.float()) / math.sqrt(D)
    return torch.einsum("hl,lhd->hd", s.softmax(-1), vc.float()).to(bf)


def fin(o):
    if isinstance(o, dict):
        if o.get("scheme") == VECTOR_SCHEME:
            return encode_xor_vectors(o)
        return {k: fin(v) for k, v in o.items()}
    if isinstance(o, list):
        return [fin(v) for v in o]
    if torch.is_tensor(o):
        return float(o.cpu().item()) if o.numel() == 1 else o.cpu().tolist()
    return o


class Log:
    def __init__(self, root, name):
        self.dir = os.path.join(root, name, "gpu_000", "ops")
        os.makedirs(self.dir, exist_ok=True)
        self.c, self.name = 0, name

    def __call__(self, fn, ins, out):
        self.c += 1
        rec = {"counter": self.c, "function": f"<{self.name}.{fn}>", "fname": f"/{self.name}/{fn}.py",
               "line_num": 10, "gpu_rank": 0,
               "inputs": {"args": [summarize_tensor(t, ignore_cpu=False) for t in ins]},
               "outputs": summarize_tensor(out, ignore_cpu=False)}
        with open(os.path.join(self.dir, f"{fn}.jsonl"), "a") as f:
            f.write(json.dumps(fin(rec)) + "\n")
        return out


def weights_phase(log, packed):
    w = torch.cat([Wq, Wk])
    log("narrow" if packed else "cat", [w] if packed else [Wq, Wk], w[:H] if packed else w)


def trainer(log, eps, poke, use_w=True):
    """Whole sequence A at once, [1, S, H]; q kept head-split [1, S, NH, D]."""
    weights_phase(log, packed=False)
    ids = A[None]
    pos = torch.arange(8, device=dev)
    x = log("embedding", [ids], E[ids])
    q = log("q_proj", [x], lin(x[0], Wq).view(1, 8, NH, D))
    k = lin(x[0], Wk).view(8, NH, D); v = lin(x[0], Wv).view(8, NH, D)
    q = log("rotary", [q, pos], rope(q[0], pos)[None])
    k = rope(k, pos)
    qn = rms(q[0], eps, use_w)
    if poke is not None:
        qn[poke, 1, 7] = (qn[poke, 1, 7].float() * 1.0078125).to(bf)
    qn = log("q_norm", [q], qn[None])
    o = torch.stack([attn(qn[0, t], k[:t + 1], v[:t + 1]) for t in range(8)])
    o = log("attention", [qn], o[None])
    log("o_proj", [o], lin(o[0].reshape(8, H), Wo)[None])


def rollout(log, eps):
    """vLLM style: varlen [T, H]; prefill 4 prompt tokens of A and B, then decode 4 steps."""
    weights_phase(log, packed=True)
    cache = {"A": ([], []), "B": ([], [])}
    steps = [[("A", p) for p in range(4)] + [("B", p) for p in range(4)]] + \
            [[("B", p), ("A", p)] for p in range(4, 8)]
    for toks in steps:
        ids = torch.stack([(A if s == "A" else B)[p] for s, p in toks])
        pos = torch.tensor([p for _, p in toks], device=dev)
        x = log("embed_tokens", [ids], E[ids])
        q = log("qkv_q", [x], lin(x, Wq))
        k = rope(lin(x, Wk).view(-1, NH, D), pos); v = lin(x, Wv).view(-1, NH, D)
        q = log("rope", [q, pos], rope(q.view(-1, NH, D), pos).reshape(-1, H))
        qn = log("rms_norm", [q], rms(q.view(-1, NH, D), eps).reshape(-1, H))
        outs = []
        for i, (s, p) in enumerate(toks):
            kc, vc = cache[s]
            kc.append(k[i]); vc.append(v[i])
            outs.append(attn(qn[i].view(NH, D), torch.stack(kc), torch.stack(vc)).reshape(H))
        o = log("paged_attn", [qn], torch.stack(outs))
        log("o_proj", [o], lin(o, Wo))


def scenario(root, name, t_eps, poke, use_w=True):
    d = os.path.join(root, name)
    shutil.rmtree(d, ignore_errors=True)
    rollout(Log(d, "rollout"), 1e-6)
    trainer(Log(d, "trainer"), t_eps, poke, use_w)
    print(f"=== {name}")
    rep = token_main([os.path.join(d, "rollout"), os.path.join(d, "trainer"), "--rank", "0"])["0"]
    return rep


if __name__ == "__main__":
    root = sys.argv[1]
    res = {"clean": scenario(root, "clean", 1e-6, None),
           "eps": scenario(root, "eps", 1e-5, None),
           "one_element": scenario(root, "one_element", 1e-6, 5),
           "missing_weight": scenario(root, "missing_weight", 1e-6, None, use_w=False)}
    sizes = {}
    for n in res:
        tot = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(root, n)) for f in fs)
        sizes[n] = tot
    json.dump({"reports": res, "trace_bytes": sizes}, open(os.path.join(root, "summary.json"), "w"), indent=1)
    print("trace bytes:", sizes)
