"""A dtype/layout tail wrapping another call is not chosen as the line's boundary."""
import os
import sys
import tempfile
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OPGUARD_MULTI_OPS_PER_LINE", "0")

from opguard.callsite.callsite_parser import CallsiteParser  # noqa: E402

SOURCE = textwrap.dedent("""\
    def forward(layer, q, x, y):
        q = layer.attn.q_norm(kernel_input(q)).float()
        a = x.float()
        b = layer.proj(x).to(y.dtype).contiguous()
        c = layer.mlp(x)
        d = x.float() + y.half()
        return q, a, b, c, d
""")


def top(line):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(SOURCE)
    try:
        return CallsiteParser().analyze_line_multi(f.name, line)[0].function
    finally:
        os.unlink(f.name)


def test_tail_float_yields_to_the_operator():
    assert top(2) == "q_norm"


def test_lone_conversion_is_still_chosen():
    assert top(3) == "float"


def test_chained_tails_yield_to_the_operator():
    assert top(4) == "proj"


def test_ordinary_line_unchanged():
    assert top(5) == "mlp"


def test_conversions_of_plain_names_are_not_tails():
    # Neither receiver contains a call, so ranking is as before.
    assert top(6) in ("float", "half")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
