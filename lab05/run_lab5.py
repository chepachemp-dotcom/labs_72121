#!/usr/bin/env python
"""Lab 5: frameworks and automatic differentiation.  Run with:  python run_lab5.py"""
import json
import platform
import statistics
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.datasets import make_moons

from micrograd import Value, vsum

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
warnings.filterwarnings("ignore")

SEED = 42
N_SAMPLES, NOISE = 200, 0.2
EPOCHS, LR, HIDDEN, RUNS = 60, 0.5, 8, 5


def save_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")


# ======================================================== tasks 1-4: gradient check
def micrograd_grads(a, b, c):
    A, B, C = ([Value(v) for v in arr] for arr in (a, b, c))
    # ((a*b + c)**2).relu().sum()  -- elementwise, then summed over the vector
    out = vsum([((x * y + z) ** 2).relu().sum() for x, y, z in zip(A, B, C)])
    out.backward()
    return out.data, np.array([v.grad for v in A]), np.array([v.grad for v in B]), np.array([v.grad for v in C])


def torch_grads(a, b, c):
    import torch
    t = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (a, b, c)]
    out = ((t[0] * t[1] + t[2]) ** 2).relu().sum()
    out.backward()
    return out.item(), *(x.grad.numpy() for x in t)


def numeric_grads(a, b, c, h=1e-5):
    def f(a, b, c):
        return np.maximum((a * b + c) ** 2, 0).sum()

    args, grads = [a.copy(), b.copy(), c.copy()], []
    for k in range(3):
        g = np.zeros_like(args[k])
        for i in range(len(g)):
            hi, lo = [x.copy() for x in args], [x.copy() for x in args]
            hi[k][i] += h
            lo[k][i] -= h
            g[i] = (f(*hi) - f(*lo)) / (2 * h)
        grads.append(g)
    return f(*args), *grads


def task_gradient_check():
    rng = np.random.default_rng(SEED)
    a, b, c = (rng.uniform(-2, 2, 8) for _ in range(3))
    m, t, n = micrograd_grads(a, b, c), None, numeric_grads(a, b, c)
    try:
        t = torch_grads(a, b, c)
    except ImportError:
        print("torch not installed: skipping PyTorch comparison")

    def maxdiff(p, q):
        return float(max(np.max(np.abs(p[i] - q[i])) for i in (1, 2, 3)))

    res = {"expression": "((a*b + c)**2).relu().sum()", "forward_value": {"micrograd": m[0], "numeric": n[0]},
           "max_abs_diff_micrograd_vs_numeric": maxdiff(m, n)}
    if t is not None:
        res["forward_value"]["torch"] = t[0]
        res["max_abs_diff_micrograd_vs_torch"] = maxdiff(m, t)
        res["max_abs_diff_torch_vs_numeric"] = maxdiff(t, n)
        res["passes_1e-6"] = res["max_abs_diff_micrograd_vs_torch"] < 1e-6 and res["max_abs_diff_micrograd_vs_numeric"] < 1e-6
    else:
        res["passes_1e-6"] = res["max_abs_diff_micrograd_vs_numeric"] < 1e-6

    # accumulation test: the same node is used several times, so gradients must be added, not overwritten
    x, y = Value(1.5), Value(-0.7)
    z = ((x * x + x * y + x) ** 3).relu()
    z.backward()
    acc = {"micrograd_dx": x.grad, "micrograd_dy": y.grad}
    try:
        import torch
        tx, ty = torch.tensor(1.5, dtype=torch.float64, requires_grad=True), torch.tensor(-0.7, dtype=torch.float64, requires_grad=True)
        ((tx * tx + tx * ty + tx) ** 3).relu().backward()
        acc.update(torch_dx=tx.grad.item(), torch_dy=ty.grad.item(),
                   max_abs_diff=max(abs(x.grad - tx.grad.item()), abs(y.grad - ty.grad.item())))
    except ImportError:
        pass
    res["accumulation_test"] = acc
    save_json("gradient_check.json", res)
    print("Gradient check:", json.dumps(res, indent=2))
    return res


# ======================================================== task 5: training comparison
def init_params(seed):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 0.7, (HIDDEN, 2)), np.full(HIDDEN, 0.1), rng.normal(0, 0.5, (1, HIDDEN)), np.zeros(1))


def np_forward(X, W1, b1, W2, b2):
    h = np.maximum(X @ W1.T + b1, 0)
    return 1 / (1 + np.exp(-(h @ W2.T + b2)[:, 0]))


def np_metrics(X, y, params):
    p = np.clip(np_forward(X, *params), 1e-12, 1 - 1e-12)
    loss = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    return loss, float(((p > 0.5) == y).mean())


def train_micrograd(X, y, params):
    W1, b1, W2, b2 = params
    pW1 = [[Value(v) for v in row] for row in W1]
    pb1 = [Value(v) for v in b1]
    pW2 = [Value(v) for v in W2[0]]
    pb2 = Value(b2[0])
    flat = [v for row in pW1 for v in row] + pb1 + pW2 + [pb2]
    Xl, yl = X.tolist(), y.tolist()
    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        losses = []
        for (x0, x1), t in zip(Xl, yl):
            h = [vsum([pW1[j][0] * x0, pW1[j][1] * x1, pb1[j]]).relu() for j in range(HIDDEN)]
            p = (vsum([pW2[j] * h[j] for j in range(HIDDEN)]) + pb2).sigmoid()
            losses.append(-(t * p.log() + (1 - t) * (1 - p).log()))
        loss = vsum(losses) * (1.0 / len(Xl))
        for v in flat:
            v.grad = 0.0
        loss.backward()
        for v in flat:
            v.data -= LR * v.grad
    elapsed = time.perf_counter() - t0
    final = (np.array([[v.data for v in r] for r in pW1]), np.array([v.data for v in pb1]),
             np.array([[v.data for v in pW2]]), np.array([pb2.data]))
    return elapsed, final


def train_torch(X, y, params):
    import torch
    import torch.nn.functional as F
    Xt, yt = torch.tensor(X), torch.tensor(y)
    ps = [torch.tensor(p, requires_grad=True) for p in params]
    W1, b1, W2, b2 = ps
    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        h = torch.relu(Xt @ W1.T + b1)
        p = torch.sigmoid(h @ W2.T + b2)[:, 0]
        loss = F.binary_cross_entropy(p, yt)
        for q in ps:
            q.grad = None
        loss.backward()
        with torch.no_grad():
            for q in ps:
                q -= LR * q.grad
    elapsed = time.perf_counter() - t0
    return elapsed, tuple(q.detach().numpy() for q in ps)


def task_training():
    X, y = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=SEED)
    y = y.astype(float)
    params = init_params(SEED)
    res = {"setup": {"samples": N_SAMPLES, "noise": NOISE, "epochs": EPOCHS, "lr": LR,
                     "architecture": "2 -> 8 (ReLU) -> 1 (sigmoid), BCE loss, full-batch SGD, float64", "runs": RUNS}}
    t_mg, last = [], None
    for _ in range(RUNS):
        t, last = train_micrograd(X, y, params)
        t_mg.append(t)
    loss, acc = np_metrics(X, y, last)
    res["micrograd"] = {"times_s": t_mg, "median_s": statistics.median(t_mg), "final_loss": loss, "accuracy": acc}
    print("micrograd:", res["micrograd"])
    try:
        t_pt, lastp = [], None
        for _ in range(RUNS):
            t, lastp = train_torch(X, y, params)
            t_pt.append(t)
        loss, acc = np_metrics(X, y, lastp)
        res["pytorch"] = {"times_s": t_pt, "median_s": statistics.median(t_pt), "final_loss": loss, "accuracy": acc}
        res["slowdown_micrograd_vs_pytorch"] = res["micrograd"]["median_s"] / res["pytorch"]["median_s"]
        res["max_abs_param_diff_after_training"] = float(max(np.max(np.abs(a - b)) for a, b in zip(last, lastp)))
        print("pytorch:", res["pytorch"])
    except ImportError:
        print("torch not installed: skipping PyTorch training")
    save_json("training_comparison.json", res)
    return res


# ======================================================== task 6: PyTorch internals
def bench(fn, iters, repeats=5, warmup=20):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        times.append((time.perf_counter() - t0) / iters)
    return statistics.median(times) * 1e6  # microseconds per call


def describe(name, t):
    return f"{name}: shape={tuple(t.shape)} dtype={t.dtype} contiguous={t.is_contiguous()} stride={t.stride()}"


def walk_graph(fn, lines, seen, depth=0):
    if fn is None or fn in seen:
        return
    seen.add(fn)
    lines.append("  " * depth + type(fn).__name__)
    for nxt, _ in fn.next_functions:
        walk_graph(nxt, lines, seen, depth + 1)


def task_torch():
    import torch
    from torch import nn
    torch.manual_seed(SEED)
    report, res = [], {"torch_version": torch.__version__}
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())

    # ---- graph and tensor inspection
    x = torch.randn(4, 2)
    out = model(x)
    report.append("== Autograd graph (walking grad_fn from the output) ==")
    walk_graph(out.sum().grad_fn, report, set())
    report.append("\n== Tensor properties ==")
    for name, t in [("input x", x), ("weight 0", model[0].weight), ("bias 0", model[0].bias),
                    ("weight 2", model[2].weight), ("output", out), ("x.t() view", x.t())]:
        report.append(describe(name, t))
    try:
        traced = torch.jit.trace(model, x)
        report.append("\n== torch.jit.trace graph ==\n" + str(traced.graph))
    except Exception as e:  # noqa: BLE001
        report.append(f"\njit.trace failed: {e}")
    (OUT / "torch_inspection.txt").write_text("\n".join(report), encoding="utf-8")

    # ---- eager vs traced vs compiled (after warm-up)
    timings = {}
    for batch, iters in [(256, 500), (65536, 100)]:
        xb = torch.randn(batch, 2)
        row = {}
        variants = {"eager": model}
        try:
            variants["jit.trace"] = torch.jit.trace(model, xb)
        except Exception as e:  # noqa: BLE001
            row["jit.trace"] = f"unsupported: {type(e).__name__}"
        try:
            comp = torch.compile(model)
            for _ in range(3):  # compilation happens on the first call(s)
                comp(xb)
            variants["torch.compile"] = comp
        except Exception as e:  # noqa: BLE001
            row["torch.compile"] = f"unsupported: {type(e).__name__}: {str(e).splitlines()[0][:120]}"
        for name, m in variants.items():
            try:
                with torch.no_grad():
                    fwd = bench(lambda: m(xb), iters)

                def step():
                    m.zero_grad()
                    m(xb).sum().backward()

                bwd = bench(step, iters)
                row[name] = {"forward_us": fwd, "forward_backward_us": bwd}
            except Exception as e:  # noqa: BLE001
                row[name] = f"failed: {type(e).__name__}: {str(e).splitlines()[0][:120]}"
        timings[f"batch_{batch}"] = row
    res["eager_vs_compiled"] = timings

    # ---- contiguous vs non-contiguous tensors
    A = torch.randn(4096, 4096)
    At, As = A.t(), A[:, ::2]
    cases = {
        "A * 2 (contiguous)": lambda: A * 2,
        "A.t() * 2 (transposed view)": lambda: At * 2,
        "A[:, ::2] * 2 (strided view)": lambda: As * 2,
        "A + A (both contiguous)": lambda: A + A,
        "A + A.t() (mixed layouts)": lambda: A + At,
        "A.t().contiguous() (copy cost)": lambda: At.contiguous(),
    }
    contig = {name: bench(fn, iters=5, repeats=5, warmup=2) / 1e3 for name, fn in cases.items()}  # ms
    res["contiguity_ms"] = contig
    res["tensor_properties"] = {"A": describe("A", A), "A.t()": describe("A.t()", At), "A[:, ::2]": describe("A[:, ::2]", As)}
    save_json("torch_timings.json", res)
    print("Torch results:", json.dumps(res, indent=2))
    return res


def main():
    meta = {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__}
    save_json("environment.json", meta)
    print("== Tasks 1-4: autograd engine vs PyTorch and numerical gradients ==")
    task_gradient_check()
    print("\n== Task 5: training micrograd vs PyTorch ==")
    task_training()
    print("\n== Task 6: PyTorch graph, eager vs compiled, contiguity ==")
    try:
        task_torch()
    except ImportError:
        print("torch not installed: run  python -m pip install torch")
    print("\nDone. Outputs in", OUT)


if __name__ == "__main__":
    main()