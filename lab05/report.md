# Lab 5: Frameworks and automatic differentiation

## 1. Goal

Implement a scalar reverse-mode automatic-differentiation engine (`micrograd.py`), check it against
PyTorch autograd and numerical gradients, train a tiny neural network with it, and compare it with
PyTorch. Then inspect PyTorch's graph and measure how eager, traced and compiled execution and
non-contiguous tensors affect speed.

## 2. Method

- **Value node:** stores `data`, `grad`, parent nodes (`_prev`), the operation name and a local derivative rule (`_backward`).
- **Operations:** addition, multiplication, powers and ReLU, plus sigmoid and log for the BCE model. Every operation creates a new node linked to its inputs, so the graph is built during the forward pass.
- **Backward pass:** topological sort (iterative DFS), output gradient set to 1, nodes visited in reverse order, and gradients accumulated with `+=` so a node used several times gets the sum of its contributions.
- **Verification:** `((a*b + c)**2).relu().sum()` on random vectors (8 elements, float64) against PyTorch autograd and central finite differences. An extra test uses one variable several times to check accumulation.
- **Training:** `make_moons` (200 samples, noise 0.2), architecture 2 → 8 (ReLU) → 1 (sigmoid), BCE loss, full-batch SGD (lr 0.5, 60 epochs), identical initial weights and float64 in both implementations. Time is the median of 5 runs.
- **PyTorch internals:** the autograd graph is inspected through `grad_fn`. Eager, `torch.jit.trace` and `torch.compile` are timed after warm-up (forward and forward+backward, batch 256 and 65536). Non-contiguous views are timed on a 4096×4096 matrix.

## 3. Results

**Gradient check** (`results/gradient_check.json`)

| Comparison | Max absolute difference |
|---|---|
| micrograd vs PyTorch autograd | FILL (must be < 1e-6) |
| micrograd vs numerical gradient | FILL (must be < 1e-6) |
| accumulation test, micrograd vs PyTorch | FILL |

**Training** (`results/training_comparison.json`)

| Implementation | Median time of 5 runs (s) | Final loss | Accuracy |
|---|---|---|---|
| micrograd (scalar) | FILL | FILL | FILL |
| PyTorch | FILL | FILL | FILL |

micrograd is FILL times slower than PyTorch.

**Eager vs traced vs compiled** (`results/torch_timings.json`, microseconds per call)

| Batch | Mode | Forward | Forward + backward |
|---|---|---|---|
| 256 | eager | FILL | FILL |
| 256 | jit.trace | FILL | FILL |
| 256 | torch.compile | FILL or "unsupported: reason" | FILL |
| 65536 | eager | FILL | FILL |
| 65536 | jit.trace | FILL | FILL |
| 65536 | torch.compile | FILL or "unsupported: reason" | FILL |

**Contiguity** (4096×4096 float32, milliseconds per operation)

| Operation | Time (ms) |
|---|---|
| A * 2 (contiguous) | FILL |
| A.t() * 2 (transposed view) | FILL |
| A[:, ::2] * 2 (strided view) | FILL |
| A + A (both contiguous) | FILL |
| A + A.t() (mixed layouts) | FILL |
| A.t().contiguous() (copy) | FILL |

The graph, tensor shapes, dtypes, contiguity and strides are in `results/torch_inspection.txt`.

## When does compilation help?

FILL: write 3–4 sentences using your timings. Suggested points: compilation or tracing removes
Python and dispatch overhead and can fuse operations, so it helps most when a model has many small
operations or is run many times (after the one-time warm-up cost). For a tiny model on a large
batch the compute dominates, so the gain is small. If `torch.compile` was unsupported on your
machine (for example no C++ compiler on Windows), say so and use the `jit.trace` numbers.

## 4. Conclusions

1. FILL (in your own words): the engine's gradients match PyTorch and numerical estimates, which shows the local rules plus topological order and `+=` accumulation are enough to implement backpropagation.
2. FILL: scalar autograd is far slower than PyTorch because every number is a Python object with its own node and closure, while PyTorch works on whole tensors in optimized C++ kernels.
3. FILL: layout and execution mode matter for speed: non-contiguous views can make operations slower, and tracing or compilation only pays off when overhead or fusable operations dominate.