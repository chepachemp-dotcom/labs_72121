"""Scalar automatic-differentiation engine (reverse mode).

Each Value stores:
  data      - the forward value
  grad      - d(output)/d(this node), accumulated during backward
  _prev     - parent nodes (the inputs of the operation that produced it)
  _op       - name of the operation (for inspection)
  _backward - the local derivative rule: pushes out.grad to the parents (with +=)
Every operation builds the computation graph during the forward pass.
"""
import math


class Value:
    __slots__ = ("data", "grad", "_prev", "_op", "_backward")

    def __init__(self, data, _prev=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_prev)
        self._op = _op
        self._backward = lambda: None

    # ---- primitive operations (each defines its local derivative rule) ----
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad          # d(a+b)/da = 1
            other.grad += out.grad         # d(a+b)/db = 1

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad   # d(ab)/da = b
            other.grad += self.data * out.grad   # d(ab)/db = a

        out._backward = _backward
        return out

    def __pow__(self, p):
        assert isinstance(p, (int, float)), "only constant exponents are supported"
        out = Value(self.data ** p, (self,), f"**{p}")

        def _backward():
            self.grad += p * (self.data ** (p - 1)) * out.grad   # d(a^p)/da = p a^(p-1)

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    # ---- extra primitives needed for the sigmoid + BCE model ----
    def sigmoid(self):
        x = self.data
        s = 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    def log(self):
        d = max(self.data, 1e-12)      # clamp for numerical safety (same idea as BCE clamping)
        out = Value(math.log(d), (self,), "log")

        def _backward():
            self.grad += out.grad / d

        out._backward = _backward
        return out

    def sum(self):
        """Sum of a single scalar is itself; use vsum() for several values."""
        return self

    # ---- derived operators (built from the primitives, so they also build the graph) ----
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else -float(other))

    def __rsub__(self, other):
        return (-self) + other

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1 if isinstance(other, Value) else 1.0 / other)

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    # ---- backward pass ----
    def backward(self):
        # 1. topological sort (iterative DFS, so deep graphs do not hit the recursion limit)
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)          # all parents are already in topo
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._prev:
                if id(p) not in visited:
                    stack.append((p, False))
        # 2. output gradient = 1
        self.grad = 1.0
        # 3. visit nodes in reverse topological order; local rules accumulate (+=) into parents
        for node in reversed(topo):
            node._backward()


def vsum(values):
    """Sum a list of Values (chained additions build the graph)."""
    it = iter(values)
    total = next(it)
    for v in it:
        total = total + v
    return total