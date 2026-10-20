"""Lab 3: Architectures and their costs.
Place at: ml_labs/lab03/src/lab3.py
Run from ml_labs root: python lab03/src/lab3.py
Writes to ml_labs/lab03/results/: results.json, acc_vs_params.png, acc_vs_latency.png
"""
import json, time, statistics, copy
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchinfo import summary
from thop import profile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB = Path(__file__).resolve().parent.parent   # ml_labs/lab03
RES = LAB / "results"
RES.mkdir(exist_ok=True)
DATA = LAB.parent / "data"                      # ml_labs/data (shared, gitignored)

torch.manual_seed(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 1. Data ----------
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
full = datasets.FashionMNIST(DATA, train=True, download=True, transform=tf)
test = datasets.FashionMNIST(DATA, train=False, download=True, transform=tf)
train, val = random_split(full, [55000, 5000], generator=torch.Generator().manual_seed(0))
BS = 128
train_dl = DataLoader(train, BS, shuffle=True)
val_dl = DataLoader(val, BS)
test_dl = DataLoader(test, BS)

# ---------- 2. Models ----------
def make_mlp():
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                         nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))

def make_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 7 * 7, 10))

class AttnNet(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.proj = nn.Linear(49, d)  # 16 patches of 7x7 -> 64
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=128,
                                           batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Linear(d, 10)
    def forward(self, x):
        B = x.shape[0]
        p = x.unfold(2, 7, 7).unfold(3, 7, 7)           # B,1,4,4,7,7
        p = p.contiguous().view(B, 16, 49)               # B,16,49
        h = self.enc(self.proj(p))                       # B,16,64
        return self.head(h.mean(dim=1))

models = {"MLP": make_mlp(), "CNN": make_cnn(), "Attention": AttnNet()}

# ---------- 3. Cost ----------
def latency(model, batch, runs, warmup=10):
    model = model.to(dev).eval()
    x = torch.randn(batch, 1, 28, 28, device=dev)
    ts = []
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        for _ in range(runs):
            if dev.type == "cuda": torch.cuda.synchronize()
            t = time.perf_counter()
            model(x)
            if dev.type == "cuda": torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)  # ms

# Hand-calculated FLOPs (per sample, 1 MAC = 2 FLOPs)
hand = {
    "MLP Linear 784->256": 2 * 784 * 256,
    "CNN Conv2 (16->32, 3x3, 14x14 out)": 2 * 3 * 3 * 16 * 32 * 14 * 14,
    "Attn encoder layer (seq16,d64,ff128)":
        2*16*64*3*64 + 2*16*16*64 + 2*16*16*64 + 2*16*64*64 + 2*(2*16*64*128),
}
print("Hand FLOPs:", hand)

results = {}
for name, m in models.items():
    print(f"\n===== {name} =====")
    info = summary(m, input_size=(1, 1, 28, 28), verbose=0)
    print(info)
    m_cpu = copy.deepcopy(m).cpu().eval()
    macs, _ = profile(m_cpu, inputs=(torch.randn(1, 1, 28, 28),), verbose=False)
    results[name] = {"params": info.total_params, "thop_macs": macs, "thop_flops": 2 * macs,
                     "lat_b1_ms": latency(m, 1, 100), "lat_b128_ms": latency(m, 128, 20)}
    print(results[name])

# ---------- 4. Train ----------
def evaluate(m, dl):
    m.eval(); c = n = 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            c += (m(x).argmax(1) == y).sum().item(); n += len(y)
    return c / n

for name, m in models.items():
    m.to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    for ep in range(5):
        m.train()
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); lossf(m(x), y).backward(); opt.step()
        print(f"{name} ep{ep+1} val_acc={evaluate(m, val_dl):.4f}")
    results[name]["val_acc"] = evaluate(m, val_dl)
    results[name]["test_acc"] = evaluate(m, test_dl)
    print(name, "TEST ACC", results[name]["test_acc"])

results["hand_flops"] = hand
json.dump(results, open(RES / "results.json", "w"), indent=2)

# ---------- 5. Plots ----------
names = list(models)
for key, xl, fn in [("params", "Parameters", "acc_vs_params.png"),
                    ("lat_b1_ms", "Latency, batch 1 (ms)", "acc_vs_latency.png")]:
    plt.figure(figsize=(5, 4))
    for n in names:
        plt.scatter(results[n][key], results[n]["test_acc"], s=80)
        plt.annotate(n, (results[n][key], results[n]["test_acc"]), xytext=(5, 5),
                     textcoords="offset points")
    plt.xlabel(xl); plt.ylabel("Test accuracy"); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(RES / fn, dpi=150); plt.close()
print("Done. See results.json and the PNGs.")
