import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# ---------------------------
# 1) SIMPLE EMBEDDING
# ---------------------------
class SimpleEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.vocab = {}
        self.dim = dim
        self.emb = nn.Embedding(5000, dim)

    def forward(self, text: str) -> torch.Tensor:
        tokens = text.lower().split()
        idxs = []
        for t in tokens:
            if t not in self.vocab:
                self.vocab[t] = len(self.vocab) % 5000
            idxs.append(self.vocab[t])

        if len(idxs) == 0:
            idxs = [0]

        x = torch.tensor(idxs, dtype=torch.long)
        return self.emb(x).mean(dim=0)


# ---------------------------
# 2) SHARDS (LOW-RANK)
# ---------------------------
class Shard(nn.Module):
    def __init__(self, dim: int = 64, rank: int = 4):
        super().__init__()
        self.U = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(rank, dim) * 0.01)

    def forward(self) -> torch.Tensor:
        return self.U @ self.V


# ---------------------------
# 3) ROUTER (KEYWORD-BASED)
# ---------------------------
class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.topics = ["rap", "math", "soft", "animal"]

    def detect(self, text: str):
        t = text.lower()
        scores = {
            "rap": sum(w in t for w in ["rap", "street", "gang", "flow"]),
            "math": sum(w in t for w in ["math", "theorem", "integral", "number"]),
            "soft": sum(w in t for w in ["love", "peace", "gentle", "kind"]),
            "animal": sum(w in t for w in ["vogel", "hund", "bär", "affe", "katze", "tier"]),
        }
        s = torch.tensor([scores[k] for k in self.topics], dtype=torch.float32)
        if s.sum() == 0:
            s = torch.ones_like(s)
        s = s / s.sum()
        return s.tolist()


# ---------------------------
# 4) NEUROSHARD LAYER
# ---------------------------
class NeuroShardLayer(nn.Module):
    def __init__(self, dim: int = 64, rank: int = 4, num_shards: int = 4):
        super().__init__()
        self.dim = dim
        self.W_base = nn.Parameter(torch.eye(dim))
        self.shards = nn.ModuleList([Shard(dim, rank) for _ in range(num_shards)])

    def forward(self, x: torch.Tensor, alpha):
        # alpha: list[float] Länge = num_shards
        delta = torch.zeros_like(self.W_base)
        for i, a in enumerate(alpha):
            delta = delta + a * self.shards[i]()
        W_eff = self.W_base + delta
        return x @ W_eff


# ---------------------------
# 5) MODEL
# ---------------------------
class NeuroShardModel(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.emb = SimpleEmbedding(dim)
        self.router = Router()
        self.layer1 = NeuroShardLayer(dim)
        self.layer2 = NeuroShardLayer(dim)

    def forward(self, text: str) -> torch.Tensor:
        x = self.emb(text)                     # [dim]
        alpha = self.router.detect(text)       # [4]
        a = torch.tensor(alpha, dtype=torch.float32)

        y = self.layer1(x, a)
        y = torch.relu(y)
        y = self.layer2(y, a)
        return y


# ---------------------------
# 6) TRAINING DATA
# ---------------------------
dataset = [
    ("street gang punchline rap", [1, 0, 0, 0]),  # rap
    ("deep love everyone peace", [0, 1, 0, 0]),   # soft
    ("advanced integral theorem math", [0, 0, 1, 0]),  # math
    ("vogel hund katze bär", [0, 0, 0, 1]),       # animal
]

X_TEXTS = [t for (t, _) in dataset]
Y_TARGETS = torch.tensor([y for (_, y) in dataset], dtype=torch.float32)


# ---------------------------
# 7) TRAIN LOOP
# ---------------------------
def train_neuroshard(num_epochs: int = 200, lr: float = 0.01, dim: int = 64, save_path: str = "neuroshard_v1.pth"):
    model = NeuroShardModel(dim=dim)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print("=== Training NeuroShard v1.0 (dim=64) ===")
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, text in enumerate(X_TEXTS):
            pred = model(text)              # [dim]
            loss = loss_fn(pred[:4], Y_TARGETS[i])

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss = {total_loss:.6f}")

    print("=== Fertig! ===")
    torch.save(model.state_dict(), save_path)
    print(f"Modell gespeichert als {save_path}")


if __name__ == "__main__":
    train_neuroshard()
