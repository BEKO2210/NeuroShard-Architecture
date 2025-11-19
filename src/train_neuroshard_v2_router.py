import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# ============================
# 1) SIMPLE EMBEDDING (wie vorher, dim=64)
# ============================
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


# ============================
# 2) SHARD (LOW-RANK)
# ============================
class Shard(nn.Module):
    def __init__(self, dim: int = 64, rank: int = 4):
        super().__init__()
        self.U = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(rank, dim) * 0.01)

    def forward(self) -> torch.Tensor:
        return self.U @ self.V


# ============================
# 3) GELERNTER ROUTER
#    statt Keywords: kleines NN über Embedding
# ============================
class LearnedRouter(nn.Module):
    def __init__(self, input_dim: int = 64, num_topics: int = 4):
        super().__init__()
        self.num_topics = num_topics
        self.fc = nn.Linear(input_dim, num_topics)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [dim]
        return: alpha [num_topics], Softmax-Gewichte
        """
        logits = self.fc(x)            # [4]
        alpha = torch.softmax(logits, dim=-1)
        return alpha


# ============================
# 4) NEUROSHARD-LAYER
# ============================
class NeuroShardLayer(nn.Module):
    def __init__(self, dim: int = 64, rank: int = 4, num_shards: int = 4):
        super().__init__()
        self.dim = dim
        self.W_base = nn.Parameter(torch.eye(dim))
        self.shards = nn.ModuleList([Shard(dim, rank) for _ in range(num_shards)])

    def forward(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        x:      [dim]
        alpha:  [num_shards]
        """
        delta = torch.zeros_like(self.W_base)
        for i, a in enumerate(alpha):
            delta = delta + a * self.shards[i]()
        W_eff = self.W_base + delta
        return x @ W_eff


# ============================
# 5) GESAMTMODELL V2
# ============================
class NeuroShardModelV2(nn.Module):
    def __init__(self, dim: int = 64, num_topics: int = 4):
        super().__init__()
        self.dim = dim
        self.num_topics = num_topics
        self.emb = SimpleEmbedding(dim)
        self.router = LearnedRouter(dim, num_topics)
        self.layer1 = NeuroShardLayer(dim, rank=4, num_shards=num_topics)
        self.layer2 = NeuroShardLayer(dim, rank=4, num_shards=num_topics)

    def forward(self, text: str, return_alpha: bool = False):
        x = self.emb(text)                        # [dim]
        alpha = self.router(x)                    # [num_topics]
        y1 = self.layer1(x, alpha)
        y1_act = torch.relu(y1)
        y2 = self.layer2(y1_act, alpha)           # [dim]

        if return_alpha:
            return y2, alpha
        return y2


# ============================
# 6) DATEN (gleich wie vorher)
# ============================
dataset = [
    ("street gang punchline rap", [1, 0, 0, 0]),        # rap
    ("deep love everyone peace", [0, 1, 0, 0]),         # soft
    ("advanced integral theorem math", [0, 0, 1, 0]),   # math
    ("vogel hund katze bär", [0, 0, 0, 1]),             # animal
]

X_TEXTS = [t for (t, _) in dataset]
Y_TARGETS = torch.tensor([y for (_, y) in dataset], dtype=torch.float32)  # [4]


# ============================
# 7) TRAINING V2 – mit Topic-Loss für Router
# ============================
def train_neuroshard_v2(
    num_epochs: int = 300,
    lr: float = 0.01,
    dim: int = 64,
    save_path: str = "neuroshard_v2_router.pth",
):
    model = NeuroShardModelV2(dim=dim, num_topics=4)
    opt = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    print("=== Training NeuroShard v2.0 (dim=64, learned router) ===")

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_mse = 0.0
        total_ce = 0.0

        for i, text in enumerate(X_TEXTS):
            target_vec = Y_TARGETS[i]        # [4]
            target_idx = torch.argmax(target_vec).unsqueeze(0)  # z.B. 0,1,2,3

            pred, alpha = model(text, return_alpha=True)  # pred: [dim], alpha: [4]

            # 1) Hauptloss: erste 4 Dimensionen sollen dem One-Hot-Ziel entsprechen
            mse = mse_loss(pred[:4], target_vec)

            # 2) Router-Loss: alpha soll Topic-Index treffen
            ce = ce_loss(alpha.unsqueeze(0), target_idx)

            loss = mse + 0.1 * ce

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_mse += mse.item()
            total_ce += ce.item()

        if epoch % 30 == 0:
            print(f"Epoch {epoch:3d} | Total={total_loss:.6f} | MSE={total_mse:.6f} | CE={total_ce:.6f}")

    print("=== Fertig (v2) ===")
    torch.save(model.state_dict(), save_path)
    print(f"Modell gespeichert als {save_path}")

    # Kleine Auswertung direkt nach dem Training
    print("\n=== Kurzer Check nach Training (Router-Alpha & Outputs) ===")
    model.eval()
    with torch.no_grad():
        for i, text in enumerate(X_TEXTS):
            pred, alpha = model(text, return_alpha=True)
            print(f"\nInput: {text}")
            print(f"  Alpha: {alpha.tolist()}")
            print(f"  Output[0:4]: {pred[:4].tolist()}")


if __name__ == "__main__":
    train_neuroshard_v2()
