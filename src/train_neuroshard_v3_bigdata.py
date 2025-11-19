import torch
import torch.nn as nn
import torch.optim as optim

from train_neuroshard_v2_router import NeuroShardModelV2

torch.manual_seed(42)

# ============================
# 1) Mehr Trainingsdaten pro Topic
# ============================

dataset = [
    # RAP
    ("street gang punchline rap",            [1, 0, 0, 0]),
    ("hard street flow on the west side",    [1, 0, 0, 0]),
    ("underground real gangsta rap",         [1, 0, 0, 0]),
    ("heavy bars and punchlines in rap",     [1, 0, 0, 0]),
    ("west coast street cypher flow",        [1, 0, 0, 0]),

    # SOFT / LOVE
    ("deep love everyone peace",             [0, 1, 0, 0]),
    ("pure love and peace for the world",    [0, 1, 0, 0]),
    ("soft words and kind hearts",           [0, 1, 0, 0]),
    ("peaceful vibes and gentle souls",      [0, 1, 0, 0]),
    ("love and harmony for all people",      [0, 1, 0, 0]),

    # MATH
    ("advanced integral theorem math",       [0, 0, 1, 0]),
    ("we analyze the integral of this function in higher math", [0, 0, 1, 0]),
    ("number theory and algebraic structures", [0, 0, 1, 0]),
    ("we prove a theorem in linear algebra", [0, 0, 1, 0]),
    ("calculus and differential equations",  [0, 0, 1, 0]),

    # ANIMAL
    ("vogel hund katze bär",                 [0, 0, 0, 1]),
    ("ich sehe einen vogel und einen hund im tierheim", [0, 0, 0, 1]),
    ("affe im zoo und viele tiere",          [0, 0, 0, 1]),
    ("katze und hund schlafen auf dem sofa", [0, 0, 0, 1]),
    ("bär und löwe sind wilde tiere",        [0, 0, 0, 1]),
]

X_TEXTS = [t for (t, _) in dataset]
Y_TARGETS = torch.tensor([y for (_, y) in dataset], dtype=torch.float32)


# ============================
# 2) Training mit gleichem Model wie v2 (gelernter Router)
# ============================

def train_neuroshard_v3(
    num_epochs: int = 400,
    lr: float = 0.001,  # ↓ sanfter
    dim: int = 64,
    save_path: str = "neuroshard_v3_bigdata.pth",
):
    model = NeuroShardModelV2(dim=dim, num_topics=4)
    opt = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    print("=== Training NeuroShard v3.0 (dim=64, learned router, mehr Daten, LR=0.001) ===")

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_mse = 0.0
        total_ce = 0.0

        for i, text in enumerate(X_TEXTS):
            target_vec = Y_TARGETS[i]
            target_idx = torch.argmax(target_vec).unsqueeze(0)

            pred, alpha = model(text, return_alpha=True)

            mse = mse_loss(pred[:4], target_vec)
            ce = ce_loss(alpha.unsqueeze(0), target_idx)

            loss = mse + 0.1 * ce

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_mse += mse.item()
            total_ce += ce.item()

        if epoch % 40 == 0:
            print(f"Epoch {epoch:3d} | Total={total_loss:.6f} | MSE={total_mse:.6f} | CE={total_ce:.6f}")

    print("=== Fertig (v3) ===")
    torch.save(model.state_dict(), save_path)
    print(f"Modell gespeichert als {save_path}")

    # Kurzer Check: ein Beispiel pro Topic
    model.eval()
    print("\n=== Kurztest v3 (ein Satz pro Topic) ===")
    test_samples = [
        "street gang punchline rap",
        "pure love and peace for the world",
        "we prove a theorem in linear algebra",
        "ich sehe einen vogel und einen hund im tierheim",
    ]
    with torch.no_grad():
        for txt in test_samples:
            pred, alpha = model(txt, return_alpha=True)
            print(f"\nInput: {txt}")
            print(f"  Alpha: {alpha.tolist()}")
            print(f"  Output[0:4]: {pred[:4].tolist()}")


if __name__ == "__main__":
    train_neuroshard_v3()
