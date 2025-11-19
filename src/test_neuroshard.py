import torch

from train_neuroshard import NeuroShardModel


def load_model(path: str = "neuroshard_v1.pth", dim: int = 64) -> NeuroShardModel:
    model = NeuroShardModel(dim=dim)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def run_example(model: NeuroShardModel, text: str):
    with torch.no_grad():
        x = model.emb(text)
        alpha = model.router.detect(text)
        a = torch.tensor(alpha, dtype=torch.float32)

        y1 = model.layer1(x, a)
        y1_act = torch.relu(y1)
        y2 = model.layer2(y1_act, a)

    print("\n=== Beispiel ===")
    print(f"Input-Text: {text}")
    print(f"Router-Alpha: {alpha}")
    print(f"Output-Vec (erste 8 Werte): {y2[:8].tolist()}")


def main():
    model = load_model()

    tests = [
        "street gang punchline rap",
        "deep love everyone peace",
        "advanced integral theorem math",
        "vogel hund katze bär",
        "neutraler text ohne spezialwort",
        "west side street flow",
        "ich liebe alle tiere und menschen",
    ]

    for t in tests:
        run_example(model, t)


if __name__ == "__main__":
    main()
