import torch
from train_neuroshard_v2_router import NeuroShardModelV2


def load_model(path: str = "neuroshard_v3_bigdata.pth", dim: int = 64) -> NeuroShardModelV2:
    model = NeuroShardModelV2(dim=dim, num_topics=4)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def run_example(model: NeuroShardModelV2, text: str):
    with torch.no_grad():
        y, alpha = model(text, return_alpha=True)

    print("\n=== Beispiel ===")
    print(f"Input-Text:  {text}")
    print(f"Router-Alpha: {alpha.tolist()}")
    print(f"Output[0:4]: {y[:4].tolist()}")


def main():
    model = load_model()

    tests = [
        # ein paar Trainingssätze
        "street gang punchline rap",
        "hard street flow on the west side",
        "pure love and peace for the world",
        "soft words and kind hearts",
        "we prove a theorem in linear algebra",
        "calculus and differential equations",
        "ich sehe einen vogel und einen hund im tierheim",
        "katze und hund schlafen auf dem sofa",

        # völlig neue Sätze zum Testen
        "real gangsta rap with heavy street flow",
        "I wish peace and love for all humans",
        "we compute the integral of this function in math",
        "viele tiere laufen im zoo herum",
        "ein komplett neutraler text ohne klare richtung",
    ]

    for t in tests:
        run_example(model, t)


if __name__ == "__main__":
    main()
