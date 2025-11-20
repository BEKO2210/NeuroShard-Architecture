# NeuroShard – Experimental Mixture-of-Experts Architecture

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.12-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-red.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Experimental-orange)](https://github.com/BEKO2210/NeuroShard-Architecture)

**Author:** Belkis Aslani (`BEKO2210`)  
**Repository:** [github.com/BEKO2210/NeuroShard-Architecture](https://github.com/BEKO2210/NeuroShard-Architecture)

📄 **Whitepaper:**  
See the full LaTeX paper here:  
[`whitepaper/NeuroShard_Whitepaper.tex`](whitepaper/NeuroShard_Whitepaper.tex)


**NeuroShard** is an experimental, lightweight **Mixture-of-Experts (MoE)** architecture.  
It is designed to answer a specific question: *Can simple learned routers, low-rank expert matrices, and compact embeddings create meaningful topic separation without Large Language Models?*

This project runs on:
* **Windows** (VS Code + Python 3.12 + PyTorch 2.9.1)
* **Android** (Termux in a pure-Python, no-torch fallback version)

## 🚀 Key Features

* **🧩 Versioned Architecture**
    * **v1:** Fixed keyword-based router.
    * **v2:** Fully *learned* router (Cross-Entropy + MSE training).
    * **v3:** Larger dataset, improved topic separation, stable outputs.

* **📉 Low-Rank Shards**
    Each "expert" is a learned low-rank adapter matrix. This makes the model efficient, fast, and extremely small in parameter count.

* **🧠 Learned Router**
    A small neural network that maps embeddings to a softmax topic distribution.

* **📦 Modular Implementation**
    Clean separation of `src` (logic), `models` (checkpoints), `data`, and `experiments` (logs).


## 🧠 Architecture Overview

### 1. Embedding Layer
A compact text embedding based on letter-frequency vectors. It requires **no external models**, works offline, and is extremely fast (mobile-friendly).

### 2. Base Transformation
A linear projection of the input:
$$h = W_{base} \cdot x$$

### 3. Shards (Experts)
Each shard provides a topic-specific direction. They are low-rank decompositions:
$$S_i = U_i \cdot V_i^T$$
$$o_i = S_i \cdot x$$

### 4. Router
A small MLP or linear layer producing the gating weights:
$$\alpha = \text{softmax}(R(x))$$

### 5. Output Fusion
The final output combines the base transformation with the weighted experts:
$$\text{output} = h + \sum_{i} \alpha_i \cdot o_i$$


## 🧪 Experimental Results (Summary)

### v2 – Learned Router
The model shows strong separation capabilities based on input semantics.

| Input | Router $\alpha$ (short) | Dominant Topic |
| :--- | :--- | :--- |
| "street gang punchline rap" | `[0.9998, ...]` | **Rap** |
| "pure love everyone peace" | `[0.00005, 0.9998, ...]` | **Soft/Soul** |
| "advanced integral theorem" | `[0.00008, 0.00014, 0.9996]` | **Math** |
| "vogel hund katze bär" | `[0.000, 0.000, 0.000, 0.999]` | **Animals** |

> **Observation:** Outputs show consistent vector direction changes per topic.

### v3 – More Data
* Better generalization to unseen words.
* Smoother routing distributions.
* Stronger cross-topic mixing capability.


## 📁 Project Structure

```text
NeuroShard-Architecture/
│
├── src/                          # Source code for training & inference
│   ├── train_neuroshard.py
│   ├── test_neuroshard.py
│   ├── train_neuroshard_v2_router.py
│   ├── neuroshard_repl.py        # Interactive REPL
│   └── ...
│
├── data/                         # Text datasets
│   ├── dataset_v1_small.txt
│   └── dataset_v3_big.txt
│
├── models/                       # Saved .pth checkpoints
│   ├── neuroshard_v1.pth
│   └── neuroshard_v2_router.pth
│
├── experiments/                  # Log files and run history
│   ├── logs_v1.txt
│   └── logs_v2.txt
│
├── whitepaper/                   # Scientific documentation
│   ├── NeuroShard_Whitepaper.tex
│   └── figures/
│
└── README.md
````

## 🛠 Installation

### 1\. Clone Repository

```bash
git clone [https://github.com/BEKO2210/NeuroShard-Architecture.git](https://github.com/BEKO2210/NeuroShard-Architecture.git)
cd NeuroShard-Architecture
```

### 2\. Create Virtual Environment (Windows)

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 3\. Install Dependencies

```bash
pip install torch numpy
```

## ▶️ Usage

**Train v1 (Basic):**

```bash
cd src
python train_neuroshard.py
```

**Test v1:**

```bash
python test_neuroshard.py
```

**Train v2 (Learned Router):**

```bash
python train_neuroshard_v2_router.py
```

**Train v3 (Big Dataset):**

```bash
python train_neuroshard_v3_bigdata.py
```


## 📄 Whitepaper

The scientific documentation and mathematical derivation can be found in:
`whitepaper/NeuroShard_Whitepaper.tex`

It includes:

  * Full mathematical formulation
  * Architecture diagrams
  * Detailed limitations and future improvements


## 🔮 Future Work

  * [ ] **Better Embeddings:** Implement subword, n-gram, or hashed embeddings.
  * [ ] **Scale Up:** Increase to 8–32 shards.
  * [ ] **Depth:** Implement multi-layer NeuroShard blocks.
  * [ ] **Optimization:** Create a specific GPU-optimized variant.
  * [ ] **LLM Integration:** Use as a preprocessing layer for larger models.


## 📜 License

Distributed under the **Apache-2.0 License**.

## ⭐ Acknowledgements

This is an independent research experiment created by **Belkis Aslani**.  
The goal is to explore extremely lightweight neural architectures that can run everywhere — even on mobile devices.
