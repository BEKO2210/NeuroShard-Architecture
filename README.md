# NeuroShard – Experimental Mixture-of-Experts Architecture

**Author:** Belkis Aslani (`BEKO2210`)  
**Repository:** https://github.com/BEKO2210/NeuroShard-Architecture  

NeuroShard is an experimental, lightweight **Mixture-of-Experts (MoE)** architecture designed to test whether:
- simple learned routers  
- low-rank expert matrices  
- and compact embeddings  
can already create meaningful topic separation and directional feature behavior without large models.

This project runs on:
- **Windows + VS Code + Python 3.12 + PyTorch 2.9.1**
- **Android (Termux) in a pure-Python, no-torch fallback version**

---

## 🚀 Key Features

- **Versioned Architecture**
  - **v1:** Fixed keyword-based router  
  - **v2:** Fully *learned* router (Cross-Entropy + MSE training)  
  - **v3:** Larger dataset, better topic separation, more stable outputs  

- **Low-Rank Shards**
  Each “expert” is a learned low-rank adapter:  
  \[
  S_i = U_i \cdot V_i^T
  \]
  Efficient, fast, extremely small in parameter count.

- **Learned Router**
  A small neural network that maps embeddings → softmax topic distribution.

- **Modular Implementation**
  All code is separated into:
  - `src/` (training + testing code)
  - `models/` (saved checkpoints)
  - `data/` (datasets)
  - `experiments/` (log files)
  - `whitepaper/` (LaTeX scientific documentation)

---

## 📁 Project Structure

```text
NeuroShard-Architecture/
│
├── src/
│   ├── train_neuroshard.py
│   ├── test_neuroshard.py
│   ├── train_neuroshard_v2_router.py
│   ├── test_neuroshard_v2_router.py
│   ├── train_neuroshard_v3_bigdata.py
│   ├── test_neuroshard_v3_bigdata.py
│   ├── neuroshard_repl.py
│   ├── neuroshard_multilayer.py
│   └── neuroshard_repl_topics.py
│
├── data/
│   ├── dataset_v1_small.txt
│   ├── dataset_v3_big.txt
│
├── models/
│   ├── neuroshard_v1.pth
│   ├── neuroshard_v2_router.pth
│   ├── neuroshard_v3_bigdata.pth
│
├── experiments/
│   ├── logs_v1.txt
│   ├── logs_v2.txt
│   ├── logs_v3.txt
│
├── whitepaper/
│   ├── NeuroShard_Whitepaper.tex
│   └── figures/
│        └── architecture_diagram.png
│
└── README.md
🧠 Architecture Overview
1. Embedding Layer
A compact text embedding:

letter-frequency vector

no external models

works offline

extremely fast (mobile-friendly)

2. Base Transformation
A linear projection:

ℎ
=
𝑊
base
⋅
𝑥
h=W 
base
​
 ⋅x
3. Shards (Experts)
Each shard is low-rank:

𝑆
𝑖
=
𝑈
𝑖
⋅
𝑉
𝑖
𝑇
S 
i
​
 =U 
i
​
 ⋅V 
i
T
​
 
They add topic-specific direction:

𝑜
𝑖
=
𝑆
𝑖
⋅
𝑥
o 
i
​
 =S 
i
​
 ⋅x
4. Router
A small MLP or linear layer producing:

𝛼
=
softmax
(
𝑅
(
𝑥
)
)
α=softmax(R(x))
5. Output Fusion
output
=
ℎ
+
∑
𝑖
𝛼
𝑖
⋅
𝑜
𝑖
output=h+ 
i
∑
​
 α 
i
​
 ⋅o 
i
​
 
🧪 Experimental Results (Summary)
v2 – Learned Router
Strong separation:

Input	Router α (short)	Dominant
"street gang punchline rap"	[0.9998, ...]	Rap
"pure love everyone peace"	[0.00005, 0.9998, ...]	Soft
"advanced integral theorem math"	[0.00008, 0.00014, 0.9996, ...]	Math
"vogel hund katze bär"	[0.00005, 0.00008, 0.00008, 0.9997]	Animals

Outputs show consistent vector direction changes per topic.

v3 – More Data
Better generalization, smoother routing, stronger cross-topic mixing.

🛠 Installation
Clone
bash
Code kopieren
git clone https://github.com/BEKO2210/NeuroShard-Architecture.git
cd NeuroShard-Architecture
Virtual Environment (Windows)
bash
Code kopieren
python -m venv .venv
.\.venv\Scripts\activate
Install Dependencies
bash
Code kopieren
pip install torch numpy
▶️ Usage
Train v1
bash
Code kopieren
cd src
python train_neuroshard.py
Test v1
bash
Code kopieren
python test_neuroshard.py
Train v2 (learned router)
bash
Code kopieren
python train_neuroshard_v2_router.py
Train v3 (big dataset)
bash
Code kopieren
python train_neuroshard_v3_bigdata.py
📄 Whitepaper
The scientific LaTeX whitepaper can be found here:

Code kopieren
whitepaper/NeuroShard_Whitepaper.tex
It includes:

full mathematical formulation

diagrams

experiments

limitations

future improvements

🔮 Future Work
better embeddings (subword, n-gram, hashed embeddings)

more experts (8–32 shards)

multi-layer NeuroShard blocks

GPU-optimized variant

integration into LLM preprocessing

📜 License
Distributed under the Apache-2.0 License.

⭐ Acknowledgements
This is an independent research experiment created by Belkis Aslani.
The goal is to explore extremely lightweight neural architectures that can run everywhere — even on mobile.