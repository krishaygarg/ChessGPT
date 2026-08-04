# ♟️ ChessGPT: Reinforcement Learning LLM Alignment for Chess

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/vLLM-Inference-green?style=for-the-badge" alt="vLLM" />
  <img src="https://img.shields.io/badge/RL-GRPO-orange?style=for-the-badge" alt="GRPO" />
  <img src="https://img.shields.io/badge/PEFT-LoRA-purple?style=for-the-badge" alt="LoRA" />
  <img src="https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge" alt="License" />
</p>

**ChessGPT** is an open-source research pipeline designed to fine-tune Large Language Models (LLMs) to output legal, strategic, and tactical chess moves. By combining **Group Relative Policy Optimization (GRPO)**, **vLLM-accelerated inference**, and **LoRA parameter-efficient fine-tuning**, ChessGPT aligns base language models into competitive chess agents capable of detailed step-by-step move reasoning.

---

## 🌟 Key Features

- 🧠 **GRPO Reinforcement Learning**: Employs Group Relative Policy Optimization to align model generation without requiring a separate critic network.
- ⚡ **Asynchronous vLLM Inference**: Decouples rollouts from gradient updates using multi-threaded vLLM nodes for high-throughput trajectory generation.
- 🏆 **Stockfish Win-Probability Rewards**: Evaluates model moves against Stockfish engine evaluations to compute dense reward signals based on win probability changes ($\Delta P_{\text{win}}$).
- 🎯 **Structured Reasoning Traces**: Enforces detailed chain-of-thought analysis within `<think>` reasoning tags prior to final move output in standard algebraic notation `\boxed{Nc3}`.
- 🛠️ **Memory-Efficient LoRA Training**: Uses Low-Rank Adaptation (LoRA) and gradient accumulation to make LLM alignment feasible on single/multi-GPU setups.
- 📦 **Automated PGN Processing**: Filters and converts millions of raw Lichess PGN games into FEN position prompts and training sequences.

---

## 📂 Repository Structure

```
ChessGPT/
├── envs/
│   └── chess_env.py          # Custom Python-Chess environment wrapper
├── train_dir/
│   ├── inference_node.py     # Asynchronous vLLM candidate generation server
│   ├── training_node.py      # PyTorch & LoRA policy update worker
│   └── main.py              # Multi-GPU orchestrator for training & inference
├── train_grpo.py             # Standalone GRPO reinforcement learning training loop
├── finetuning.py             # Supervised & RL fine-tuning routines
├── training_loop.py          # Custom PyTorch training loop
├── reward.py                 # Stockfish-backed reward evaluation system
├── vllm.py                   # Custom vLLM sampling configuration
├── data_sampler.py           # PGN position sampler and FEN parser
├── filter_lichess_games.py  # Utility to parse and filter high-Elo Lichess PGN databases
├── get_move_sequences.py     # Move sequence extraction helper
├── preprocess_position.py   # Board position preprocessing & prompt formatter
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites

- **Python**: 3.10 or higher
- **GPU**: NVIDIA GPU with CUDA support (Recommended: 24GB+ VRAM)
- **Stockfish Engine**: Installed and accessible in your system PATH

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/krishaygarg/ChessGPT.git
cd ChessGPT
pip install -r requirements.txt
```

Make sure Stockfish is installed on your operating system:
- **macOS**: `brew install stockfish`
- **Linux (Ubuntu)**: `sudo apt-get install stockfish`

### 3. Running the Training Pipeline

#### Step A: Filter & Sample PGN Datasets
Parse raw Lichess database dumps to extract high-quality candidate games and convert positions into standard FEN format:

```bash
python filter_lichess_games.py --input lichess_db.pgn.zst --output filtered_games.pgn
```

#### Step B: Launch GRPO Training
Run the multi-node training pipeline connecting the vLLM rollout generator and PyTorch LoRA optimizer:

```bash
python train_grpo.py --model_name_or_path "Qwen/Qwen2.5-Coder-7B-Instruct"
```

Or orchestrate distributed training across GPUs:

```bash
python train_dir/main.py
```

---

## 🔬 Reinforcement Learning Details

### Reward Formulation

ChessGPT uses a composite reward function $R(s, a)$:

$$R(s, a) = R_{\text{validity}}(a) + \Delta P_{\text{win}}(s, a) + R_{\text{format}}(a)$$

1. **Move Validity Reward ($R_{\text{validity}}$)**: $+1.0$ for generating legal moves according to official chess rules, $-2.0$ for illegal move attempts.
2. **Win Probability Change ($\Delta P_{\text{win}}$)**: Measures Stockfish Centipawn evaluation shift converted to expected win probability:
   $$\Delta P = \sigma(E_{\text{post}} / 400) - \sigma(E_{\text{pre}} / 400)$$
3. **Format Reward ($R_{\text{format}}$)**: Bonus for adhering to structured `<think>` reasoning tags and enclosed LaTeX move tags (`\boxed{move}`).

---

## 📜 License

Distributed under the [MIT License](LICENSE).
