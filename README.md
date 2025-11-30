# Nano-HOPE: High-Fidelity Reference Implementation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zWa65q2WK5lgeeYwJBCNrc4v_AueRFrd?usp=sharing)

**Author:** Shushank

## Abstract

**Nano-HOPE** is a pristine, educational, and high-fidelity implementation of the **HOPE** (Higher-Order Policy Engine) architecture. Unlike standard Transformers, HOPE employs a **Self-Modifying Titans Core** that updates its own weights at inference time, allowing it to "learn" from the context dynamically.

This repository implements the exact **L2 Regression / Delta Rule** update mechanism described in the research, optimized for readability and educational value.

## Architecture

### The Self-Modifying Core (TitansL2)

At the heart of Nano-HOPE is the `TitansL2` layer. Instead of static attention, it maintains a memory matrix $M_t$ that evolves according to the **Delta Rule**:

$$ M_{t+1} = M_t - \alpha (M_t k_t) k_t^T + \beta v_t k_t^T $$

Where:
*   $M_t$: The memory state (effectively a weight matrix).
*   $k_t, v_t$: Key and Value vectors at step $t$.
*   $\alpha$: Forget rate (learnable).
*   $\beta$: Write rate (learnable).

This allows the model to explicitly **overwrite** old information with new data, solving the "capacity" problem of standard Linear Attention.

```mermaid
graph LR
    Input[Input Token] --> Proj[Q, K, V Projections]
    Proj --> Read[Read: y = M @ q]
    Proj --> Update[Update: M_new = M - Forget + Write]
    Read --> Output[Output Token]
    Update --> Memory((Memory State M))
    Memory --> Read
    Memory --> Update
```

### Continuum Memory System (CMS)

Surrounding the Titans core are **CMS Blocks**—MLPs that operate at different timescales to capture long-term structural knowledge.

## Installation

```bash
git clone https://github.com/Sk16er/hope_nano.git
cd hope
pip install -r requirements.txt
```

## Quick Start

### 1. Training
Train the model on `TinyStories` with advanced features (Cosine Scheduler, Gradient Clipping):

```bash
python train.py
```

### 2. Stateful Inference (The "Magic")
Run the generation script to see **Stateful Inference** in action. Unlike Transformers that re-process the whole history, HOPE passes a compact state forward.

```bash
python generate.py
```

## Advanced Usage

### Config
Modify `config.py` to change model size or update rules.

```python
@dataclass
class HOPEConfig:
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    # ...
```

### Model API
The `HOPE` class in `model.py` supports explicit state passing:

```python
# Initial Step
logits, loss, states = model(input_ids)

# Next Step (Pass 'states' to avoid re-computation)
logits, loss, states = model(next_token, states=states)
```

