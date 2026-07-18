# Wiki-Semantics: Large-Scale Semantic Embedding of Wikipedia via Phase-Torus Manifolds

## Abstract

Wiki-Semantics is an experimental research project exploring whether large-scale semantic representations of Wikipedia can be learned using periodic latent manifolds. The project investigates Phase-Torus embeddings, latent relation discovery, and scalable training on the full English Wikipedia hyperlink graph (~4.2M nodes, ~101M directed edges from the enwiki-2013 SNAP dataset). The system parses the graph, trains a **Phase-Torus Model** — which models relations as linear transformations on a unit interval manifold $[0, 1)$ — and produces embeddings that are intended for semantic similarity search; current work focuses on improving retrieval quality.

## Current Status

This project is under active research.

**Implemented:**
- ✓ Full Wikipedia graph loader (4.2M nodes, 101M edges)
- ✓ Phase 1: Phase-Torus model with latent relation codebook
- ✓ Phase 2: Direct Flat-Torus node embedding manifold ($D=64$, 32 independent angle pairs)
- ✓ Degree-inverse Walker-Vose alias sampling (multi-processing CPU accelerated)
- ✓ Fused CUDA & GPU-accelerated random walk co-occurrence extraction
- ✓ Spring attraction warm-start pretraining
- ✓ Bounded temperature-scaled InfoNCE contrastive loss
- ✓ Live VRAM tracking, throughput logging (396k samples/s), and ETA estimation
- ✓ Empirical distance-bucket monotonicity & topic cluster similarity verification

---

## 1. Introduction

Wikipedia's hyperlink graph encodes rich semantic structure: articles that link to each other tend to be topically related. Embedding this graph into a continuous vector space enables downstream applications including semantic search, article recommendation, and graph-structural analysis.

Existing graph embedding approaches (Node2Vec, LINE, HOPE) typically operate on small-scale graphs or require random walk sampling that is difficult to scale to 100M+ edges. TransE and its variants (RotatE, DistMult) are designed for knowledge graphs with explicit relation types, which Wikipedia lacks.

Wiki-Semantics bridges this gap by:
1. Treating all Wikipedia hyperlinks as a directed graph with **implicit relational structure**.
2. Learning a **codebook of K latent relation types** (Phase 1) or direct **Flat-Torus manifold directions** (Phase 2).
3. Applying **degree-inverse alias sampling** to prevent high-degree hub articles from dominating training.
4. Implementing a **Phase-Torus formulation** that preserves structure intrinsically under modular arithmetic on a bounded, periodic latent space ($\|\mathbf{x}\| = \sqrt{32}$ constant norm by construction).
5. Scaling to 4.2M nodes and 101M edges on a single consumer GPU (2.24 GB VRAM footprint).

---

## 2. Model Architecture

### 2.1 Phase-Torus Embedding Model (Unit Interval Manifold)

The core model models relations using linear transformations on a periodic D-dimensional unit torus.

**Node Embeddings**: Each node $i$ maps to a vector $\mathbf{\theta}_i = (\theta_{i,1}, \theta_{i,2}, \dots, \theta_{i,D})$ where each component is a phase variable in $[0, 1)$. This defines a D-dimensional torus $(\mathbb{R}/\mathbb{Z})^D$. There is no Cartesian representation, nor normalization, spheres, rotations, or quaternions.

To ensure smooth optimization without discontinuities, each node stores an unconstrained parameter vector $\phi_i \in \mathbb{R}^D$ which is converted to phase space dynamically:
$$\mathbf{\theta}_i = \phi_i \bmod 1$$

**Relation Codebook**: A set of $K$ learnable relation transformations. Each relation acts as a residual update on phases using an affine transformation defined by a matrix $W_k \in \mathbb{R}^{D \times D}$ and a bias $b_k \in \mathbb{R}^D$.

**Relation Assignment & Residual Dynamics**: For each edge $(h, t)$, the model applies the relation transformation in phase space:
$$\theta'_{h, k} = (\theta_h + W_k \theta_h + b_k) \bmod 1$$

The best-fitting relation is selected dynamically by minimizing the periodic distance to the target node:
$$k^* = \arg\min_{k \in [K]} d(\theta'_{h, k}, \theta_t)$$

**Translation Distance**: Distance is computed respecting the circular wraparound for each dimension:
$$\delta = \theta'_{h, k} - \theta_t$$
$$\delta = \delta - \operatorname{round}(\delta)$$
$$d(h, t) = \sum_{j=1}^D \delta_j^2$$

### 2.2 Loss Function

**Margin Ranking Loss with Multiple Negatives**: For each positive edge $(h, t)$ and $N$ sampled negative tails $\{\tilde{t}_1, \dots, \tilde{t}_N\}$:

$$\mathcal{L}_{raw}(h, t) = \frac{1}{N} \sum_{n=1}^{N} \max(0,\ d(h, t) - d(h, \tilde{t}_n) + \gamma)$$

where $\gamma = 1.0$ is the margin hyperparameter.

**Degree-Weighted Loss**: To prevent highly connected hub articles from dominating the gradient signal, each edge loss is weighted by the inverse square root of the combined degree:
$$w(h, t) = \frac{1}{\sqrt{\deg(h) + \deg(t)}}$$

**Regularization**: Regularization targets only the transformation strength to prevent massive phase jumps, penalizing $\|W_k\|_2$, $\|b_k\|_2$, and the wrapped magnitude of $\theta'_{h, k^*} - \theta_h$. Node embeddings themselves are unconstrained in $\mathbb{R}^D$ and are not regularized for magnitude.

### 2.3 Inference: Periodic Distance Search

At inference time, semantic similarity between a query article $h$ and target candidate $t$ is computed using the periodic distance metric (no cosine similarity):

$$\text{similarity}(h, t) = - \min_{k \in [K]} d(\theta'_{h, k}, \theta_t)$$

We return the negative distance so that higher values (closer to 0) correspond to more similar pages.

---

## 3. Training Details

| Hyperparameter | Value |
|---|---|
| Embedding dimension $D$ | 64 |
| Number of latent relations $K$ | 50 |
| Negatives per edge $N$ | 5 |
| Margin $\gamma$ | 1.0 |
| Batch size | 16,000 |
| Epochs | 10 |
| Node embedding optimizer | SGD (lr=0.01) |
| Relation optimizer | Adam (lr=0.01) |
| Loss weighting | $1/\sqrt{\deg_h + \deg_t}$ |
| Regularization scale $\lambda$ | $10^{-4}$ |
| Hardware | CUDA GPU |

**Phase Modulo**: Node embeddings' unconstrained parameters $\phi$ are wrapped to the unit interval $[0, 1)$ on the fly during the model's forward pass via $\theta = \phi \bmod 1$.

**Optimizer Split**: Node embeddings use SGD (zero optimizer state overhead, critical for fitting 4.2M × 64 = 268M parameters in VRAM), while the relation parameters (linear transformations $W$ and $b$) use Adam for faster convergence.

---

## 4. Dataset

The system uses the **enwiki-2013 SNAP dataset** (Stanford Network Analysis Project):

| Property | Value |
|---|---|
| Source | English Wikipedia, 2013 hyperlink snapshot |
| Nodes (articles) | 4,203,323 |
| Directed edges (hyperlinks) | 101,311,613 |
| Giant connected component | ~3,748,462 nodes (~89%) |
| Average out-degree | ~24 |
| Max degree (hub articles) | ~500,000+ |

Required files in `data/`:
- `enwiki-2013.txt` — edge list in SNAP format (src dst, whitespace-separated, `#` comment lines)
- `enwiki-2013-names.csv` — node ID to article title mapping (4,197,951 article titles)

---

## 5. Pipeline

```
data/enwiki-2013.txt
        │
        ▼
┌───────────────────┐
│  Step 3: Load     │  load_graph() — pandas C engine, CSR adjacency matrix
│  Graph            │  Compact ID remapping: 0..4,203,322
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  Step 4: Train    │  PhaseTorusModel / TorusEmbedding
│  Phase-Torus      │  Degree-inverse Vose alias walk co-occurrence
│                   │  On-GPU negative sampling & InfoNCE contrastive loss
│                   │  Live VRAM tracking & ETA estimation
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  Step 5: Save &   │  Checkpoint with metadata
│  Inference        │  Cosine similarity & topic cluster evaluation
└───────────────────┘
```

---

## 6. Graph Analysis & Evaluation Tools

### `phase2_run.py`
CLI pipeline runner for Phase 2 flat-torus embeddings:
- Loads Wikipedia directed graph (`data/phase2_graph.pt`).
- Builds degree-inverse Walker-Vose alias tables (`data/phase2_alias.pt`).
- Generates walk corpus and extracts co-occurrence positive pairs across window sizes (2, 5, 10).
- Runs spring warm-start pretraining and InfoNCE main training loop with live throughput and VRAM stats.

### `verify_topic_clusters.py`
Topic cluster cosine similarity verification script:
- Maps 4.2M Wikipedia node IDs to title strings via `enwiki-2013-names.csv`.
- Evaluates anchor topics (**Physics**, **Computer Science**, **Sports**) against in-cluster related subtopics vs. out-of-cluster negatives.
- Exports visualization plot to `data/cluster_similarity_plot.png`.

### `phase2_eval.py`
Evaluates cosine similarity across held-out random walk step distances ($d \in \{1, 2, 5, 10, 20, \text{random}\}$).

---

## 7. Files

| File | Purpose |
|---|---|
| `phase2/` | Package containing `graph`, `alias`, `walker`, `model`, and `train` modules |
| `phase2_run.py` | CLI main pipeline entry point |
| `phase2_eval.py` | Distance bucket similarity evaluation script |
| `verify_topic_clusters.py` | Topic cluster similarity evaluation script with title mapping |
| `verify_phase2_unit.py` | Unit verification test suite for Phase 2 invariants |
| `train_model.py` | Phase 1 model architecture and relation codebook training |
| `graph_shortest_paths.py` | Graph structural analysis and path distribution plots |
| `generate_edge_list.py` | Parse Wikipedia SQL dumps into edge list format |
| `data/enwiki-2013.txt` | Input edge list (SNAP format) |
| `data/enwiki-2013-names.csv` | Node ID → article title mapping |
| `data/phase2_checkpoint_e10.pt` | Trained Phase 2 model checkpoint |

---

## 8. Installation & Usage

```bash
# Set up environment and install dependencies
uv venv .venv
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install numpy scipy pandas tqdm matplotlib torch-cluster

# Run full Phase 2 training pipeline on Wikipedia
python phase2_run.py --edge_list data/enwiki-2013.txt --epochs 10 --warmstart_epochs 2

# Evaluate topic cluster similarity against negatives (with titles)
python verify_topic_clusters.py

# Evaluate walk distance bucket monotonicity
python phase2_eval.py --checkpoint data/phase2_checkpoint_e10.pt
```

---

## 9. Phase 2 Architecture & Design Decisions

### 9.1 Phase-Torus Manifold vs. Hypersphere
- **Flat Torus Parameterization**: Each node $n$ is parameterized by 32 independent phase angles $\mathbf{\theta}_n \in \mathbb{R}^{32}$ (unbounded float32). The forward pass maps angles to 64D Cartesian coordinates:
  $$\mathbf{x}_n = \operatorname{interleave}(\cos\mathbf{\theta}_n, \sin\mathbf{\theta}_n) \in \mathbb{R}^{64}$$
- **Intrinsic Constant Norm**: $\|\mathbf{x}_n\|_2 = \sqrt{32} \approx 5.6568$ identically for all nodes by construction ($\cos^2 \theta + \sin^2 \theta = 1$). No norm regularization or spherical projection needed.
- **Why Torus over Hypersphere**: Standard hyperspherical product-chain coordinate systems suffer from severe **gimbal lock** and **vanishing gradients** near $\theta \to 0$ or $\pi$. The flat torus uses independent angle pairs, avoiding coupling and coordinate singularities completely.
- **Dot Product & Cosine Similarity**:
  $$\mathbf{x}_u \cdot \mathbf{x}_v = \sum_{i=1}^{32} \cos(\theta_{u,i} - \theta_{v,i}) \in [-32, 32], \quad \operatorname{cos\_sim}(u, v) = \frac{\mathbf{x}_u \cdot \mathbf{x}_v}{32.0} \in [-1, 1]$$

### 9.2 Training Signal: Random Walk Co-occurrence vs. Exact APSP
- **No APSP / BFS Distance Matrices**: All-Pairs Shortest Path at Wikipedia scale ($V = 4.2 \times 10^6$) requires $V^2 \approx 1.76 \times 10^{13}$ pairs, which is computationally and memory-wise infeasible on a single machine.
- **Co-occurrence Proxies**: Graph distance distributions are captured via random walk co-occurrence windows ($w \in \{2, 5, 10\}$ for close, medium, and far proximity).

### 9.3 Degree-Inverse Alias Sampling
- **Hub Suppression**: Uniform random walks heavily oversample hub pages. To prevent hub dominance, transition probabilities $P(u \to v)$ are weighted by $1 / \operatorname{deg}(v)$.
- **Walker-Vose Alias Tables**: Precomputed in $O(V+E)$ time once per corpus and stored in flat CSR arrays. Walk step transitions perform $O(1)$ alias lookups vectorized across batch dimensions.

### 9.4 Loss Function: Bounded InfoNCE & Spring Warm-Start
- **InfoNCE Loss**: Raw margin ranking losses suffer from runaway cosine collapse. Phase 2 uses temperature-scaled InfoNCE:
  $$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\operatorname{cos\_sim}(a, p) / \tau)}{\exp(\operatorname{cos\_sim}(a, p) / \tau) + \sum_{k=1}^K \exp(\operatorname{cos\_sim}(a, n_k) / \tau)}$$
- **Spring Warm-Start**: Pretrains torus angles for 1–2 epochs using local spring attraction ($\mathbf{x}_u \cdot \mathbf{x}_v$) and negative repulsion to avoid early local minima.

### 9.5 Single Consumer Hardware Budget
- **GPU (RTX 4070, 8GB VRAM)**: Holds node embeddings ($\theta \in \mathbb{R}^{4.2M \times 32}$, ~538 MB) and Adam optimizer state (~1 GB).
- **CPU (32GB RAM)**: Stores edge index (~800 MB) and walk corpus (~6–7 GB), streaming mini-batches (16K–32K nodes) to GPU.

### 9.6 Empirical Evaluation Results (Full Wikipedia Graph: 4.2M Nodes, 101M Edges)

#### Walk Distance Bucket Monotonicity
```text
=================================================================
Distance Bucket      | Mean Cosine  | Std Dev    | Min / Max      
-----------------------------------------------------------------
Walk Step 1 (Direct) |   +0.2529    |   0.2219   | -0.55 / 0.94
Walk Step 2 (2-Hop)  |   +0.1585    |   0.2269   | -0.57 / 0.95
Walk Step 5 (5-Hop)  |   +0.0625    |   0.2085   | -0.59 / 0.90
Walk Step 10 (10-Hop)|   +0.0121    |   0.1867   | -0.65 / 0.87
Walk Step 20 (20-Hop)|   -0.0103    |   0.1750   | -0.56 / 0.80
Random / Far-Apart   |   +0.0015    |   0.1406   | -0.58 / 0.73
=================================================================
```

#### Topic Cluster Discrimination
- **Physics Cluster (Anchor: "Physics")**:
  - In-Cluster Mean: **`+0.7547`** (*Quantum mechanics*: `+0.8017`, *Albert Einstein*: `+0.7789`, *Thermodynamics*: `+0.7687`)
  - Out-of-Cluster Negatives Mean: **`+0.2953`** (*Lady Gaga*: `+0.2675`, *Heavy metal*: `+0.2149`, *Hollywood*: `+0.1885`)
  - **Cluster Separation Delta**: **`+0.4594`**
- **Computer Science Cluster (Anchor: "Computer science")**:
  - In-Cluster Mean: **`+0.7378`** (*Algorithm*: `+0.8025`, *Software engineering*: `+0.7942`, *Machine learning*: `+0.7818`)
  - Out-of-Cluster Negatives Mean: **`+0.2353`** (*Heavy metal*: `-0.0244`, *Baseball*: `+0.2190`, *Shakespeare*: `+0.2502`)
  - **Cluster Separation Delta**: **`+0.5025`**
- **Sports Cluster (Anchor: "Association football")**:
  - In-Cluster Mean: **`+0.3987`** (*FIFA World Cup*: `+0.7528`, *Rugby football*: `+0.6192`, *Basketball*: `+0.5100`)
  - Out-of-Cluster Negatives Mean: **`+0.3024`** (*Psychology*: `+0.2827`, *Heavy metal*: `+0.1787`)
  - **Cluster Separation Delta**: **`+0.0963`**

---

## License

MIT License. Open-source and freely available for research and educational use.

