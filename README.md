# Wiki-Semantics: Large-Scale Semantic Embedding of Wikipedia via Phase-Torus Manifolds

## Abstract

Wiki-Semantics is a framework for learning dense semantic representations of Wikipedia articles at scale. The system parses the full English Wikipedia hyperlink graph (~4.2M nodes, ~101M directed edges from the enwiki-2013 SNAP dataset), trains a **Phase-Torus Model** — which models relations as linear transformations on a unit interval manifold $[0, 1)$ — on the resulting directed graph, and produces embeddings suitable for semantic similarity search across all Wikipedia articles. Unlike approaches based on text content, Wiki-Semantics derives semantic structure purely from hyperlink connectivity, capturing relational proximity (e.g., `Mars` → `Solar System` → `Jupiter`) without requiring raw article text.

---

## 1. Introduction

Wikipedia's hyperlink graph encodes rich semantic structure: articles that link to each other tend to be topically related. Embedding this graph into a continuous vector space enables downstream applications including semantic search, article recommendation, and graph-structural analysis.

Existing graph embedding approaches (Node2Vec, LINE, HOPE) typically operate on small-scale graphs or require random walk sampling that is difficult to scale to 100M+ edges. TransE and its variants (RotatE, DistMult) are designed for knowledge graphs with explicit relation types, which Wikipedia lacks.

Wiki-Semantics bridges this gap by:
1. Treating all Wikipedia hyperlinks as a directed graph with **implicit relational structure**.
2. Learning a **codebook of K latent relation types** automatically from data, without manual relation annotation.
3. Applying **degree-weighted loss** to prevent high-degree hub articles from dominating training.
4. Implementing a **Phase-Torus formulation** that preserves structure intrinsically under modular arithmetic on a bounded, periodic latent space.
5. Scaling to 4.2M nodes and 101M edges on a single consumer GPU.

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
- `enwiki-2013-names.csv` — node ID to article title mapping

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
│  Step 4: Train    │  PhaseTorusModel
│  Phase-Torus      │  Degree-weighted margin-ranking loss
│                   │  On-GPU negative sampling & L2 scale regularization
│                   │  Relation usage monitoring (k_star histogram)
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  Step 5: Save &   │  Checkpoint with metadata (epochs, batch size, timestamp)
│  Inference        │  Periodic distance search (chunked 50K at a time)
└───────────────────┘
```

---

## 6. Graph Analysis Tools

### `graph_shortest_paths.py`

A companion analysis script for studying the structural properties of the Wikipedia hyperlink graph:

- **BFS Shortest Paths**: Computes shortest path distances from a source node to all reachable nodes using SciPy's C-level BFS.
- **Node-Disjoint Alternative Paths**: Finds up to $N$ node-disjoint shortest paths between pairs using a greedy intermediate-node blocking strategy.
- **Path Length Distribution**: Plots the distribution of the 1st through $N$-th shortest path lengths across target nodes.
- **Variance Analysis**: Plots the variance of the $k$-th shortest path length across randomly sampled $(src, target)$ pairs.

### `verify_similar.py`

Interactive similarity verification and model comparison tool:
- Automatically discovers and loads the latest trained `PTM_*.pt` checkpoints from the `data/` directory.
- Queries the embedding space using periodic minimum energy distance.
- Displays comparative, side-by-side results for multiple models.
- Filters and displays only results where all compared models agree with a score above a configurable threshold (default: -1.5, representing energy/similarity).

---

## 7. Files

| File | Purpose |
|---|---|
| `train_model.py` | Graph loading, PhaseTorus architecture, training loop, periodic distance search |
| `verify_similar.py` | Load checkpoints and run side-by-side similarity comparisons |
| `graph_shortest_paths.py` | Graph structural analysis and path distribution plots |
| `generate_edge_list.py` | Parse Wikipedia SQL dumps into edge list format |
| `data/enwiki-2013.txt` | Input edge list (SNAP format) |
| `data/enwiki-2013-names.csv` | Node ID → article title mapping (also supports SQLite `.db` load in `get_title_map`) |
| `data/PTM_*.pt` | Trained checkpoint (includes metadata) |

---

## 8. Installation & Usage

```bash
# Install dependencies
pip install numpy scipy pandas scikit-learn torch tqdm matplotlib

# Train the model (uses PhaseTorusModel)
python train_model.py

# Run side-by-side similarity comparisons on trained checkpoints
python verify_similar.py

# Analyze graph structure and path distributions
python graph_shortest_paths.py
```

**Checkpoint naming convention**: Saved checkpoints include training metadata in the filename:
```
PTM_YYYYMMDD_HHMM_e{epoch}_of_{epochs}_b{batch_size}.pt
```

---

## License

MIT License. Open-source and freely available for research and educational use.
