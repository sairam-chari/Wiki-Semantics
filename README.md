# Wiki-Semantics: Large-Scale Semantic Embedding of Wikipedia via Discrete Latent Translation & Rotation

## Abstract

Wiki-Semantics is a framework for learning dense semantic representations of Wikipedia articles at scale. The system parses the full English Wikipedia hyperlink graph (~4.2M nodes, ~101M directed edges from the enwiki-2013 SNAP dataset), trains a **Discrete Latent Translation + Rotation Model** (TransRot) — which models relations as complex-space rotations combined with translations — on the resulting directed graph, and produces embeddings suitable for semantic similarity search across all Wikipedia articles. Unlike approaches based on text content, Wiki-Semantics derives semantic structure purely from hyperlink connectivity, capturing relational proximity (e.g., `Mars` → `Solar System` → `Jupiter`) without requiring raw article text.

---

## 1. Introduction

Wikipedia's hyperlink graph encodes rich semantic structure: articles that link to each other tend to be topically related. Embedding this graph into a continuous vector space enables downstream applications including semantic search, article recommendation, and graph-structural analysis.

Existing graph embedding approaches (Node2Vec, LINE, HOPE) typically operate on small-scale graphs or require random walk sampling that is difficult to scale to 100M+ edges. TransE and its variants (RotatE, DistMult) are designed for knowledge graphs with explicit relation types, which Wikipedia lacks.

Wiki-Semantics bridges this gap by:
1. Treating all Wikipedia hyperlinks as a directed graph with **implicit relational structure**.
2. Learning a **codebook of K latent relation types** automatically from data, without manual relation annotation.
3. Applying **degree-weighted loss** to prevent high-degree hub articles from dominating training.
4. Implementing a **complex-space rotation + translation** model that resolves symmetries and hierarchical relationships.
5. Scaling to 4.2M nodes and 101M edges on a single consumer GPU.

---

## 2. Model Architecture

### 2.1 Discrete Latent Translation & Rotation Model (TransRot)

The core model models relations using complex-space rotations combined with translations. For each directed edge $(h, t)$:

**Node Embeddings**: Each node $i$ maps to a vector $\mathbf{e}_i \in \mathbb{R}^D$ where $D$ is the embedding dimension. The vector is normalized to the unit sphere on the fly during the forward pass and treated as $D/2$ complex numbers $\mathbf{e}_i \in \mathbb{C}^{D/2}$.

**Relation Codebook**: A set of $K$ learnable relation rotations $\{\theta_1, \dots, \theta_K\} \subset \mathbb{R}^{D/2}$ and relation translation vectors $\{r_1, \dots, r_K\} \subset \mathbb{R}^D$. These capture different types of Wikipedia linkage (e.g., "is an instance of", "is related to", "is part of") without supervision. Unlike models with unit-normalized translations, the scale of translation vectors $r_k$ is unconstrained to allow them to shrink to zero during training, which encourages close clustering.

**Relation Assignment**: For each edge $(h, t)$, the model dynamically selects the best-fitting relation:
$$k^* = \arg\min_{k \in [K]} \| \text{rotate}(\mathbf{e}_h, \theta_k) + r_k - \mathbf{e}_t \|_2$$

where $\text{rotate}(\mathbf{e}_h, \theta_k)$ rotates the $D/2$ complex coordinates of $h$ by angles $\theta_k$.

**Translation Distance**: Using the selected relation:
$$d(h, t) = \| \text{rotate}(\mathbf{e}_h, \theta_{k^*}) + r_{k^*} - \mathbf{e}_t \|_2$$

### 2.1.1 Non-Commutativity and Relation Composition

A critical mathematical feature of the TransRot architecture is that relationship composition is **non-commutative (order-dependent)**. This distinguishes it from models like TransE (translation-only) and RotatE (rotation-only) where relationship composition is commutative.

#### Mathematical Derivation:
If we apply two relationship transformations sequentially to a node embedding $\mathbf{h}$:
1. **Path A (Relation 1 then Relation 2)**:
   $$T_2(T_1(\mathbf{h})) = \text{rotate}(\mathbf{h}, \theta_1 + \theta_2) + \text{rotate}(r_1, \theta_2) + r_2$$
2. **Path B (Relation 2 then Relation 1)**:
   $$T_1(T_2(\mathbf{h})) = \text{rotate}(\mathbf{h}, \theta_2 + \theta_1) + \text{rotate}(r_2, \theta_1) + r_1$$

Since rotating a translation vector changes its direction:
$$\text{rotate}(r_1, \theta_2) + r_2 \neq \text{rotate}(r_2, \theta_1) + r_1$$

Therefore, $T_2(T_1(\mathbf{h})) \neq T_1(T_2(\mathbf{h}))$.

#### Semantic Importance:
In real-world semantics, the order of relationship application matters. For instance:
* `Seattle` $\xrightarrow{\text{located in}}$ `Washington` $\xrightarrow{\text{is a state of}}$ `United States` (Valid hierarchy, whereas the reverse order is semantically invalid).
* In family relationships, $\text{Brother} \circ \text{Father} \implies \text{Uncle}$, whereas $\text{Father} \circ \text{Brother} \implies \text{Father}$.

By maintaining non-commutative composition, TransRot avoids false semantic equivalences (like collapsing Uncle and Father) and can model complex directed paths and hierarchies.

### 2.2 Loss Function

**Margin Ranking Loss with Multiple Negatives**: For each positive edge $(h, t)$ and $N$ sampled negative tails $\{\tilde{t}_1, \dots, \tilde{t}_N\}$:

$$\mathcal{L}_{raw}(h, t) = \frac{1}{N} \sum_{n=1}^{N} \max(0,\ d(h, t) - d(h, \tilde{t}_n) + \gamma)$$

where $\gamma = 1.0$ is the margin hyperparameter.

**Negative Distance Evaluation**: The distance for negative samples is evaluated under the same relation $k^*$ selected for the positive edge:
$$d(h, \tilde{t}_n) = \| \text{rotate}(\mathbf{e}_h, \theta_{k^*}) + r_{k^*} - \mathbf{e}_{\tilde{t}_n} \|_2$$

**Degree-Weighted Loss**: To prevent highly connected hub articles (e.g., "United States", "World War II") from dominating the gradient signal, each edge loss is weighted by the inverse square root of the combined degree:

$$w(h, t) = \frac{1}{\sqrt{\deg(h) + \deg(t)}}$$

where $\deg(v) = \text{in-degree}(v) + \text{out-degree}(v)$. Weights are normalized per batch to sum to 1:

$$\mathcal{L}_{margin} = \sum_{(h,t) \in \mathcal{B}} \frac{w(h,t)}{\sum_{(h',t') \in \mathcal{B}} w(h',t')} \cdot \mathcal{L}_{raw}(h, t)$$

**L2 Scale Regularization**: To reduce the scale of the relationship transformations and force related nodes to cluster closer together in the embedding space (so that $\text{rotate}(\mathbf{e}_h, \theta_k) + r_k \approx \mathbf{e}_h$, implying $\mathbf{e}_h \approx \mathbf{e}_t$), we add an L2 regularization term on both relation rotation angles ($\theta$) and relation translations ($r$):

$$\mathcal{L}_{reg} = \lambda \left( \|\Theta\|_2 + \|R\|_2 \right)$$

where $\lambda = 10^{-4}$ is the regularization scale factor.

The final loss optimized is:
$$\mathcal{L} = \mathcal{L}_{margin} + \mathcal{L}_{reg}$$

### 2.3 Inference: TransRot Energy Minimum Search

At inference time, rather than using raw cosine similarity (which ignores the learned rotations and translations), semantic similarity between a query article $h$ and target candidate $t$ is computed using the model's actual energy distance metric:

$$\text{similarity}(h, t) = -\min_{k \in [K]} \| \text{rotate}(\mathbf{e}_h, \theta_k) + r_k - \mathbf{e}_t \|_2$$

We return the negative distance so that higher values (closer to 0) correspond to more similar pages. For efficient top-$k$ retrieval over 4.2M articles, the distance is vectorized using PyTorch `cdist` and evaluated in chunks of 50,000 targets to prevent GPU memory depletion (OOM).

---

## 3. Training Details

| Hyperparameter | Value |
|---|---|
| Embedding dimension $D$ | 64 (must be even for TransRot) |
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

**Embedding Normalization**: Node embeddings are normalized to the unit sphere on the fly during the model's forward pass to maintain numerical stability and ensure consistency.

**Optimizer Split**: Node embeddings use SGD (zero optimizer state overhead, critical for fitting 4.2M × 64 = 268M parameters in VRAM), while the relation parameters (rotations and translations) use Adam for faster convergence.

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
│  Step 4: Train    │  DiscreteLatentTransRotModel (Exclusive)
│  TransRot         │  Degree-weighted margin-ranking loss
│                   │  On-GPU negative sampling & L2 scale regularization
│                   │  Relation usage monitoring (k_star histogram)
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  Step 5: Save &   │  Checkpoint with metadata (epochs, batch size, timestamp)
│  Inference        │  Cosine similarity search (chunked 500K at a time)
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
- Automatically discovers and loads the latest trained `transrot_model_*.pt` checkpoints from the `data/` directory.
- Queries the embedding space using cosine similarity.
- Displays comparative, side-by-side results for multiple models.
- Filters and displays only results where all compared models agree with a score above a configurable threshold (default: -1.5, representing energy/similarity).

---

## 7. Files

| File | Purpose |
|---|---|
| `train_model.py` | Graph loading, TransRot architecture, training loop, cosine similarity search |
| `verify_similar.py` | Load checkpoints and run side-by-side similarity comparisons |
| `graph_shortest_paths.py` | Graph structural analysis and path distribution plots |
| `generate_edge_list.py` | Parse Wikipedia SQL dumps into edge list format |
| `data/enwiki-2013.txt` | Input edge list (SNAP format) |
| `data/enwiki-2013-names.csv` | Node ID → article title mapping (also supports SQLite `.db` load in `get_title_map`) |
| `data/transrot_model_*.pt` | Trained checkpoint (includes metadata) |

---

## 8. Installation & Usage

```bash
# Install dependencies
pip install numpy scipy pandas scikit-learn torch tqdm matplotlib

# Train the model (uses DiscreteLatentTransRotModel exclusively)
python train_model.py

# Run side-by-side similarity comparisons on trained checkpoints
python verify_similar.py

# Analyze graph structure and path distributions
python graph_shortest_paths.py
```

**Checkpoint naming convention**: Saved checkpoints include training metadata in the filename:
```
transrot_model_YYYYMMDD_HHMM_e{epochs}_b{batch_size}.pt
```

---

## License

MIT License. Open-source and freely available for research and educational use.
