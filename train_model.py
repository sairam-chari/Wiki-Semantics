import os
import time
import sqlite3
import numpy as np
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd

def load_graph(edge_list_path):
    """Load directed graph with compact ID remapping to minimize memory.
    
    Returns:
        edges: (N, 2) int32 array of edges using compact IDs
        adj_matrix: CSR adjacency matrix using compact IDs
        all_nodes: sorted array of compact node IDs (0..num_unique-1)
        orig_to_compact: dict mapping original page IDs -> compact IDs
        compact_to_orig: dict mapping compact IDs -> original page IDs
    """
    print(f"\n[Step 3] Loading graph from {edge_list_path}...")
    t0 = time.time()
    
    # Phase 1: Load the entire edge list using pandas C engine
    print("  Reading file with pandas C engine (fast)...")
    df = pd.read_csv(
        edge_list_path, 
        sep=r'\s+', 
        header=None, 
        names=['src', 'dst'],
        dtype=np.int32,
        engine='c',
        comment='#'
    )
    
    # Phase 2: Compute unique nodes and remapping efficiently
    print("  Computing unique nodes and vectorizing remapping...")
    # np.unique sorts the unique values, so compact IDs will simply be 0..N-1
    edges_array = df.values
    all_nodes, compact_edges_flat = np.unique(edges_array, return_inverse=True)
    
    # Reshape back to (N, 2)
    edges = compact_edges_flat.reshape(-1, 2).astype(np.int32)
    
    # Free memory
    del df
    del edges_array
    del compact_edges_flat
    
    num_compact = len(all_nodes)
    print(f"  Found {num_compact:,} unique nodes. Building mapping dictionaries...")
    
    # Create the mappings
    orig_to_compact = {orig: compact for compact, orig in enumerate(all_nodes)}
    compact_to_orig = {compact: orig for compact, orig in enumerate(all_nodes)}
    
    # Replace all_nodes with compact IDs for consistent downstream use
    # (negative sampling, adjacency matrix indexing, embedding lookups)
    all_nodes = np.arange(num_compact, dtype=np.int32)
    
    print("  Building Compressed Sparse Row (CSR) adjacency matrix...")
    adj_matrix = csr_matrix(
        (np.ones(len(edges), dtype=bool), (edges[:, 0], edges[:, 1])),
        shape=(num_compact, num_compact)
    )
    adj_matrix.sort_indices()
    
    print(f"Graph loaded successfully in {time.time() - t0:.2f}s")
    print(f"Total nodes: {num_compact:,}, Total edges: {len(edges):,}")
    
    return edges, adj_matrix, all_nodes, orig_to_compact, compact_to_orig


class DiscreteLatentTransRotModel(nn.Module):
    """Discrete Latent Translation + Rotation model.

    Extends the DLTM by replacing the additive mask with complex-space rotation:
      d(h, t) = || rotate(h, theta_k) + r_k - t ||_2

    where rotate() applies element-wise complex multiplication using
    angles theta_k in R^(D/2), and r_k is the translation vector.
    No masking — the rotation inherently gates dimension relevance
    (small angles ~ identity, large angles ~ active transform).

    Relation codebook of K types is auto-assigned (same as DLTM).
    Requires embedding_dim to be even (for complex pairing).
    """
    def __init__(self, num_nodes, num_relations, embedding_dim):
        assert embedding_dim % 2 == 0, "embedding_dim must be even for complex rotation"
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_relations = num_relations
        self.D2 = embedding_dim // 2          # complex dimensionality

        self.node_embeddings  = nn.Embedding(num_nodes,    embedding_dim)
        self.relation_rot     = nn.Embedding(num_relations, self.D2)       # rotation angles theta
        self.relation_trans   = nn.Embedding(num_relations, embedding_dim) # translation vectors r

        nn.init.xavier_uniform_(self.node_embeddings.weight)
        nn.init.uniform_(self.relation_rot.weight, -3.14159, 3.14159)      # angles in (-pi, pi)
        nn.init.xavier_uniform_(self.relation_trans.weight)

    def _rotate(self, x, theta):
        """Apply complex rotation to x using angles theta.
        x:     (B, D)    treated as D/2 complex numbers
        theta: (K, D/2)  rotation angles per relation
        Returns: (B, K, D) rotated embeddings for all K relations.
        """
        B  = x.shape[0]
        K  = theta.shape[0]
        D2 = self.D2

        x_re = x[:, ::2].unsqueeze(1)   # (B, 1, D/2)
        x_im = x[:, 1::2].unsqueeze(1)  # (B, 1, D/2)

        cos_t = torch.cos(theta).unsqueeze(0)  # (1, K, D/2)
        sin_t = torch.sin(theta).unsqueeze(0)  # (1, K, D/2)

        rot_re = x_re * cos_t - x_im * sin_t  # (B, K, D/2)
        rot_im = x_re * sin_t + x_im * cos_t  # (B, K, D/2)

        # Interleave real and imag back into shape (B, K, D)
        out = torch.zeros(B, K, self.embedding_dim, device=x.device)
        out[:, :, ::2]  = rot_re
        out[:, :, 1::2] = rot_im
        return out

    def forward(self, heads, tails, neg_tails=None):
        h = self.node_embeddings(heads)  # (B, D)
        t = self.node_embeddings(tails)  # (B, D)

        # Unit-normalize node embeddings
        h = h / (h.norm(dim=1, keepdim=True) + 1e-9)
        t = t / (t.norm(dim=1, keepdim=True) + 1e-9)

        theta = self.relation_rot.weight    # (K, D/2)
        r     = self.relation_trans.weight  # (K, D)

        # Rotate h under all K relations: (B, K, D)
        h_rot = self._rotate(h, theta)

        # Score all K relations: (B, K)
        t_exp = t.unsqueeze(1)    # (B, 1, D)
        r_exp = r.unsqueeze(0)    # (1, K, D)
        dists = torch.norm(h_rot + r_exp - t_exp, dim=2)  # (B, K)

        # Select best-fitting relation
        k_star = torch.argmin(dists, dim=1)              # (B,)
        h_rot_best = h_rot[torch.arange(len(heads), device=heads.device), k_star]  # (B, D)
        r_best     = r[k_star]                           # (B, D)

        pos_dist = torch.norm(h_rot_best + r_best - t, dim=1)  # (B,)

        if neg_tails is not None:
            t_neg = self.node_embeddings(neg_tails)            # (B, N, D)
            t_neg = t_neg / (t_neg.norm(dim=2, keepdim=True) + 1e-9)

            h_rot_exp = h_rot_best.unsqueeze(1)  # (B, 1, D)
            r_exp_b   = r_best.unsqueeze(1)      # (B, 1, D)
            neg_dist  = torch.norm(h_rot_exp + r_exp_b - t_neg, dim=2)  # (B, N)
            return pos_dist, neg_dist, k_star

        return pos_dist

    def translation_distance(self, heads, tails):
        return self.forward(heads, tails)


def sample_negatives_vectorized(heads, all_nodes, adj_matrix):
    """Vectorized sampling of fake tails that are NOT real neighbors of heads.
    Uses SciPy CSR fancy indexing (compiled C) to detect collisions without Python loops.
    """
    batch_size = len(heads)
    neg_tails = np.random.choice(all_nodes, size=batch_size)
    
    # Vectorized collision check via CSR fancy indexing (C-level, no Python loop)
    collisions = np.asarray(adj_matrix[heads, neg_tails]).flatten().astype(bool)
    
    # Resample only the collisions until none remain
    while np.any(collisions):
        collision_indices = np.where(collisions)[0]
        neg_tails[collision_indices] = np.random.choice(all_nodes, size=len(collision_indices))
        # Recheck only the resampled entries
        collisions[collision_indices] = np.asarray(
            adj_matrix[heads[collision_indices], neg_tails[collision_indices]]
        ).flatten().astype(bool)
    
    return neg_tails

def train(model, edges, adj_matrix, all_nodes, epochs=10, batch_size=1024, lr=0.01, margin=1.0, num_negatives=5, reg_scale=1e-6):
    # Ensure GPU/CUDA device is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available. Please ensure a CUDA device is present.")
    device = torch.device("cuda")
    print(f"Training on device: {device} with {num_negatives} negatives per edge.")
    model.to(device)
    
    # Precompute total degree (in + out) for every node — used for loss weighting.
    # Weight for edge a->b = 1 / (degree_a + degree_b): penalizes hub-heavy edges less.
    out_degrees = np.array(adj_matrix.sum(axis=1)).flatten().astype(np.float32)  # row sums
    in_degrees  = np.array(adj_matrix.sum(axis=0)).flatten().astype(np.float32)  # col sums
    total_degrees = torch.from_numpy(out_degrees + in_degrees).to(device)        # shape (N,)
    print(f"Degree weighting enabled. Max degree: {int(total_degrees.max().item()):,}, Mean: {total_degrees.mean().item():.1f}")
    
    num_nodes = len(all_nodes)
    num_edges = len(edges)
    
    # Split optimizer to fit 8GB VRAM:
    # - Node embeddings (~2 GB): SGD (zero optimizer state overhead)
    # - Relation params (tiny, model-agnostic): Adam for better convergence
    # Uses named_parameters() so this works for both DLTM and TransRot architectures.
    embedding_params = [model.node_embeddings.weight]
    relation_params  = [p for name, p in model.named_parameters() if 'node_embeddings' not in name]
    optimizer = torch.optim.SGD(embedding_params, lr=lr)
    relation_optimizer = torch.optim.Adam(relation_params, lr=lr)
    
    for epoch in range(epochs):
        t_epoch_start = time.time()
        indices = np.random.permutation(num_edges)
        total_loss = 0.0
        
        relation_counts = torch.zeros(model.num_relations, dtype=torch.long, device=device)
        
        num_batches = (num_edges + batch_size - 1) // batch_size
        with tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for i in range(0, num_edges, batch_size):
                batch_edges = edges[indices[i:i + batch_size]]
                
                heads = torch.from_numpy(batch_edges[:, 0].astype(np.int64)).to(device)
                tails = torch.from_numpy(batch_edges[:, 1].astype(np.int64)).to(device)
                
                # GPU-native negative sampling: Shape (batch_size, num_negatives)
                neg_tails = torch.randint(0, num_nodes, (len(heads), num_negatives), device=device)
                
                # Forward pass using the model
                pos_dist, neg_dist, k_star = model(heads, tails, neg_tails)
                
                # Accumulate k_star counts to monitor relation usage
                relation_counts += torch.bincount(k_star, minlength=model.num_relations)
                
                # pos_dist shape: (B,), neg_dist shape: (B, num_negatives)
                # Per-sample margin loss: mean over negatives → shape (B,)
                per_sample_loss = torch.relu(pos_dist.unsqueeze(1) - neg_dist + margin).mean(dim=1)
                
                # Degree-based weighting: weight = 1 / (deg_a + deg_b)
                # Rare specific pages get higher weight; hub nodes (high degree) contribute less
                deg_sum = total_degrees[heads] + total_degrees[tails]   # shape (B,)
                weights = 1.0 / deg_sum.clamp(min=1.0).sqrt()           # 1/sqrt(deg_a + deg_b)
                weights = weights / weights.sum()                        # normalize to sum=1
                
                base_loss = (per_sample_loss * weights).sum()
                
                # Add a factor in the loss that reduces the scale of the relationship transformations
                reg_loss = reg_scale * (model.relation_rot.weight.norm(p=2) + model.relation_trans.weight.norm(p=2))
                loss = base_loss + reg_loss
                
                optimizer.zero_grad(set_to_none=True)
                relation_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                relation_optimizer.step()
                
                total_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                
        # Print relation usage histogram in a compact format (5 per line)
        total_relations_used = relation_counts.sum().item()
        print(f"Epoch {epoch+1}/{epochs} completed in {time.time() - t_epoch_start:.2f}s, Loss: {total_loss:.4f}")
        print("Relation usage histogram:")
        lines = []
        for r_id in range(model.num_relations):
            pct = (relation_counts[r_id].item() / total_relations_used) * 100
            lines.append(f"Rel {r_id:<2}: {pct:5.1f}%")
        for chunk in range(0, len(lines), 5):
            print("  " + " | ".join(lines[chunk:chunk+5]))

def get_title_map(names_csv_path):
    """Load the mapping from page IDs to titles from the enwiki-2013-names.csv file or SQLite database."""
    print(f"Loading title map from {names_csv_path}...")
    if names_csv_path.endswith('.db'):
        conn = sqlite3.connect(names_csv_path)
        df = pd.read_sql_query("SELECT id AS node_id, title AS name FROM pages", conn)
        conn.close()
    else:
        df = pd.read_csv(names_csv_path, dtype={'node_id': 'Int64', 'name': 'str'}, on_bad_lines='skip')
    df = df.dropna()
    id_to_title = dict(zip(df['node_id'].astype(int), df['name']))
    title_to_id = dict(zip(df['name'], df['node_id'].astype(int)))
    return id_to_title, title_to_id

def most_similar(model, compact_node_id, compact_to_orig, id_to_title, all_nodes, topn=5):
    """Find nearest nodes by minimum energy (distance) across all latent relations.
    
    Instead of raw cosine similarity, this computes the proper translation+rotation
    distance: d(h, t) = min_k || rotate(h, theta_k) + r_k - t ||_2
    
    Uses chunked computation to prevent GPU memory depletion.
    """
    with torch.no_grad():
        target_idx = torch.tensor([compact_node_id], dtype=torch.long, device=model.node_embeddings.weight.device)
        h = model.node_embeddings(target_idx)
        h = h / (h.norm(dim=1, keepdim=True) + 1e-9)
        
        theta = model.relation_rot.weight
        r = model.relation_trans.weight
        
        h_rot = model._rotate(h, theta) # shape (1, K, D)
        h_transformed = (h_rot.squeeze(0) + r) # shape (K, D)
        
        # Process in chunks to cap GPU memory usage
        chunk_size = 50_000
        best_sims = torch.full((topn,), -1000.0, device=h.device)
        best_ids = torch.full((topn,), -1, dtype=torch.long, device=h.device)
        
        num_nodes = len(all_nodes)
        for start in range(0, num_nodes, chunk_size):
            end = min(start + chunk_size, num_nodes)
            chunk_emb = model.node_embeddings.weight[start:end]
            chunk_normed = chunk_emb / (chunk_emb.norm(dim=1, keepdim=True) + 1e-9)
            
            # Compute distance between all targets in chunk and all K relation transforms
            dists = torch.cdist(chunk_normed, h_transformed, p=2.0) # (N, K)
            min_dists, _ = torch.min(dists, dim=1)                  # (N,)
            chunk_sims = -min_dists                                 # Negative distance = similarity
            
            # Zero out self-similarity
            if start <= compact_node_id < end:
                chunk_sims[compact_node_id - start] = -1000.0
            
            # Merge this chunk's top candidates with running best
            combined_sims = torch.cat([best_sims, chunk_sims])
            combined_ids = torch.cat([best_ids, torch.arange(start, end, device=h.device)])
            top_k = torch.topk(combined_sims, topn)
            best_sims = top_k.values
            best_ids = combined_ids[top_k.indices]
        
        result_ids = best_ids.tolist()
        result_sims = best_sims.tolist()
        
    return [(id_to_title.get(compact_to_orig.get(i, i), str(compact_to_orig.get(i, i))), s) 
            for i, s in zip(result_ids, result_sims)]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Wiki Semantics Model")
    args = parser.parse_args()

    working_db_path = r"data\wiki_graph_working.db"
    edge_list_path  = r"data\enwiki-2013.txt"

    # Run Step 3: Load graph as CSR adjacency matrix
    edges, adj_matrix, all_nodes, orig_to_compact, compact_to_orig = load_graph(edge_list_path)

    # ── Training config ───────────────────────────────────────────────────────
    MODEL_TYPE    = "transrot"
    num_nodes     = len(all_nodes)
    num_relations = 50
    embedding_dim = 64    # must be even for transrot
    epochs        = 10
    batch_size    = 16000
    num_negatives = 5

    print("\n[Step 4] Initializing and training Translation+Rotation Model (no masking)...")
    model = DiscreteLatentTransRotModel(
        num_nodes=num_nodes,
        num_relations=num_relations,
        embedding_dim=embedding_dim
    ).to("cuda")
    model_prefix = "transrot_model"

    train(
        model=model,
        edges=edges,
        adj_matrix=adj_matrix,
        all_nodes=all_nodes,
        epochs=epochs,
        batch_size=batch_size,
        num_negatives=num_negatives
    )

    # Save checkpoint with metadata
    from datetime import datetime
    timestamp_str  = datetime.now().strftime("%Y%m%d_%H%M")
    model_filename = f"{model_prefix}_{timestamp_str}_e{epochs}_b{batch_size}.pt"
    model_path     = os.path.join(r"data", model_filename)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    checkpoint = {
        'state_dict':    model.state_dict(),
        'model_type':    MODEL_TYPE,
        'epochs':        epochs,
        'batch_size':    batch_size,
        'num_relations': num_relations,
        'embedding_dim': embedding_dim,
        'num_negatives': num_negatives,
        'timestamp':     datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    torch.save(checkpoint, model_path)
    print(f"Model saved to {model_path}")

    # Run Step 5: quick sanity-check similarity query
    print("\n[Step 5] Loading title map and verifying similarity query...")
    names_csv_path = r"data\enwiki-2013-names.csv"
    id_to_title, title_to_id = get_title_map(names_csv_path)

    if all_nodes.size > 0:
        example_compact_id = int(all_nodes[0])
        example_orig_id    = compact_to_orig.get(example_compact_id, example_compact_id)
        example_title      = id_to_title.get(example_orig_id, str(example_orig_id))
        print(f"Finding articles most similar to: '{example_title}' (ID: {example_orig_id})")
        similar = most_similar(model, example_compact_id, compact_to_orig, id_to_title, all_nodes, topn=5)
        for rank, (title, sim) in enumerate(similar, 1):
            print(f"  {rank}. {title} (Similarity: {sim:.4f})")