import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from phase2.graph import load_phase2_graph
from phase2.alias import build_alias_tables
from phase2.walker import generate_walks_biased, extract_cooccurrence_pairs
from phase2.model import TorusEmbedding
from phase2.deepwalk import DeepWalkEmbedding
from phase2.train import BFloat16Adam, get_vram_stats, _format_time
from verify_topic_clusters import load_title_mappings, find_compact_id

def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def compute_link_prediction_auc(model, test_pos_src, test_pos_dst, num_nodes, num_samples=50000, seed=42, device="cuda"):
    """Compute held-out Link Prediction ROC-AUC on 100% unseen test edges vs random negative pairs."""
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    
    num_test_edges = len(test_pos_src)
    pos_perm = torch.randperm(num_test_edges, generator=g)[:num_samples]
    p_src = test_pos_src[pos_perm].to(device)
    p_dst = test_pos_dst[pos_perm].to(device)
    
    n_src = torch.randint(0, num_nodes, (num_samples,), generator=g, device=device)
    n_dst = torch.randint(0, num_nodes, (num_samples,), generator=g, device=device)
    
    with torch.no_grad():
        p_src_c = model(p_src)
        p_dst_c = model(p_dst)
        pos_sims = model.compute_cosine_sim(p_src_c, p_dst_c)
        
        n_src_c = model(n_src)
        n_dst_c = model(n_dst)
        neg_sims = model.compute_cosine_sim(n_src_c, n_dst_c)
        
    chunk = 5000
    gt_count = 0.0
    eq_count = 0.0
    total_pairs = float(num_samples * num_samples)
    
    for i in range(0, num_samples, chunk):
        p_chunk = pos_sims[i:i+chunk].unsqueeze(1) # [chunk, 1]
        for j in range(0, num_samples, chunk):
            n_chunk = neg_sims[j:j+chunk].unsqueeze(0) # [1, chunk]
            gt_count += (p_chunk > n_chunk).sum().item()
            eq_count += (p_chunk == n_chunk).sum().item()
            
    return (gt_count + 0.5 * eq_count) / total_pairs

def evaluate_topic_clusters(model, compact_to_title, title_to_compact, device="cuda"):
    """Evaluate topic cluster cosine similarities for a model."""
    clusters = {
        "Sports": {
            "anchor": "Association football",
            "positives": ["Basketball", "Tennis", "Baseball", "Cricket", "Rugby football", "Olympic Games", "FIFA World Cup", "Golf", "Ice hockey"],
            "negatives": ["Quantum mechanics", "Computer science", "Calculus", "France", "Japan", "Heavy metal music", "Algebra", "Psychology"]
        },
        "Physics": {
            "anchor": "Physics",
            "positives": ["Quantum mechanics", "General relativity", "Particle physics", "Thermodynamics", "Electromagnetism", "Albert Einstein", "Classical mechanics"],
            "negatives": ["Association football", "Basketball", "Tennis", "Heavy metal music", "Hollywood", "Lady Gaga", "Baseball", "Cricket"]
        },
        "Computer Science": {
            "anchor": "Computer science",
            "positives": ["Algorithm", "Programming language", "Software engineering", "Artificial intelligence", "Python (programming language)", "Machine learning", "Operating system"],
            "negatives": ["Tennis", "Cricket", "Roman Empire", "William Shakespeare", "Heavy metal music", "Baseball", "Olympic Games"]
        }
    }
    
    model.eval()
    results = {}
    with torch.no_grad():
        for cluster_name, data in clusters.items():
            anchor_cid = find_compact_id(data["anchor"], title_to_compact)
            if anchor_cid is None:
                continue
            anchor_coord = model(torch.tensor([anchor_cid], device=device))
            
            pos_sims = []
            for t in data["positives"]:
                cid = find_compact_id(t, title_to_compact)
                if cid is not None:
                    coord = model(torch.tensor([cid], device=device))
                    sim = float(model.compute_cosine_sim(anchor_coord, coord).item())
                    pos_sims.append(sim)
                    
            neg_sims = []
            for t in data["negatives"]:
                cid = find_compact_id(t, title_to_compact)
                if cid is not None:
                    coord = model(torch.tensor([cid], device=device))
                    sim = float(model.compute_cosine_sim(anchor_coord, coord).item())
                    neg_sims.append(sim)
                    
            mean_pos = float(np.mean(pos_sims)) if pos_sims else 0.0
            mean_neg = float(np.mean(neg_sims)) if neg_sims else 0.0
            delta = mean_pos - mean_neg
            results[cluster_name] = {'pos': mean_pos, 'neg': mean_neg, 'delta': delta}
            
    return results

def train_rigorous_model(
    model,
    model_name,
    pairs_dict,
    num_nodes,
    epochs=10,
    batch_size=16384,
    lr=0.01,
    num_negatives=10,
    temperature=0.1,
    seed=42,
    device="cuda"
):
    """Rigorous training loop ensuring 100% identical updates, seeds, loss, and negative sampling."""
    print(f"\n[{model_name} Train] Starting leak-free training ({epochs} epochs, lr={lr}, temp={temperature}, batch_size={batch_size:,})...")
    t0_start = time.time()
    
    cleanup_cuda()
    model.to(device)
    model.train()
    
    optimizer = BFloat16Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    
    anchors = pairs_dict['anchors']
    positives = pairs_dict['positives']
    num_pairs = len(anchors)
    
    if not anchors.is_pinned():
        anchors = anchors.pin_memory()
        positives = positives.pin_memory()
        
    for epoch in range(epochs):
        t_ep_start = time.time()
        
        # Fixed random generator seed per epoch for identical mini-batch permutations and negative sampling
        g = torch.Generator(device="cpu").manual_seed(seed + epoch * 1000)
        perm = torch.randperm(num_pairs, generator=g)
        
        total_loss = 0.0
        num_batches = (num_pairs + batch_size - 1) // batch_size
        samples_processed = 0
        
        with tqdm(total=num_batches, desc=f"{model_name} Ep {epoch+1}/{epochs}") as pbar:
            for b in range(0, num_pairs, batch_size):
                b_idx = perm[b:b + batch_size]
                a_ids = anchors[b_idx].to(device, non_blocking=True)
                p_ids = positives[b_idx].to(device, non_blocking=True)
                curr_batch = len(a_ids)
                samples_processed += curr_batch
                
                # Identical GPU negative generator
                g_cuda = torch.Generator(device=device).manual_seed(seed + epoch * 1000 + b)
                n_ids = torch.randint(0, num_nodes, (curr_batch, num_negatives), generator=g_cuda, device=device)
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                    a_coords = model(a_ids)
                    p_coords = model(p_ids)
                    n_coords = model(n_ids)
                    
                    pos_sim = model.compute_cosine_sim(a_coords, p_coords).unsqueeze(1)
                    neg_sim = model.compute_cosine_sim(a_coords, n_coords)
                    
                    logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature
                    targets = torch.zeros(curr_batch, dtype=torch.long, device=device)
                    
                    loss = F.cross_entropy(logits, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                pbar.update(1)
                
                if b % (batch_size * 10) == 0:
                    alloc_gb, res_gb, _ = get_vram_stats(device)
                    pbar.set_postfix(loss=f"{loss.item():.4f}", vram=f"{alloc_gb:.2f}/{res_gb:.2f}GB")

        ep_duration = time.time() - t_ep_start
        avg_loss = total_loss / num_batches
        throughput = samples_processed / ep_duration
        alloc_gb, res_gb, peak_gb = get_vram_stats(device)
        print(f"Epoch {epoch+1}/{epochs} done in {_format_time(ep_duration)} | Loss: {avg_loss:.4f} | Throughput: {throughput/1e3:.1f}k/s | VRAM: {alloc_gb:.2f}GB alloc")

    ckpt_path = f"data/leakfree_{model_name.lower().replace(' ', '_')}_e{epochs}.pt"
    torch.save({'state_dict': model.state_dict(), 'dim': getattr(model, 'dim', 32)}, ckpt_path)
    print(f"[{model_name}] Training complete. Saved checkpoint to {ckpt_path}")
    return model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== RIGOROUS LEAK-FREE BENCHMARK (Torus-32 vs DeepWalk-32) on Device: {device} ===")
    
    edge_list_path = "data/enwiki-2013.txt"
    names_path = "data/enwiki-2013-names.csv"
    
    # Step 1: Load Full Graph
    graph_data = load_phase2_graph(edge_list_path)
    edge_index = graph_data['edge_index']
    num_nodes = graph_data['num_nodes']
    orig_to_compact = graph_data['orig_to_compact']
    title_to_compact, compact_to_title = load_title_mappings(names_path, orig_to_compact)
    
    total_edges = edge_index.shape[1]
    print(f"\n[Leak-Free Split] Splitting total {total_edges:,} edges into 90% Train (~{int(total_edges*0.9):,}) and 10% Test (~{int(total_edges*0.1):,})...")
    
    # Strict Train / Test Edge Split
    g_split = torch.Generator().manual_seed(42)
    edge_perm = torch.randperm(total_edges, generator=g_split)
    num_test = int(0.10 * total_edges)
    num_train = total_edges - num_test
    
    train_edge_idx = edge_perm[:num_train]
    test_edge_idx = edge_perm[num_train:]
    
    train_edge_index = edge_index[:, train_edge_idx]
    test_pos_src = edge_index[0, test_edge_idx]
    test_pos_dst = edge_index[1, test_edge_idx]
    
    # Step 2: Generate Walk Corpus on TRAIN EDGES ONLY (10% Test Edges are Hidden!)
    train_walks_path = "data/phase2_walks_leakfree_train.pt"
    if os.path.exists(train_walks_path):
        print(f"[Walk Corpus] Loading cached leak-free train walks from {train_walks_path}...")
        walks_train = torch.load(train_walks_path, weights_only=False)
    else:
        print("[Walk Corpus] Building alias tables on 90% train edges ONLY...")
        # Compute degrees on train edges
        train_deg = torch.zeros(num_nodes, dtype=torch.float32)
        train_deg.index_add_(0, train_edge_index[0], torch.ones(num_train))
        train_deg.index_add_(0, train_edge_index[1], torch.ones(num_train))
        
        node_weights = 1.0 / train_deg.clamp(min=1.0)
        alias_data = build_alias_tables(train_edge_index, num_nodes, node_weights, cache_path="data/phase2_alias_train.pt")
        
        print("[Walk Corpus] Generating degree-inverse random walks on 90% train graph ONLY...")
        walks_train = generate_walks_biased(alias_data, num_nodes, walk_length=40, walks_per_node=1, device=device)
        torch.save(walks_train, train_walks_path)
        
    pairs_dict = extract_cooccurrence_pairs(walks_train, window_sizes=(2, 5, 10))
    
    # Step 3: Train Both Models with 100% Identical Controls
    print("\n" + "="*85)
    print("TRAINING MODEL 1: Torus-32 Angles (32 Trainable Angles/Node -> 64D Coordinates)")
    print("="*85)
    torus_model = TorusEmbedding(num_nodes=num_nodes, num_angles=32)
    train_rigorous_model(torus_model, "Torus-32 Angles", pairs_dict, num_nodes, epochs=10, batch_size=16384, lr=0.01, seed=42, device=device)
    
    print("\n" + "="*85)
    print("TRAINING MODEL 2: DeepWalk-32 (32 Trainable Floats/Node -> 32D Vector)")
    print("="*85)
    dw32_model = DeepWalkEmbedding(num_nodes=num_nodes, dim=32, normalize=False)
    train_rigorous_model(dw32_model, "DeepWalk-32", pairs_dict, num_nodes, epochs=10, batch_size=16384, lr=0.01, seed=42, device=device)
    
    # Step 4: Evaluate Held-Out Link Prediction & Topic Separation
    print("\n" + "="*85)
    print("STRICT LEAK-FREE EVALUATION RESULTS (10% UNSEEN HELD-OUT TEST EDGES)")
    print("="*85)
    
    torus_auc = compute_link_prediction_auc(torus_model, test_pos_src, test_pos_dst, num_nodes, num_samples=50000, seed=42, device=device)
    dw32_auc = compute_link_prediction_auc(dw32_model, test_pos_src, test_pos_dst, num_nodes, num_samples=50000, seed=42, device=device)
    
    torus_clusters = evaluate_topic_clusters(torus_model, compact_to_title, title_to_compact, device=device)
    dw32_clusters = evaluate_topic_clusters(dw32_model, compact_to_title, title_to_compact, device=device)
    
    print(f"\n{'Metric':<35} | {'Torus-32 Angles':<18} | {'DeepWalk-32':<18} | {'Winner':<10}")
    print("-" * 85)
    print(f"{'Held-Out Link Prediction ROC-AUC':<35} | {torus_auc:^18.4f} | {dw32_auc:^18.4f} | {'TORUS' if torus_auc > dw32_auc else 'DEEPWALK'}")
    
    for cname in ["Physics", "Computer Science", "Sports"]:
        t_delta = torus_clusters[cname]['delta']
        d_delta = dw32_clusters[cname]['delta']
        w = "TORUS" if t_delta > d_delta else "DEEPWALK"
        print(f"{cname + ' Separation Delta':<35} | {t_delta:^+18.4f} | {d_delta:^+18.4f} | {w}")

    # Step 5: Save Single Consolidated Output Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False)
    fig.suptitle("Strict Leak-Free Benchmark (10% Unseen Test Edges Hidden During Walk Gen)\nEqual 32 Trainable Parameters per Node on 4.2M Wikipedia Graph", fontsize=13, fontweight='bold')

    m_labels = ['Torus-32 Angles\n(32 Angles → 64D)', 'DeepWalk-32\n(32 Floats → 32D)']
    colors = ['#9b59b6', '#3498db']
    x_pos = np.arange(len(m_labels))

    # Subplot 1: Link Prediction ROC-AUC
    ax0 = axes[0]
    auc_vals = [torus_auc, dw32_auc]
    bars0 = ax0.bar(x_pos, auc_vals, color=colors, width=0.45, edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars0, auc_vals):
        ax0.annotate(f"{val:.4f}", xy=(bar.get_x() + bar.get_width()/2, val), xytext=(0, 4),
                     textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax0.set_xticks(x_pos)
    ax0.set_xticklabels(m_labels, fontsize=9.5)
    ax0.set_title("Held-Out Link Prediction\nROC-AUC (10% Hidden Edges)", fontsize=11, fontweight='bold')
    ax0.set_ylabel("ROC-AUC Score")
    ax0.set_ylim(0.5, 1.0)
    ax0.grid(axis='y', linestyle=':', alpha=0.6)

    # Subplots 2, 3, 4: Topic Cluster Separation Deltas
    c_keys = ["Physics", "Computer Science", "Sports"]
    for i, cname in enumerate(c_keys, start=1):
        ax = axes[i]
        t_delta = torus_clusters[cname]['delta']
        d_delta = dw32_clusters[cname]['delta']
        vals = [t_delta, d_delta]
        
        bars = ax.bar(x_pos, vals, color=colors, width=0.45, edgecolor='black', linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.annotate(f"{val:+.4f}", xy=(bar.get_x() + bar.get_width()/2, val), xytext=(0, 4 if val>=0 else -14),
                         textcoords="offset points", ha='center', va='bottom' if val>=0 else 'top', fontsize=10, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(m_labels, fontsize=9.5)
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_title(f"Topic Separation Delta:\n{cname}", fontsize=11, fontweight='bold')
        ax.set_ylabel("Pos Sim - Neg Sim" if i == 1 else "")
        ax.set_ylim(-0.25, 0.7)
        ax.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plot_path = "data/rigorous_leakfree_benchmark_plot.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved leak-free benchmark plot to {plot_path}")

if __name__ == "__main__":
    main()
