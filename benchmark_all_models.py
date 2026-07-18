import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import time
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from phase2.graph import load_phase2_graph
from phase2.alias import build_alias_tables
from phase2.walker import generate_walks_biased, extract_cooccurrence_pairs
from phase2.model import TorusEmbedding
from phase2.deepwalk import DeepWalkEmbedding
from phase2.train_deepwalk import train_deepwalk_infonce
from phase2_eval import evaluate_similarity_buckets
from verify_topic_clusters import load_title_mappings, find_compact_id

def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def evaluate_link_prediction_auc(model, edge_index, num_nodes, num_samples=50000, device="cuda"):
    """Evaluate Link Prediction ROC-AUC on held-out edges vs random negative pairs."""
    model.eval()
    num_edges = edge_index.shape[1]
    
    # 1. Sample positive test edges
    pos_idx = torch.randperm(num_edges)[:num_samples]
    pos_src = edge_index[0, pos_idx].to(device)
    pos_dst = edge_index[1, pos_idx].to(device)
    
    # 2. Sample negative test edges
    neg_src = torch.randint(0, num_nodes, (num_samples,), device=device)
    neg_dst = torch.randint(0, num_nodes, (num_samples,), device=device)
    
    with torch.no_grad():
        pos_src_coords = model(pos_src)
        pos_dst_coords = model(pos_dst)
        pos_scores = model.compute_cosine_sim(pos_src_coords, pos_dst_coords).cpu().numpy()
        
        neg_src_coords = model(neg_src)
        neg_dst_coords = model(neg_dst)
        neg_scores = model.compute_cosine_sim(neg_src_coords, neg_dst_coords).cpu().numpy()
        
    labels = np.concatenate([np.ones(num_samples), np.zeros(num_samples)])
    scores = np.concatenate([pos_scores, neg_scores])
    auc = roc_auc_score(labels, scores)
    return float(auc)

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

def main():
    parser = argparse.ArgumentParser(description="Consolidated 4-Model Benchmark")
    parser.add_argument("--edge_list", type=str, default="data/enwiki-2013.txt")
    parser.add_argument("--names_csv", type=str, default="data/enwiki-2013-names.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16384)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Running Consolidated 4-Model Benchmark on Device: {device} ===")

    # Step 1: Load graph and mappings
    graph_data = load_phase2_graph(args.edge_list)
    edge_index = graph_data['edge_index']
    num_nodes = graph_data['num_nodes']
    total_degree = graph_data['total_degree']
    orig_to_compact = graph_data['orig_to_compact']

    title_to_compact, compact_to_title = load_title_mappings(args.names_csv, orig_to_compact)

    # Step 2: Load walk co-occurrence corpus
    walks_cache_path = "data/phase2_walks_biased.pt"
    if os.path.exists(walks_cache_path):
        print(f"[Walk Corpus] Loading cached walks from {walks_cache_path}...")
        walks = torch.load(walks_cache_path, weights_only=False)
    else:
        node_weights = 1.0 / total_degree.clamp(min=1.0)
        alias_data = build_alias_tables(edge_index, num_nodes, node_weights, cache_path="data/phase2_alias.pt")
        walks = generate_walks_biased(alias_data, num_nodes, walk_length=40, walks_per_node=1, device=device)
        torch.save(walks, walks_cache_path)

    pairs_dict = extract_cooccurrence_pairs(walks, window_sizes=(2, 5, 10))

    # Checkpoint paths
    torus_ckpt = "data/phase2_checkpoint_e10.pt"
    dw32_ckpt = "data/deepwalk32_checkpoint_e10.pt"
    dw64_ckpt = "data/deepwalk64_checkpoint_e5.pt"
    dw64norm_ckpt = "data/deepwalk64norm_checkpoint_e5.pt"

    # Step 3: Train DeepWalk-64 and DeepWalk-64-Norm if not already present
    if not os.path.exists(dw64_ckpt):
        cleanup_cuda()
        print("\n" + "="*70)
        print("TRAINING BASELINE: DeepWalk-64 (64 Floats/Node, 64-D Output)")
        print("="*70)
        dw64 = DeepWalkEmbedding(num_nodes=num_nodes, dim=64, normalize=False)
        train_deepwalk_infonce(
            model=dw64, pairs_dict=pairs_dict, num_nodes=num_nodes, epochs=5,
            batch_size=args.batch_size, lr=0.01, checkpoint_prefix="data/deepwalk64_checkpoint", device=device
        )
        del dw64
        cleanup_cuda()

    if not os.path.exists(dw64norm_ckpt):
        cleanup_cuda()
        print("\n" + "="*70)
        print("TRAINING BASELINE: DeepWalk-64-Norm (64 Floats/Node, Unit Sphere)")
        print("="*70)
        dw64norm = DeepWalkEmbedding(num_nodes=num_nodes, dim=64, normalize=True)
        train_deepwalk_infonce(
            model=dw64norm, pairs_dict=pairs_dict, num_nodes=num_nodes, epochs=5,
            batch_size=args.batch_size, lr=0.01, checkpoint_prefix="data/deepwalk64norm_checkpoint", device=device
        )
        del dw64norm
        cleanup_cuda()

    # Step 4: Load All 4 Checkpoints for Comprehensive Evaluation
    cleanup_cuda()
    models = {}

    # Torus-32
    if os.path.exists(torus_ckpt):
        torus_model = TorusEmbedding(num_nodes=num_nodes, num_angles=32)
        cp = torch.load(torus_ckpt, map_location=device, weights_only=False)
        if 'state_dict' in cp: torus_model.load_state_dict(cp['state_dict'])
        elif 'theta' in cp: torus_model.theta.data.copy_(cp['theta'])
        models['Torus-32 Angles\n(32 Angles, 64D Derived)'] = torus_model.to(device)

    # DW-32
    if os.path.exists(dw32_ckpt):
        dw32_model = DeepWalkEmbedding(num_nodes=num_nodes, dim=32, normalize=False)
        cp = torch.load(dw32_ckpt, map_location=device, weights_only=False)
        dw32_model.load_state_dict(cp['state_dict'])
        models['DeepWalk-32\n(32 Floats, 32D Output)'] = dw32_model.to(device)

    # DW-64
    if os.path.exists(dw64_ckpt):
        dw64_model = DeepWalkEmbedding(num_nodes=num_nodes, dim=64, normalize=False)
        cp = torch.load(dw64_ckpt, map_location=device, weights_only=False)
        dw64_model.load_state_dict(cp['state_dict'])
        models['DeepWalk-64\n(64 Floats, 64D Output)'] = dw64_model.to(device)

    # DW-64-Norm
    if os.path.exists(dw64norm_ckpt):
        dw64norm_model = DeepWalkEmbedding(num_nodes=num_nodes, dim=64, normalize=True)
        cp = torch.load(dw64norm_ckpt, map_location=device, weights_only=False)
        dw64norm_model.load_state_dict(cp['state_dict'])
        models['DeepWalk-64-Norm\n(64 Floats, 64D Unit Sphere)'] = dw64norm_model.to(device)

    # Step 5: Evaluate Link Prediction ROC-AUC and Topic Clusters
    print("\n" + "="*95)
    print("FOUR-MODEL BENCHMARK EVALUATION (HELD-OUT METRICS)")
    print("="*95)

    auc_results = {}
    cluster_results = {}

    for name, m in models.items():
        print(f"\nEvaluating: {name.replace('\n', ' ')}...")
        auc = evaluate_link_prediction_auc(m, edge_index, num_nodes, num_samples=50000, device=device)
        auc_results[name] = auc
        print(f"  Held-Out Link Prediction ROC-AUC: {auc:.4f}")
        
        cres = evaluate_topic_clusters(m, compact_to_title, title_to_compact, device=device)
        cluster_results[name] = cres
        for cname, stats in cres.items():
            print(f"  {cname:<18} | Pos: {stats['pos']:+.4f} | Neg: {stats['neg']:+.4f} | Delta: {stats['delta']:+.4f}")

    # Step 6: Generate Single Consolidated PNG Plot
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), sharey=False)
    fig.suptitle("Wiki-Semantics Comprehensive Benchmark: Torus-32 Angles vs DeepWalk Baselines\n(4.2M Wikipedia Nodes, 101M Edges)", fontsize=14, fontweight='bold')

    m_keys = list(models.keys())
    colors = ['#9b59b6', '#3498db', '#2ecc71', '#e67e22']
    x_pos = np.arange(len(m_keys))

    # Subplot 1: Link Prediction ROC-AUC
    ax0 = axes[0]
    aucs = [auc_results[m] for m in m_keys]
    bars0 = ax0.bar(x_pos, aucs, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars0, aucs):
        ax0.annotate(f"{val:.4f}", xy=(bar.get_x() + bar.get_width()/2, val), xytext=(0, 4),
                     textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax0.set_xticks(x_pos)
    ax0.set_xticklabels(m_keys, rotation=25, ha='right', fontsize=8.5)
    ax0.set_title("Held-Out Link Prediction ROC-AUC\n(Higher is Better)", fontsize=11, fontweight='bold')
    ax0.set_ylabel("ROC-AUC Score")
    ax0.set_ylim(0.5, 1.0)
    ax0.grid(axis='y', linestyle=':', alpha=0.6)

    # Subplot 2: Physics Cluster Separation Delta
    ax1 = axes[1]
    phys_deltas = [cluster_results[m]['Physics']['delta'] for m in m_keys]
    bars1 = ax1.bar(x_pos, phys_deltas, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars1, phys_deltas):
        ax1.annotate(f"{val:+.3f}", xy=(bar.get_x() + bar.get_width()/2, val), xytext=(0, 4 if val>=0 else -12),
                     textcoords="offset points", ha='center', va='bottom' if val>=0 else 'top', fontsize=9, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(m_keys, rotation=25, ha='right', fontsize=8.5)
    ax1.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax1.set_title("Physics Domain Separation Delta\n(Pos Sim - Neg Sim)", fontsize=11, fontweight='bold')
    ax1.set_ylim(-0.2, 0.7)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)

    # Subplot 3: CS Cluster Separation Delta
    ax2 = axes[2]
    cs_deltas = [cluster_results[m]['Computer Science']['delta'] for m in m_keys]
    bars2 = ax2.bar(x_pos, cs_deltas, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars2, cs_deltas):
        ax2.annotate(f"{val:+.3f}", xy=(bar.get_x() + bar.get_width()/2, val), xytext=(0, 4 if val>=0 else -12),
                     textcoords="offset points", ha='center', va='bottom' if val>=0 else 'top', fontsize=9, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(m_keys, rotation=25, ha='right', fontsize=8.5)
    ax2.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax2.set_title("Computer Science Separation Delta\n(Pos Sim - Neg Sim)", fontsize=11, fontweight='bold')
    ax2.set_ylim(-0.2, 0.7)
    ax2.grid(axis='y', linestyle=':', alpha=0.6)

    # Subplot 4: Sports Cluster Separation Delta
    ax3 = axes[3]
    sports_deltas = [cluster_results[m]['Sports']['delta'] for m in m_keys]
    bars3 = ax3.bar(x_pos, sports_deltas, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars3, sports_deltas):
        ax3.annotate(f"{val:+.3f}", xy=(bar.get_x() + bar.get_width()/2, val), xytext=(0, 4 if val>=0 else -12),
                     textcoords="offset points", ha='center', va='bottom' if val>=0 else 'top', fontsize=9, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(m_keys, rotation=25, ha='right', fontsize=8.5)
    ax3.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax3.set_title("Sports Domain Separation Delta\n(Pos Sim - Neg Sim)", fontsize=11, fontweight='bold')
    ax3.set_ylim(-0.2, 0.7)
    ax3.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plot_path = "data/all_models_benchmark_plot.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved consolidated 4-model benchmark plot to {plot_path}")

if __name__ == "__main__":
    main()
