import os
import time
import csv
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from phase2.graph import load_phase2_graph
from phase2.model import TorusEmbedding
from phase2.deepwalk import DeepWalkEmbedding
from phase2_eval import evaluate_similarity_buckets
from verify_topic_clusters import load_title_mappings, find_compact_id

def compute_link_prediction_auc(model, edge_index, num_nodes, num_samples=50000, device="cuda"):
    """Compute held-out Link Prediction ROC-AUC using vector rank-sum math without sklearn."""
    model.eval()
    num_edges = edge_index.shape[1]
    
    # 1. Sample held-out positive test edges
    pos_idx = torch.randperm(num_edges)[:num_samples]
    pos_src = edge_index[0, pos_idx].to(device)
    pos_dst = edge_index[1, pos_idx].to(device)
    
    # 2. Sample negative test edges
    neg_src = torch.randint(0, num_nodes, (num_samples,), device=device)
    neg_dst = torch.randint(0, num_nodes, (num_samples,), device=device)
    
    with torch.no_grad():
        pos_src_coords = model(pos_src)
        pos_dst_coords = model(pos_dst)
        pos_sims = model.compute_cosine_sim(pos_src_coords, pos_dst_coords)
        
        neg_src_coords = model(neg_src)
        neg_dst_coords = model(neg_dst)
        neg_sims = model.compute_cosine_sim(neg_src_coords, neg_dst_coords)
        
    # Wilcoxon-Mann-Whitney rank AUC score: P(pos > neg) + 0.5 * P(pos == neg)
    # Compute in chunks of 5000 to avoid [50k, 50k] matrix memory
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
            
    auc = (gt_count + 0.5 * eq_count) / total_pairs
    return auc

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph_path = "data/enwiki-2013.txt"
    names_path = "data/enwiki-2013-names.csv"
    torus_ckpt = "data/phase2_checkpoint_e10.pt"
    dw32_ckpt = "data/deepwalk32_checkpoint_e10.pt"

    print("=== Torus-32 Angles vs. DeepWalk-32 Benchmark (32 Parameters / Node) ===")
    
    # 1. Load graph and title mappings
    graph_data = load_phase2_graph(graph_path)
    edge_index = graph_data['edge_index']
    num_nodes = graph_data['num_nodes']
    orig_to_compact = graph_data['orig_to_compact']

    title_to_compact, compact_to_title = load_title_mappings(names_path, orig_to_compact)

    # 2. Load models
    print(f"\nLoading Torus-32 model from {torus_ckpt}...")
    torus_model = TorusEmbedding(num_nodes=num_nodes, num_angles=32)
    cp_t = torch.load(torus_ckpt, map_location=device, weights_only=False)
    if 'state_dict' in cp_t: torus_model.load_state_dict(cp_t['state_dict'])
    elif 'theta' in cp_t: torus_model.theta.data.copy_(cp_t['theta'])
    torus_model.to(device)

    print(f"Loading DeepWalk-32 model from {dw32_ckpt}...")
    dw32_model = DeepWalkEmbedding(num_nodes=num_nodes, dim=32, normalize=False)
    cp_dw = torch.load(dw32_ckpt, map_location=device, weights_only=False)
    dw32_model.load_state_dict(cp_dw['state_dict'])
    dw32_model.to(device)

    # 3. Evaluate Held-Out Link Prediction ROC-AUC
    print("\n[Metric 1] Evaluating Held-Out Link Prediction ROC-AUC (50,000 test edges)...")
    torus_auc = compute_link_prediction_auc(torus_model, edge_index, num_nodes, num_samples=50000, device=device)
    dw32_auc = compute_link_prediction_auc(dw32_model, edge_index, num_nodes, num_samples=50000, device=device)
    print(f"  Torus-32 Angles ROC-AUC : {torus_auc:.4f}")
    print(f"  DeepWalk-32 ROC-AUC     : {dw32_auc:.4f}")

    # 4. Evaluate Topic Cluster Separation
    print("\n[Metric 2] Evaluating Topic Cluster Separation Deltas...")
    torus_clusters = evaluate_topic_clusters(torus_model, compact_to_title, title_to_compact, device=device)
    dw32_clusters = evaluate_topic_clusters(dw32_model, compact_to_title, title_to_compact, device=device)

    print("\n" + "="*85)
    print("HEAD-TO-HEAD BENCHMARK (32 LEARNED PARAMETERS / NODE)")
    print("="*85)
    print(f"{'Metric':<32} | {'Torus-32 Angles':<18} | {'DeepWalk-32':<18} | {'Winner':<10}")
    print("-" * 85)
    print(f"{'Link Prediction ROC-AUC':<32} | {torus_auc:^18.4f} | {dw32_auc:^18.4f} | {'TORUS' if torus_auc > dw32_auc else 'DEEPWALK'}")
    
    for cname in ["Physics", "Computer Science", "Sports"]:
        t_delta = torus_clusters[cname]['delta']
        d_delta = dw32_clusters[cname]['delta']
        w = "TORUS" if t_delta > d_delta else "DEEPWALK"
        print(f"{cname + ' Separation Delta':<32} | {t_delta:^+18.4f} | {d_delta:^+18.4f} | {w}")

    # 5. Generate Single Consolidated PNG Figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False)
    fig.suptitle("Parameter Efficiency Benchmark (Equal 32 Learned Parameters per Node)\nTorus-32 Angles vs DeepWalk-32 on 4.2M Wikipedia Graph", fontsize=13, fontweight='bold')

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
    ax0.set_title("Held-Out Link Prediction\nROC-AUC Score", fontsize=11, fontweight='bold')
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
    os.makedirs("data", exist_ok=True)
    plot_path = "data/torus32_vs_deepwalk32_final_plot.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved consolidated benchmark plot to {plot_path}")

if __name__ == "__main__":
    main()
