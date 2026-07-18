import os
import csv
import time
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

def evaluate_clusters(model, compact_to_title, title_to_compact, device="cuda"):
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
                    pos_sims.append((t, sim))
                    
            neg_sims = []
            for t in data["negatives"]:
                cid = find_compact_id(t, title_to_compact)
                if cid is not None:
                    coord = model(torch.tensor([cid], device=device))
                    sim = float(model.compute_cosine_sim(anchor_coord, coord).item())
                    neg_sims.append((t, sim))
                    
            mean_pos = float(np.mean([x[1] for x in pos_sims])) if pos_sims else 0.0
            mean_neg = float(np.mean([x[1] for x in neg_sims])) if neg_sims else 0.0
            delta = mean_pos - mean_neg
            results[cluster_name] = {
                'pos_sims': pos_sims, 'neg_sims': neg_sims,
                'mean_pos': mean_pos, 'mean_neg': mean_neg, 'delta': delta
            }
            
    return results

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph_path = "data/enwiki-2013.txt"
    names_path = "data/enwiki-2013-names.csv"
    torus_ckpt = "data/phase2_checkpoint_e10.pt"
    dw32_ckpt = "data/deepwalk32_checkpoint_e10.pt"

    print("=== Parameter Efficiency Benchmark: Torus-32 Angles vs DeepWalk-32 ===")
    print("Graph: English Wikipedia (~4.2M nodes, ~101M edges)")
    print("Parameter Budget: 32 Trainable Parameters / Node\n")

    # Step 1: Load graph and title mappings
    graph_data = load_phase2_graph(graph_path)
    edge_index = graph_data['edge_index']
    orig_to_compact = graph_data['orig_to_compact']
    num_nodes = graph_data['num_nodes']
    
    title_to_compact, compact_to_title = load_title_mappings(names_path, orig_to_compact)

    # Step 2: Load models
    print(f"Loading Torus-32 model from {torus_ckpt}...")
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

    # Step 3: Evaluate Topic Clusters
    torus_cres = evaluate_clusters(torus_model, compact_to_title, title_to_compact, device=device)
    dw32_cres = evaluate_clusters(dw32_model, compact_to_title, title_to_compact, device=device)

    print("\n" + "="*85)
    print("TOPIC CLUSTER SEPARATION COMPARISON (32 PARAMETERS PER NODE)")
    print("="*85)
    print(f"{'Cluster Domain':<18} | {'Metric':<18} | {'Torus-32 Angles':<18} | {'DeepWalk-32':<18} | {'Winner':<10}")
    print("-" * 85)

    for cname in ["Physics", "Computer Science", "Sports"]:
        t_pos = torus_cres[cname]['mean_pos']
        t_neg = torus_cres[cname]['mean_neg']
        t_delta = torus_cres[cname]['delta']

        d_pos = dw32_cres[cname]['mean_pos']
        d_neg = dw32_cres[cname]['mean_neg']
        d_delta = dw32_cres[cname]['delta']

        winner = "TORUS" if t_delta > d_delta else "DEEPWALK"
        print(f"{cname:<18} | Mean Pos Sim      | {t_pos:^+18.4f} | {d_pos:^+18.4f} |")
        print(f"{'':<18} | Mean Neg Sim      | {t_neg:^+18.4f} | {d_neg:^+18.4f} |")
        print(f"{'':<18} | Separation Delta | {t_delta:^+18.4f} | {d_delta:^+18.4f} | {winner}")
        print("-" * 85)

    # Step 4: Evaluate Walk Distance Monotonicity
    print("\n" + "="*85)
    print("DISTANCE BUCKET SIMILARITY MONOTONICITY")
    print("="*85)
    print("\n--- Torus-32 Angles Distance Buckets ---")
    evaluate_similarity_buckets(torus_model, edge_index, num_nodes, walk_length=30, device=device)

    print("\n--- DeepWalk-32 Distance Buckets ---")
    evaluate_similarity_buckets(dw32_model, edge_index, num_nodes, walk_length=30, device=device)

    # Step 5: Save Comparison Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle("Parameter Efficiency Benchmark (32 Trainable Parameters / Node)\nTorus-32 Angles vs DeepWalk-32 Topic Cluster Separation", fontsize=13, fontweight='bold')

    clusters_list = ["Physics", "Computer Science", "Sports"]
    for ax, cname in zip(axes, clusters_list):
        t_delta = torus_cres[cname]['delta']
        d_delta = dw32_cres[cname]['delta']
        
        bars = ax.bar(['Torus-32 Angles\n(32 Angles → 64D)', 'DeepWalk-32\n(32 Floats → 32D)'],
                      [t_delta, d_delta],
                      color=['#9b59b6', '#3498db'], width=0.5, edgecolor='black', linewidth=0.8)
        
        for bar, val in zip(bars, [t_delta, d_delta]):
            height = bar.get_height()
            ax.annotate(f"{val:+.4f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4 if height >= 0 else -14),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top', fontsize=10, fontweight='bold')

        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_title(f"Domain: {cname}", fontsize=11)
        ax.set_ylabel("Separation Delta (Pos - Neg)" if ax == axes[0] else "")
        ax.set_ylim(-0.1, 0.7)
        ax.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plot_path = "data/torus32_vs_deepwalk32_plot.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved parameter efficiency comparison plot to {plot_path}")

if __name__ == "__main__":
    main()
