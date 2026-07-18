import os
import csv
import time
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg') # Headless backend for saving PNG
import matplotlib.pyplot as plt

from phase2.graph import load_phase2_graph
from phase2.model import TorusEmbedding

def load_title_mappings(names_csv_path, orig_to_compact):
    """Load enwiki-2013-names.csv using robust csv.reader and map title strings to compact node IDs."""
    print(f"Loading page titles from {names_csv_path}...")
    t0 = time.time()
    
    title_to_compact = {}
    compact_to_title = {}
    
    with open(names_csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        header = next(reader, None) # Skip header line ("node_id","name")
        
        for row in reader:
            if len(row) < 2:
                continue
            try:
                orig_id = int(row[0])
                title = row[1]
                if orig_id in orig_to_compact:
                    c_id = orig_to_compact[orig_id]
                    title_to_compact[title.lower()] = c_id
                    compact_to_title[c_id] = title
            except ValueError:
                continue
            
    print(f"Mapped {len(title_to_compact):,} titles to compact node IDs in {time.time()-t0:.2f}s.")
    return title_to_compact, compact_to_title

def find_compact_id(query, title_to_compact):
    """Find compact node ID for a title query string."""
    q_low = query.lower()
    if q_low in title_to_compact:
        return title_to_compact[q_low]
    for t, c_id in title_to_compact.items():
        if t.startswith(q_low):
            return c_id
    for t, c_id in title_to_compact.items():
        if q_low in t:
            return c_id
    return None

def main():
    parser = argparse.ArgumentParser(description="Verify Topic Cluster Cosine Similarities")
    parser.add_argument("--checkpoint", type=str, default="data/phase2_checkpoint_128d_e10.pt", help="Path to model checkpoint")
    parser.add_argument("--edge_list", type=str, default="data/enwiki-2013.txt", help="Path to edge list text file")
    parser.add_argument("--names_csv", type=str, default="data/enwiki-2013-names.csv", help="Path to title CSV file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 1: Load graph and mappings
    graph_data = load_phase2_graph(args.edge_list)
    orig_to_compact = graph_data['orig_to_compact']
    num_nodes = graph_data['num_nodes']
    
    title_to_compact, compact_to_title = load_title_mappings(args.names_csv, orig_to_compact)

    # Step 2: Load trained Phase 2 model
    print(f"Loading trained model checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    num_angles = checkpoint.get('num_angles', 64)
    
    model = TorusEmbedding(num_nodes=num_nodes, num_angles=num_angles)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'theta' in checkpoint:
        model.theta.data.copy_(checkpoint['theta'])
    model.to(device)
    model.eval()

    # Step 3: Define Known Topic Clusters (Sports, Physics, Computer Science)
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

    # Step 4: Evaluate Cluster Cosine Similarities
    print("\n" + "="*85)
    print(f"PHASE 2 FLAT-TORUS EMBEDDING TOPIC CLUSTER EVALUATION (128D / {num_angles} Angles)")
    print("="*85)

    plot_data = {}

    with torch.no_grad():
        for cluster_name, data in clusters.items():
            anchor_title = data["anchor"]
            anchor_cid = find_compact_id(anchor_title, title_to_compact)
            if anchor_cid is None:
                print(f"Warning: Anchor '{anchor_title}' not found in dataset!")
                continue

            anchor_real_name = compact_to_title.get(anchor_cid, anchor_title)
            anchor_coord = model(torch.tensor([anchor_cid], device=device)) # [1, 128]

            print(f"\n--- Cluster: {cluster_name} (Anchor: '{anchor_real_name}') ---")
            print(f"{'Category':<15} | {'Article Title':<35} | {'Cosine Sim':<10}")
            print("-" * 65)

            pos_sims = []
            for pos_title in data["positives"]:
                pos_cid = find_compact_id(pos_title, title_to_compact)
                if pos_cid is not None:
                    pos_real_name = compact_to_title.get(pos_cid, pos_title)
                    pos_coord = model(torch.tensor([pos_cid], device=device))
                    sim = float(model.compute_cosine_sim(anchor_coord, pos_coord).item())
                    pos_sims.append((pos_real_name, sim))
                    print(f"{'In-Cluster':<15} | {pos_real_name:<35} | {sim:^+10.4f}")

            neg_sims = []
            for neg_title in data["negatives"]:
                neg_cid = find_compact_id(neg_title, title_to_compact)
                if neg_cid is not None:
                    neg_real_name = compact_to_title.get(neg_cid, neg_title)
                    neg_coord = model(torch.tensor([neg_cid], device=device))
                    sim = float(model.compute_cosine_sim(anchor_coord, neg_coord).item())
                    neg_sims.append((neg_real_name, sim))
                    print(f"{'Out-of-Cluster':<15} | {neg_real_name:<35} | {sim:^+10.4f}")

            mean_pos = float(np.mean([s[1] for s in pos_sims])) if pos_sims else 0.0
            mean_neg = float(np.mean([s[1] for s in neg_sims])) if neg_sims else 0.0
            print("-" * 65)
            print(f"Mean In-Cluster Sim : {mean_pos:+.4f}")
            print(f"Mean Out-Cluster Sim: {mean_neg:+.4f}  (Difference Delta: {mean_pos - mean_neg:+.4f})")
            
            plot_data[cluster_name] = {
                'anchor': anchor_real_name,
                'pos_sims': pos_sims,
                'neg_sims': neg_sims,
                'mean_pos': mean_pos,
                'mean_neg': mean_neg
            }

    # Step 5: Plot Topic Cluster Similarity Chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(f"Phase 2 Flat-Torus Manifold ({num_angles*2}D): Topic Cluster Cosine Similarity vs Negatives", fontsize=14, fontweight='bold')

    for ax, (cluster_name, pdata) in zip(axes, plot_data.items()):
        pos_titles = [x[0][:18] for x in pdata['pos_sims'][:5]]
        pos_vals = [x[1] for x in pdata['pos_sims'][:5]]
        
        neg_titles = [x[0][:18] for x in pdata['neg_sims'][:5]]
        neg_vals = [x[1] for x in pdata['neg_sims'][:5]]

        labels = pos_titles + [""] + neg_titles
        values = pos_vals + [0] + neg_vals
        colors = ['#2ecc71']*len(pos_vals) + ['white'] + ['#e74c3c']*len(neg_vals)

        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, values, color=colors, width=0.6, edgecolor='black', linewidth=0.8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_title(f"Anchor: '{pdata['anchor']}'\nPos Mean: {pdata['mean_pos']:+.3f} | Neg Mean: {pdata['mean_neg']:+.3f}", fontsize=11)
        ax.set_ylabel("Cosine Similarity" if ax == axes[0] else "")
        ax.set_ylim(-0.4, 1.0)
        ax.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    os.makedirs("data", exist_ok=True)
    plot_path = f"data/cluster_similarity_plot_{num_angles*2}d.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved similarity visualization plot to {plot_path}")

if __name__ == "__main__":
    main()
