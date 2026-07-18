import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import argparse
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from phase2.graph import load_phase2_graph
from phase2.alias import build_alias_tables
from phase2.walker import generate_walks_biased, extract_cooccurrence_pairs
from phase2.model import TorusEmbedding
from phase2.deepwalk import DeepWalkEmbedding
from phase2.train import train_infonce, train_warmstart
from phase2.train_deepwalk import train_deepwalk_infonce
from phase2_eval import evaluate_similarity_buckets
from verify_topic_clusters import load_title_mappings, find_compact_id

def cleanup_cuda():
    """Clear CUDA cache and run garbage collection between models."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def evaluate_model_clusters(model, compact_to_title, title_to_compact, device="cuda"):
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
    parser = argparse.ArgumentParser(description="Baseline Benchmark Execution & Analysis")
    parser.add_argument("--edge_list", type=str, default="data/enwiki-2013.txt")
    parser.add_argument("--names_csv", type=str, default="data/enwiki-2013-names.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--skip_training", action="store_true", help="Skip training and evaluate existing checkpoints")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Starting DeepWalk vs. Torus Baseline Benchmark on Device: {device} ===")

    # Step 1: Load graph and mappings
    graph_data = load_phase2_graph(args.edge_list)
    edge_index = graph_data['edge_index']
    num_nodes = graph_data['num_nodes']
    total_degree = graph_data['total_degree']
    orig_to_compact = graph_data['orig_to_compact']

    title_to_compact, compact_to_title = load_title_mappings(args.names_csv, orig_to_compact)

    # Step 2: Load or generate walk co-occurrence corpus
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
    dw64_ckpt = "data/deepwalk64_checkpoint_e10.pt"
    dw64norm_ckpt = "data/deepwalk64norm_checkpoint_e10.pt"

    # Step 3: Train DeepWalk models if not skipping
    if not args.skip_training:
        cleanup_cuda()
        # DeepWalk-32
        if not os.path.exists(dw32_ckpt):
            print("\n" + "="*70)
            print("TRAINING BASELINE 1: DeepWalk-32 (32 Floats/Node, 32-D Output)")
            print("="*70)
            dw32 = DeepWalkEmbedding(num_nodes=num_nodes, dim=32, normalize=False)
            train_deepwalk_infonce(
                model=dw32, pairs_dict=pairs_dict, num_nodes=num_nodes, epochs=args.epochs,
                batch_size=32768, lr=args.lr, checkpoint_prefix="data/deepwalk32_checkpoint", device=device
            )
            del dw32
            cleanup_cuda()

        # DeepWalk-64
        if not os.path.exists(dw64_ckpt):
            cleanup_cuda()
            print("\n" + "="*70)
            print("TRAINING BASELINE 2: DeepWalk-64 (64 Floats/Node, 64-D Output)")
            print("="*70)
            dw64 = DeepWalkEmbedding(num_nodes=num_nodes, dim=64, normalize=False)
            train_deepwalk_infonce(
                model=dw64, pairs_dict=pairs_dict, num_nodes=num_nodes, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr, checkpoint_prefix="data/deepwalk64_checkpoint", device=device
            )
            del dw64
            cleanup_cuda()

        # DeepWalk-64-Norm
        if not os.path.exists(dw64norm_ckpt):
            cleanup_cuda()
            print("\n" + "="*70)
            print("TRAINING BASELINE 3: DeepWalk-64-Norm (64 Floats/Node, Unit-Sphere Normalization)")
            print("="*70)
            dw64norm = DeepWalkEmbedding(num_nodes=num_nodes, dim=64, normalize=True)
            train_deepwalk_infonce(
                model=dw64norm, pairs_dict=pairs_dict, num_nodes=num_nodes, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr, checkpoint_prefix="data/deepwalk64norm_checkpoint", device=device
            )
            del dw64norm
            cleanup_cuda()

    # Step 4: Load All Checkpoints for Evaluation
    cleanup_cuda()
    models = {}

    # Torus-32
    if os.path.exists(torus_ckpt):
        torus_model = TorusEmbedding(num_nodes=num_nodes, num_angles=32)
        cp = torch.load(torus_ckpt, map_location=device, weights_only=False)
        if 'state_dict' in cp: torus_model.load_state_dict(cp['state_dict'])
        elif 'theta' in cp: torus_model.theta.data.copy_(cp['theta'])
        models['Torus-32 Angles (64D Derived)'] = torus_model.to(device)

    # DW-32
    if os.path.exists(dw32_ckpt):
        dw32_model = DeepWalkEmbedding(num_nodes=num_nodes, dim=32, normalize=False)
        cp = torch.load(dw32_ckpt, map_location=device, weights_only=False)
        dw32_model.load_state_dict(cp['state_dict'])
        models['DeepWalk-32 (32D Unbounded)'] = dw32_model.to(device)

    # DW-64
    if os.path.exists(dw64_ckpt):
        dw64_model = DeepWalkEmbedding(num_nodes=num_nodes, dim=64, normalize=False)
        cp = torch.load(dw64_ckpt, map_location=device, weights_only=False)
        dw64_model.load_state_dict(cp['state_dict'])
        models['DeepWalk-64 (64D Unbounded)'] = dw64_model.to(device)

    # DW-64-Norm
    if os.path.exists(dw64norm_ckpt):
        dw64norm_model = DeepWalkEmbedding(num_nodes=num_nodes, dim=64, normalize=True)
        cp = torch.load(dw64norm_ckpt, map_location=device, weights_only=False)
        dw64norm_model.load_state_dict(cp['state_dict'])
        models['DeepWalk-64-Norm (64D Unit Sphere)'] = dw64norm_model.to(device)

    # Step 5: Comparative Evaluation & Results Display
    print("\n" + "="*95)
    print("EMPIRICAL BENCHMARK EVALUATION RESULTS")
    print("="*95)

    all_cluster_results = {}
    for name, m in models.items():
        print(f"\nEvaluating Topic Clusters for: {name}...")
        cres = evaluate_model_clusters(m, compact_to_title, title_to_compact, device=device)
        all_cluster_results[name] = cres
        for cname, stats in cres.items():
            print(f"  {cname:<18} | Pos: {stats['pos']:+.4f} | Neg: {stats['neg']:+.4f} | Delta: {stats['delta']:+.4f}")

    # Plot Comparative Bar Chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("Model Benchmark: Torus-32 Angles vs. DeepWalk Baselines\n(Topic Cluster Separation Delta: Pos - Neg)", fontsize=14, fontweight='bold')

    m_names = list(models.keys())
    cluster_keys = ["Physics", "Computer Science", "Sports"]

    for ax, ckey in zip(axes, cluster_keys):
        deltas = [all_cluster_results[m][ckey]['delta'] if ckey in all_cluster_results[m] else 0.0 for m in m_names]
        short_names = ["Torus-32\n(32 Angles, 64D)", "DeepWalk-32\n(32 Floats, 32D)", "DeepWalk-64\n(64 Floats, 64D)", "DeepWalk-64-Norm\n(64 Floats, 64D)"]
        
        colors = ['#9b59b6', '#3498db', '#2ecc71', '#e67e22']
        x_pos = np.arange(len(short_names))
        
        bars = ax.bar(x_pos, deltas, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
        
        for bar, val in zip(bars, deltas):
            height = bar.get_height()
            ax.annotate(f"{val:+.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -12),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top', fontsize=9, fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(short_names, rotation=25, ha='right', fontsize=9)
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_title(f"Cluster: {ckey}", fontsize=12)
        ax.set_ylabel("Separation Delta (Pos - Neg)" if ax == axes[0] else "")
        ax.set_ylim(-0.2, 0.8)
        ax.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    os.makedirs("data", exist_ok=True)
    plot_path = "data/baseline_comparison_plot.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved comparative benchmark plot to {plot_path}")

if __name__ == "__main__":
    main()
