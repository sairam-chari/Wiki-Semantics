import os
import argparse
import numpy as np
import torch
from phase2.graph import load_phase2_graph
from phase2.walker import generate_walks_uniform
from phase2.model import TorusEmbedding

def evaluate_similarity_buckets(model, edge_index, num_nodes, walk_length=30, num_eval_walks=50000, device="cuda"):
    """Evaluate trained TorusEmbedding cosine similarity across walk-distance buckets.
    
    Args:
        model (TorusEmbedding): Loaded model.
        edge_index (torch.LongTensor): Graph edge index.
        num_nodes (int): Node count.
        walk_length (int): Evaluation walk length.
        num_eval_walks (int): Sampled walks for evaluation.
        device (str): Compute device.
    """
    model.to(device)
    model.eval()
    
    print(f"\n[Phase2 Eval] Generating {num_eval_walks:,} evaluation walks (length={walk_length})...")
    # Sample random start nodes for evaluation
    eval_start_nodes = torch.randint(0, num_nodes, (num_eval_walks,))
    
    try:
        from torch_cluster import random_walk as tc_random_walk
        row, col = edge_index[0], edge_index[1]
        eval_walks = tc_random_walk(row, col, eval_start_nodes, walk_length=walk_length)
    except ImportError:
        # Fallback to uniform walker
        eval_walks = generate_walks_uniform(edge_index, num_nodes, walk_length=walk_length, walks_per_node=1)[:num_eval_walks]

    eval_distances = [1, 2, 5, 10, 20]
    results = {}
    
    print("\n[Phase2 Eval] Computing cosine similarity metrics across distance buckets:")
    print("=" * 65)
    print(f"{'Distance Bucket':<20} | {'Mean Cosine':<12} | {'Std Dev':<10} | {'Min / Max':<15}")
    print("-" * 65)
    
    with torch.no_grad():
        for d in eval_distances:
            if d > walk_length:
                continue
            anchors = eval_walks[:, 0]
            targets = eval_walks[:, d]
            
            # Filter self-loops
            valid = anchors != targets
            a_ids = anchors[valid].to(device)
            t_ids = targets[valid].to(device)
            
            a_coords = model(a_ids)
            t_coords = model(t_ids)
            
            cos_sims = model.compute_cosine_sim(a_coords, t_coords).cpu().numpy()
            
            mean_sim = float(np.mean(cos_sims))
            std_sim = float(np.std(cos_sims))
            min_sim = float(np.min(cos_sims))
            max_sim = float(np.max(cos_sims))
            
            results[f"dist_{d}"] = mean_sim
            print(f"{f'Walk Step {d}':<20} | {mean_sim:^12.4f} | {std_sim:^10.4f} | {min_sim:.2f} / {max_sim:.2f}")

        # Random negative pair bucket
        rand_a = torch.randint(0, num_nodes, (num_eval_walks,), device=device)
        rand_b = torch.randint(0, num_nodes, (num_eval_walks,), device=device)
        valid = rand_a != rand_b
        
        a_coords = model(rand_a[valid])
        b_coords = model(rand_b[valid])
        rand_sims = model.compute_cosine_sim(a_coords, b_coords).cpu().numpy()
        
        mean_rand = float(np.mean(rand_sims))
        std_rand = float(np.std(rand_sims))
        min_rand = float(np.min(rand_sims))
        max_rand = float(np.max(rand_sims))
        results["random"] = mean_rand
        
        print(f"{'Random / Far-Apart':<20} | {mean_rand:^12.4f} | {std_rand:^10.4f} | {min_rand:.2f} / {max_rand:.2f}")
        print("=" * 65)

    # Sanity check monotonicity
    monotonic = True
    d_keys = sorted([k for k in results.keys() if k.startswith("dist_")])
    for i in range(len(d_keys) - 1):
        if results[d_keys[i]] < results[d_keys[i+1]]:
            monotonic = False
            break
    if results.get("dist_1", 0) <= results.get("random", 0):
        monotonic = False

    if monotonic:
        print("[Phase2 Eval PASSED] Cosine similarity monotonically decreases with graph distance!")
    else:
        print("[Phase2 Eval WARNING] Cosine similarity is not strictly monotonic across distance buckets; check training duration.")
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 Torus Embedding Distance Buckets")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint .pt file")
    parser.add_argument("--edge_list", type=str, default="data/enwiki-2013.txt", help="Edge list file path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load graph
    graph_data = load_phase2_graph(args.edge_list)
    edge_index = graph_data['edge_index']
    num_nodes = graph_data['num_nodes']
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    num_angles = checkpoint.get('num_angles', 32)
    model = TorusEmbedding(num_nodes=num_nodes, num_angles=num_angles)
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'theta' in checkpoint:
        model.theta.data.copy_(checkpoint['theta'])
        
    evaluate_similarity_buckets(model, edge_index, num_nodes, device=device)
