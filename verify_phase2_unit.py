import os
import torch
from phase2.model import TorusEmbedding
from phase2.alias import build_alias_tables
from phase2.walker import generate_walks_uniform, generate_walks_biased, extract_cooccurrence_pairs
from phase2.train import train_warmstart, train_infonce
from phase2_eval import evaluate_similarity_buckets

def test_unit():
    print("=== Testing TorusEmbedding Manifold ===")
    num_nodes = 100
    model = TorusEmbedding(num_nodes=num_nodes, num_angles=32)
    
    node_ids = torch.arange(10)
    coords = model.get_coords(node_ids)
    
    assert coords.shape == (10, 64), f"Expected shape (10, 64), got {coords.shape}"
    
    # Check norm is identically sqrt(32)
    norms = coords.pow(2).sum(dim=-1).sqrt()
    expected_norm = (32.0 ** 0.5)
    max_norm_diff = (norms - expected_norm).abs().max().item()
    print(f"Norm check: max diff from sqrt(32) is {max_norm_diff:.8f}")
    assert max_norm_diff < 1e-5, f"Norm check failed! Max diff: {max_norm_diff}"

    # Check dot product equivalence
    u = coords[0]
    v = coords[1]
    dot_val = model.compute_dot(u, v)
    theta_u = model.theta[0]
    theta_v = model.theta[1]
    cos_sum = torch.cos(theta_u - theta_v).sum()
    dot_diff = (dot_val - cos_sum).abs().item()
    print(f"Dot product vs sum cos(theta_u - theta_v) diff: {dot_diff:.8f}")
    assert dot_diff < 1e-5, f"Dot product equivalence failed! Diff: {dot_diff}"

    print("\n=== Testing Graph, Alias & Walker on Synthetic Graph ===")
    # Simple synthetic directed ring graph with 50 nodes
    src = torch.cat([torch.arange(50), torch.arange(50)])
    dst = torch.cat([(torch.arange(50) + 1) % 50, (torch.arange(50) + 2) % 50])
    edge_index = torch.stack([src, dst], dim=0)
    
    total_degree = torch.bincount(src, minlength=50) + torch.bincount(dst, minlength=50)
    node_weights = 1.0 / total_degree.float().clamp(min=1.0)
    
    # Alias tables
    alias_data = build_alias_tables(edge_index, 50, node_weights)
    
    # Walks
    u_walks = generate_walks_uniform(edge_index, 50, walk_length=10, walks_per_node=2)
    b_walks = generate_walks_biased(alias_data, 50, walk_length=10, walks_per_node=2)
    
    assert u_walks.shape == (100, 11), f"Uniform walks shape mismatch: {u_walks.shape}"
    assert b_walks.shape == (100, 11), f"Biased walks shape mismatch: {b_walks.shape}"
    
    # Co-occurrence pairs
    pairs = extract_cooccurrence_pairs(b_walks, window_sizes=(1, 2, 5))
    assert 'anchors' in pairs and 'positives' in pairs
    print(f"Extracted {len(pairs['anchors'])} synthetic pairs.")

    print("\n=== Testing Warm-Start Pretraining ===")
    train_warmstart(model, edge_index, num_nodes=50, epochs=1, batch_size=16, device="cpu")

    print("\n=== Testing InfoNCE Training ===")
    train_infonce(model, pairs, num_nodes=50, epochs=1, batch_size=16, device="cpu", checkpoint_prefix="data/test_ckpt")

    print("\n=== Testing Distance Bucket Evaluation ===")
    results = evaluate_similarity_buckets(model, edge_index, num_nodes=50, walk_length=5, num_eval_walks=20, device="cpu")
    
    print("\nUnit test verification passed successfully!")

if __name__ == "__main__":
    test_unit()
