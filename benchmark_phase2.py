import time
import torch
from phase2.alias import build_alias_tables
from phase2.walker import generate_walks_uniform, generate_walks_biased, extract_cooccurrence_pairs
from phase2.model import TorusEmbedding
from phase2.train import train_warmstart, train_infonce
from phase2_eval import evaluate_similarity_buckets

def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Phase 2 Performance & VRAM Benchmark on {device} ===")
    
    num_nodes = 1_000_000 # 1 Million nodes
    num_edges = 20_000_000 # 20 Million edges
    
    print(f"Generating synthetic graph benchmark ({num_nodes:,} nodes, {num_edges:,} directed edges)...")
    t0 = time.time()
    src = torch.randint(0, num_nodes, (num_edges,), dtype=torch.int64)
    dst = torch.randint(0, num_nodes, (num_edges,), dtype=torch.int64)
    edge_index = torch.stack([src, dst], dim=0)
    
    total_degree = torch.bincount(src, minlength=num_nodes) + torch.bincount(dst, minlength=num_nodes)
    node_weights = 1.0 / total_degree.float().clamp(min=1.0)
    print(f"Graph generated in {time.time()-t0:.2f}s.")
    
    # 1. Alias Tables
    print("\n1. Testing Alias Table Construction...")
    alias_data = build_alias_tables(edge_index, num_nodes, node_weights)
    
    # 2. Walk Generation
    print("\n2. Testing GPU-Accelerated Biased Walk Generation...")
    walks = generate_walks_biased(
        alias_data, num_nodes, walk_length=20, walks_per_node=1, batch_size=256000, device=device
    )
    
    # 3. Pair Extraction
    print("\n3. Extracting Walk Co-occurrence Pairs...")
    pairs_dict = extract_cooccurrence_pairs(walks, window_sizes=(2, 5, 10))
    
    # 4. Model Initialization & Memory Overhead
    print("\n4. Initializing TorusEmbedding Model...")
    model = TorusEmbedding(num_nodes=num_nodes, num_angles=32)
    
    if torch.cuda.is_available():
        alloc_gb = torch.cuda.memory_allocated(device) / 1e9
        res_gb = torch.cuda.memory_reserved(device) / 1e9
        print(f"Model VRAM Allocation: {alloc_gb:.2f} GB (Reserved: {res_gb:.2f} GB)")

    # 5. Warm-Start Training Benchmark
    print("\n5. Running Spring Warm-Start Benchmark (1 Epoch)...")
    train_warmstart(
        model=model,
        edge_index=edge_index,
        num_nodes=num_nodes,
        epochs=1,
        batch_size=32768,
        lr=0.01,
        num_negatives=5,
        use_amp=True,
        device=device
    )
    
    # 6. InfoNCE Main Training Benchmark
    print("\n6. Running InfoNCE Training Benchmark (2 Epochs)...")
    train_infonce(
        model=model,
        pairs_dict=pairs_dict,
        num_nodes=num_nodes,
        epochs=2,
        batch_size=16384,
        lr=0.01,
        num_negatives=10,
        temperature=0.1,
        use_amp=True,
        device=device,
        checkpoint_prefix="data/benchmark_ckpt"
    )
    
    # 7. Evaluation Benchmark
    print("\n7. Running Similarity Bucket Evaluation...")
    evaluate_similarity_buckets(
        model=model,
        edge_index=edge_index,
        num_nodes=num_nodes,
        walk_length=20,
        num_eval_walks=50000,
        device=device
    )
    
    print("\n=== Benchmark Completed Successfully ===")

if __name__ == "__main__":
    run_benchmark()
