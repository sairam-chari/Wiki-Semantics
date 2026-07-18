import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import time
import torch

from phase2.graph import load_phase2_graph
from phase2.alias import build_alias_tables
from phase2.walker import generate_walks_uniform, generate_walks_biased, extract_cooccurrence_pairs
from phase2.model import TorusEmbedding
from phase2.train import train_warmstart, train_infonce
from phase2_eval import evaluate_similarity_buckets

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Flat-Torus Node Embedding Pipeline")
    parser.add_argument("--edge_list", type=str, default="data/enwiki-2013.txt", help="Path to edge list text file")
    parser.add_argument("--uniform_only", action="store_true", help="Use pure uniform random walks instead of degree-inverse biased walks")
    parser.add_argument("--skip_warmstart", action="store_true", help="Skip spring warm-start pretraining phase")
    parser.add_argument("--warmstart_epochs", type=int, default=1, help="Number of spring warm-start epochs")
    parser.add_argument("--epochs", type=int, default=10, help="Number of InfoNCE training epochs")
    parser.add_argument("--batch_size", type=int, default=16384, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--num_angles", type=int, default=64, help="Number of independent phase angles (dim = 2 * num_angles)")
    parser.add_argument("--temperature", type=float, default=0.1, help="InfoNCE loss temperature scaling factor")
    parser.add_argument("--num_negatives", type=int, default=10, help="Negative samples per positive pair")
    parser.add_argument("--walk_length", type=int, default=40, help="Random walk length")
    parser.add_argument("--walks_per_node", type=int, default=1, help="Number of random walks starting per node")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam"], help="Optimizer type for node parameters")
    parser.add_argument("--no_amp", action="store_true", help="Disable Automatic Mixed Precision (AMP)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Phase 2 Pipeline Starting on Device: {device} (128D / {args.num_angles} Angles | Opt: {args.optimizer} | Batch: {args.batch_size:,}) ===")

    # Step 1: Load graph
    graph_data = load_phase2_graph(args.edge_list)
    edge_index = graph_data['edge_index']
    num_nodes = graph_data['num_nodes']
    total_degree = graph_data['total_degree']

    # Step 2: Walk Generation
    walks_cache_path = f"data/phase2_walks_{'uniform' if args.uniform_only else 'biased'}.pt"
    if os.path.exists(walks_cache_path):
        print(f"[Walk Corpus] Loading cached walks from {walks_cache_path}...")
        walks = torch.load(walks_cache_path, weights_only=False)
    else:
        if args.uniform_only:
            walks = generate_walks_uniform(
                edge_index, num_nodes, walk_length=args.walk_length, walks_per_node=args.walks_per_node, device=device
            )
        else:
            # Step 2b: Build Alias Tables
            node_weights = 1.0 / total_degree.clamp(min=1.0)
            alias_data = build_alias_tables(
                edge_index, num_nodes, node_weights, cache_path="data/phase2_alias.pt"
            )
            walks = generate_walks_biased(
                alias_data, num_nodes, walk_length=args.walk_length, walks_per_node=args.walks_per_node, device=device
            )
        print(f"[Walk Corpus] Saving walk corpus to {walks_cache_path}...")
        torch.save(walks, walks_cache_path)

    # Step 3: Extract Co-occurrence Pairs
    pairs_dict = extract_cooccurrence_pairs(walks, window_sizes=(2, 5, 10))

    # Step 4: Initialize Torus Embedding Model
    print(f"\n[Model Init] Initializing TorusEmbedding ({num_nodes:,} nodes, {args.num_angles} angles -> {args.num_angles*2}D coordinates)...")
    model = TorusEmbedding(num_nodes=num_nodes, num_angles=args.num_angles)

    # Step 5: Warm-Start Pretraining (Spring Init)
    if not args.skip_warmstart and args.warmstart_epochs > 0:
        train_warmstart(
            model=model,
            edge_index=edge_index,
            num_nodes=num_nodes,
            epochs=args.warmstart_epochs,
            batch_size=131072,
            lr=args.lr,
            num_negatives=5,
            use_amp=(not args.no_amp),
            device=device
        )

    # Step 6: Main InfoNCE Training
    train_infonce(
        model=model,
        pairs_dict=pairs_dict,
        num_nodes=num_nodes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_negatives=args.num_negatives,
        temperature=args.temperature,
        use_amp=(not args.no_amp),
        optimizer_type=args.optimizer,
        device=device,
        checkpoint_prefix="data/phase2_checkpoint_128d"
    )

    # Step 7: Evaluation
    evaluate_similarity_buckets(
        model=model,
        edge_index=edge_index,
        num_nodes=num_nodes,
        walk_length=30,
        device=device
    )

if __name__ == "__main__":
    main()
