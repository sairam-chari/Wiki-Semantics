import os
import time
import numpy as np
import pandas as pd
import torch

def load_phase2_graph(edge_list_path, cache_path=None, force_reload=False):
    """Load Wikipedia directed hyperlink graph into PyTorch edge_index [2, E] and degree arrays.
    
    Args:
        edge_list_path (str): Path to whitespace-separated edge list text file (src dst).
        cache_path (str, optional): Path to saved .pt binary cache. Defaults to data/phase2_graph.pt.
        force_reload (bool): If True, ignores cache and parses text file from scratch.
        
    Returns:
        dict: Graph data containing:
            - 'edge_index': torch.LongTensor [2, num_edges]
            - 'in_degree': torch.FloatTensor [num_nodes]
            - 'out_degree': torch.FloatTensor [num_nodes]
            - 'total_degree': torch.FloatTensor [num_nodes]
            - 'num_nodes': int
            - 'num_edges': int
            - 'orig_to_compact': dict (original ID -> compact ID 0..N-1)
            - 'compact_to_orig': dict (compact ID 0..N-1 -> original ID)
    """
    if cache_path is None:
        cache_dir = os.path.dirname(edge_list_path) if os.path.dirname(edge_list_path) else "data"
        cache_path = os.path.join(cache_dir, "phase2_graph.pt")
        
    if not force_reload and os.path.exists(cache_path):
        print(f"[Phase2 Graph] Loading cached graph data from {cache_path}...")
        t0 = time.time()
        graph_data = torch.load(cache_path, weights_only=False)
        print(f"[Phase2 Graph] Loaded {graph_data['num_nodes']:,} nodes and {graph_data['num_edges']:,} edges in {time.time()-t0:.2f}s.")
        return graph_data

    print(f"[Phase2 Graph] Parsing edge list from {edge_list_path}...")
    t0 = time.time()
    
    # Read edges using pandas C engine (fastest)
    df = pd.read_csv(
        edge_list_path,
        sep=r'\s+',
        header=None,
        names=['src', 'dst'],
        dtype=np.int32,
        engine='c',
        comment='#'
    )
    
    edges_raw = df.values
    all_nodes, compact_edges_flat = np.unique(edges_raw, return_inverse=True)
    edges_compact = compact_edges_flat.reshape(-1, 2)
    
    num_nodes = len(all_nodes)
    num_edges = len(edges_compact)
    print(f"[Phase2 Graph] Found {num_nodes:,} unique nodes and {num_edges:,} edges.")
    
    # Mappings
    orig_to_compact = {orig: compact for compact, orig in enumerate(all_nodes)}
    compact_to_orig = {compact: orig for compact, orig in enumerate(all_nodes)}
    
    src = torch.from_numpy(edges_compact[:, 0].astype(np.int64))
    dst = torch.from_numpy(edges_compact[:, 1].astype(np.int64))
    edge_index = torch.stack([src, dst], dim=0) # [2, E]
    
    # Degrees
    out_degree = torch.zeros(num_nodes, dtype=torch.float32)
    in_degree = torch.zeros(num_nodes, dtype=torch.float32)
    
    out_degree.scatter_add_(0, src, torch.ones(num_edges, dtype=torch.float32))
    in_degree.scatter_add_(0, dst, torch.ones(num_edges, dtype=torch.float32))
    total_degree = out_degree + in_degree
    
    graph_data = {
        'edge_index': edge_index,
        'in_degree': in_degree,
        'out_degree': out_degree,
        'total_degree': total_degree,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'orig_to_compact': orig_to_compact,
        'compact_to_orig': compact_to_orig,
    }
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(f"[Phase2 Graph] Saving preprocessed graph to {cache_path}...")
    torch.save(graph_data, cache_path)
    print(f"[Phase2 Graph] Completed in {time.time()-t0:.2f}s.")
    
    return graph_data
