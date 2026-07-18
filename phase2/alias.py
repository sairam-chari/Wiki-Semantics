import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
from tqdm import tqdm

def _vose_alias_setup(weights):
    """Vose's Alias method for a single discrete probability distribution."""
    K = len(weights)
    prob = np.zeros(K, dtype=np.float32)
    alias = np.zeros(K, dtype=np.int64)

    sum_w = float(weights.sum())
    if sum_w <= 0 or K == 0:
        return prob, alias

    scaled_p = (weights / sum_w) * K
    small = []
    large = []

    for i, p in enumerate(scaled_p):
        if p < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        s = small.pop()
        l = large.pop()
        prob[s] = scaled_p[s]
        alias[s] = l
        scaled_p[l] = (scaled_p[l] + scaled_p[s]) - 1.0
        if scaled_p[l] < 1.0:
            small.append(l)
        else:
            large.append(l)

    while large:
        l = large.pop()
        prob[l] = 1.0
        alias[l] = l

    while small:
        s = small.pop()
        prob[s] = 1.0
        alias[s] = s

    return prob, alias


def _process_alias_chunk(args):
    """Worker process function for parallel alias table construction."""
    chunk_start, chunk_end, row_ptr, dst_sorted, weights_np = args
    sub_start = row_ptr[chunk_start]
    sub_end = row_ptr[chunk_end]
    num_sub_edges = sub_end - sub_start

    sub_prob = np.zeros(num_sub_edges, dtype=np.float32)
    sub_alias = np.zeros(num_sub_edges, dtype=np.int64)

    for u in range(chunk_start, chunk_end):
        start = row_ptr[u]
        end = row_ptr[u + 1]
        deg = end - start
        if deg == 0:
            continue

        nbrs = dst_sorted[start:end]
        nbr_w = weights_np[nbrs]

        prob_u, alias_u = _vose_alias_setup(nbr_w)
        local_start = start - sub_start
        local_end = end - sub_start
        sub_prob[local_start:local_end] = prob_u
        sub_alias[local_start:local_end] = alias_u

    return chunk_start, sub_start, sub_end, sub_prob, sub_alias


def build_alias_tables(edge_index, num_nodes, node_weights, cache_path=None, force_reload=False, num_workers=None):
    """Build precomputed CSR Walker-Vose alias tables for inverse-degree weighted transitions.
    
    Uses multi-processing CPU parallelism for fast construction on multi-core CPUs.
    """
    if cache_path and not force_reload and os.path.exists(cache_path):
        print(f"[Phase2 Alias] Loading cached alias tables from {cache_path}...")
        t0 = time.time()
        alias_data = torch.load(cache_path, weights_only=False)
        print(f"[Phase2 Alias] Loaded alias tables in {time.time()-t0:.2f}s.")
        return alias_data

    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    print(f"[Phase2 Alias] Building Vose alias tables for {num_nodes:,} nodes ({num_workers} parallel workers)...")
    t0 = time.time()

    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    num_edges = len(src)

    # Sort edges by source node to build CSR representation
    sort_order = np.argsort(src, kind='stable')
    src_sorted = src[sort_order]
    dst_sorted = dst[sort_order]

    # Count out-degree per node
    node_counts = np.bincount(src_sorted, minlength=num_nodes)
    row_ptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(node_counts, out=row_ptr[1:])

    weights_np = node_weights.numpy()

    alias_prob = np.zeros(num_edges, dtype=np.float32)
    alias_table = np.zeros(num_edges, dtype=np.int64)

    # Chunk nodes across parallel worker processes
    chunk_size = max(10000, num_nodes // (num_workers * 4))
    chunks = []
    for c_start in range(0, num_nodes, chunk_size):
        c_end = min(c_start + chunk_size, num_nodes)
        chunks.append((c_start, c_end, row_ptr, dst_sorted, weights_np))

    if num_workers > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_process_alias_chunk, chunk) for chunk in chunks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Alias Table Construction"):
                c_start, sub_start, sub_end, sub_prob, sub_alias = future.result()
                alias_prob[sub_start:sub_end] = sub_prob
                alias_table[sub_start:sub_end] = sub_alias
    else:
        for chunk in tqdm(chunks, desc="Alias Table Construction"):
            c_start, sub_start, sub_end, sub_prob, sub_alias = _process_alias_chunk(chunk)
            alias_prob[sub_start:sub_end] = sub_prob
            alias_table[sub_start:sub_end] = sub_alias

    alias_data = {
        'row_ptr': torch.from_numpy(row_ptr),
        'neighbors': torch.from_numpy(dst_sorted),
        'alias_prob': torch.from_numpy(alias_prob),
        'alias_table': torch.from_numpy(alias_table),
    }

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print(f"[Phase2 Alias] Saving precomputed alias tables to {cache_path}...")
        torch.save(alias_data, cache_path)

    print(f"[Phase2 Alias] Built alias tables in {time.time()-t0:.2f}s.")
    return alias_data
