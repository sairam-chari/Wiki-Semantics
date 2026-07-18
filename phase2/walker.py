import os
import time
import torch
from tqdm import tqdm

try:
    from torch_cluster import random_walk as tc_random_walk
    HAS_TORCH_CLUSTER = True
except ImportError:
    HAS_TORCH_CLUSTER = False

def _format_time(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes:02d}m {secs:02d}s"

def generate_walks_uniform(edge_index, num_nodes, walk_length=40, walks_per_node=1, device="cuda"):
    """Generate unbiased random walks using torch_cluster.random_walk CUDA kernel.
    
    Args:
        edge_index (torch.LongTensor): [2, E] graph edges.
        num_nodes (int): Total node count.
        walk_length (int): Steps per walk.
        walks_per_node (int): Number of walks starting from each node.
        device (str): Device to run walker on.
        
    Returns:
        torch.LongTensor: Walk paths tensor of shape [num_nodes * walks_per_node, walk_length + 1].
    """
    print(f"[Phase2 Walker] Generating uniform walks (length={walk_length}, walks/node={walks_per_node})...")
    t0 = time.time()
    
    is_cuda = torch.cuda.is_available() and device.startswith("cuda")
    start_nodes = torch.arange(num_nodes, dtype=torch.int64).repeat(walks_per_node)
    
    if HAS_TORCH_CLUSTER and is_cuda:
        edge_index_dev = edge_index.to(device)
        start_nodes_dev = start_nodes.to(device)
        row, col = edge_index_dev[0], edge_index_dev[1]
        walks = tc_random_walk(row, col, start_nodes_dev, walk_length=walk_length).cpu()
    else:
        print("[Phase2 Walker] Using PyTorch vectorized fallback walker.")
        src = edge_index[0]
        dst = edge_index[1]
        sort_idx = torch.argsort(src)
        src_s = src[sort_idx]
        dst_s = dst[sort_idx]
        row_ptr = torch.zeros(num_nodes + 1, dtype=torch.int64)
        counts = torch.bincount(src_s, minlength=num_nodes)
        torch.cumsum(counts, dim=0, out=row_ptr[1:])
        
        total_walks = len(start_nodes)
        walks = torch.zeros((total_walks, walk_length + 1), dtype=torch.int64)
        walks[:, 0] = start_nodes
        
        curr = start_nodes.clone()
        for step in range(1, walk_length + 1):
            s = row_ptr[curr]
            e = row_ptr[curr + 1]
            deg = e - s
            has_nbrs = deg > 0
            
            next_nodes = curr.clone()
            if has_nbrs.any():
                active_indices = torch.where(has_nbrs)[0]
                active_s = s[active_indices]
                active_deg = deg[active_indices]
                
                rand_offset = (torch.rand(len(active_indices)) * active_deg.float()).long()
                next_nodes[active_indices] = dst_s[active_s + rand_offset]
                
            walks[:, step] = next_nodes
            curr = next_nodes

    duration = time.time() - t0
    print(f"[Phase2 Walker] Completed uniform walks ({len(walks):,} walks) in {_format_time(duration)} ({len(walks)/duration:.0f} walks/s).")
    return walks


def _step_alias_sampling(curr_nodes, row_ptr, neighbors, alias_prob, alias_table):
    """Vectorized PyTorch step using precomputed alias tables."""
    s = row_ptr[curr_nodes]
    e = row_ptr[curr_nodes + 1]
    deg = e - s
    has_nbrs = deg > 0
    
    next_nodes = curr_nodes.clone()
    if not has_nbrs.any():
        return next_nodes

    active_indices = torch.where(has_nbrs)[0]
    active_s = s[active_indices]
    active_deg = deg[active_indices]
    
    rand_offset = (torch.rand(len(active_indices), device=curr_nodes.device) * active_deg.float()).long()
    sample_ptr = active_s + rand_offset
    
    u = torch.rand(len(active_indices), device=curr_nodes.device)
    probs = alias_prob[sample_ptr]
    
    accept = u < probs
    chosen_ptr = torch.where(accept, sample_ptr, active_s + alias_table[sample_ptr])
    
    next_nodes[active_indices] = neighbors[chosen_ptr]
    return next_nodes


def generate_walks_biased(alias_data, num_nodes, walk_length=40, walks_per_node=1, batch_size=256000, device="cuda"):
    """Generate inverse-degree biased random walks using Vose alias tables.
    
    Automatically loads alias tables to CUDA VRAM if sufficient memory is available.
    """
    print(f"[Phase2 Walker] Generating degree-inverse biased walks (length={walk_length}, walks/node={walks_per_node})...")
    t0 = time.time()
    
    row_ptr = alias_data['row_ptr']
    neighbors = alias_data['neighbors']
    alias_prob = alias_data['alias_prob']
    alias_table = alias_data['alias_table']
    
    is_cuda = torch.cuda.is_available() and device.startswith("cuda")
    if is_cuda:
        free_vram, _ = torch.cuda.mem_get_info(device)
        required_vram = (row_ptr.element_size() * len(row_ptr) +
                         neighbors.element_size() * len(neighbors) +
                         alias_prob.element_size() * len(alias_prob) +
                         alias_table.element_size() * len(alias_table))
        if free_vram > required_vram + 1e9: # Keep 1GB safety margin
            print(f"[Phase2 Walker] Loading alias tables ({required_vram/1e9:.2f} GB) into GPU VRAM for maximum speed...")
            row_ptr = row_ptr.to(device)
            neighbors = neighbors.to(device)
            alias_prob = alias_prob.to(device)
            alias_table = alias_table.to(device)
        else:
            print(f"[Phase2 Walker] VRAM constrained ({free_vram/1e9:.2f} GB free). Running alias walker on CPU host RAM.")
            device = "cpu"
            
    start_nodes = torch.arange(num_nodes, dtype=torch.int64).repeat(walks_per_node)
    total_walks = len(start_nodes)
    num_batches = (total_walks + batch_size - 1) // batch_size
    
    walks = torch.zeros((total_walks, walk_length + 1), dtype=torch.int64)
    
    with tqdm(total=num_batches, desc="Biased Walk Generation") as pbar:
        for b_start in range(0, total_walks, batch_size):
            b_end = min(b_start + batch_size, total_walks)
            curr = start_nodes[b_start:b_end].to(device)
            
            b_walks = torch.zeros((b_end - b_start, walk_length + 1), dtype=torch.int64, device=device)
            b_walks[:, 0] = curr
            
            for step in range(1, walk_length + 1):
                curr = _step_alias_sampling(curr, row_ptr, neighbors, alias_prob, alias_table)
                b_walks[:, step] = curr

            walks[b_start:b_end] = b_walks.cpu()
            pbar.update(1)

    duration = time.time() - t0
    print(f"[Phase2 Walker] Completed biased walks ({total_walks:,} walks) in {_format_time(duration)} ({total_walks/duration:.0f} walks/s).")
    return walks


def extract_cooccurrence_pairs(walks, window_sizes=(2, 5, 10), pairs_per_walk=20):
    """Extract (anchor, positive, window_distance) co-occurrence pairs from walk paths."""
    print(f"[Phase2 Walker] Extracting co-occurrence pairs from {len(walks):,} walks...")
    t0 = time.time()
    
    num_walks, walk_len_plus_1 = walks.shape
    walk_len = walk_len_plus_1 - 1
    
    all_anchors = []
    all_positives = []
    all_window_dists = []
    
    for w in window_sizes:
        if w > walk_len:
            continue
        max_start = walk_len - w
        i_indices = torch.randint(0, max_start + 1, (num_walks,))
        j_indices = i_indices + w
        
        walk_idx = torch.arange(num_walks)
        anchors = walks[walk_idx, i_indices]
        positives = walks[walk_idx, j_indices]
        
        valid = anchors != positives
        anchors = anchors[valid]
        positives = positives[valid]
        
        all_anchors.append(anchors)
        all_positives.append(positives)
        all_window_dists.append(torch.full_like(anchors, w))

    anchors_cat = torch.cat(all_anchors)
    positives_cat = torch.cat(all_positives)
    window_dists_cat = torch.cat(all_window_dists)
    
    duration = time.time() - t0
    print(f"[Phase2 Walker] Extracted {len(anchors_cat):,} pairs in {_format_time(duration)} ({len(anchors_cat)/duration:.0f} pairs/s).")
    return {
        'anchors': anchors_cat,
        'positives': positives_cat,
        'window_dists': window_dists_cat,
    }
