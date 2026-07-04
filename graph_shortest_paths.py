import os
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
import matplotlib.pyplot as plt

def load_graph_as_csr(edge_list_path):
    print(f"Loading edge list from {edge_list_path}...")
    t0 = time.time()
    df = pd.read_csv(edge_list_path, sep=r'\s+', header=None,
                     names=['src','dst'], dtype=np.int32, engine='c', comment='#')
    edges_array = df.values
    all_nodes, flat = np.unique(edges_array, return_inverse=True)
    edges = flat.reshape(-1, 2).astype(np.int32)
    N = len(all_nodes)
    adj = sp.csr_matrix(
        (np.ones(len(edges), dtype=np.float32), (edges[:, 0], edges[:, 1])),
        shape=(N, N)
    )
    print(f"Loaded in {time.time()-t0:.1f}s. Nodes: {N:,}, Edges: {adj.nnz:,}")
    return adj

def reconstruct_path(pred, src, target):
    path = []
    node = int(target)
    while node != -9999 and node != src:
        path.append(node)
        node = int(pred[node])
    if node == src:
        path.append(src)
        path.reverse()
        return path
    return None

def block_nodes(adj, blocked):
    if len(blocked) == 0:
        return adj
    m = adj.copy()
    for node in blocked:
        m.data[m.indptr[node]:m.indptr[node+1]] = 0.0
    mc = m.tocsc()
    for node in blocked:
        mc.data[mc.indptr[node]:mc.indptr[node+1]] = 0.0
    out = mc.tocsr()
    out.eliminate_zeros()
    return out

def main():
    edge_list_path = r"data\enwiki-2013.txt"
    output_plot_path = r"data\shortest_paths_plot.png"

    if not os.path.exists(edge_list_path):
        print(f"Error: {edge_list_path} not found!")
        return

    adj = load_graph_as_csr(edge_list_path)
    N = adj.shape[0]
    num_levels = 10
    num_pairs  = 5   # small for speed

    out_deg = np.diff(adj.indptr)
    high_deg = np.where(out_deg >= num_levels)[0]

    # Pick one source, sample num_pairs random targets from it
    src = int(np.random.choice(high_deg))
    print(f"\nSource: {src} (out-degree: {out_deg[src]})")

    print("Running global BFS from source (1 scipy call)...")
    t0 = time.time()
    dist0, pred0 = shortest_path(adj, method='BF', directed=True,
                                  indices=src, return_predecessors=True)
    print(f"Done in {time.time()-t0:.1f}s")

    reachable = np.where(np.isfinite(dist0) & (np.arange(N) != src))[0]
    targets = np.random.choice(reachable, size=min(num_pairs, len(reachable)), replace=False)
    print(f"Sampled {len(targets)} targets: {targets}")

    # pair_lengths[j, k] = length of k-th disjoint path for pair j
    pair_lengths = np.full((num_pairs, num_levels), np.nan)

    blocked_global = set()
    cur_adj = adj
    cur_pred = pred0
    cur_dist = dist0

    for k in range(num_levels):
        print(f"\nLevel {k+1}/{num_levels} — 1 scipy BFS call...")
        t0 = time.time()

        for j, tgt in enumerate(targets):
            pair_lengths[j, k] = cur_dist[tgt] if np.isfinite(cur_dist[tgt]) else np.nan

        # Collect intermediate nodes from all paths at this level → block them next round
        new_blocked = set()
        for tgt in targets:
            if np.isfinite(cur_dist[tgt]):
                path = reconstruct_path(cur_pred, src, int(tgt))
                if path and len(path) > 2:
                    new_blocked.update(path[1:-1])

        if not new_blocked - blocked_global:
            print("  No new intermediate nodes to block. Stopping early.")
            for remaining in range(k + 1, num_levels):
                pair_lengths[:, remaining] = pair_lengths[:, k]
            break

        blocked_global |= new_blocked
        cur_adj = block_nodes(adj, blocked_global)

        cur_dist, cur_pred = shortest_path(cur_adj, method='BF', directed=True,
                                            indices=src, return_predecessors=True)
        print(f"  Done in {time.time()-t0:.1f}s, blocked {len(blocked_global)} nodes so far")

    # Variance and mean across pairs at each level k
    var_per_level  = np.nanvar(pair_lengths, axis=0)
    mean_per_level = np.nanmean(pair_lengths, axis=0)

    print("\nPath lengths matrix (rows=pairs, cols=levels):")
    print(pair_lengths)
    print(f"\nVariance per level: {var_per_level}")
    print(f"Mean per level:     {mean_per_level}")

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(1, num_levels + 1)

    bars = ax1.bar(x, var_per_level, color='steelblue', alpha=0.75, label='Variance')
    ax1.set_xlabel("k-th Disjoint Path Level", fontsize=12)
    ax1.set_ylabel("Variance of Path Length Across Pairs", fontsize=12, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    ax2 = ax1.twinx()
    ax2.plot(x, mean_per_level, marker='o', color='darkorange',
             linewidth=2.5, label='Mean path length')
    ax2.set_ylabel("Mean Path Length (Hops)", fontsize=12, color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')

    ax1.set_title(f"Variance of k-th Disjoint Path Length Across {num_pairs} Pairs\n"
                  f"(source={src}, n={num_levels} levels)", fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.grid(True, linestyle='--', alpha=0.4)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    print(f"\nVariance plot saved to: {output_plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
