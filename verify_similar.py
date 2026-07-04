import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from train_model import PhaseTorusModel, most_similar

# Reconfigure stdout to prevent CP1252 encoding errors on Windows terminals
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def load_names_map(csv_path):
    print(f"Loading titles mapping from {csv_path}...")
    df = pd.read_csv(csv_path, dtype={'node_id': 'Int64', 'name': 'str'}, on_bad_lines='skip')
    df = df.dropna()
    id_to_title = dict(zip(df['node_id'].astype(int), df['name']))
    title_to_id = dict(zip(df['name'], df['node_id'].astype(int)))
    print(f"Loaded {len(id_to_title):,} page names.")
    return id_to_title, title_to_id

def load_graph_compact_mapping(edge_list_path):
    print(f"Loading edge list {edge_list_path} to build compact ID mapping...")
    df = pd.read_csv(
        edge_list_path,
        sep=r'\s+',
        header=None,
        names=['src', 'dst'],
        dtype=np.int32,
        engine='c',
        comment='#'
    )
    edges_array = df.values
    all_nodes = np.unique(edges_array)
    num_compact = len(all_nodes)
    orig_to_compact = {orig: compact for compact, orig in enumerate(all_nodes)}
    compact_to_orig = {compact: orig for compact, orig in enumerate(all_nodes)}
    all_nodes_compact = np.arange(num_compact, dtype=np.int32)
    return all_nodes_compact, orig_to_compact, compact_to_orig

def load_model(model_path, num_nodes, device):
    """Load a checkpoint, auto-detect metadata, return (model, label)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        num_relations = checkpoint.get('num_relations', 20)
        embedding_dim = checkpoint.get('embedding_dim', 64)
        model_type    = checkpoint.get('model_type', 'phase_torus')
        state_dict    = checkpoint['state_dict']
        label = (
            f"{os.path.basename(model_path)}\n"
            f"  epochs={checkpoint.get('epochs')}  "
            f"batch={checkpoint.get('batch_size')}  dim={embedding_dim}  "
            f"rels={num_relations}  negs={checkpoint.get('num_negatives', 1)}  "
            f"trained={checkpoint.get('timestamp')}"
        )
    else:
        num_relations = 20
        embedding_dim = 64
        model_type    = 'phase_torus'
        state_dict    = checkpoint
        label = f"{os.path.basename(model_path)} (legacy)"

    # Instantiate PhaseTorusModel
    model = PhaseTorusModel(
        num_nodes=num_nodes,
        num_relations=num_relations,
        embedding_dim=embedding_dim
    ).to(device)
        
    model.load_state_dict(state_dict)
    model.eval()
    return model, label

def raw_cosine_similarity(model, compact_node_id, compact_to_orig, id_to_title, all_nodes, topn=5):
    """Compute similarity as negative distance in phase space."""
    with torch.no_grad():
        target_idx = torch.tensor([compact_node_id], dtype=torch.long, device=model.node_embeddings.weight.device)
        # target_theta shape: (1, D)
        target_phi = model.node_embeddings(target_idx)
        
        chunk_size = 50_000
        best_sims = torch.full((topn,), -1000.0, device=target_phi.device)
        best_ids = torch.full((topn,), -1, dtype=torch.long, device=target_phi.device)
        
        num_nodes = len(all_nodes)
        for start in range(0, num_nodes, chunk_size):
            end = min(start + chunk_size, num_nodes)
            chunk_tensor = torch.arange(start, end, device=model.node_embeddings.weight.device)
            chunk_phi = model.node_embeddings(chunk_tensor) # (C, D)
            
            diff = chunk_phi - target_phi # (C, D)
            delta = diff - torch.round(diff)
            dists = torch.sum(delta ** 2, dim=-1) # (C,)
            sims = -dists
            
            # Zero out self-similarity
            if start <= compact_node_id < end:
                sims[compact_node_id - start] = -1000.0
            
            combined_sims = torch.cat([best_sims, sims])
            combined_ids = torch.cat([best_ids, torch.arange(start, end, device=target_phi.device)])
            top_k = torch.topk(combined_sims, topn)
            best_sims = top_k.values
            best_ids = combined_ids[top_k.indices]
            
        result_ids = best_ids.tolist()
        result_sims = best_sims.tolist()
        
    return [(id_to_title.get(compact_to_orig.get(i, i), str(compact_to_orig.get(i, i))), s) 
            for i, s in zip(result_ids, result_sims)]

def relation_specific_similarity(model, compact_node_id, relation_id, compact_to_orig, id_to_title, all_nodes, topn=5):
    """Compute similarity under a specific relation in angle space."""
    with torch.no_grad():
        target_idx = torch.tensor([compact_node_id], dtype=torch.long, device=model.node_embeddings.weight.device)
        h = model.node_embeddings(target_idx) # (1, D)
        
        W = model.relation_W[relation_id] # (D, D)
        b = model.relation_b[relation_id] # (D,)
        
        delta_theta = torch.matmul(h, W.t()) + b # (1, D)
        h_transformed = h + delta_theta # (1, D)
        
        chunk_size = 50_000
        best_sims = torch.full((topn,), -1000.0, device=h.device)
        best_ids = torch.full((topn,), -1, dtype=torch.long, device=h.device)
        
        num_nodes = len(all_nodes)
        for start in range(0, num_nodes, chunk_size):
            end = min(start + chunk_size, num_nodes)
            chunk_emb = model.node_embeddings.weight[start:end] # (C, D)
            
            diff = chunk_emb - h_transformed # (C, D)
            wrapped_diff = (diff + 3.141592653589793) % (2 * 3.141592653589793) - 3.141592653589793
            dists = torch.sqrt(torch.sum(wrapped_diff ** 2, dim=-1) + 1e-9) # (C,)
            sims = -dists
            
            if start <= compact_node_id < end:
                sims[compact_node_id - start] = -1000.0
                
            combined_sims = torch.cat([best_sims, sims])
            combined_ids = torch.cat([best_ids, torch.arange(start, end, device=h.device)])
            top_k = torch.topk(combined_sims, topn)
            best_sims = top_k.values
            best_ids = combined_ids[top_k.indices]
            
        result_ids = best_ids.tolist()
        result_sims = best_sims.tolist()
        
    return [(id_to_title.get(compact_to_orig.get(i, i), str(compact_to_orig.get(i, i))), s) 
            for i, s in zip(result_ids, result_sims)]

def inspect_all_relations(model, compact_node_id, compact_to_orig, id_to_title, all_nodes, topn=3):
    with torch.no_grad():
        target_idx = torch.tensor([compact_node_id], dtype=torch.long, device=model.node_embeddings.weight.device)
        h = model.node_embeddings(target_idx) # (1, D)
        
        # W: (K, D, D), b: (K, D)
        delta_theta = torch.einsum('bd,kcd->bkc', h, model.relation_W) + model.relation_b.unsqueeze(0) # (1, K, D)
        h_transformed = h.unsqueeze(1) + delta_theta # (1, K, D)
        h_transformed = h_transformed.squeeze(0) # (K, D)
        
        K = model.num_relations
        best_sims = torch.full((K, topn), -1000.0, device=h.device)
        best_ids = torch.full((K, topn), -1, dtype=torch.long, device=h.device)
        
        num_nodes = len(all_nodes)
        chunk_size = 50_000
        for start in range(0, num_nodes, chunk_size):
            end = min(start + chunk_size, num_nodes)
            chunk_emb = model.node_embeddings.weight[start:end] # (C, D)
            
            # diff shape: (C, K, D)
            diff = chunk_emb.unsqueeze(1) - h_transformed.unsqueeze(0)
            wrapped_diff = (diff + 3.141592653589793) % (2 * 3.141592653589793) - 3.141592653589793
            dists = torch.sqrt(torch.sum(wrapped_diff ** 2, dim=-1) + 1e-9) # (C, K)
            sims = -dists
            
            if start <= compact_node_id < end:
                sims[compact_node_id - start, :] = -1000.0
                
            combined_sims = torch.cat([best_sims, sims.t()], dim=1) # (K, topn + C)
            chunk_indices = torch.arange(start, end, device=h.device).unsqueeze(0).repeat(K, 1) # (K, C)
            combined_ids = torch.cat([best_ids, chunk_indices], dim=1) # (K, topn + C)
            
            top_k = torch.topk(combined_sims, topn, dim=1)
            best_sims = top_k.values
            best_ids = torch.gather(combined_ids, 1, top_k.indices)
            
        result_ids = best_ids.tolist()
        result_sims = best_sims.tolist()
        
    relation_results = {}
    for k in range(K):
        relation_results[k] = [
            (id_to_title.get(compact_to_orig.get(i, i), str(compact_to_orig.get(i, i))), s)
            for i, s in zip(result_ids[k], result_sims[k])
        ]
    return relation_results

def show_similar_for_id(c_id, models, compact_to_orig, id_to_title, all_nodes, topn=5):
    orig_id = compact_to_orig[c_id]
    title   = id_to_title.get(orig_id, f"ID: {orig_id}")
    
    print(f"\n{'-'*90}")
    print(f"Query: '{title}'  (orig_id={orig_id}, compact_id={c_id})")
    print(f"{'-'*90}")
    
    results = [
        most_similar(m, int(c_id), compact_to_orig, id_to_title, all_nodes, topn=topn)
        for m in models
    ]
    
    col_w = max(30, 80 // len(models))
    col_letters = "ABCDEFGH"
    header = f"{'Rank':<5}" + "".join(f"  {'MODEL '+col_letters[i]:<{col_w}}" for i in range(len(models)))
    print(header)
    print(f"{'':-<5}" + "".join(f"  {'':-<{col_w}}" for _ in models))
    
    for rank in range(topn):
        def fmt(res):
            if rank < len(res):
                t, s = res[rank]
                return f"{t[:col_w-18]} (Score:{s:+.3f})"
            return "—"
        row = f"{rank+1:<5}" + "".join(f"  {fmt(r):<{col_w}}" for r in results)
        print(row)

def process_query(query, models, labels, compact_to_orig, id_to_title, all_nodes, title_to_id, orig_to_compact, topn=5):
    col_letters = "ABCDEFGH"
    if query.lower() == 'random':
        c_id = np.random.choice(all_nodes)
        show_similar_for_id(c_id, models, compact_to_orig, id_to_title, all_nodes, topn)
        return

    # Parse prefixes
    query_lower = query.lower()
    if query_lower.startswith("raw "):
        mode = "raw"
        search_term = query[4:].strip()
    elif query_lower.startswith("cosine "):
        mode = "raw"
        search_term = query[7:].strip()
    elif query_lower.startswith("inspect "):
        mode = "inspect"
        search_term = query[8:].strip()
    elif query_lower.startswith("rel:") or query_lower.startswith("r:"):
        prefix = "rel:" if query_lower.startswith("rel:") else "r:"
        rest = query[len(prefix):].strip()
        parts = rest.split(maxsplit=1)
        if len(parts) == 2:
            rel_str, search_term = parts
            if rel_str.isdigit():
                mode = f"relation_{rel_str}"
                relation_id = int(rel_str)
            else:
                print(f"Invalid relation ID: {rel_str}")
                return
        else:
            print("Usage: rel:<ID> <Article Title>")
            return
    else:
        mode = "default"
        search_term = query

    # Look up title
    orig_id = title_to_id.get(search_term)
    
    # Try case-insensitive matching if exact match not found
    if orig_id is None:
        search_term_lower = search_term.lower()
        matches = [t for t in title_to_id if search_term_lower in t.lower()]
        if matches:
            exact_ci = [m for m in matches if m.lower() == search_term_lower]
            if exact_ci:
                search_term = exact_ci[0]
                orig_id = title_to_id[search_term]
            else:
                print(f"Article '{search_term}' not found. Did you mean one of these?")
                for m in matches[:10]:
                    print(f"  - {m}")
                return
        else:
            print(f"Article '{search_term}' not found (no partial matches found either).")
            return

    # Map to compact ID
    c_id = orig_to_compact.get(orig_id)
    if c_id is None:
        print(f"Article '{search_term}' (ID: {orig_id}) is not present in the graph edge list.")
        return
        
    if mode == "default":
        show_similar_for_id(c_id, models, compact_to_orig, id_to_title, all_nodes, topn)
    elif mode == "raw":
        print(f"\n{'-'*90}")
        print(f"RAW EMBEDDING COSINE SIMILARITY for: '{search_term}'")
        print(f"{'-'*90}")
        col_w = max(30, 80 // len(models))
        header = f"{'Rank':<5}" + "".join(f"  {'MODEL '+col_letters[i]:<{col_w}}" for i in range(len(models)))
        print(header)
        print(f"{'':-<5}" + "".join(f"  {'':-<{col_w}}" for _ in models))
        
        results = []
        for m in models:
            res = raw_cosine_similarity(m, c_id, compact_to_orig, id_to_title, all_nodes, topn)
            results.append(res)
            
        for rank in range(topn):
            def fmt(res):
                if rank < len(res):
                    t, s = res[rank]
                    return f"{t[:col_w-18]} (Score:{s:+.3f})"
                return "—"
            row = f"{rank+1:<5}" + "".join(f"  {fmt(r):<{col_w}}" for r in results)
            print(row)
            
    elif mode.startswith("relation_"):
        print(f"\n{'-'*90}")
        print(f"RELATION {relation_id} SIMILARITY for: '{search_term}'")
        print(f"{'-'*90}")
        
        col_w = max(30, 80 // len(models))
        header = f"{'Rank':<5}" + "".join(f"  {'MODEL '+col_letters[i]:<{col_w}}" for i in range(len(models)))
        print(header)
        print(f"{'':-<5}" + "".join(f"  {'':-<{col_w}}" for _ in models))
        
        results = []
        for m in models:
            if relation_id >= m.num_relations:
                print(f"Model relation ID out of bounds ({relation_id} >= {m.num_relations})")
                res = []
            else:
                res = relation_specific_similarity(m, c_id, relation_id, compact_to_orig, id_to_title, all_nodes, topn)
            results.append(res)
            
        for rank in range(topn):
            def fmt(res):
                if rank < len(res):
                    t, s = res[rank]
                    return f"{t[:col_w-18]} (Score:{s:+.3f})"
                return "-"
            row = f"{rank+1:<5}" + "".join(f"  {fmt(r):<{col_w}}" for r in results)
            print(row)
            
    elif mode == "inspect":
        print(f"\n{'-'*90}")
        print(f"RELATION INSPECT (ALL RELATIONS) for: '{search_term}'")
        print(f"{'-'*90}")
        
        for model_idx, m in enumerate(models):
            label = labels[model_idx].split('\n')[0]
            print(f"\nMODEL {col_letters[model_idx]} ({label}):")
            print("-" * 40)
            
            # Retrieve relation_results
            relation_results = inspect_all_relations(m, c_id, compact_to_orig, id_to_title, all_nodes, topn=3)
            
            # Print them compactly
            for k in range(m.num_relations):
                res = relation_results[k]
                res_strings = []
                for title, score in res:
                    res_strings.append(f"{title[:25]} ({score:+.2f})")
                print(f"  Rel {k:<2}: " + " | ".join(res_strings))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify Similar Articles")
    parser.add_argument("--query", type=str, help="Search query (e.g., 'Mars', 'raw Mars', 'rel:12 Mars', 'inspect Mars')")
    parser.add_argument("--topn", type=int, default=5, help="Number of similar articles to show")
    args = parser.parse_args()

    names_csv_path = r"data\enwiki-2013-names.csv"
    edge_list_path = r"data\enwiki-2013.txt"

    # Discover all model checkpoints in data/
    data_dir = r"data"
    model_files = []
    if os.path.exists(data_dir):
        model_files = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if (f.startswith("transrot_model") or f.startswith("latent_angle_model")) and f.endswith(".pt")
        ])
    
    # Compare up to the 4 latest checkpoints
    model_paths = model_files[-4:] if model_files else []
    if not model_paths:
        print("No trained checkpoints found in data/.")
        return
 
    # ── Shared setup ──────────────────────────────────────────────────────────
    id_to_title, title_to_id = load_names_map(names_csv_path)
    all_nodes, orig_to_compact, compact_to_orig = load_graph_compact_mapping(edge_list_path)
    num_nodes = len(all_nodes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    # ── Load models (skip any that are missing) ───────────────────────────────
    models = []
    labels = []
    col_letters = "ABCDEFGH"
    for path in model_paths:
        if not os.path.exists(path):
            print(f"  [SKIP] Not found: {path}")
            continue
        print(f"\nLoading: {path}")
        try:
            m, lbl = load_model(path, num_nodes, device)
            models.append(m)
            labels.append(lbl)
            print(f"  -> {lbl}")
            
            # Print embedding similarity stats
            with torch.no_grad():
                sample_size = min(5000, len(m.node_embeddings.weight))
                idx = torch.randperm(len(m.node_embeddings.weight))[:sample_size]
                theta_sample = m.node_embeddings.weight[idx]
                emb = theta_to_cartesian(theta_sample)
                S = torch.matmul(emb, emb.t())
                mask = ~torch.eye(len(idx), dtype=torch.bool, device=device)
                
                sim_vals = S[mask]
                mean_sim = sim_vals.mean().item()
                std_sim = sim_vals.std().item()
                min_sim = sim_vals.min().item()
                max_sim = sim_vals.max().item()
                
                print(f"  Embedding space stats (sample size={sample_size}):")
                print(f"    Mean Similarity : {mean_sim:+.4f}")
                print(f"    Std Dev         : {std_sim:.4f}")
                print(f"    Min Similarity  : {min_sim:+.4f}")
                print(f"    Max Similarity  : {max_sim:+.4f}")
        except Exception as e:
            print(f"  [ERROR] {e}")

    if not models:
        print("No models were successfully loaded.")
        return

    print(f"\n{'='*90}")
    print("MODEL LEGEND")
    print(f"{'='*90}")
    for i, lbl in enumerate(labels):
        print(f"  MODEL {col_letters[i]}: {lbl}")

    topn = args.topn

    # If query is passed, run it once and exit
    if args.query:
        query = args.query.strip()
        process_query(query, models, labels, compact_to_orig, id_to_title, all_nodes, title_to_id, orig_to_compact, topn)
        return

    # ── Interactive Query Loop ────────────────────────────────────────────────
    print("\n" + "="*90)
    print("SEARCH MODE")
    print("="*90)
    print("Enter a Wikipedia article title to find similar articles.")
    print("Prefixes supported:")
    print("  - 'raw <title>'    : Cosine similarity on raw entity embeddings")
    print("  - 'rel:<k> <title>': Distance under a specific relation k")
    print("  - 'inspect <title>': Top-3 neighbors for all 50 relations")
    print("  - 'random'         : Run comparison on a random page")
    print("  - 'q' or 'quit'    : Exit search mode")
    
    while True:
        try:
            query = input("\nEnter search: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
            
        if not query:
            continue
        if query.lower() in ('q', 'quit', 'exit'):
            print("Exiting...")
            break
            
        process_query(query, models, labels, compact_to_orig, id_to_title, all_nodes, title_to_id, orig_to_compact, topn)

if __name__ == "__main__":
    main()
