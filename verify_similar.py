import os
import torch
import numpy as np
import pandas as pd
from train_model import DiscreteLatentTransRotModel, most_similar

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
        model_type    = checkpoint.get('model_type', 'transrot')
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
        model_type    = 'transrot'
        state_dict    = checkpoint
        label = f"{os.path.basename(model_path)} (legacy)"

    # Exclusively instantiate Translation+Rotation Model
    model = DiscreteLatentTransRotModel(
        num_nodes=num_nodes,
        num_relations=num_relations,
        embedding_dim=embedding_dim
    ).to(device)
        
    model.load_state_dict(state_dict)
    model.eval()
    return model, label

def show_similar_for_id(c_id, models, compact_to_orig, id_to_title, all_nodes, topn=5):
    orig_id = compact_to_orig[c_id]
    title   = id_to_title.get(orig_id, f"ID: {orig_id}")
    
    print(f"\n{'─'*90}")
    print(f"Query: '{title}'  (orig_id={orig_id}, compact_id={c_id})")
    print(f"{'─'*90}")
    
    results = [
        most_similar(m, int(c_id), compact_to_orig, id_to_title, all_nodes, topn=topn)
        for m in models
    ]
    
    col_w = max(30, 80 // len(models))
    col_letters = "ABCDEFGH"
    header = f"{'Rank':<5}" + "".join(f"  {'MODEL '+col_letters[i]:<{col_w}}" for i in range(len(models)))
    print(header)
    print(f"{'':─<5}" + "".join(f"  {'':─<{col_w}}" for _ in models))
    
    for rank in range(topn):
        def fmt(res):
            if rank < len(res):
                t, s = res[rank]
                return f"{t[:col_w-18]} (Score:{s:+.3f})"
            return "—"
        row = f"{rank+1:<5}" + "".join(f"  {fmt(r):<{col_w}}" for r in results)
        print(row)

def main():
    names_csv_path = r"data\enwiki-2013-names.csv"
    edge_list_path = r"data\enwiki-2013.txt"

    # Discover all model checkpoints in data/
    data_dir = r"data"
    model_files = []
    if os.path.exists(data_dir):
        model_files = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.startswith("transrot_model") and f.endswith(".pt")
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
            print(f"  → {lbl}")
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

    # ── Interactive Query Loop ────────────────────────────────────────────────
    print("\n" + "="*90)
    print("SEARCH MODE")
    print("="*90)
    print("Enter a Wikipedia article title to find similar articles.")
    print("Type 'random' to see a comparison on random pages, or 'q' to quit.")
    
    topn = 5
    while True:
        try:
            query = input("\nEnter article title: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
            
        if not query:
            continue
        if query.lower() in ('q', 'quit', 'exit'):
            print("Exiting...")
            break
            
        if query.lower() == 'random':
            c_id = np.random.choice(all_nodes)
            show_similar_for_id(c_id, models, compact_to_orig, id_to_title, all_nodes, topn)
            continue

        # Look up title
        orig_id = title_to_id.get(query)
        
        # Try case-insensitive matching if exact match not found
        if orig_id is None:
            query_lower = query.lower()
            matches = [t for t in title_to_id if query_lower in t.lower()]
            if matches:
                exact_ci = [m for m in matches if m.lower() == query_lower]
                if exact_ci:
                    query = exact_ci[0]
                    orig_id = title_to_id[query]
                else:
                    print(f"Article '{query}' not found. Did you mean one of these?")
                    for m in matches[:10]:
                        print(f"  - {m}")
                    continue
            else:
                print(f"Article '{query}' not found (no partial matches found either).")
                continue

        # Map to compact ID
        c_id = orig_to_compact.get(orig_id)
        if c_id is None:
            print(f"Article '{query}' (ID: {orig_id}) is not present in the graph edge list.")
            continue
            
        show_similar_for_id(c_id, models, compact_to_orig, id_to_title, all_nodes, topn)

if __name__ == "__main__":
    main()
