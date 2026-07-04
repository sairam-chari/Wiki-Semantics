import os
import gzip
import sqlite3
import shutil
import numpy as np
import torch
from generate_edge_list import parse_and_load, export_edges_to_file
from train_model import load_graph, DiscreteLatentTransRotModel, train, get_title_map, most_similar

def create_mock_files():
    mock_dir = "mock_data"
    os.makedirs(mock_dir, exist_ok=True)
    
    # 1. Mock page SQL file:
    # Page IDs: 10 ('AccessibleComputing', redirect=1), 12 ('Anarchism', redirect=0), 20 ('TargetPage', redirect=0)
    page_content = (
        "/* Header comments */\n"
        "DROP TABLE IF EXISTS `page`;\n"
        "CREATE TABLE `page` ( ... );\n"
        "INSERT INTO `page` VALUES "
        "(10,0,'AccessibleComputing',1,0,0.5,'20260601','20260601',100,100,'wikitext',NULL),"
        "(12,0,'Anarchism',0,0,0.6,'20260601','20260601',101,200,'wikitext',NULL),"
        "(20,0,'TargetPage',0,0,0.7,'20260601','20260601',102,150,'wikitext',NULL);\n"
    )
    page_path = os.path.join(mock_dir, "mock-page.sql.gz")
    with gzip.open(page_path, "wt", encoding="utf-8") as f:
        f.write(page_content)
        
    # 2. Mock linktarget SQL file:
    # lt_id 2 maps to Namespace 0, 'TargetPage'
    linktarget_content = (
        "/* Header comments */\n"
        "DROP TABLE IF EXISTS `linktarget`;\n"
        "CREATE TABLE `linktarget` ( ... );\n"
        "INSERT INTO `linktarget` VALUES "
        "(2,0,'TargetPage'),"
        "(5,1,'Talk:TargetPage');\n"
    )
    linktarget_path = os.path.join(mock_dir, "mock-linktarget.sql.gz")
    with gzip.open(linktarget_path, "wt", encoding="utf-8") as f:
        f.write(linktarget_content)
        
    # 3. Mock pagelinks SQL file:
    # Source page ID 12 (Anarchism), source namespace 0, links to lt_id 2
    pagelinks_content = (
        "/* Header comments */\n"
        "DROP TABLE IF EXISTS `pagelinks`;\n"
        "CREATE TABLE `pagelinks` ( ... );\n"
        "INSERT INTO `pagelinks` VALUES "
        "(12,0,2),"
        "(12,1,5);\n"  # This link should be ignored because source namespace is 1, not 0
    )
    pagelinks_path = os.path.join(mock_dir, "mock-pagelinks.sql.gz")
    with gzip.open(pagelinks_path, "wt", encoding="utf-8") as f:
        f.write(pagelinks_content)
        
    return page_path, linktarget_path, pagelinks_path

def verify():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available. Please ensure a CUDA device is present.")
    
    page_path, linktarget_path, pagelinks_path = create_mock_files()
    db_path = os.path.join("mock_data", "mock_wiki.db")
    edge_list_path = os.path.join("mock_data", "mock_edge_list.txt")
    
    # Remove existing DB if any
    if os.path.exists(db_path):
        os.remove(db_path)
        
    print("Running parse_and_load pipeline on mock files...")
    parse_and_load(db_path, page_path, linktarget_path, pagelinks_path)
    
    # Verify Database tables and content
    print("\n--- Verifying SQLite database contents ---")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check tables that should exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]
    print(f"Tables in DB: {tables}")
    assert "pages" in tables, "Table 'pages' should exist!"
    assert "links" in tables, "Table 'links' should exist!"
    assert "linktarget" not in tables, "Temporary table 'linktarget' should have been dropped!"
    assert "temp_links" not in tables, "Temporary table 'temp_links' should have been dropped!"
    
    # Verify pages contents
    cursor.execute("SELECT * FROM pages ORDER BY id")
    pages = cursor.fetchall()
    print(f"Pages loaded: {pages}")
    assert len(pages) == 3, f"Expected 3 pages, found {len(pages)}"
    assert pages[0] == (10, 'AccessibleComputing', 1)
    assert pages[1] == (12, 'Anarchism', 0)
    assert pages[2] == (20, 'TargetPage', 0)
    
    # Verify links contents (Anarchism 12 -> TargetPage 20)
    cursor.execute("SELECT * FROM links")
    links = cursor.fetchall()
    print(f"Resolved links: {links}")
    assert len(links) == 1, f"Expected 1 resolved link, found {len(links)}"
    assert links[0] == (12, 20), f"Expected link (12, 20), found {links[0]}"
    
    conn.close()

    # Step 2 Verification: Export to edge list file
    export_edges_to_file(db_path, edge_list_path)
    assert os.path.exists(edge_list_path), "Edge list file should be created!"
    with open(edge_list_path, "r", encoding="utf-8") as f:
        content = f.read()
    print(f"Edge list file content: {repr(content)}")
    assert content.strip() == "12 20", f"Expected content '12 20', got {repr(content)}"

    # Step 3 Verification: Load graph into CSR matrix
    edges, adj_matrix, all_nodes, orig_to_compact, compact_to_orig = load_graph(edge_list_path)
    
    print("\n--- Verifying CSR Graph Matrix ---")
    print(f"Loaded Edges shape: {edges.shape}")
    print(f"Unique nodes list: {all_nodes}")
    print(f"Adjacency matrix shape: {adj_matrix.shape}")
    
    # Assertions for edges
    assert len(edges) == 1, "Expected 1 edge"
    assert edges[0, 0] == 0 and edges[0, 1] == 1, "Expected edge 0 -> 1"
    
    # Assertions for unique nodes
    import numpy as np
    assert np.array_equal(all_nodes, np.array([0, 1])), "Expected unique nodes [0, 1]"
    
    # Assertions for CSR adjacency matrix
    assert adj_matrix.shape == (2, 2), f"Expected shape (2, 2), got {adj_matrix.shape}"
    
    # Check edge existence check is True for (0, 1) and False for others
    assert adj_matrix[0, 1] == True, "Adjacency matrix[0, 1] should be True"
    assert adj_matrix[1, 0] == False, "Adjacency matrix[1, 0] should be False"
    
    # Step 4 Verification: Initialize and train DiscreteLatentTransRotModel
    print("\n--- Verifying DiscreteLatentTransRotModel ---")
    num_nodes = int(all_nodes.max() + 1)
    num_relations = 3
    embedding_dim = 16
    
    model = DiscreteLatentTransRotModel(
        num_nodes=num_nodes, 
        num_relations=num_relations, 
        embedding_dim=embedding_dim
    ).to("cuda")
    
    # Run 2 epochs of training on mock data
    print("Running training loop for 2 epochs...")
    train(
        model=model,
        edges=edges,
        adj_matrix=adj_matrix,
        all_nodes=all_nodes,
        epochs=2,
        batch_size=2,
        lr=0.05
    )
    

    # Verify that model weights can be saved and loaded
    model_save_path = os.path.join("mock_data", "mock_model.pt")
    torch.save(model.state_dict(), model_save_path)
    assert os.path.exists(model_save_path), "Model weights file should be saved!"
    
    # Load weights into a new model instance
    new_model = DiscreteLatentTransRotModel(
        num_nodes=num_nodes, 
        num_relations=num_relations, 
        embedding_dim=embedding_dim
    ).to("cuda")
    new_model.load_state_dict(torch.load(model_save_path))
    print("Model state dict successfully saved and loaded!")
    
    # Verify model forward pass runs
    h_idx = torch.tensor([0], dtype=torch.long, device="cuda")
    t_idx = torch.tensor([1], dtype=torch.long, device="cuda")
    pos_d = new_model(h_idx, t_idx)
    assert pos_d.shape == (1,), "Forward output distance should have shape (1,)"
    print(f"Forward pass output translation distance for edge 0 -> 1: {pos_d.item():.4f}")
    
    # Step 5 Verification: Title mapping and similarity retrieval
    print("\n--- Verifying Step 5: Similarity Inference ---")
    mock_db_path = os.path.join("mock_data", "mock_wiki.db")
    id_to_title, title_to_id = get_title_map(mock_db_path)
    print(f"Loaded mock title map: {id_to_title}")
    assert 12 in id_to_title and id_to_title[12] == "Anarchism", "Mapping for 12 should be Anarchism!"
    assert 20 in id_to_title and id_to_title[20] == "TargetPage", "Mapping for 20 should be TargetPage!"
    
    similar = most_similar(new_model, 0, compact_to_orig, id_to_title, all_nodes, topn=2)
    print(f"Nearest nodes to ID 12 (Anarchism): {similar}")
    assert len(similar) == 2, "Expected 2 similar items (topn=2)"
    
    # Cleanup mock data directory
    shutil.rmtree("mock_data")
    print("\nAll tests passed successfully! Pipeline is fully correct and ready.")

if __name__ == "__main__":
    verify()
