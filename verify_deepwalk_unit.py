import torch
from phase2.deepwalk import DeepWalkEmbedding
from phase2.train_deepwalk import train_deepwalk_infonce

def test_deepwalk_unit():
    print("=== Testing DeepWalk Unit Invariants ===")
    num_nodes = 100
    
    # 1. Unnormalized model test
    dw32 = DeepWalkEmbedding(num_nodes=num_nodes, dim=32, normalize=False)
    ids = torch.tensor([0, 1, 2])
    coords = dw32(ids)
    assert coords.shape == (3, 32), f"Expected shape (3, 32), got {coords.shape}"
    
    sim = dw32.compute_cosine_sim(coords[0], coords[1])
    assert -1.0 <= sim.item() <= 1.0, f"Cosine sim out of bounds: {sim.item()}"
    print("✓ DeepWalk-32 shapes and cosine similarity verified.")

    # 2. Normalized model test
    dw64_norm = DeepWalkEmbedding(num_nodes=num_nodes, dim=64, normalize=True)
    coords_norm = dw64_norm(ids)
    norms = torch.norm(coords_norm, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), f"Expected unit norm, got {norms}"
    print("✓ DeepWalk-64-Norm unit norm invariant verified.")

    # 3. Mini training loop test
    pairs_dict = {
        'anchors': torch.randint(0, num_nodes, (500,)),
        'positives': torch.randint(0, num_nodes, (500,))
    }
    train_deepwalk_infonce(
        model=dw32,
        pairs_dict=pairs_dict,
        num_nodes=num_nodes,
        epochs=1,
        batch_size=128,
        num_negatives=5,
        device="cpu",
        checkpoint_prefix="data/test_dw"
    )
    print("✓ DeepWalk training loop execution verified.")
    print("=== ALL DEEPWALK UNIT TESTS PASSED ===")

if __name__ == "__main__":
    test_deepwalk_unit()
