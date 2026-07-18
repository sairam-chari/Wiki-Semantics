import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepWalkEmbedding(nn.Module):
    """DeepWalk node embedding model supporting Euclidean and L2-normalized hyperspherical vectors."""
    def __init__(self, num_nodes: int, dim: int = 32, normalize: bool = False):
        super().__init__()
        self.num_nodes = num_nodes
        self.dim = dim
        self.normalize = normalize
        
        # Standard Gaussian initialization scaled by 1/sqrt(dim)
        self.emb = nn.Embedding(num_nodes, dim)
        nn.init.normal_(self.emb.weight, std=1.0 / (dim ** 0.5))
        
    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        """
        Input:
            node_ids: Tensor of shape [...]
        Output:
            coords: Tensor of shape [..., dim]
        """
        x = self.emb(node_ids)
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        return x

    def compute_cosine_sim(self, a_coords: torch.Tensor, b_coords: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between anchor and candidate embeddings.
        Supports broadcasting for [B, 1, dim] vs [B, K, dim].
        """
        if self.normalize:
            # If vectors are L2 normalized, cosine similarity is simply the dot product
            if a_coords.dim() == 2 and b_coords.dim() == 3:
                return (a_coords.unsqueeze(1) * b_coords).sum(dim=-1)
            return (a_coords * b_coords).sum(dim=-1)
        else:
            if a_coords.dim() == 2 and b_coords.dim() == 3:
                return F.cosine_similarity(a_coords.unsqueeze(1), b_coords, dim=-1)
            return F.cosine_similarity(a_coords, b_coords, dim=-1)

    def compute_dot(self, a_coords: torch.Tensor, b_coords: torch.Tensor) -> torch.Tensor:
        """Raw dot product for similarity search."""
        if a_coords.dim() == 2 and b_coords.dim() == 3:
            return (a_coords.unsqueeze(1) * b_coords).sum(dim=-1)
        return (a_coords * b_coords).sum(dim=-1)
