import math
import torch
import torch.nn as nn

class TorusEmbedding(nn.Module):
    """Flat-Torus Node Embedding Manifold.
    
    Each node has 32 independent phase angles theta in R^32.
    The forward mapping converts theta to 64D Cartesian coordinates:
      coords = interleave(cos(theta), sin(theta))
      
    Properties:
    - Norm of coords is identically sqrt(32) for all nodes by construction.
    - Pairwise dot product dot(u, v) = sum_i cos(theta_u_i - theta_v_i) in [-32, 32].
    - Cosine similarity cos_sim(u, v) = dot(u, v) / 32.0 in [-1.0, 1.0].
    - No gimbal lock or vanishing gradients at angle boundaries.
    """
    def __init__(self, num_nodes, num_angles=32):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_angles = num_angles
        self.dim = num_angles * 2 # 64
        
        # Raw unbounded angle parameters [num_nodes, 32]
        self.theta = nn.Parameter(torch.empty(num_nodes, num_angles))
        # Uniform initialization in [0, 2*pi)
        nn.init.uniform_(self.theta, 0.0, 2.0 * math.pi)
        
    def get_coords(self, node_ids=None):
        """Map raw theta parameters to 64D unit-norm coordinate representations.
        
        Args:
            node_ids (torch.LongTensor, optional): Node indices [B] or [B, N].
            
        Returns:
            torch.FloatTensor: 64D coordinates [..., 64].
        """
        if node_ids is None:
            t = self.theta
        else:
            t = self.theta[node_ids]
            
        cos_t = torch.cos(t) # [..., 32]
        sin_t = torch.sin(t) # [..., 32]
        
        # Interleave cos and sin to form 64D vector
        coords = torch.stack([cos_t, sin_t], dim=-1).flatten(-2, -1) # [..., 64]
        return coords

    def forward(self, node_ids=None):
        return self.get_coords(node_ids)

    def compute_dot(self, coords_u, coords_v):
        """Compute pairwise or matrix dot product between coordinate vectors.
        
        Args:
            coords_u (torch.FloatTensor): Shape [B, 64] or [B, 1, 64]
            coords_v (torch.FloatTensor): Shape [B, 64] or [B, K, 64]
            
        Returns:
            torch.FloatTensor: Dot product in range [-32, 32].
        """
        if coords_u.dim() == 2 and coords_v.dim() == 2:
            # 1-to-1 pair dot product: sum over dim -1
            return (coords_u * coords_v).sum(dim=-1)
        elif coords_u.dim() == 2 and coords_v.dim() == 3:
            # Broadcasted dot product: (B, 1, 64) * (B, K, 64) -> (B, K)
            return (coords_u.unsqueeze(1) * coords_v).sum(dim=-1)
        elif coords_u.dim() == 3 and coords_v.dim() == 3:
            return (coords_u * coords_v).sum(dim=-1)
        else:
            return (coords_u * coords_v).sum(dim=-1)

    def compute_cosine_sim(self, coords_u, coords_v):
        """Compute cosine similarity between coordinate vectors.
        
        Returns:
            torch.FloatTensor: Cosine similarity in range [-1.0, 1.0].
        """
        return self.compute_dot(coords_u, coords_v) / float(self.num_angles)
