"""Graph convolutional network for task priority prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """Compute D^{-1/2} (A + I) D^{-1/2} normalization.

    Args:
        adjacency: Adjacency matrix A.

    Returns:
        Normalized adjacency matrix.
    """
    # Add identity matrix to adjacency matrix (A + I)
    identity = torch.eye(adjacency.size(0), device=adjacency.device)
    adjacency_hat = adjacency + identity
    
    # Calculate degree matrix D
    degree = torch.sum(adjacency_hat, dim=1)
    
    # D^{-1/2}
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0  # Handle division by zero
    
    # Create diagonal matrix for D^{-1/2}
    degree_inv_sqrt_mat = torch.diag(degree_inv_sqrt)
    
    # D^{-1/2} * (A + I) * D^{-1/2}
    normalized_adjacency = torch.mm(
        torch.mm(degree_inv_sqrt_mat, adjacency_hat), degree_inv_sqrt_mat
    )
    return normalized_adjacency

class GCNLayer(nn.Module):
    """Single GCN layer for X^{(l+1)} = σ(D^{-1/2}(A+I)D^{-1/2}X^{(l)}W^{(l)})."""

    def __init__(self, in_features: int, out_features: int):
        """Initialize the GCN layer.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
        """
        super(GCNLayer, self).__init__()
        # Trainable weight matrix W^{(l)}
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # Initialize weights
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor, normalized_adjacency: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Node feature matrix.
            normalized_adjacency: Normalized adjacency matrix.

        Returns:
            Updated node features.
        """
        # X * W
        support = torch.mm(features, self.weight)
        # normalized_A * (X * W)
        output = torch.mm(normalized_adjacency, support)
        return output

class TaskPriorityGCN(nn.Module):
    """3-layer GCN architecture for subtask priority prediction."""

    def __init__(self, num_features: int, hidden_dim: int = 32):
        """Initialize the GCN model.

        Args:
            num_features: Input node feature dimension.
            hidden_dim: Hidden layer width.
        """
        super(TaskPriorityGCN, self).__init__()
        # Layer 1: Input to Hidden
        self.gcn1 = GCNLayer(num_features, hidden_dim)
        # Layer 2: Hidden to Hidden
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        # Layer 3: Hidden to Output (1 scalar priority value per subtask)
        self.gcn3 = GCNLayer(hidden_dim, 1)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GCN.

        Args:
            features: Node feature matrix.
            adjacency: DAG adjacency matrix.

        Returns:
            Priority score per node.
        """
        # Precompute the normalized adjacency matrix
        normalized_adjacency = normalize_adjacency(adjacency)
        
        # Layer 1 + ReLU
        x = self.gcn1(features, normalized_adjacency)
        x = F.relu(x)
        
        # Layer 2 + ReLU
        x = self.gcn2(x, normalized_adjacency)
        x = F.relu(x)
        
        # Layer 3 (Output layer to predict priority score)
        x = self.gcn3(x, normalized_adjacency)
        
        return x

# --- Training Setup Example ---
# If you are testing this isolated from the rest of the code:
# model = TaskPriorityGCN(num_features=6, hidden_dim=32)
# optimizer = optim.Adam(model.parameters(), lr=0.01) # Learning rate from paper
# criterion = nn.MSELoss() # MSE Loss from paper
