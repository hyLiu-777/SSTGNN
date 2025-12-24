import torch

class SpectralFeatureExtractor(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, feat_dim, dropout=0.1):
        super().__init__()
        self.activation = torch.nn.ReLU()
        self.layernrom = torch.nn.LayerNorm(feat_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, in_dim)
        )
        self.out_dropout = torch.nn.Dropout(dropout) 
        
    def batch_laplacian_spectral_decomposition(self, node_features, edge_indexd, edge_weights):
        device = node_features.device
        batch_size, num_nodes, _ = node_features.shape

        eigvecs_all = []
        eigvals_all = []

        for i in range(batch_size):
            edge_weight = edge_weights[i]
            edge_weight = torch.clamp(edge_weight, min=0.0, max=1.0)

            A = torch.sparse_coo_tensor(
                indices=edge_indexd,
                values=edge_weight,
                size=(num_nodes, num_nodes)
            ).to_dense()
            
            A = (A + A.T) / 2

            deg = A.sum(dim=1)
            deg = torch.clamp(deg, min=1e-6)
            deg_inv_sqrt = torch.pow(deg, -0.5)
            deg_inv_sqrt = torch.nan_to_num(deg_inv_sqrt, nan=0.0, posinf=0.0, neginf=0.0)
            
            D_inv_sqrt = torch.diag(deg_inv_sqrt)
            L = torch.eye(num_nodes, device=device) - D_inv_sqrt @ A @ D_inv_sqrt

            L = (L + L.T) / 2
            L = L + torch.eye(num_nodes, device=device) * 1e-6

            try:
                eigvals, eigvecs = torch.linalg.eigh(L.detach())
            except RuntimeError as e:
                try:
                    U, S, Vh = torch.linalg.svd(L.detach())
                    eigvecs = U
                    eigvals = S
                except RuntimeError:
                    eigvecs = torch.eye(num_nodes, device=device)
                    eigvals = torch.ones(num_nodes, device=device)

            if torch.isnan(eigvecs).any() or torch.isinf(eigvecs).any():
                eigvecs = torch.eye(num_nodes, device=device)
            if torch.isnan(eigvals).any() or torch.isinf(eigvals).any():
                eigvals = torch.ones(num_nodes, device=device)

            eigvecs_all.append(eigvecs)
            eigvals_all.append(eigvals)

        eigvecs_all = torch.stack(eigvecs_all, dim=0)
        eigvals_all = torch.stack(eigvals_all, dim=0)

        return eigvecs_all, eigvals_all

    def forward(self, node_features, edge_indexd, edge_weights):
        eigvecs, eigvals = self.batch_laplacian_spectral_decomposition(node_features, edge_indexd, edge_weights)
        x = self.layernrom(node_features)
        x_residual = x
        x = self.activation(x)

        eigvecs_T = eigvecs.transpose(1, 2)
        x1 = torch.bmm(eigvecs_T, x)

        eigval_features = self.mlp(eigvals)  
        eigval_features = eigval_features.unsqueeze(-1).expand_as(x1)  # [B, k, F]
         
        x1 = x1 * eigval_features
        out = torch.bmm(eigvecs, x1)

        out += x_residual
        out = self.out_dropout(out)

        return out  # [B, N, F]