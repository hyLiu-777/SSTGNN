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
        
    def batch_laplacian_spectral_decomposition(self, node_featrues, edge_indexd, edge_weights):
        device = node_featrues.device
        batch_size = len(node_featrues)
        max_nodes = max(len(i) for i in node_featrues)

        eigvecs_all = []
        eigvals_all = []

        for i in range(batch_size):
            num_nodes = len(node_featrues[i])
            edge_weight = edge_weights[i]

            A = torch.sparse_coo_tensor(
                indices=edge_indexd,
                values=edge_weight,
                size=(num_nodes, num_nodes)
            ).to_dense()

            deg = A.sum(dim=1)
            deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
            D_inv_sqrt = torch.diag(deg_inv_sqrt)
            L = torch.eye(num_nodes, device=device) - D_inv_sqrt @ A @ D_inv_sqrt

            L = (L + L.T) / 2
            eigvals, eigvecs = torch.linalg.eigh(L.detach())

            pad_vecs = torch.zeros((max_nodes, max_nodes), device=device)
            pad_vals = torch.zeros((max_nodes,), device=device)
            pad_vecs[:num_nodes, :num_nodes] = eigvecs
            pad_vals[:num_nodes] = eigvals

            eigvecs_all.append(pad_vecs)
            eigvals_all.append(pad_vals)

        eigvecs_all = torch.stack(eigvecs_all, dim=0)
        eigvals_all = torch.stack(eigvals_all, dim=0)

        return eigvecs_all, eigvals_all

    def forward(self, x, node_featrues, edge_indexd, edge_weight):
        eigvecs, eigvals = self.batch_laplacian_spectral_decomposition(node_featrues, edge_indexd, edge_weight)
        x = self.layernrom(x)
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
    