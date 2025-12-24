import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from models import MMTGNN
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from dataset.utils import read_video_frames, video2patches


def predict_with_intermediates(model, patch_images, device):
    with torch.no_grad():
        patch_images = patch_images.to(device)
        
        logits, intermediates = model(patch_images, return_intermediates=True)
        probs = F.softmax(logits, dim=1)
        
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
        
        prediction = pred_class
        confidence = confidence
        
        result = {
            'prediction': 'fake' if pred_class == 0 else 'real',
            'confidence': confidence,
            'logits': logits.cpu().numpy(),
            'probs': probs.cpu().numpy()
        }
        
        return result, intermediates


def build_spatial_similarity_graph(patches_flat, edge_index, num_frames, grid_size, device):
    B = patches_flat.shape[0]
    num_patches = grid_size * grid_size
    num_edges = edge_index.shape[1]
    
    def get_8_neighbors(node_idx, frame_idx):
        """获取某节点在同一帧内的8邻域节点集合"""
        local_idx = node_idx - frame_idx * num_patches
        row, col = local_idx // grid_size, local_idx % grid_size
        
        neighbors = set()
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = row + di, col + dj
                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                    neighbor_local = ni * grid_size + nj
                    neighbor_global = frame_idx * num_patches + neighbor_local
                    neighbors.add(neighbor_global)
        return neighbors
    
    total_nodes = num_frames * num_patches
    all_neighbors = {}
    for node in range(total_nodes):
        frame_idx = node // num_patches
        all_neighbors[node] = get_8_neighbors(node, frame_idx)
    
    src_nodes = edge_index[0].cpu().numpy()
    dst_nodes = edge_index[1].cpu().numpy()
    
    is_neighbor = np.zeros(num_edges, dtype=bool)
    for e in range(num_edges):
        src, dst = src_nodes[e], dst_nodes[e]
        if dst in all_neighbors.get(src, set()):
            is_neighbor[e] = True
    
    is_neighbor = torch.tensor(is_neighbor, device=device)
    
    patches_norm = F.normalize(patches_flat, dim=-1)
    
    src_features = patches_norm[:, edge_index[0], :]
    dst_features = patches_norm[:, edge_index[1], :]
    
    similarity = (src_features * dst_features).sum(dim=-1)
    similarity = (similarity + 1) / 2
    

    edge_weights = similarity * is_neighbor.unsqueeze(0).float()
    
    neighbor_count = is_neighbor.sum().item()
    
    return edge_weights

def visualize_spectral(
    model, 
    frames,
    patches,
    node_features,
    edge_index,
    edge_weights,
    num_patches,
    grid_size,
    patch_size=32,
    output_dir='./spectral_vis_simple',
    device='cpu'
):
    with torch.no_grad():
        B, T, C, H, W = frames.shape
        os.makedirs(output_dir, exist_ok=True)
        num_effective_frames = 7

        eigvecs, eigvals = model.spectral_feature_extractor.batch_laplacian_spectral_decomposition(
            node_features, edge_index, edge_weights
        )
        
        filter_response = model.spectral_feature_extractor.mlp(eigvals)
        filter_response = torch.sigmoid(filter_response)
        
        patches_flat = patches.view(B, -1, C * patch_size * patch_size).to(device)
        patches_flat = patches_flat[:, :num_effective_frames * num_patches, :]
        
        eigvecs_pixel, eigvals_pixel = model.spectral_feature_extractor.batch_laplacian_spectral_decomposition(
            patches_flat, edge_index, edge_weights
        )
        
        patches_freq = torch.bmm(eigvecs_pixel.transpose(1, 2), patches_flat)

        patches_filtered = patches_freq * filter_response.unsqueeze(-1)

        patches_recon = torch.bmm(eigvecs_pixel, patches_filtered)

        def reconstruct_frame(patches_data, frame_idx):
            p = patches_data.view(num_effective_frames, num_patches, 3, patch_size, patch_size)
            frame_patches = p[frame_idx].view(grid_size, grid_size, 3, patch_size, patch_size)
            
            img = torch.zeros(3, grid_size * patch_size, grid_size * patch_size, device=device)
            for i in range(grid_size):
                for j in range(grid_size):
                    img[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = frame_patches[i, j]
            return img

        fig, axes = plt.subplots(2, num_effective_frames, figsize=(3 * num_effective_frames, 6))
        
        for frame_idx in range(num_effective_frames):
            img_original = reconstruct_frame(patches_flat[0], frame_idx)
            img_filtered = reconstruct_frame(patches_recon[0], frame_idx)

            orig_np = img_original.cpu().numpy().transpose(1, 2, 0)
            orig_np = np.clip(orig_np / 255.0, 0, 1)[..., ::-1]
            axes[0, frame_idx].imshow(orig_np)
            axes[0, frame_idx].set_title(f'Frame {frame_idx}', fontsize=10)
            axes[0, frame_idx].axis('off')

            filt_np = img_filtered.cpu().numpy().transpose(1, 2, 0)
            filt_np = (filt_np - filt_np.min()) / (filt_np.max() - filt_np.min() + 1e-8)
            filt_np = filt_np[..., ::-1]
            axes[1, frame_idx].imshow(filt_np)
            axes[1, frame_idx].axis('off')
        
        axes[0, 0].set_ylabel('Original', fontsize=12)
        axes[1, 0].set_ylabel('Filtered', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'spectral_filtering_all_frames.png'), dpi=150)
        plt.close()
        print(f"Visualization saved to {output_dir}/")


def main():
    video_dir = "data/FaceForensics++/manipulated_sequences/NeuralTextures/c23/frames/022_489"
    num_frames = 8
    patch_size = 32
    output_dir = './spectral_vis'
    

    with open("./configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    model = MMTGNN(
        num_features=cfg["model"]["num_features"],
        num_classes=cfg["model"]["num_classes"],
        back_method=cfg["model"]["back_method"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_frame_layers=cfg["model"]["num_frame_layers"],
        num_diff_layers=cfg["model"]["num_diff_layers"],
        num_frames=cfg["model"]["num_frames"],
        num_patches=cfg["model"]["num_patches"],
        similarity_threshold=cfg["model"]["similarity_threshold"],
        dropout_rate=cfg["model"]["dropout_rate"],
        num_attention_heads=cfg["model"]["multi_head"],
        temporal_window=cfg["model"]["temporal_window"]
    ).to(device)
    
    weights_path = os.path.join(cfg["save_path"], "MMTGNN_best.pth")
    assert os.path.exists(weights_path), f"Checkpoint not found at {weights_path}"
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Loaded weights from: {weights_path}")
    
    
    frames = read_video_frames(video_dir, num_frames)
    frames = frames[0]
    patches = video2patches(frames,32,32)
    frames = torch.from_numpy(frames).unsqueeze(0)
    frames = frames.permute(0,1,4,2,3)
    patches = patches.unsqueeze(0)
    print(f"Loaded {patches.shape} from: {video_dir}")


    model.eval()
    result, intermediates = predict_with_intermediates(model, patches, device)
    
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    
    visualize_spectral(
        model=model,
        frames=frames,
        patches=patches,
        node_features=intermediates['node_features'],
        edge_index=intermediates['full_spatial_edge_index'],
        edge_weights=intermediates['spatial_edge_weights'],
        num_patches=cfg["model"]["num_patches"],
        grid_size=7,
        patch_size=patch_size,
        output_dir=output_dir,
        device=device
    )
    
if __name__ == "__main__":
    main()
