import torch
from torch.nn import ReLU
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import timm
import torchvision.models as models
from .spectral import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class MMTGNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, back_method="resnet18", hidden_dim=64, 
                 num_frame_layers=3, num_diff_layers=1, num_frames=8, num_patches=49, 
                 similarity_threshold=0.6, dropout_rate=0.3, num_attention_heads=-1, 
                 temporal_window=2):

        super(MMTGNN, self).__init__()
        
        self.num_frame_layers = num_frame_layers
        self.num_diff_layers = num_diff_layers
        self.num_frames = num_frames
        self.num_patches = num_patches
        self.temporal_window = temporal_window
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        
        self.spatial_threshold = torch.nn.Threshold(similarity_threshold, 0)
        self.temporal_threshold = torch.nn.Threshold(similarity_threshold, 0)
        
        self.spatial_gat_conv1 = GATConv(hidden_dim, hidden_dim)
        self.spatial_relu1 = ReLU()
        self.spatial_dropout1 = torch.nn.Dropout(p=dropout_rate)
        
        self.spatial_gat_conv2 = GATConv(hidden_dim, hidden_dim)
        self.spatial_relu2 = ReLU()
        self.spatial_dropout2 = torch.nn.Dropout(p=dropout_rate)
        
        self.temporal_gat_conv = GATConv(hidden_dim, hidden_dim)
        self.temporal_relu = ReLU()
        self.temporal_dropout = torch.nn.Dropout(p=dropout_rate)
        
        self.feature_dropout = torch.nn.Dropout(p=dropout_rate)
        
        if back_method == "resnet18":            
            pretrain = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
            self.backbone = torch.nn.Sequential(
                pretrain.conv1,
                pretrain.bn1,
                pretrain.relu,
                pretrain.maxpool,
                pretrain.layer1,
                pretrain.layer2,
                pretrain.layer3,
                pretrain.layer4,
                pretrain.avgpool
            )
            self.align = torch.nn.Linear(512, hidden_dim).to(device)

        elif back_method == "resnet50":
            pretrain = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
            self.backbone = torch.nn.Sequential(pretrain.conv1,pretrain.bn1,pretrain.relu,pretrain.maxpool,pretrain.layer1,pretrain.layer2,pretrain.avgpool)    
            self.align = torch.nn.Linear(512,hidden_dim).to(device)
        
        self.motion_encoders = []
        for layer_idx in range(num_diff_layers):
            motion_encoder = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim * 2, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            ).to(device)
            self.motion_encoders.append(motion_encoder)
            self.register_module(f"motion_encoder_{layer_idx}", motion_encoder)
        

        self.spatial_edge_index = self._build_spatial_edges(num_patches).to(device)
        self.temporal_edge_index = self._build_temporal_edges(
            num_frames, num_patches, num_diff_layers, temporal_window
        ).to(device)
        
        if num_attention_heads != -1:
            self.head_dim = hidden_dim // num_attention_heads
            self.attention_head_weights = []

            for weight_idx in range(5):
                head_weights = []
                for head_idx in range(num_attention_heads):
                    weight = torch.nn.Linear(self.head_dim, self.head_dim).to(device)
                    head_weights.append(weight)
                    self.register_module(f"attention_weight_{weight_idx}_{head_idx}", weight)
                self.attention_head_weights.append(head_weights)
        else:
            self.projection_weights = []
            for weight_idx in range(5):
                weight = torch.nn.Linear(hidden_dim, hidden_dim).to(device)
                self.projection_weights.append(weight)
                self.register_module(f"projection_weight_{weight_idx}", weight)
        
        num_effective_frames = num_frames - 1
        self.spectral_feature_extractor = SpectralFeatureExtractor(
            in_dim=num_effective_frames * num_patches,
            hidden_dim=64,
            feat_dim=hidden_dim
        )
        
        self.classifier = torch.nn.Linear(hidden_dim * 4, num_classes).to(device)
    
    def _build_spatial_edges(self, num_patches):
        edge_list = [[], []]
        for src_patch in range(num_patches):
            for dst_patch in range(num_patches):
                edge_list[0].append(src_patch)
                edge_list[1].append(dst_patch)
        return torch.tensor(edge_list, dtype=torch.int64)
    
    def _build_temporal_edges(self, num_frames, num_patches, num_diff_layers, temporal_window):
        frame_offset = num_diff_layers - 1
        num_effective_frames = num_frames - frame_offset - 1
        edge_list = [[], []]
        
        for patch_idx in range(num_patches):
            for frame_t1 in range(num_effective_frames):
                if temporal_window == -1:
                    frame_range = range(num_effective_frames)
                else:
                    frame_range = range(
                        frame_t1 + 1,
                        min(frame_t1 + temporal_window + 1, num_effective_frames)
                    )
                
                for frame_t2 in frame_range:
                    node_t1 = frame_t1 * num_patches + patch_idx
                    node_t2 = frame_t2 * num_patches + patch_idx
                    
                    edge_list[0].extend([node_t1, node_t2])
                    edge_list[1].extend([node_t2, node_t1])
        
        return torch.tensor(edge_list, dtype=torch.int64)
    
    def patches2rep(self, patch):
        rep = self.backbone(patch)
        rep = torch.flatten(rep, 1)
        rep = self.feature_dropout(rep)
        rep = self.align(rep)
        return rep
    
    def interpolate(self, img, factor):
        return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True), scale_factor=1/factor, mode='nearest', recompute_scale_factor=True)

    
    def construct_dual_graphs(self, patch_images, use_npr=False):
        batch_size, num_frames, num_patches, channels, height, width = patch_images.shape
        flat_patches = patch_images.view(batch_size * num_frames * num_patches, channels, height, width)
        x_in = flat_patches

        if use_npr:
            x_in = x_in - self.interpolate(x_in, 0.5)

        patch_features = self.patches2rep(x_in)
        
        patch_features = patch_features.view(batch_size, num_frames, num_patches, -1)
        patch_features = patch_features.permute(1, 2, 0, 3)
        
        frame_offset = self.num_diff_layers - 1
        num_effective_frames = num_frames - frame_offset - 1

        frame_pairs = []
        for frame_idx in range(num_effective_frames):
            current_frame_feat = patch_features[frame_idx]
            future_frame_feat = patch_features[frame_idx + frame_offset + 1]
            concatenated = torch.cat([current_frame_feat, future_frame_feat], dim=-1)
            frame_pairs.append(concatenated)
        
        frame_pairs = torch.stack(frame_pairs, dim=0)
        frame_pairs_flat = frame_pairs.view(-1, frame_pairs.size(-1))

        motion_features = self.motion_encoders[frame_offset](frame_pairs_flat)

        motion_features = motion_features.view(
            num_effective_frames * num_patches, 
            batch_size, 
            self.hidden_dim
        )
        node_features = motion_features.permute(1, 0, 2)
        
        spatial_edge_weights_list = []
        
        for batch_idx in range(batch_size):
            batch_spatial_weights = []
            
            for frame_idx in range(num_effective_frames):
                fused_frame_features = node_features[batch_idx, frame_idx * num_patches : (frame_idx + 1) * num_patches, :]
                normalized_features = F.normalize(fused_frame_features, p=2, dim=1)
                similarity_matrix = torch.matmul(normalized_features, normalized_features.t())
                
                thresholded_similarity = self.spatial_threshold(similarity_matrix)

                batch_spatial_weights.append(thresholded_similarity.view(-1))

            spatial_edge_weights_list.append(torch.cat(batch_spatial_weights))
            
        
        spatial_edge_weights = torch.stack(spatial_edge_weights_list)  # [batch, total_spatial_edges]

        temporal_edge_weights_list = []
        
        for batch_idx in range(batch_size):
            temporal_weights = []
            
            for patch_idx in range(num_patches):
                avg_patch_features = node_features[batch_idx, patch_idx::num_patches, :]
                normalized_patch_features = F.normalize(avg_patch_features, p=2, dim=1)

                for frame_t1 in range(num_effective_frames):
                    if self.temporal_window == -1:
                        frame_range = range(num_effective_frames)
                    else:
                        frame_range = range(
                            frame_t1 + 1,
                            min(frame_t1 + self.temporal_window + 1, num_effective_frames)
                        )
                    
                    for frame_t2 in frame_range:
                        similarity = torch.dot(
                            normalized_patch_features[frame_t1],
                            normalized_patch_features[frame_t2]
                        )
                        
                        thresholded_difference = (1 - similarity)
                        temporal_weights.extend([thresholded_difference, thresholded_difference])
            
            temporal_edge_weights_list.append(torch.stack(temporal_weights))
        
        temporal_edge_weights = torch.stack(temporal_edge_weights_list)  # [batch, total_temporal_edges]
        
        spatial_data_list = []
        temporal_data_list = []
        
        for batch_idx in range(batch_size):
            batch_node_features = node_features[batch_idx]

            spatial_edges_all_frames = []
            for frame_idx in range(num_effective_frames):
                frame_offset_nodes = frame_idx * num_patches
                shifted_edge_index = self.spatial_edge_index + frame_offset_nodes
                spatial_edges_all_frames.append(shifted_edge_index)
            
            spatial_edges_combined = torch.cat(spatial_edges_all_frames, dim=1)

            batch_spatial_edge_weights = spatial_edge_weights[batch_idx]
            
            spatial_data_list.append(Data(
                x=batch_node_features,
                edge_index=spatial_edges_combined,
                edge_attr=batch_spatial_edge_weights
            ))
            
            batch_temporal_edge_weights = temporal_edge_weights[batch_idx]
            
            temporal_data_list.append(Data(
                x=batch_node_features,
                edge_index=self.temporal_edge_index,
                edge_attr=batch_temporal_edge_weights
            ))
        
        spatial_batch = Batch.from_data_list(spatial_data_list)
        temporal_batch = Batch.from_data_list(temporal_data_list)

        
        return spatial_batch, temporal_batch, node_features, spatial_edge_weights
    
    def _get_full_spatial_edge_index(self, num_effective_frames, num_patches):
        spatial_edges_all_frames = []
        for frame_idx in range(num_effective_frames):
            frame_offset_nodes = frame_idx * num_patches
            shifted_edge_index = self.spatial_edge_index + frame_offset_nodes
            spatial_edges_all_frames.append(shifted_edge_index)
        
        return torch.cat(spatial_edges_all_frames, dim=1)
    
    def forward(self, patch_images, return_intermediates=False):
        spatial_batch, temporal_batch, node_features, spatial_edge_weights = \
            self.construct_dual_graphs(patch_images, use_npr=True)
        
        full_spatial_edge_index = self._get_full_spatial_edge_index(
            node_features.shape[1] // self.num_patches,
            self.num_patches
        )
        
        spectral_features = self.spectral_feature_extractor(
            node_features,
            full_spatial_edge_index,
            spatial_edge_weights
        )
        
        spatial_out1 = self.spatial_gat_conv1(
            spatial_batch.x,
            spatial_batch.edge_index,
            spatial_batch.edge_attr
        )
        
        if self.num_attention_heads != -1:
            head_outputs = []
            for head_idx in range(self.num_attention_heads):
                head_start = head_idx * self.head_dim
                head_end = (head_idx + 1) * self.head_dim
                head_out = self.attention_head_weights[0][head_idx](
                    spatial_out1[:, head_start:head_end]
                )
                head_outputs.append(head_out)
            spatial_out1 = torch.cat(head_outputs, dim=1)
        else:
            spatial_out1 = self.projection_weights[0](spatial_out1)
        
        spatial_out1 = F.normalize(spatial_out1, p=2, dim=1)
        spatial_out1 = self.spatial_relu1(spatial_out1)
        spatial_out1 = self.spatial_dropout1(spatial_out1)
        
        spatial_out2 = self.spatial_gat_conv2(
            spatial_out1,
            spatial_batch.edge_index,
            spatial_batch.edge_attr
        )
        
        if self.num_attention_heads != -1:
            head_outputs = []
            for head_idx in range(self.num_attention_heads):
                head_start = head_idx * self.head_dim
                head_end = (head_idx + 1) * self.head_dim
                head_out = self.attention_head_weights[1][head_idx](
                    spatial_out2[:, head_start:head_end]
                )
                head_outputs.append(head_out)
            spatial_out2 = torch.cat(head_outputs, dim=1)
        else:
            spatial_out2 = self.projection_weights[1](spatial_out2)
        
        spatial_out2 = F.normalize(spatial_out2, p=2, dim=1)
        spatial_out2 = self.spatial_relu2(spatial_out2)
        
        spatial_features_combined = torch.cat([spatial_out1, spatial_out2], dim=1)
        spatial_features_pooled = global_mean_pool(spatial_features_combined, batch=spatial_batch.batch)
        
        temporal_out = self.temporal_gat_conv(
            temporal_batch.x,
            temporal_batch.edge_index,
            temporal_batch.edge_attr
        )
        
        if self.num_attention_heads != -1:
            head_outputs = []
            for head_idx in range(self.num_attention_heads):
                head_start = head_idx * self.head_dim
                head_end = (head_idx + 1) * self.head_dim
                head_out = self.attention_head_weights[2][head_idx](
                    temporal_out[:, head_start:head_end]
                )
                head_outputs.append(head_out)
            temporal_out = torch.cat(head_outputs, dim=1)
        else:
            temporal_out = self.projection_weights[2](temporal_out)
        
        temporal_out = F.normalize(temporal_out, p=2, dim=1)
        temporal_out = self.temporal_relu(temporal_out)
        temporal_out = self.temporal_dropout(temporal_out)
        
        temporal_features_pooled = global_mean_pool(temporal_out, batch=temporal_batch.batch)
        
        spectral_features = spectral_features.reshape(-1, self.hidden_dim)
        spectral_features_pooled = global_mean_pool(spectral_features, batch=spatial_batch.batch)
        
        final_embedding = torch.cat([
            spatial_features_pooled,
            spectral_features_pooled,
            temporal_features_pooled
        ], dim=1)
        
        final_embedding = self.spatial_dropout2(final_embedding)
        
        logits = self.classifier(final_embedding)
        
        if return_intermediates:
            intermediates = {
                'node_features': node_features,
                'full_spatial_edge_index': full_spatial_edge_index,
                'spatial_edge_weights': spatial_edge_weights
            }
            return logits, intermediates
        else:
            return logits
