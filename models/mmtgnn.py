import torch
from torch.nn import ReLU
from torch_geometric.nn import GCNConv,  global_mean_pool, GATConv

import torch.nn.functional as F
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

import timm
import torchvision.models as models
from .spectral import *

device =  "cuda:0" if torch.cuda.is_available() else "cpu"
class MMTGNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, back_method="resnet18", hidden=64, Lf=3, Ld=1, T=8, Ns=49, threshold = 0.6, dropout=0.3, multi_head=-1):
        super(MMTGNN,self).__init__()
        self.Lf = Lf
        self.Ld = Ld
        self.threshold = torch.nn.Threshold(threshold,0)
        self.conv1 = GATConv(hidden, hidden)
        self.relu1 = ReLU()
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.conv2 = GATConv(hidden, hidden)
        self.relu2 = ReLU()
        self.dropout2 = torch.nn.Dropout(p=dropout)
        self.conv3 = GCNConv(hidden, hidden)
        self.relu3 = ReLU()

        
        self.dropout3 = torch.nn.Dropout(p=dropout)

        #use different kind of backbones to preprocess the pixel features
        if back_method == "resnet50":
            pretrain = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
            self.backbone = torch.nn.Sequential(pretrain.conv1,pretrain.bn1,pretrain.relu,pretrain.maxpool,pretrain.layer1,pretrain.layer2,pretrain.avgpool)    
            self.align = torch.nn.Linear(512,hidden).to(device)

        elif back_method == "resnet50_f":    
            pretrain = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
            self.backbone = torch.nn.Sequential(*list(pretrain.children())[:-1])
            self.align = torch.nn.Linear(2048, hidden).to(device)
        
        elif back_method == "xception":
            self.backbone = timm.create_model('xception', pretrained=True)
            self.backbone.fc = torch.nn.Identity()
            self.align = torch.nn.Linear(2048, hidden).to(device)
        
        self.motion_layers = []
        for t in range(Ld):        
            self.motion_layers.append(torch.nn.Sequential(torch.nn.Linear(hidden*2,hidden),torch.nn.ReLU(),torch.nn.Linear(hidden,hidden)).to(device))
            self.register_module("motion_layer_{}".format(t), self.motion_layers[-1])
            
        
        e_i= [[],[]]
        start = 0
        for d in range(self.Ld):
            N = (T-d-1)*Ns
            for u in range(N):
                for v in range(N):
                    e_i[0].append(u+start)
                    e_i[1].append(v+start)
            start = start+N
        self.edge_indexd =  torch.tensor(e_i,dtype=torch.int64).to(device)                                                                                            
        self.multi_head = multi_head
        self.hidden = hidden
        
        if multi_head!=-1:
            self.head_dim = hidden//multi_head
            self.head_weight = []
            for i in range(4):
                self.head_weight.append([])
                for j in range(multi_head):
                    self.head_weight[i].append(torch.nn.Linear( self.head_dim , self.head_dim ).to(device))
                    self.register_module("head_weight_{}_{}".format(i,j), self.head_weight[i][-1])
        else:
            self.process_weight=[]
            for i in range(4):
                self.process_weight.append(torch.nn.Linear( self.hidden , self.hidden ).to(device))
                self.register_module("process_weight{}".format(i), self.process_weight[-1])

        self.spectral_layer = SpectralFeatureExtractor(in_dim=(T-1)*Ns, hidden_dim=64, feat_dim=self.hidden)
        self.final_layer = torch.nn.Linear(hidden*3,num_classes).to(device)
        
        
    def patches2rep(self, patch):
        rep = self.backbone(patch)
        rep = torch.flatten(rep, 1)
        rep = self.dropout3(rep)
        rep = self.align(rep)
        return rep
    
    def interpolate(self, img, factor):
        return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True), scale_factor=1/factor, mode='nearest', recompute_scale_factor=True)

        
    def construct_differential(self, in_patches, is_NPR=False):
        # node: (frame_number-d-1) * patches
        # output: node_features->
        # output: e_w
        batch_size, frame_number, patches, c, h, w = in_patches.shape
        
        in_patches = in_patches.view(batch_size * frame_number * patches, c, h, w)
        
        if is_NPR == True:
            NPR = in_patches - self.interpolate(in_patches, 0.5)
            reps = self.patches2rep(NPR)
        else:
            reps = self.patches2rep(in_patches)
        
        reps = reps.view(batch_size, frame_number, patches, -1)
        
        reps = reps.permute(1, 2, 0, 3)
        
        node_featrues = None
        d = self.Ld-1

        cat_feature = []
        for t in range(frame_number-d-1):
            concatenated = torch.cat([reps[t], reps[t + d + 1]], dim=-1)
            cat_feature.append(concatenated)
        cat_features = torch.stack(cat_feature, dim=0)
        cat_features = cat_features.view(-1, cat_features.size(-1))
        node_featrues = self.motion_layers[d](cat_features)
        node_featrues = node_featrues.view((frame_number-d-1) * patches, batch_size, self.hidden)
        node_featrues = node_featrues.permute(1, 0, 2)
        

        X_f = reps[:frame_number-d-1]
        X_t = reps[d+1:]
            
        norms = torch.linalg.norm(X_f, axis=3, keepdims=True)  
        Xf_normed = X_f / (norms+1e-4)
        Xf_normed = Xf_normed.view((frame_number-d-1) * patches, batch_size, self.hidden)
        Sf = torch.matmul(Xf_normed.permute(1,0,2), Xf_normed.permute(1,2,0)) # [batch_size, (frame_number-d-1) * patches, (frame_number-d-1) * patches]
            
        norms = torch.linalg.norm(X_t, axis=3, keepdims=True) 
        Xt_normed = X_t / (norms+1e-4)  
        Xt_normed = Xt_normed.view((frame_number-d-1) * patches, batch_size, self.hidden)
        St = torch.matmul(Xt_normed.permute(1,0,2), Xt_normed.permute(1,2,0)) # [batch_size, (frame_number-d-1) * patches, (frame_number-d-1) * patches]

        max_S, _ = torch.max(torch.stack([Sf, St], dim=0), dim=0)
            
        max_S = self.threshold(max_S)
        
        edge_weight = max_S.view(batch_size,-1)
        data = []
        for i in range(batch_size):
            data.append(Data(x=node_featrues[i], edge_index=self.edge_indexd, edge_attr=edge_weight[i]))
        batch = Batch.from_data_list(data)
        return batch, node_featrues, edge_weight
    
    
    
    def forward(self, patches):
        
        batch_graph, node_featrues, edge_weight = self.construct_differential(patches, is_NPR = True)
        spectral_feat = self.spectral_layer(node_featrues, node_featrues, self.edge_indexd, edge_weight)
        
        stackd = []

        if self.multi_head!=-1:
            out1d = self.conv1(batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr)
            out1d = torch.cat([self.head_weight[1][i](out1d[:,i*self.head_dim:(i+1)*self.head_dim]) for i in range(self.multi_head)],dim=1)
        else:
            out1d = self.conv1(batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr)
            out1d = self.process_weight[1](out1d)
        out1d = torch.nn.functional.normalize(out1d, p=2, dim=1) 
        out1d = self.relu1(out1d)
        out1d = self.dropout1(out1d)
        stackd.append(out1d)

        if self.multi_head!=-1:
            out2d = self.conv2(out1d, batch_graph.edge_index, batch_graph.edge_attr)
            out2d = torch.cat([self.head_weight[3][i](out2d[:,i*self.head_dim:(i+1)*self.head_dim]) for i in range(self.multi_head)],dim=1)
            out2d = torch.nn.functional.normalize(out2d, p=2, dim=1) 
            out2d = self.relu2(out2d)
        else:
            out2d = self.conv2(out1d, batch_graph.edge_index, batch_graph.edge_attr)
            out2d = self.process_weight[4](out2d)
            out2d = torch.nn.functional.normalize(out2d, p=2, dim=1) 
            out2d = self.relu2(out2d)
        
        stackd.append(out2d)
        
        
        """
        out3d = self.conv3(out2d, self.edge_indexd,edge_weight_d)
        out3d = torch.nn.functional.normalize(out3d, p=2, dim=1) 
        out3d = self.relu3(out3d)
        stackd.append(out3d)
        """    
        stackd = torch.cat(stackd, dim=1)
        spatial_feat = global_mean_pool(stackd, batch=batch_graph.batch)
        spectral_feat = spectral_feat.reshape(-1, self.hidden)
        spectral_feat = global_mean_pool(spectral_feat, batch=batch_graph.batch)
        embedding = torch.cat([spatial_feat, spectral_feat], dim=1)  # [B, 3*hidden]

        embedding = self.dropout2(embedding)
        
        out = self.final_layer(embedding)
        #if turn to a simple linear transform of preprocess patches
        return out