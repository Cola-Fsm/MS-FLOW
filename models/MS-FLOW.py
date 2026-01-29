import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """
    (Reversible Instance Normalization)
    (Non-stationarity)
    """

    def __init__(self, num_features, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + 1e-10)
        x = x * self.stdev
        x = x + self.mean
        return x


class PatchEmbedding(nn.Module):


    def __init__(self, patch_len, stride, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=stride,
            padding=0,
            bias=False
        )

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        B, L, M = x.shape
        x = x.permute(0, 2, 1).reshape(B * M, 1, L)

        #  Padding
        if (L - self.patch_len) % self.stride != 0:
            pad_l = self.stride - ((L - self.patch_len) % self.stride)
            x = F.pad(x, (0, pad_l), mode='replicate')

        x_emb = self.proj(x)  # -> [B*M, d_model, Num_Patches]
        _, d, n = x_emb.shape
        x_emb = x_emb.reshape(B, M, d, n).permute(0, 1, 3, 2)
        # Output: [Batch, Channels, Num_Patches, d_model]
        return x_emb


class DynamicGraphLearner(nn.Module):


    def __init__(self, num_channels, d_model, top_k=10, dropout=0.1):
        super().__init__()
        self.top_k = min(top_k, num_channels)


        self.lin_q = nn.Linear(d_model, 64)
        self.lin_k = nn.Linear(d_model, 64)

   
        self.gcn_weight = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        # x Input: [Batch, Num_Patches, Channels, d_model]

        B, N, C, D = x.shape  #torch.Size([64, 11, 7, 256])


        # node_feat: [B, C, D]
        node_feat = x.mean(dim=1)

        Q = self.lin_q(node_feat)  # [B, C, 64]
        K = self.lin_k(node_feat)  # [B, C, 64]

        # logits: [B, C, C]
        adj = torch.matmul(Q, K.transpose(-2, -1)) / 8.0  # /sqrt(64) torch.Size([64, 7, 7])


        if self.top_k < C:
            topk_val, topk_idx = torch.topk(adj, k=self.top_k, dim=-1)   #torch.Size([64, 7, 5])
            mask = torch.zeros_like(adj)#torch.Size([64, 7, 7])
            mask.scatter_(-1, topk_idx, 1)
            adj = adj.masked_fill(mask == 0, -1e9)

        adj = F.softmax(adj, dim=-1)  # [B, C, C] #torch.Size([16, 862, 862])
        adj = self.dropout(adj)

        # 3. Dynamic Graph Convolution)

        # x: [B, N, C, D] -> [B, N, C, D]
        x_trans = self.gcn_weight(x)


        # x_trans: [B, N, C, D] -> Permute to [B, N, D, C] [B, C, C] 
        x_trans = x_trans.permute(0, 1, 3, 2)

        # [B, N, D, C] x [B, 1, C, C] (Broadcast adj) -> [B, N, D, C]
        out = torch.matmul(x_trans, adj.unsqueeze(1).transpose(-1, -2))

        # [B, N, C, D]
        out = out.permute(0, 1, 3, 2)

        return self.act(out)


class GraphPatchBlock(nn.Module):
    def __init__(self, num_patches, num_channels, d_model, top_k=10, dropout=0.1):
        super().__init__()

        # 1. Time Mixing (Patch Interaction) - MLP
        self.norm1 = nn.BatchNorm2d(num_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(num_patches, num_patches),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_patches, num_patches)
        )

        # 2. Channel Mixing (Variable Interaction) - Dynamic Graph

        self.norm2 = nn.BatchNorm2d(num_channels)
        self.graph_learner = DynamicGraphLearner(num_channels, d_model, top_k, dropout)

        # 3. FFN (Feature Mixing)
        self.norm3 = nn.BatchNorm2d(num_channels)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x):
        # x Input: [Batch, Channels, Num_Patches, d_model]
        B, M, N, D = x.shape

        # --- Part 1: Time Mixing (MLP along N) ---
        res = x
        # [B, M, D, N]  
        x_time = x.permute(0, 1, 3, 2)
        x_time = self.time_mlp(x_time)#torch.Size([64, 7, 256, 11])
        x = res + x_time.permute(0, 1, 3, 2)    #torch.Size([64, 7, 11, 256])
        x = self.norm1(x)

        # --- Part 2: Channel Mixing (Graph along M) ---
        res = x

        x_graph = x.permute(0, 2, 1, 3)
        x_graph = self.graph_learner(x_graph)

        x_graph = x_graph.permute(0, 2, 1, 3)
        x = res + x_graph
        x = self.norm2(x)

        # --- Part 3: FFN (MLP along D) ---
        res = x
        x_ffn = self.ffn(x)
        x = res + x_ffn
        x = self.norm3(x)

        return x


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        enc_in = self.enc_in = args.enc_in
        seq_len = self.seq_len = args.seq_len
        pred_len = self.pred_len = args.pred_len
        patch_len = self.patch_len = args.patch_len if hasattr(args, 'patch_len') else 16
        stride = self.stride = args.stride if hasattr(args, 'stride') else 8
        d_model = self.d_model = args.d_model if hasattr(args, 'd_model') else 128
        e_layers = self.e_layers = args.e_layers if hasattr(args, 'e_layers') else 3
        top_k = self.top_k = args.top_k if hasattr(args, 'top_k') else 5
        dropout = self.dropout = args.dropout if hasattr(args, 'dropout') else 0.1


        self.revin = RevIN(enc_in)


        pad_l = 0
        if (seq_len - patch_len) % stride != 0:
            pad_l = stride - ((seq_len - patch_len) % stride)
        self.num_patches = (seq_len + pad_l - patch_len) // stride + 1

        # Embedding
        self.patch_embed = PatchEmbedding(patch_len, stride, d_model)

        # Backbone 
        self.encoder = nn.ModuleList([
            GraphPatchBlock(self.num_patches, enc_in, d_model, top_k, dropout)
            for _ in range(e_layers)
        ])

        # Prediction Head
        # Flatten Patch -> Linear -> Output
        self.head = nn.Linear(self.num_patches * d_model, pred_len)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        B, L, M = x.shape

        # 1. Instance Normalization
        x = self.revin(x, 'norm') #torch.Size([16, 96, 862])

        # 2. Patching & Embedding
        # Output: [Batch, Channels, Num_Patches, d_model]
        x_enc = self.patch_embed(x) #torch.Size([16, 862, 11, 256])

        # 3. Graph Patch Layers
        for layer in self.encoder:
            x_enc = layer(x_enc)

        # 4. Prediction Head
        # Flatten: [Batch, Channels, Num_Patches * d_model]
        out = x_enc.reshape(B, M, -1)

        # Linear Projection: [Batch, Channels, Pred_Len]
        out = self.head(out)

        # Permute: [Batch, Pred_Len, Channels]
        out = out.permute(0, 2, 1)

        # 5. De-Normalization
        out = self.revin(out, 'denorm')

        return out
