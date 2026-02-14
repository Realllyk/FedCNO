import torch
import torch.nn as nn
import torch.nn.functional as F


class MANDONet(nn.Module):
    """
    Lightweight MANDO-style encoder for cached contract graphs.
    Expects x1 as:
      - dict(node_features: [B, N, F], mask: [B, N])
      - or tensor [B, N, F]
    x2 is unused and kept for interface compatibility.
    """

    def __init__(self, input_dim=80, hidden_dim=128, embed_dim=128, num_classes=2, dropout=0.2):
        super().__init__()
        # inter_outputs 给 LGV 的 KNN 逻辑复用
        self.inter_outputs = None
        self.node_mlp_1 = nn.Linear(input_dim, hidden_dim)
        self.node_mlp_2 = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def encode_graph(self, x1):
        if isinstance(x1, dict):
            # 推荐输入: {"node_features": [B,N,F], "mask": [B,N]}
            node_features = x1["node_features"]
            mask = x1.get("mask", None)
        else:
            # 兼容输入: 直接传 [B,N,F]
            node_features = x1
            mask = None

        h = F.relu(self.node_mlp_1(node_features))
        h = self.dropout(h)
        h = F.relu(self.node_mlp_2(h))

        if mask is None:
            # 无 mask 时直接均值池化
            pooled = h.mean(dim=1)
        else:
            # 有 mask 时忽略补齐节点
            mask = mask.float().unsqueeze(-1)
            denom = torch.clamp(mask.sum(dim=1), min=1.0)
            pooled = (h * mask).sum(dim=1) / denom

        emb = F.relu(self.proj(pooled))
        return emb

    def forward(self, x1, x2=None):
        # x2 为接口占位参数，保持与 CBGRU/CGE forward(x1, x2) 一致
        emb = self.encode_graph(x1)
        self.inter_outputs = emb.detach()
        return self.classifier(emb)
