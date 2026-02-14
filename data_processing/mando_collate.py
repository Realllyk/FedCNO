import torch


def _pad_node_features(node_features_list):
    # 将不同图的节点特征补齐到同一长度，便于组成 batch 张量
    max_nodes = max(x.shape[0] for x in node_features_list)
    feat_dim = node_features_list[0].shape[1]
    batch_size = len(node_features_list)

    padded = torch.zeros((batch_size, max_nodes, feat_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_nodes), dtype=torch.float32)
    for i, feat in enumerate(node_features_list):
        n = feat.shape[0]
        padded[i, :n, :] = feat
        # mask=1 表示真实节点，0 表示补齐节点
        mask[i, :n] = 1.0
    return padded, mask


def mando_collate_fn(batch):
    # 普通训练: (graph_item, aux, label) -> (x1, x2, y)
    graphs, aux_list, labels = zip(*batch)
    node_features_list = [g["node_features"].float() for g in graphs]
    padded, mask = _pad_node_features(node_features_list)
    # x1 用字典保存图特征与 mask，便于模型内部做掩码池化
    x1 = {"node_features": padded, "mask": mask}
    x2 = torch.stack(aux_list, dim=0).float()
    y = torch.stack(labels, dim=0).long()
    return x1, x2, y


def lgv_mando_collate_fn(batch):
    # LGV 训练: 额外携带 agreement_ratio
    graphs, aux_list, labels, agr_list = zip(*batch)
    node_features_list = [g["node_features"].float() for g in graphs]
    padded, mask = _pad_node_features(node_features_list)
    x1 = {"node_features": padded, "mask": mask}
    x2 = torch.stack(aux_list, dim=0).float()
    y = torch.stack(labels, dim=0).long()
    agr = torch.stack(
        [a.float() if torch.is_tensor(a) else torch.tensor(a, dtype=torch.float32) for a in agr_list],
        dim=0
    )
    return x1, x2, y, agr
