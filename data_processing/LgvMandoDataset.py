import torch
from data_processing.MandoDataset import MandoDataset


class LgvMandoDataset(MandoDataset):
    def __init__(self, graph_dir, labels_path, names_path, feature_dim=80):
        super().__init__(graph_dir, labels_path, names_path, feature_dim=feature_dim)
        # 与 LgvCgeDataset 行为对齐，默认一致性权重全 1
        self.agreement_ratio = torch.ones(len(self.labels), dtype=torch.float32)

    def __getitem__(self, idx):
        # LGV 训练额外返回 agreement_ratio
        graph_item, aux, label = super().__getitem__(idx)
        return graph_item, aux, label, self.agreement_ratio[idx]

    def set_ag_rt(self, agreement_ratio):
        # 供 Fed_LGV_client 在训练过程中动态更新
        self.agreement_ratio = torch.tensor(agreement_ratio, dtype=torch.float32)
