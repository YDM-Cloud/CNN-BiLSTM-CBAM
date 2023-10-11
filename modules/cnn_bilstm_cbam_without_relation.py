import torch
import torch.nn as nn
from .bilstm import BiLSTM
from .attention import CBAM


class CNN_BiLSTM_CBAM_Without_Relation(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # cloud
        self.cloud_feat = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        # control
        self.control_feat = BiLSTM(3, 256, kwargs['device'])
        self.attention = CBAM(256)
        self.fc1 = nn.Linear(1248, 1208)
        self.output1 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1)
        )
        self.fc2 = nn.Linear(1248, 20)
        self.output2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1)
        )

    def forward(self, control_point, control_changed_point, point_cloud, relation):
        control_point_feat = self.control_feat(control_point)
        control_point_feat = control_point_feat.permute(1, 2, 0)
        control_changed_point_feat = self.control_feat(control_changed_point)
        control_changed_point_feat = control_changed_point_feat.permute(1, 2, 0)
        point_cloud_feat = self.cloud_feat(point_cloud.permute(0, 2, 1))

        concat_feat = torch.cat([control_point_feat, control_changed_point_feat,
                                 point_cloud_feat], dim=2)

        concat_attn = self.attention(concat_feat.permute(0, 2, 1))
        concat_feat = concat_feat + concat_attn.permute(0, 2, 1)

        out1 = self.fc1(concat_feat)
        out1 = self.output1(out1).permute(0, 2, 1)

        out2 = self.fc2(concat_feat)
        out2 = self.output2(out2).permute(0, 2, 1)
        return out1, out2
