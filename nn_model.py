import torch
from torch import nn
from torch.nn import functional as F

# -----------------------------
# 2-Layer 1D CNN
# -----------------------------
class CNN_1D_2L(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.n_in = n_in

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )

        # After two /2 pools, time length = n_in // 4; channels = 128
        self.linear1 = nn.Linear(self.n_in * 128 // 4, 4)

    def forward(self, x, return_feats: bool = False):
        # x: (B, n_in) -> (B, 1, n_in)
        x = x.view(-1, 1, self.n_in)
        x = self.layer1(x)
        x = self.layer2(x)

        # --- Features for CORAL: GAP over time => (B, 128) ---
        feats = x.mean(dim=2)

        # --- Logits for classification: use flattened activations (unchanged) ---
        logits = self.linear1(x.view(x.size(0), -1))

        if return_feats:
            return logits, feats
        return logits


# -----------------------------
# 3-Layer 1D CNN
# -----------------------------
class CNN_1D_3L(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.n_in = n_in

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # After three /2 pools, time length = n_in // 8; channels = 128
        self.linear1 = nn.Linear(self.n_in * 128 // 8, 4)

    def forward(self, x, return_feats: bool = False):
        # x: (B, n_in) -> (B, 1, n_in)
        x = x.view(-1, 1, self.n_in)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # --- Features for CORAL: GAP over time => (B, 128) ---
        feats = x.mean(dim=2)

        # --- Logits for classification: use flattened activations (unchanged) ---
        logits = self.linear1(x.view(x.size(0), -1))

        if return_feats:
            return logits, feats
        return logits