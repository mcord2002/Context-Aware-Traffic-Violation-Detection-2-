import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

class CNNLSTM(nn.Module):
    """
    Temporal binary classifier.
    Input: [batch, time, channels, height, width]
    Output logits: [batch, 2] where class 0=normal, class 1=accident.
    """
    def __init__(self, hidden_size=256, num_layers=1, pretrained=True, dropout=0.3):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.cnn = backbone
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, x):
        batch, time, channels, height, width = x.shape
        x = x.reshape(batch * time, channels, height, width)
        features = self.cnn(x).reshape(batch, time, -1)
        sequence, _ = self.lstm(features)
        return self.classifier(self.dropout(sequence[:, -1, :]))
