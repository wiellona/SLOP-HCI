# app/models/siformer_model.py
import torch
import torch.nn as nn

class Siformer(nn.Module):
    def __init__(self, in_channels=3, num_joints=21, num_frames=30, num_classes=18):
        super(Siformer, self).__init__()
        
        self.embed_dim = 128
        
        # Linear projection
        self.input_projection = nn.Linear(num_joints * in_channels, self.embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Classification head (match dengan checkpoint: 18 classes)
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        
    def forward(self, x):
        # x shape: (Batch, Frames, Joints * Channels) -> (B, 30, 63)
        
        # Project input
        x = self.input_projection(x)  # (B, 30, 128)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (B, 30, 128)
        
        # Global average pooling
        x = x.mean(dim=1)  # (B, 128)
        
        # Classification
        logits = self.classifier(x)  # (B, num_classes)
        
        return logits