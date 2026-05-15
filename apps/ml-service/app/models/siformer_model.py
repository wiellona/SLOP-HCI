import torch
import torch.nn as nn

class Siformer(nn.Module):
    def __init__(self, in_channels=3, num_joints=21, num_frames=30, num_classes=10):
        super(Siformer, self).__init__()
        
        # 1. Parameter Dasar
        # in_channels: 3 (X, Y, Z)
        # num_joints: 21 (titik pada satu tangan) atau 42 (dua tangan)
        # num_frames: 30 (panjang sekuens)
        # num_classes: Jumlah kata/isyarat yang bisa dikenali
        
        self.embed_dim = 128
        
        # 2. Linear Projection
        # Mengubah input mentah (21*3 = 63) menjadi representasi 128 dimensi
        self.input_projection = nn.Linear(num_joints * in_channels, self.embed_dim)
        
        # 3. Transformer Encoder
        # Menggunakan struktur standar PyTorch untuk Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=8,                 # 8 mekanisme perhatian (Multi-head attention)
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # 4. Classification Head
        # Mengubah output Transformer menjadi probabilitas kelas (kata)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        # x memiliki bentuk: (Batch, Frames, Features) -> (1, 30, 63)
        
        # Flatten koordinat jika input masih berupa (Batch, Frames, Joints, Channels)
        if len(x.shape) == 4:
            batch, frames, joints, channels = x.shape
            x = x.view(batch, frames, joints * channels)
            
        # Proses Projeksi
        x = self.input_projection(x) # Hasil: (1, 30, 128)
        
        # Proses Transformer
        x = self.transformer_encoder(x) # Hasil: (1, 30, 128)
        
        # Global Average Pooling (Mengambil rata-rata dari 30 frame)
        x = x.mean(dim=1) # Hasil: (1, 128)
        
        # Prediksi
        logits = self.classifier(x) # Hasil: (1, num_classes)
        return logits