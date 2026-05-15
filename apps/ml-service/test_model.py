import torch
from app.models.siformer_model import Siformer

# 1. Inisialisasi Model
# Kita asumsikan ada 10 kata BISINDO yang ingin dideteksi
model = Siformer(num_classes=10)
model.eval() # Mengatur model ke mode evaluasi (bukan training)

# 2. Simulasi Input dari Tahap 2
# Tensor berukuran [1, 30, 63] -> 1 Batch, 30 Frame, 63 Koordinat (21 titik * 3)
dummy_input = torch.randn(1, 30, 63)

# 3. Melakukan Forward Pass (Proses inferensi)
with torch.no_grad(): # Menonaktifkan perhitungan gradient agar lebih ringan
    output = model(dummy_input)

# 4. Melihat Hasil
print("--- Hasil Inisialisasi Model Siformer ---")
print(f"Bentuk Input: {dummy_input.shape}")
print(f"Bentuk Output (Logits): {output.shape}")
print(f"Prediksi Mentah: \n{output}")

# Mencari indeks dengan nilai tertinggi sebagai hasil prediksi sementara
predicted_class = torch.argmax(output, dim=1)
print(f"\nIndeks Kelas Terprediksi: {predicted_class.item()}")