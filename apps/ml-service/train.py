import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob

# Mengimport arsitektur Siformer yang sudah kita buat di Tahap 3
from app.models.siformer_model import Siformer

# --- 1. DEFINISI DATASET CUSTOM ---
# Kelas ini bertugas mencari dan memuat file .npy yang sudah kamu ekstrak
class BISINDODataset(Dataset):
    def __init__(self, data_path, sequence_length=30):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.samples = []
        self.labels = []

        if not os.path.isdir(data_path):
            raise FileNotFoundError(
                "Folder dataset tidak ditemukan: "
                f"{data_path}. Pastikan path benar dan cwd: {os.getcwd()}"
            )
        
        # Mencari semua folder kelas (Halo, Terima Kasih, dll)
        self.classes = sorted(
            [
                name
                for name in os.listdir(data_path)
                if os.path.isdir(os.path.join(data_path, name))
            ]
        )
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"Ditemukan {len(self.classes)} kelas: {self.classes}")

        for cls_name in self.classes:
            cls_folder = os.path.join(data_path, cls_name)
            for npy_file in glob.glob(os.path.join(cls_folder, "*.npy")):
                self.samples.append(npy_file)
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Memuat matriks .npy
        data = np.load(self.samples[idx])
        
        # Penyesuaian Panjang Sekuens (Padding/Truncating)
        # Jika frame video > 30, kita potong. Jika < 30, kita tambah angka nol.
        if len(data) > self.sequence_length:
            data = data[:self.sequence_length]
        elif len(data) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(data), 63))
            data = np.vstack((data, padding))
            
        return torch.tensor(data, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# --- 2. FUNGSI UTAMA PELATIHAN ---
def train_model():
    # Parameter Dasar
    base_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(base_dir, "data", "wlasl_npy")
    BATCH_SIZE = 32
    EPOCHS = 50 # Bisa ditambah sesuai kebutuhan
    LEARNING_RATE = 0.001
    EXPECTED_NUM_CLASSES = 17
    
    # Inisialisasi Dataset dan Loader
    dataset = BISINDODataset(DATA_PATH)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Cetak urutan kelas untuk mengetahui indeks 0..n
    print("Urutan kelas (index: nama):")
    for idx, name in enumerate(dataset.classes):
        print(f"  {idx}: {name}")

    num_classes = len(dataset.classes)
    if num_classes != EXPECTED_NUM_CLASSES:
        print(
            "Peringatan: jumlah kelas ditemukan "
            f"{num_classes}, bukan {EXPECTED_NUM_CLASSES}."
        )
    
    # Inisialisasi Model, Loss Function, dan Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Siformer(num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss() # Menghitung selisih prediksi dengan label asli
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Algoritma pengubah bobot

    print(f"Memulai pelatihan di device: {device}")

    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Reset gradien agar tidak menumpuk dari perhitungan sebelumnya
            optimizer.zero_grad()
            
            # Forward: Masukkan data ke model untuk dapat prediksi
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward: Hitung seberapa salah prediksi tersebut dan balikkan ke belakang
            loss.backward()
            
            # Update: Ubah bobot matriks model berdasarkan hasil backward
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

    # Simpan hasil akhir "otak" model
    os.makedirs("app/models/weights", exist_ok=True)
    torch.save(model.state_dict(), "app/models/weights/siformer_wlasl_cafe.pth")
    print("Pelatihan selesai! File bobot disimpan di app/models/weights/siformer_wlasl_cafe.pth")

if __name__ == "__main__":
    train_model()