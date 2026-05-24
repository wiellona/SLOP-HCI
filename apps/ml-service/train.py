import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import json

from app.models.siformer_model import Siformer
from app.hand_preprocess import FEATURE_SIZE_ONE_HAND, FEATURE_SIZE_TWO_HANDS

def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if isinstance(state_dict, dict):
        return {k.replace("module.", ""): v for k, v in state_dict.items()}
    return {}


def _resize_input_projection(weight: torch.Tensor, expected_in: int):
    if weight.ndim != 2:
        return None
    old_in = weight.shape[1]
    if old_in == expected_in:
        return weight
    if expected_in == old_in * 2:
        return torch.cat([weight, weight], dim=1)
    if old_in == expected_in * 2:
        return weight[:, :expected_in]
    if old_in < expected_in:
        pad = torch.zeros(weight.shape[0], expected_in - old_in, dtype=weight.dtype)
        return torch.cat([weight, pad], dim=1)
    if old_in > expected_in:
        return weight[:, :expected_in]
    return None


def _prepare_state_dict(state_dict, expected_in: int, model_state: dict):
    weight = state_dict.get("input_projection.weight")
    if isinstance(weight, torch.Tensor) and weight.shape[1] != expected_in:
        resized = _resize_input_projection(weight, expected_in)
        if resized is None:
            state_dict.pop("input_projection.weight", None)
        else:
            state_dict["input_projection.weight"] = resized

    filtered = {}
    for key, value in state_dict.items():
        if key in model_state and value.shape == model_state[key].shape:
            filtered[key] = value
    return filtered


USE_TWO_HANDS = _env_flag("SIGN_USE_TWO_HANDS", True)
FREEZE_ENCODER = _env_flag("SIGN_FREEZE_ENCODER", True)
FEATURE_SIZE = FEATURE_SIZE_TWO_HANDS if USE_TWO_HANDS else FEATURE_SIZE_ONE_HAND
NUM_JOINTS = 42 if USE_TWO_HANDS else 21

# --- 1. DEFINISI DATASET CUSTOM ---
class WLASLdataset(Dataset):
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
        
        if data.ndim != 2 or data.shape[1] != FEATURE_SIZE:
            raise ValueError(
                "Ukuran fitur tidak sesuai. "
                "Jalankan ulang preprocess_videos.py dengan konfigurasi USE_TWO_HANDS yang sama."
            )

        if len(data) > self.sequence_length:
            data = data[:self.sequence_length]
        elif len(data) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(data), FEATURE_SIZE))
            data = np.vstack((data, padding))
            
        return torch.tensor(data, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# --- 2. FUNGSI UTAMA PELATIHAN ---
def train_model():
    # Parameter Dasar
    base_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(base_dir, "data", "wlasl_npy")
    BATCH_SIZE = _env_int("SIGN_BATCH_SIZE", 32)
    EPOCHS = _env_int("SIGN_EPOCHS", 50)
    LEARNING_RATE = _env_float("SIGN_LR", 0.001)
    EXPECTED_NUM_CLASSES = 22
    
    # Inisialisasi Dataset dan Loader
    dataset = WLASLdataset(DATA_PATH)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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
    model = Siformer(num_joints=NUM_JOINTS, num_classes=num_classes).to(device)
    pretrained_path = os.getenv("SIGN_PRETRAINED_PATH") or os.path.join(
        "app",
        "models",
        "weights",
        "siformer_wlasl_cafe.pth",
    )
    
    if os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        raw_state_dict = _extract_state_dict(checkpoint)
        model_state = model.state_dict()
        filtered_state = _prepare_state_dict(raw_state_dict, FEATURE_SIZE, model_state)
        incompatible = model.load_state_dict(filtered_state, strict=False)
        print(
            f"Loaded {len(filtered_state)}/{len(model_state)} layers from checkpoint."
        )
        if incompatible.missing_keys:
            print(f"Missing keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"Unexpected keys: {incompatible.unexpected_keys}")

    if FREEZE_ENCODER:
        for name, param in model.named_parameters():
            if name.startswith("transformer_encoder"):
                param.requires_grad = False
    
    criterion = nn.CrossEntropyLoss() # Menghitung selisih prediksi dengan label asli
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE * 0.1)

    print(f"Memulai pelatihan di device: {device}")

    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

    os.makedirs("app/models/weights", exist_ok=True)
    weights_dir = os.path.join("app", "models", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(weights_dir, "siformer_wlasl_cafe.pth"))
    with open(os.path.join(weights_dir, "siformer_labels.json"), "w", encoding="utf-8") as f:
        json.dump(dataset.classes, f, ensure_ascii=True, indent=2)
    print("Pelatihan selesai!")

if __name__ == "__main__":
    train_model()