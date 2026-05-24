import argparse
import glob
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from app.models.siformer_model import Siformer


def _load_checkpoint(path: str) -> Dict[str, torch.Tensor] | None:
    if not os.path.exists(path):
        return None
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if isinstance(state_dict, dict):
        return {k.replace("module.", ""): v for k, v in state_dict.items()}
    return None


def _load_labels(path: str, fallback: List[str]) -> List[str]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, list) and data:
                return data
    return fallback


def _pad_or_truncate(data: np.ndarray, sequence_length: int, feature_size: int) -> np.ndarray:
    if data.shape[0] > sequence_length:
        return data[:sequence_length]
    if data.shape[0] < sequence_length:
        padding = np.zeros((sequence_length - data.shape[0], feature_size), dtype=np.float32)
        return np.vstack([data, padding])
    return data


def _collect_samples(data_root: str, class_to_idx: Dict[str, int]) -> List[Tuple[str, int]]:
    samples: List[Tuple[str, int]] = []
    for class_name in sorted(os.listdir(data_root)):
        class_dir = os.path.join(data_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        if class_name not in class_to_idx:
            continue
        for npy_file in glob.glob(os.path.join(class_dir, "*.npy")):
            samples.append((npy_file, class_to_idx[class_name]))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sign model accuracy and latency.")
    parser.add_argument("--data-root", default="data/wlasl_npy")
    parser.add_argument("--weights", default="app/models/weights/siformer_wlasl_cafe.pth")
    parser.add_argument("--labels", default="app/models/weights/siformer_labels.json")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"Dataset folder not found: {args.data_root}")

    fallback_labels = sorted(
        [name for name in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, name))]
    )
    label_map = _load_labels(args.labels, fallback_labels)
    class_to_idx = {name: idx for idx, name in enumerate(label_map)}

    samples = _collect_samples(args.data_root, class_to_idx)
    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]

    if not samples:
        raise RuntimeError("No samples found for benchmarking.")

    first_sample = np.load(samples[0][0])
    if first_sample.ndim != 2:
        raise ValueError("Invalid sample shape; expected 2D array.")

    feature_size = first_sample.shape[1]
    if feature_size % 3 != 0:
        raise ValueError("Feature size must be divisible by 3.")

    num_joints = feature_size // 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Siformer(
        num_joints=num_joints,
        num_classes=len(label_map),
        num_frames=args.sequence_length,
    ).to(device)

    state_dict = _load_checkpoint(args.weights)
    if state_dict:
        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"Weights not found at {args.weights}")

    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    latencies: List[float] = []

    for path, label in samples:
        data = np.load(path)
        data = _pad_or_truncate(data, args.sequence_length, feature_size)
        input_tensor = torch.from_numpy(data).float().unsqueeze(0).to(device)

        start = time.perf_counter()
        with torch.inference_mode():
            output_logits = model(input_tensor)
        latencies.append((time.perf_counter() - start) * 1000)

        pred_idx = int(torch.argmax(output_logits, dim=1).item())
        y_true.append(label)
        y_pred.append(pred_idx)

    print(classification_report(y_true, y_pred, target_names=label_map))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        p95_index = max(0, int(len(latencies) * 0.95) - 1)
        p95_latency = sorted(latencies)[p95_index]
        print(f"Average latency (ms): {avg_latency:.2f}")
        print(f"P95 latency (ms): {p95_latency:.2f}")


if __name__ == "__main__":
    main()
