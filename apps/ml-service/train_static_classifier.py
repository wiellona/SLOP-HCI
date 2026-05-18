import glob
import json
import os

import joblib
import numpy as np
from typing import List, Optional, Tuple
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from app.hand_preprocess import FEATURE_SIZE_ONE_HAND, FEATURE_SIZE_TWO_HANDS

USE_TWO_HANDS = False
FEATURE_SIZE = FEATURE_SIZE_TWO_HANDS if USE_TWO_HANDS else FEATURE_SIZE_ONE_HAND


def aggregate_sequence(sequence: np.ndarray) -> Optional[np.ndarray]:
    if sequence.ndim != 2 or sequence.shape[1] != FEATURE_SIZE:
        return None

    valid = sequence[np.any(sequence != 0.0, axis=1)]
    if len(valid) == 0:
        return None

    return np.median(valid, axis=0)


def load_dataset(data_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    classes = sorted(
        [
            name
            for name in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, name))
        ]
    )
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    features = []
    labels = []

    for cls_name in classes:
        cls_folder = os.path.join(data_path, cls_name)
        for npy_file in glob.glob(os.path.join(cls_folder, "*.npy")):
            sequence = np.load(npy_file)
            sample = aggregate_sequence(sequence)
            if sample is None:
                continue

            features.append(sample)
            labels.append(class_to_idx[cls_name])

    if not features:
        raise RuntimeError(f"No samples found under {data_path}")

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64), classes


def train_static_classifier() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "wlasl_npy")

    features, labels, classes = load_dataset(data_path)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale", probability=True)),
        ]
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=classes))

    weights_dir = os.path.join(base_dir, "app", "models", "weights")
    os.makedirs(weights_dir, exist_ok=True)

    model_path = os.path.join(weights_dir, "static_svm.joblib")
    labels_path = os.path.join(weights_dir, "static_labels.json")

    joblib.dump(model, model_path)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=True, indent=2)

    print(f"Model saved to: {model_path}")
    print(f"Labels saved to: {labels_path}")


if __name__ == "__main__":
    train_static_classifier()
