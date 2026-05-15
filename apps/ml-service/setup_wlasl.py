import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import zipfile

import kagglehub

TARGET_WORDS = [
    "hello",
    "want",
    "order",
    "coffee",
    "tea",
    "milk",
    "sugar",
    "water",
    "no",
    "ice",
    "hot",
    "one",
    "two",
    "three",
    "four",
    "five",
    "thank you",
    "please",
]

METADATA_HINTS = (
    "wlasl",
    "class",
    "label",
    "gloss",
    "metadata",
    "annotation",
)

VIDEO_EXTS = (".mp4",)


def normalize_word(text: str) -> str:
    text = text.lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_token_sequence(tokens, target_tokens) -> bool:
    if not tokens or not target_tokens:
        return False
    max_i = len(tokens) - len(target_tokens)
    for i in range(max_i + 1):
        if tokens[i : i + len(target_tokens)] == target_tokens:
            return True
    return False


def build_target_index(words):
    norm_to_word = {normalize_word(word): word for word in words}
    targets_sorted = sorted(
        norm_to_word.keys(),
        key=lambda s: (-len(s.split()), -len(s)),
    )
    target_tokens = {norm: norm.split() for norm in norm_to_word}
    return norm_to_word, targets_sorted, target_tokens


def match_target_in_text(text, targets_sorted, target_tokens):
    tokens = normalize_word(text).split()
    for norm in targets_sorted:
        if contains_token_sequence(tokens, target_tokens[norm]):
            return norm
    return None


def find_metadata_files(dataset_root: Path):
    candidates = []
    for root, _, files in os.walk(dataset_root):
        for name in files:
            lower = name.lower()
            if not lower.endswith((".json", ".csv", ".tsv")):
                continue
            if any(hint in lower for hint in METADATA_HINTS):
                candidates.append(Path(root) / name)
    return candidates


def _map_from_json_data(data, target_norms):
    mapping = {}
    if isinstance(data, dict):
        for key in ("annotations", "data", "videos", "instances"):
            if key in data:
                mapping.update(_map_from_json_data(data[key], target_norms))
        if data and all(isinstance(v, str) for v in data.values()):
            for key, value in data.items():
                norm = normalize_word(value)
                if norm in target_norms:
                    mapping[str(key)] = norm
        return mapping

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                gloss = (
                    item.get("gloss")
                    or item.get("word")
                    or item.get("label")
                    or item.get("name")
                )
                if gloss:
                    norm = normalize_word(gloss)
                    if norm in target_norms:
                        for key in (
                            "video_id",
                            "id",
                            "uid",
                            "name",
                            "file",
                            "filename",
                            "path",
                        ):
                            if item.get(key):
                                mapping[Path(str(item[key])).stem] = norm
                instances = (
                    item.get("instances")
                    or item.get("videos")
                    or item.get("samples")
                    or item.get("data")
                )
                if gloss and isinstance(instances, list):
                    norm = normalize_word(gloss)
                    if norm in target_norms:
                        for inst in instances:
                            if isinstance(inst, dict):
                                for key in (
                                    "video_id",
                                    "id",
                                    "uid",
                                    "name",
                                    "file",
                                    "filename",
                                    "path",
                                ):
                                    if inst.get(key):
                                        mapping[Path(str(inst[key])).stem] = norm
                            elif isinstance(inst, str):
                                mapping[Path(inst).stem] = norm
            elif isinstance(item, str):
                pass
    return mapping


def parse_json_metadata(path: Path, target_norms):
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return _map_from_json_data(data, target_norms)


def parse_csv_metadata(path: Path, target_norms):
    try:
        with path.open("r", encoding="utf-8") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            dialect = csv.Sniffer().sniff(sample)
            reader = csv.DictReader(handle, dialect=dialect)
            fieldnames = [name.lower() for name in reader.fieldnames or []]
            gloss_fields = [
                name
                for name in fieldnames
                if name in ("gloss", "word", "label", "name", "sign")
            ]
            video_fields = [
                name
                for name in fieldnames
                if name in ("video_id", "id", "uid", "file", "filename", "path", "video")
            ]
            if not gloss_fields or not video_fields:
                return {}
            gloss_field = gloss_fields[0]
            video_field = video_fields[0]
            mapping = {}
            for row in reader:
                gloss = row.get(gloss_field) or ""
                video = row.get(video_field) or ""
                norm = normalize_word(gloss)
                if norm in target_norms and video:
                    mapping[Path(video).stem] = norm
            return mapping
    except (OSError, csv.Error):
        return {}


def build_metadata_mapping(dataset_root: Path, target_norms):
    mapping = {}
    for path in find_metadata_files(dataset_root):
        if path.suffix.lower() == ".json":
            mapping.update(parse_json_metadata(path, target_norms))
        elif path.suffix.lower() in {".csv", ".tsv"}:
            mapping.update(parse_csv_metadata(path, target_norms))
    return mapping


def infer_gloss(video_path: Path, metadata_map, targets_sorted, target_tokens, target_norms):
    stem = video_path.stem
    if stem in metadata_map:
        return metadata_map[stem]

    for part in [
        video_path.parent.name,
        video_path.parent.parent.name,
        video_path.name,
    ]:
        match = match_target_in_text(part, targets_sorted, target_tokens)
        if match in target_norms:
            return match

    return None


def extract_zip_if_needed(dataset_path: Path) -> Path:
    if dataset_path.is_dir():
        return dataset_path
    if dataset_path.is_file() and dataset_path.suffix.lower() == ".zip":
        extract_dir = dataset_path.parent / dataset_path.stem
        if not extract_dir.exists():
            with zipfile.ZipFile(dataset_path, "r") as handle:
                handle.extractall(extract_dir)
        return extract_dir
    if dataset_path.is_file():
        return dataset_path.parent
    return dataset_path


def main():
    parser = argparse.ArgumentParser(
        description="Download WLASL dataset and extract selected words into data/wlasl_cafe."
    )
    parser.add_argument(
        "--dataset",
        default="risangbaskoro/wlasl-processed",
        help="Kaggle dataset handle (default: risangbaskoro/wlasl-processed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report matches without copying files",
    )
    args = parser.parse_args()

    norm_to_word, targets_sorted, target_tokens = build_target_index(TARGET_WORDS)
    target_norms = set(norm_to_word.keys())

    print("Mengunduh dataset via kagglehub...")
    dataset_path = Path(kagglehub.dataset_download(args.dataset))
    dataset_root = extract_zip_if_needed(dataset_path)
    print(f"Dataset ditemukan di: {dataset_root}")

    output_root = Path(__file__).resolve().parent / "data" / "wlasl_cafe"
    output_root.mkdir(parents=True, exist_ok=True)

    for word in TARGET_WORDS:
        (output_root / word).mkdir(parents=True, exist_ok=True)

    print("Memindai metadata...")
    metadata_map = build_metadata_mapping(dataset_root, target_norms)
    if metadata_map:
        print(f"Metadata map ditemukan: {len(metadata_map)} entri")
    else:
        print("Metadata map tidak ditemukan, menggunakan folder/file name")

    print("Mencari file .mp4...")
    video_files = [
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    ]
    print(f"Total video ditemukan: {len(video_files)}")

    copied = 0
    matched = 0
    per_word = {norm: 0 for norm in target_norms}

    for video_path in video_files:
        gloss_norm = infer_gloss(
            video_path,
            metadata_map,
            targets_sorted,
            target_tokens,
            target_norms,
        )
        if not gloss_norm:
            continue

        matched += 1
        per_word[gloss_norm] += 1

        if args.dry_run:
            continue

        dest_dir = output_root / norm_to_word[gloss_norm]
        dest_path = dest_dir / video_path.name

        if dest_path.exists():
            continue

        shutil.copy2(video_path, dest_path)
        copied += 1

    print("Selesai.")
    print(f"Matched: {matched}")
    print(f"Copied: {copied}")

    missing = [norm_to_word[norm] for norm, count in per_word.items() if count == 0]
    if missing:
        print("Tidak ditemukan video untuk kata: " + ", ".join(missing))


if __name__ == "__main__":
    main()
