import argparse
import csv
import json
from pathlib import Path


def video_id_from_feature(feature_path):
    return feature_path.stem.rsplit("__", 1)[0]


def relative_path_from_original(original_path, label):
    feature_name = Path(original_path).name
    return Path(label, feature_name).as_posix()


def build_feature_index(feature_root):
    feature_root = Path(feature_root).resolve()
    index = {}
    for feature_path in feature_root.glob("*/*.npy"):
        rel_path = feature_path.relative_to(feature_root).as_posix()
        index.setdefault(feature_path.name, []).append(rel_path)
    return index


def relative_path_from_feature_index(original_path, label, feature_index):
    feature_name = Path(original_path).name
    candidates = feature_index.get(feature_name, [])
    if len(candidates) == 1:
        return candidates[0]
    for candidate in candidates:
        if candidate.startswith(f"{label}/"):
            return candidate
    if candidates:
        return candidates[0]
    return relative_path_from_original(original_path, label)


def load_described_video_ids(description_json):
    with open(description_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {
        record["video_id"]
        for record in records
        if record.get("video_id") and (record.get("gpt_description") or "").strip()
    }


def convert_source_csv(source_csv, output_csv, described_video_ids=None, feature_index=None):
    output_csv = Path(output_csv)
    rows = []
    with open(source_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"]
            if feature_index is None:
                rel_path = relative_path_from_original(row["path"], label)
            else:
                rel_path = relative_path_from_feature_index(row["path"], label, feature_index)
            video_id = video_id_from_feature(Path(rel_path))
            if described_video_ids is not None and video_id not in described_video_ids:
                continue
            rows.append((rel_path, label, video_id))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "video_id"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")


def write_relative_csv(feature_root, output_csv, split_filter=None):
    feature_root = Path(feature_root).resolve()
    output_csv = Path(output_csv)
    rows = []

    for feature_path in sorted(feature_root.glob("*/*.npy")):
        label = feature_path.parent.name
        video_id = video_id_from_feature(feature_path)
        if split_filter and video_id not in split_filter:
            continue
        rel_path = feature_path.relative_to(feature_root).as_posix()
        rows.append((rel_path, label, video_id))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "video_id"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")


def load_video_ids_from_txt(txt_path):
    video_ids = set()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name = Path(line).stem
            video_ids.add(name)
    return video_ids


def main():
    parser = argparse.ArgumentParser(description="Create UCF-Crime relative feature CSV files.")
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--source-train-csv", default="ucf_CLIP_rgb.csv")
    parser.add_argument("--source-test-csv", default="ucf_CLIP_rgbtest.csv")
    parser.add_argument("--train-txt", default="Anomaly_Train.txt")
    parser.add_argument("--test-txt", default="Anomaly_Test.txt")
    parser.add_argument("--train-output", default="ucf_CLIP_rgb_description.csv")
    parser.add_argument("--test-output", default="ucf_CLIP_rgbtest_description.csv")
    parser.add_argument("--description-json", default=None)
    parser.add_argument("--filter-described", action="store_true")
    args = parser.parse_args()

    described_video_ids = None
    if args.filter_described:
        if args.description_json is None:
            raise ValueError("--description-json is required when --filter-described is used.")
        described_video_ids = load_described_video_ids(args.description_json)
        print(f"Loaded {len(described_video_ids)} described videos from {args.description_json}")

    if Path(args.source_train_csv).exists() and Path(args.source_test_csv).exists():
        feature_index = build_feature_index(args.feature_root)
        convert_source_csv(args.source_train_csv, args.train_output, described_video_ids, feature_index)
        convert_source_csv(args.source_test_csv, args.test_output, described_video_ids, feature_index)
    else:
        train_ids = load_video_ids_from_txt(args.train_txt)
        test_ids = load_video_ids_from_txt(args.test_txt)
        write_relative_csv(args.feature_root, args.train_output, train_ids)
        write_relative_csv(args.feature_root, args.test_output, test_ids)


if __name__ == "__main__":
    main()
