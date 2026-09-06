"""Sinh cả ba file ground truth của UCF-Crime trong một lượt.

    python list/make_gt_ucf_relative.py --feature-root /content/UCFClipFeatures

Bản này thay cho `make_gt_ucf.py` + `make_gt_mAP_ucf.py`, vốn không chạy được ngoài máy của
tác giả gốc vì hai lý do:

* Chúng lọc `'__0.npy' not in name: continue`, trong khi cả hai file list test của dự án đều
  dùng crop `__5.npy` — bộ lọc đó bỏ qua toàn bộ 290 dòng và cho ra file rỗng.
* Chúng đọc `list/ucf_CLIP_rgbtest.csv` chứa đường dẫn tuyệt đối trên máy tác giả, nên
  `np.load` không tìm thấy file.

Phần tính toán được giữ nguyên hoàn toàn so với hai script gốc:

* `gt_ucf.npy` — nhãn theo khung hình, nối tiếp nhau theo đúng thứ tự file list. Mỗi video
  đóng góp `N * 16` khung, với N là số dòng đặc trưng. (Bản gốc dựng vector dài `(N+1)*16`
  rồi cắt bỏ 16 khung cuối; kết quả y hệt.)
* `gt_segment_ucf.npy` / `gt_label_ucf.npy` — mỗi video một mục, dùng cho việc tính mAP.
  Video Normal cho đúng một đoạn `[0, N*16]` với nhãn `'A'` (nhãn này cố ý không nằm trong
  danh sách lớp, đúng như bản gốc). Video bất thường lấy đoạn từ file chú thích, giữ nguyên
  dạng chuỗi vì `ucf_detectionMAP.py` tự ép kiểu.

Khác biệt duy nhất về hành vi: việc ghép video với dòng chú thích dùng **so khớp chính xác**
trên trường đầu tiên, thay vì `if name in gt_line` của bản gốc. So khớp chuỗi con có thể ghép
nhầm khi một tên là tiền tố của tên khác.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

CLIP_LEN = 16


def load_annotations(path):
    """Trả về {video_id: [class_name, s1, e1, s2, e2]}."""
    annotations = {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = [f for f in line.replace("\r", "").strip("\n").split("  ") if f != ""]
            if len(fields) < 6:
                continue
            annotations[fields[0]] = fields[1:6]
    return annotations


def main():
    parser = argparse.ArgumentParser(description="Sinh gt_ucf.npy, gt_segment_ucf.npy, gt_label_ucf.npy")
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--test-list", default="list/ucf_CLIP_rgbtest_relative.csv")
    parser.add_argument("--annotation", default="list/Temporal_Anomaly_Annotation.txt")
    parser.add_argument("--output-dir", default="list")
    args = parser.parse_args()

    feature_root = Path(args.feature_root)
    output_dir = Path(args.output_dir)
    annotations = load_annotations(args.annotation)
    print(f"Đã đọc {len(annotations)} dòng chú thích.")

    with open(args.test_list, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    print(f"Đã đọc {len(rows)} dòng từ {args.test_list}.")

    frame_labels = []
    segments = []
    segment_labels = []
    missing_annotation = []

    for row in rows:
        relative = row["path"]
        label_text = row["label"]
        video_id = row.get("video_id") or Path(relative).name.rsplit("__", 1)[0]

        feature_path = feature_root / relative
        if not feature_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy đặc trưng: {feature_path}\n"
                "Kiểm tra lại --feature-root; nó phải trỏ tới thư mục chứa Abuse/, Arrest/, ..."
            )
        num_clips = int(np.load(feature_path).shape[0])
        num_frames = num_clips * CLIP_LEN

        vector = np.zeros(num_frames, dtype=np.float32)

        if "Normal" in label_text:
            segments.append([[0, num_frames]])
            segment_labels.append(["A"])
        else:
            entry = annotations.get(video_id)
            if entry is None:
                missing_annotation.append(video_id)
                segments.append([])
                segment_labels.append([])
            else:
                class_name, s1, e1, s2, e2 = entry
                video_segments, video_labels = [], []
                for start, end in ((s1, e1), (s2, e2)):
                    if start == "-1" or end == "-1":
                        continue
                    video_segments.append([start, end])
                    video_labels.append(class_name)
                    vector[int(start): int(end)] = 1.0
                segments.append(video_segments)
                segment_labels.append(video_labels)

        frame_labels.extend(vector.tolist())

    if missing_annotation:
        print(f"CẢNH BÁO: {len(missing_annotation)} video không có chú thích: "
              f"{missing_annotation[:10]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "gt_ucf.npy", np.array(frame_labels, dtype=np.float32))
    np.save(output_dir / "gt_segment_ucf.npy", np.array(segments, dtype=object), allow_pickle=True)
    np.save(output_dir / "gt_label_ucf.npy", np.array(segment_labels, dtype=object), allow_pickle=True)

    positives = int(np.sum(frame_labels))
    print(f"\nĐã lưu vào {output_dir}/")
    print(f"  gt_ucf.npy          {len(frame_labels):,} khung, "
          f"{positives:,} khung bất thường ({100 * positives / max(1, len(frame_labels)):.2f}%)")
    print(f"  gt_segment_ucf.npy  {len(segments)} video")
    print(f"  gt_label_ucf.npy    {len(segment_labels)} video")
    print("\nSố khung phải khớp với tổng số đặc trưng nhân 16. Nếu script chấm điểm báo "
          "\"Scored N frames but the ground truth has M\" thì file list và feature không khớp nhau.")


if __name__ == "__main__":
    main()
