# Finetune VadCLIP (UCF-Crime)

Repo này dùng để fine-tune mô hình `VadCLIP` cho bài toán video anomaly detection trên `UCF-Crime` với sự hướng dẫn ngữ nghĩa từ **video descriptions** (tổng hợp từ annotation theo timestamp).

Code gốc của `VadCLIP` nằm trong thư mục `VadCLIP/` (xem `VadCLIP/README.md` để biết cách train/test theo pipeline gốc).

Tài liệu trong `docs/` ghi lại các ý tưởng/thử nghiệm fine-tune với description (ví dụ: guided finetuning, semantic alignment).

## Cấu trúc chính

- `VadCLIP/`: implementation và scripts train/test của VadCLIP gốc
- `code/`: các notebook / artifact phục vụ tạo prototype/description và chạy thí nghiệm
- `docs/`: ghi chú về hướng fine-tune và quyết định pipeline

## Data & artifact (các thứ KHÔNG nên đẩy lên GitHub)

Các file/thư mục nặng đã được ignore trong `/.gitignore`, bao gồm:

- Feature + annotation theo UCF:
  - `UCF Annotation/`
  - `UCFClipFeatures/`
- Checkpoint/diagnostics dump:
  - `ucf_checkpoint_diagnostics*/`
- Model weights:
  - `model_ucf.pth` và các file `*.pth`, `*.pt`, `*.ckpt`, ...
- Các cache/embedding/caption dump sinh ra trong `code/`:
  - `code/*embeddings_cache*.json`
  - `code/*gpt_video_descriptions*.(json|csv)`

Lưu ý: nếu trước đó bạn đã `git add`/push các file nặng này lên remote, `.gitignore` sẽ không tự động gỡ chúng khỏi lịch sử/index. Khi đó bạn cần `git rm --cached ...` (tuỳ workflow của bạn).

## Hướng dẫn nhanh

1. Đọc `VadCLIP/README.md` để setup & chạy train/test theo pipeline gốc.
2. Với phần fine-tune/guide bằng description:
   - tham khảo các notebook trong `code/`
   - đọc thêm các tài liệu trong `docs/` để hiểu mục tiêu supervision (description dùng như tín hiệu training, không dùng lúc inference).

## References

- `VadCLIP` (AAAI 2024): xem `VadCLIP/README.md`

