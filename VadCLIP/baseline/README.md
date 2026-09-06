# VadCLIP baseline — code gốc, tách riêng

Thư mục này chứa **chỉ code gốc của VadCLIP upstream**, tách khỏi `VadCLIP/src/` (nơi đã trộn
lẫn nhiều nhánh thí nghiệm: description, class-prototype, multi-prompt, shift-consistency…).

Mục đích: có một chỗ chạy lại VadCLIP nguyên bản mà **không thể vô tình dính vào code thí nghiệm**,
để làm mốc so sánh đáng tin.

Chạy trên Colab bằng `code/train_baseline_vadclip_colab.ipynb`.

---

## Nội dung

```
baseline/
├── README.md                     ← file này
└── src/
    ├── clip/                     VERBATIM  (thư viện CLIP của OpenAI)
    ├── utils/
    │   ├── dataset.py            ĐÃ SỬA    (+ feature_root)
    │   ├── layers.py             VERBATIM
    │   ├── lr_warmup.py          VERBATIM
    │   ├── tools.py              VERBATIM
    │   ├── ucf_detectionMAP.py   VERBATIM
    │   └── xd_detectionMAP.py    VERBATIM
    ├── crop.py                   VERBATIM
    ├── model.py                  VERBATIM
    ├── ucf_option.py             ĐÃ SỬA
    ├── ucf_train.py              ĐÃ SỬA
    ├── ucf_test.py               ĐÃ SỬA
    └── xd_option.py / xd_train.py / xd_test.py   VERBATIM (XD-Violence, không dùng)
```

Danh sách train/test và ground truth **không** copy vào đây — dùng chung `VadCLIP/list/`
để chỉ có một nguồn sự thật duy nhất.

Kiểm chứng lại các file VERBATIM bất cứ lúc nào:

```bash
cd VadCLIP/baseline/src
for f in crop.py model.py utils/tools.py utils/layers.py utils/ucf_detectionMAP.py; do
  a=$(git show HEAD:VadCLIP/src/$f | tr -d '\r' | md5sum | cut -c1-8)
  b=$(tr -d '\r' < $f | md5sum | cut -c1-8)
  [ "$a" = "$b" ] && echo "VERBATIM $f" || echo "KHAC     $f"
done
```

---

## Toàn bộ sửa đổi so với upstream

Chỉ **4 file** bị sửa. Không có sửa đổi nào chạm vào hàm mất mát, vòng lặp huấn luyện, kiến trúc
model, hay bất kỳ dòng nào tiêu thụ số ngẫu nhiên — nên quỹ đạo huấn luyện không đổi.

### 1. `utils/dataset.py` — hỗ trợ `feature_root`

Upstream đọc thẳng cột `path` của CSV, mà CSV upstream chứa đường dẫn tuyệt đối trên máy tác giả
(`/home/xbgydx/Desktop/UCFClipFeatures/...`). Thêm tham số `feature_root` (mặc định `''`) được nối
vào các đường dẫn tương đối.

- `feature_root=''` → hành vi **y hệt upstream**.
- Đường dẫn tuyệt đối trong CSV **luôn được ưu tiên**, nên list cũ vẫn chạy đúng.

### 2. `ucf_option.py`

| Sửa đổi | Lý do | Có đổi hành vi mặc định? |
|---|---|---|
| `--lr`, `--scheduler-rate` thêm `type=float` | Upstream không khai báo type → `--lr 1e-5` tới `AdamW` dưới dạng **chuỗi** và crash. Giá trị mặc định giữ nguyên | Không |
| `--scheduler-milestones` thêm `nargs='+' type=int` | Cùng lý do | Không |
| `--use-checkpoint` đổi `type=bool` → `str2bool` | `type=bool` khiến argparse coi **mọi chuỗi khác rỗng** là True — kể cả `--use-checkpoint False` | Không (mặc định vẫn False) |
| Đường dẫn mặc định `list/…` → `../../list/…` | Upstream giả định chạy từ thư mục `VadCLIP/`; ở đây chạy từ `baseline/src/` | Không (trỏ tới cùng file) |
| Thêm `--feature-root` (mặc định `''`) | Xem mục 1 | Không |
| Thêm `--epoch-checkpoint-dir` (mặc định `''`) | Xem mục 3 | Không (mặc định tắt) |
| Thêm `--save-cur-path` (mặc định `model/model_cur.pth`) | Upstream hardcode chuỗi này | Không |

### 3. `ucf_train.py`

```diff
+ import os

- checkpoint = torch.load(args.checkpoint_path)
+ checkpoint = torch.load(args.checkpoint_path, weights_only=False)      (3 chỗ)

- torch.save(model.state_dict(), 'model/model_cur.pth')
+ torch.save(model.state_dict(), args.save_cur_path)

+ if args.epoch_checkpoint_dir:            # mặc định tắt
+     torch.save(model.state_dict(), .../model_epoch_XXX.pth)
+ print('=== end of epoch', e+1, '| best AUC so far:', ap_best, '===')
+ print('saved final model:', args.model_path)

+ os.makedirs(...)                         # tạo thư mục model/ trước khi train

- UCFDataset(args.visual_length, args.train_list, False, label_map, True)
+ UCFDataset(args.visual_length, args.train_list, False, label_map, True, args.feature_root)
```

Vì sao từng cái là an toàn:

- **`weights_only=False`** — bắt buộc từ PyTorch 2.6. File checkpoint chứa optimizer state nên
  không unpickle được với mặc định mới. Không đổi dữ liệu được đọc.
- **`os.makedirs`** — upstream giả định `model/` đã tồn tại. Không có nó thì crash ở cuối epoch 1.
- **`epoch-checkpoint-dir`** — upstream chỉ giữ `model_cur.pth` bị ghi đè mỗi epoch, nên Colab rớt
  giữa chừng là mất trắng. `torch.save` không tiêu thụ RNG → quỹ đạo huấn luyện giống hệt dù bật
  hay tắt. Mặc định tắt.
- **`print`** — log upstream chỉ in ở step 1280, không phân biệt được ranh giới epoch trong một log
  hơn 1000 dòng.
- **`feature_root`** — chỉ thêm đối số cuối. **Thứ tự khởi tạo dataset/loader/model giữ nguyên
  tuyệt đối**, vì chính thứ tự đó quyết định luồng số ngẫu nhiên.

### 4. `ucf_test.py` — sửa một lỗi thật của upstream

`test()` **verbatim, không sửa gì**. Chỉ sửa phần `__main__`:

```diff
- label_map = dict({'Normal': 'Normal', 'Abuse': 'Abuse', ...})     # viết HOA
+ label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', ...})     # viết thường, khớp ucf_train.py
```

**Đây là lỗi upstream, không phải lựa chọn thẩm mỹ.** `ucf_train.py` huấn luyện với prompt viết
thường (`'normal'`, `'abuse'`, …) còn `ucf_test.py` chạy độc lập lại dùng viết hoa. Hai chuỗi khác
nhau → tokenize khác nhau → text feature khác nhau. Nghĩa là bản upstream **chấm điểm checkpoint
bằng bộ prompt mà nó chưa từng được huấn luyện với**, và cho ra con số khác với chính lần đánh giá
chạy bên trong vòng train.

Ngoài ra: `torch.load(..., weights_only=False)` + chấp nhận cả state_dict trần lẫn checkpoint dict,
và truyền `args.feature_root`.

---

## Cấu hình: repo vs paper

`ucf_option.py` giữ nguyên mặc định của repo, trong đó có **một chỗ lệch với paper**:

| Tham số | Paper (mục *Implementation Details*, UCF-Crime) | Repo & thư mục này |
|---|---|---|
| **Learning rate** | **1 × 10⁻⁵** | **2 × 10⁻⁵** ← lệch, gấp đôi |
| Epoch | 10 | 10 ✓ |
| Optimizer | AdamW | AdamW ✓ |
| Batch size | 64 | 64 ✓ |
| Window (LGT-Adapter) | 8 | 8 ✓ |
| Context length *l* | 20 | 10 + 10 = 20 ✓ |
| λ (Eq.10) | 1 × 10⁻¹ | `loss3 / 13 * 1e-1` ✓ |

`MultiStepLR([4, 8], gamma=0.1)` và `seed=234` **không hề xuất hiện trong paper** — chỉ repo mới có.

Chạy đúng config paper:

```bash
python ucf_train.py --lr 1e-5 ...
```

Đáng chạy cả hai: không lần train-from-scratch nào trong `docs/` của dự án chạm được 88.02 AUC mà
paper công bố, và learning rate có thể chính là mảnh còn thiếu.

---

## Chạy thủ công (ngoài Colab)

```bash
cd VadCLIP/baseline/src

python ucf_train.py \
  --feature-root /đường/dẫn/tới/UCFClipFeatures \
  --train-list ../../list/ucf_CLIP_rgb_relative.csv \
  --test-list  ../../list/ucf_CLIP_rgbtest_relative.csv \
  --gt-path ../../list/gt_ucf.npy \
  --gt-segment-path ../../list/gt_segment_ucf.npy \
  --gt-label-path ../../list/gt_label_ucf.npy \
  --seed 234 --lr 2e-5 --max-epoch 10 --batch-size 64 \
  --model-path model/model_ucf.pth \
  --checkpoint-path model/checkpoint.pth

python ucf_test.py \
  --feature-root /đường/dẫn/tới/UCFClipFeatures \
  --test-list ../../list/ucf_CLIP_rgbtest_relative.csv \
  --gt-path ../../list/gt_ucf.npy \
  --gt-segment-path ../../list/gt_segment_ucf.npy \
  --gt-label-path ../../list/gt_label_ucf.npy \
  --model-path model/model_ucf.pth
```

Bỏ `--feature-root` và dùng `ucf_CLIP_rgb.csv` (list tuyệt đối) nếu muốn đúng hành vi upstream
nguyên bản.

---

## Những thứ **cố ý không** thêm vào

Để giữ thư mục này là mốc so sánh sạch:

- **`--num-workers` mặc định 0.** Đây là mặc định của upstream. Cờ này *có* tồn tại để chạy nhiều
  seed cho nhanh, nhưng **nó không phải cờ tăng tốc thuần tuý**: đổi nó làm đổi thứ tự video trong
  từng lô, và do đó ra một model khác. Hai lần chạy chỉ khác con số này là **hai lần rút thăm khác
  nhau**, không phải cùng một lần chạy nhanh hơn. Luôn ghi lại giá trị đã dùng.
  Đây chính là chỗ khiến `ucf_train_augment.py` (`--num-workers 4`) không trùng khít đường gốc.
- **Không sửa `layers.py`.** Bản trong `src/utils/layers.py` có cache ma trận khoảng cách (tiết kiệm
  bộ nhớ, giá trị không đổi). Ở đây dùng bản upstream nguyên vẹn. Hệ quả: `DistanceAdj` hardcode
  `.to('cuda')` nên **bắt buộc phải có GPU**.
- **Không gọi lại `model.train()` sau `test()`.** Upstream quên, nên model ở chế độ eval suốt phần
  còn lại của epoch sau lần đánh giá đầu. Vô hại — `CLIPVAD` không có Dropout hay BatchNorm, chỉ có
  LayerNorm — nên giữ nguyên để bám sát upstream.
- **Không sửa resume.** `--use-checkpoint` của upstream nạp lại `epoch` nhưng vòng lặp vẫn là
  `for e in range(args.max_epoch)`, tức luôn chạy lại từ epoch 0. Đây là lỗi upstream; không sửa vì
  không cần cho lần chạy 10 epoch liền mạch.

---

## Lưu ý về ground truth

`list/gt_ucf.npy`, `gt_segment_ucf.npy`, `gt_label_ucf.npy` bị `.gitignore` nên không có trong repo.
Chúng phải có sẵn (trên Drive) trước khi train.

Cảnh báo nếu định sinh lại: `list/make_gt_ucf.py` lọc `if '__0.npy' not in name: continue`, trong
khi `ucf_CLIP_rgbtest.csv` toàn bộ là `__5.npy` → chạy thẳng sẽ cho ra **file rỗng**. Cần sửa bộ lọc
thành `__5.npy` (hoặc dùng lại các file gt đã có).
