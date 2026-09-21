# Thống Kê Thực Nghiệm: Cắt Đặc Trưng & Nhất Quán Theo Dịch Chuyển

Tổng hợp **những gì đã thực sự chạy** trong hai vòng thí nghiệm shift-consistency, đọc lại
từ báo cáo và từ file kết quả gốc. Mỗi con số đều kèm nguồn.

Nguồn: `docs/BaoCao_VadCLIP_Shift_Consistency.docx`, `Result/Result/shift_consistency_all_metrics.csv`,
`Result/shift_v2_metrics.csv`, `Result/logs_shift_v2/*.log`,
`Result/Result/logs_shift_consistency/*.log`.

---

## 1. Đã thực nghiệm những cách cắt nào

**Đúng một cách.** Đây là kết luận quan trọng nhất của bản thống kê này.

| Tham số | Giá trị đã chạy | Các giá trị khác mà code hỗ trợ nhưng **chưa từng chạy** |
|---|---|---|
| `--shift-offset` | `26` (≈10% của lưới T = 256) | bất kỳ số nào khác |
| `--shift-direction` | `head` (bỏ 26 ô đầu) | `tail`, `both` |
| `--random-shift` | `false` (luôn dùng đúng 26) | `true` (bốc trong [1, 26]) |
| `--shift-ratio` | `0.0` (cắt theo lưới) | `> 0` (cắt theo % độ dài **của chính video đó**) |
| `--shift-ratio-warmup` | `0` (không có curriculum) | `> 0` |
| `--consistency-branch` | `c` | `a`, `both` |
| `--consistency-detach` | `false` | `true` |

Cả hai vòng đều dùng y hệt cấu hình cắt này. Kiểm chứng: mọi dòng trong
`Result/shift_v2_metrics.csv` đều có `shift_ratio = 0.0`, `shift_direction = head`,
`random_shift = 0`; và dòng config đầu mỗi log vòng 1 cũng vậy.

### Một điều dễ hiểu nhầm về cấu hình này

`--lambda-consistency 0` **không phải** là "có augment mà không có loss". Ba hàm mất mát
gốc (`L₁`, `L₂`, `L₃`) chỉ được tính trên khung nhìn **đầy đủ**; khung nhìn dịch chỉ tồn
tại để so sánh trong `L_shift`. Nhân `L_shift` với 0 thì nó không đóng góp gradient nào —
tức `λ = 0` **là VadCLIP gốc**, không hơn.

Nói cách khác: **hướng "chỉ thêm augment data" chưa từng được thực nghiệm.** Nó cần cờ
`--augment-task-loss` (mới thêm), khiến `L₁`/`L₂` tính trên cả hai khung nhìn.

### Thống kê về phép cắt (đo thật, không ước lượng)

| Đại lượng | Giá trị |
|---|---|
| Tổng số tệp huấn luyện | 16.100 |
| `N` nhỏ nhất / lớn nhất | 6 / 61.032 |
| `N` trung bình / trung vị | 490,9 / 138,5 |
| Tỉ lệ đi qua nhánh **đệm không** (`N ≤ 256`) | 72,0% |
| Tỉ lệ đi qua nhánh **nén đều** (`N > 256`) | 28,0% |
| Tệp mất hẳn vùng chồng lấn ở Δ = 26 | 630 / 16.100 = **3,91%** |

Hệ quả đáng chú ý: 26 ô là ~10% của **lưới**, nhưng với video có độ dài trung vị
(ℓ = 138) nó là **~19% của video**. Liều lượng dịch chuyển vì vậy không đều giữa các
video — đó chính là lý do `--shift-ratio` tồn tại, và cũng là lý do nó đáng được thử.

---

## 2. Trọng số cho loss 4 (`λ`) đã dùng bao nhiêu

### Vòng 1 — đặt tay, tuyệt đối

| Lần chạy | `λ` | Cách đặt |
|---|---|---|
| `baseline_ctrl` | `0` | — |
| `v0` | `0,01` | Đặt tay |

`0,01` là con số **đoán**. Và nó nhỏ hơn nhiều so với cảm giác: ở bước 0,
`L_task ≈ 2,11` còn `L_shift ≈ 3,5 × 10⁻³` — chênh khoảng **600 lần**. Nhân thêm 0,01 thì
số hạng thứ tư chiếm khoảng **1/60.000** tổng hàm mục tiêu.

### Vòng 2 — đặt theo tỉ lệ, giải tự động

Vòng 2 đổi sang `--lambda-auto r`: chạy 100 bước với số hạng tắt, đo cả hai vế, rồi giải
`λ = r · L_task / L_shift`. Giá trị `λ` thực tế được giải ra:

| Lần chạy | `r` (mục tiêu) | `λ` giải ra | Tỉ lệ **thực tế đạt được** cuối run |
|---|---|---|---|
| `v2_ctrl_s234` | — | `0` | — |
| `v2_ctrl_s1234` | — | `0` | — |
| `v2_lam0.01` | 0,01 | `6,0154` | **1,9%** |
| `v2_lam0.03` | 0,03 | `18,046` | **2,6%** |
| `v2_lam0.1` | 0,10 | `60,154` | **4,0%** |

Cột cuối là vấn đề của vòng 2. Giải `λ` **một lần** ở bước 100 rồi giữ nguyên thì tỉ lệ
trôi: `L_task` giảm khoảng 5 lần trong khi `L_shift` giảm 10–40 lần, nên tỉ lệ thực tế mà
số hạng nắm giữ tụt xuống. Ba lần chạy định trải **một bậc thập phân** (1% → 10%) kết thúc
chỉ còn cách nhau **hai lần** (1,9% → 4,0%). Không tách được gì.

Đó là lý do `--lambda-auto-recalibrate` tồn tại (giải lại mỗi epoch, có `--lambda-auto-max-growth`
làm dây cương). **Cờ này chưa từng được dùng trong một lần chạy đầy đủ nào.**

---

## 3. Kết quả đạt được

### 3.1. Vòng 1 — `--select-metric auc` (giữ checkpoint AUC cao nhất)

Seed 234, 10 epoch, lr 2e-5, lô 64, train từ đầu, nhánh C, warmup λ 1 epoch.

| Mô hình | AUC nhánh C | AP nhánh C | AUC nhánh A | AP nhánh A | mAP TB |
|---|---:|---:|---:|---:|---:|
| Checkpoint tác giả | 88,02 | 33,56 | 85,69 | 26,50 | 6,68 |
| `baseline_ctrl` (λ = 0) | **88,13** | 33,10 | 86,67 | 29,03 | **8,50** |
| `v0` (λ = 0,01) | 87,50 | 31,10 | 86,24 | 27,81 | **9,34** |

mAP chi tiết:

| Mô hình | @0,1 | @0,2 | @0,3 | @0,4 | @0,5 |
|---|---:|---:|---:|---:|---:|
| Checkpoint tác giả | 11,72 | 7,83 | 6,40 | 4,53 | 2,93 |
| `baseline_ctrl` | 15,79 | 11,09 | 7,14 | 4,96 | 3,54 |
| `v0` | 16,60 | 12,25 | 8,10 | 5,41 | 4,37 |

> **Lưu ý về hai con số 88,13 và 87,69 cùng nói về `baseline_ctrl`.** 88,13 là lần chấm
> tốt nhất trong ~120 lần chấm giữa epoch (`shift_consistency_all_metrics.csv` không ghi
> nó; nó nằm ở Bảng 3 của báo cáo). 87,69 là điểm của **checkpoint epoch cuối**, lấy từ
> `ucf_checkpoint_diagnostics_baseline_ctrl/metrics_by_checkpoint.csv`. Cả hai đều đúng,
> chỉ khác độ mịn. Đây chính là chỗ hay bị đọc nhầm.

### 3.2. Vòng 1 — độ nhạy dịch chuyển (chỉ số **có ý nghĩa nhất** của cả hướng)

266 video (loại 24 video ngắn hơn độ dịch lớn nhất), tương quan trung bình giữa chuỗi
điểm số gốc và chuỗi sau khi dịch rồi căn lại, trên nhánh C.

| Mô hình | Δ = 8 | Δ = 16 | Δ = 32 | Biên độ dao động AUC |
|---|---:|---:|---:|---:|
| Checkpoint tác giả | 0,734 | 0,697 | 0,638 | **0,152** |
| `baseline_ctrl` (λ = 0) | 0,702 | 0,635 | 0,612 | 0,366 |
| `v0` (λ = 0,01) | **0,720** | **0,645** | **0,636** | **0,268** |

`v0` tốt hơn `baseline_ctrl` ở **cả ba** mức dịch và ở biên độ dao động. Đây là tác động
đúng hướng thiết kế, và là bằng chứng nhất quán duy nhất của vòng 1. Nhưng cả hai mô hình
huấn luyện lại đều **kém ổn định hơn checkpoint gốc của tác giả** (0,152).

### 3.3. Vòng 2 — `--select-metric none` (giữ trọng số cuối epoch 10)

| Lần chạy | seed | `λ` | AUC nhánh C | AP nhánh C | AUC nhánh A | mAP TB |
|---|---:|---:|---:|---:|---:|---:|
| `v2_ctrl_s234` | 234 | 0 | **87,84** | 31,11 | 86,56 | 8,15 |
| `v2_ctrl_s1234` | 1234 | 0 | **85,28** | 27,08 | 83,90 | 6,57 |
| `v2_lam0.01` | 234 | 6,015 | 87,69 | 31,34 | 86,16 | 7,77 |
| `v2_lam0.03` | 234 | 18,046 | 87,67 | 31,92 | 86,16 | 7,05 |
| `v2_lam0.1` | 234 | 60,154 | 87,27 | 31,78 | 85,91 | 7,03 |

Đọc bảng này: **hai lần đối chứng cùng cấu hình chỉ khác seed lệch nhau 2,56 điểm AUC**
(87,84 vs 85,28). Khoảng cách đó lớn hơn mọi hiệu ứng của λ trong cùng bảng. Ba lần chạy
λ nằm gọn trong dải 87,27–87,69, tức không tách khỏi nhau và cũng không tách khỏi đối
chứng seed 234.

---

## 4. Sàn nhiễu — con số phải nhớ khi đọc mọi bảng trên

Hai lần huấn luyện **giống hệt nhau về mặt toán học** (cùng seed, cùng cấu hình, λ = 0,
không áp dụng phương pháp nào) cho ra:

| | AUC nhánh C | mAP TB |
|---|---:|---:|
| Lần 1 | 88,13 | 8,50 |
| Lần 2 | 87,55 | 6,95 |
| **Chênh lệch** | **0,58** | **1,55** |

Nguyên nhân: GPU không tất định, cộng với việc chọn checkpoint theo AUC cao nhất qua ~120
lần chấm **trên chính tập test**.

**Quy tắc bắt buộc khi đọc:** mọi chênh lệch AUC dưới **0,58 điểm** và mAP dưới
**1,55 điểm** đều không được tuyên bố là cải thiện.

Áp quy tắc này vào Bảng 3.1: chênh lệch AUC giữa `v0` và `baseline_ctrl` là **0,63 điểm**
— xấp xỉ đúng sàn nhiễu. Chênh lệch mAP là 0,84 điểm — **nhỏ hơn** sàn nhiễu. Vì vậy
Bảng 3.1 và 3.2 **chưa kết luận được gì** về chất lượng phát hiện. Bằng chứng duy nhất
đứng vững là Bảng 3.2 (độ nhạy dịch chuyển).

---

## 5. Những khoảng trống còn lại

| Khoảng trống | Vì sao đáng làm |
|---|---|
| **Chưa thử kiểu cắt nào khác ngoài `head 26` cố định** | `tail` không mất nội dung với 72% video; `ratio` làm đều liều lượng; `random` ràng buộc ở nhiều khoảng cách. Cả ba đều đã có trong code, chưa lần nào được chạy. |
| **Chưa tách được "augment" khỏi "consistency"** | `λ = 0` là VadCLIP gốc, không phải augment. Cần `--augment-task-loss`. |
| **Chưa dùng `--lambda-auto-recalibrate`** | Đây là thứ sửa đúng cái trôi tỉ lệ đã đo được ở vòng 2. |
| **Chưa cân λ theo chuẩn gradient** | Optimizer đi theo gradient chứ không theo giá trị loss. Hai tỉ lệ này không thay thế được cho nhau. |
| **Chưa có nhiều seed cho mỗi cấu hình** | Vòng 2 đo được 2,56 điểm chênh lệch trên đúng **hai** mẫu. Muốn nói gì về λ thì phải ≥ 3 seed mỗi cấu hình. |
| **Siêu tham số vẫn được chọn trên tập test** | Cách sạch là cắt validation **theo video** từ tập train. Chưa làm. Khi viết báo cáo phải ghi thẳng điều này. |

Notebook `code/train_shift_consistency_kaggle.ipynb` (bản viết lại) nhắm vào bốn khoảng
trống đầu.
