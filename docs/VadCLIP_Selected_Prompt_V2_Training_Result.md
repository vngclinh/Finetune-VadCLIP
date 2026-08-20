# Nhật Ký Thí Nghiệm: Chọn Một Prompt Đại Diện Cho Mỗi Class

## 1. Mục Tiêu

Sau các thí nghiệm dùng nhiều prompt cho mỗi class, một hướng đơn giản hơn được kiểm tra: **chọn một prompt đại diện tốt nhất cho mỗi class** rồi dùng bộ prompt này để train VadCLIP.

Câu hỏi chính của thí nghiệm:

> Nếu mỗi class chỉ dùng một mô tả ngắn, rõ ràng và có tính đại diện cao, model có học ổn định hơn so với việc dùng nhiều prompt hay không?

Động lực của hướng này đến từ trade-off giữa hai yếu tố:

- Nhiều prompt giúp bao phủ nhiều biến thể thị giác của cùng một class.
- Nhiều prompt cũng có thể đưa thêm nhiễu nếu một số prompt quá rộng, quá giống class khác, hoặc không phù hợp với biểu diễn ảnh của CLIP.

Vì vậy, thí nghiệm này dùng bước chấm điểm prompt để lọc ra một prompt tiêu biểu nhất cho từng class, sau đó đánh giá xem cách làm này có đủ tốt để thay thế multi-prompt hay không.

## 2. Cách Làm

### 2.1. Tạo Nhiều Prompt Ứng Viên

Với mỗi class trong UCF-Crime, nhiều prompt ứng viên được chuẩn bị trước. Các prompt được viết theo phong cách gần với caption ảnh, ưu tiên hành động và đối tượng có thể quan sát được trong frame.

Ví dụ với class `Explosion`, các prompt ứng viên có thể mô tả:

- một vụ nổ mạnh phá hủy vật thể hoặc công trình;
- ánh sáng, khói và mảnh vỡ xuất hiện sau vụ nổ;
- khối lửa hoặc sóng xung kích lan ra từ điểm nổ.

Mục tiêu của bước này là tạo một tập prompt đủ đa dạng trước khi thực hiện chấm điểm và chọn lọc.

### 2.2. Encode Prompt Bằng CLIP Text Encoder

Toàn bộ prompt ứng viên và tên class gốc được đưa qua CLIP text encoder để thu được text embedding. Các embedding được L2-normalize trước khi tính cosine similarity.

Ở bước chọn prompt, embedding được lấy từ CLIP text encoder gốc, không dùng phần learnable prompt của VadCLIP. Cách này giúp quá trình chọn prompt phản ánh semantic tự nhiên của CLIP, tránh bị bias bởi prompt context đã học trong checkpoint VadCLIP trước đó.

### 2.3. Chấm Điểm Prompt

Mỗi prompt của class `c` được đánh giá theo ba thành phần:

1. **Class anchor similarity**: prompt gần với embedding của tên class gốc.
2. **Intra-class similarity**: prompt gần với các prompt khác thuộc cùng class.
3. **Inter-class similarity**: prompt xa các prompt thuộc class khác.

Score tổng hợp:

```text
score =
  0.50 * similarity(prompt, class_name)
+ 0.30 * similarity(prompt, same_class_prompts)
- 0.20 * similarity(prompt, other_class_prompts)
```

Trọng số cho class anchor được đặt cao nhất vì tên class đóng vai trò là điểm neo semantic. Prompt được chọn không chỉ cần mô tả tốt một tình huống cụ thể, mà còn phải giữ được ý nghĩa tổng quát của class.

### 2.4. Chọn Prompt Đại Diện

Với mỗi class, prompt có score cao nhất được chọn làm prompt đại diện.

Một số ví dụ:

| Class | Prompt đại diện |
|---|---|
| Arson | a person lighting material before flames spread |
| Explosion | a powerful blast destroying an object or structure |
| Fighting | a public fight where both sides are physically attacking |
| RoadAccidents | a vehicle collision causing roadway impact and disruption |
| Shoplifting | a person secretly taking merchandise from a store without paying |

Khi đưa vào VadCLIP, hai cách sử dụng prompt đại diện được kiểm tra:

- `class name + prompt`: thêm tên class trước prompt để giữ class anchor.
- `prototype_only`: dùng trực tiếp prompt đại diện, không thêm tên class.

Ví dụ với `class name + prompt`:

```text
explosion a powerful blast destroying an object or structure
```

Ví dụ với `prototype_only`:

```text
a powerful blast destroying an object or structure
```

## 3. Cấu Hình Training

Kiến trúc chính của VadCLIP được giữ nguyên. Thay đổi nằm ở phần text prompt dùng để tạo class embedding.

Thay vì dùng nhiều prompt/class, thí nghiệm này dùng:

```text
1 prompt / class
```

Thiết lập chính:

| Thành phần | Cấu hình |
|---|---|
| Dataset | UCF-Crime |
| Số class | 14 |
| Số prompt mỗi class | 1 |
| Biến thể prompt | `class name + prompt`, `prototype_only` |
| Text encoder | CLIP text encoder, frozen |
| Visual/model-specific layers | train từ scratch |
| Learning rate | 2e-5 |
| Số epoch | 3 |
| Scheduler milestone | epoch 2 |
| Inference | giống VadCLIP gốc, không dùng description |

Loss chính:

```text
L_total =
  L_cls
+ L_mil
+ L_text_reg
+ lambda_consistency * L_consistency
```

Tuy nhiên, do mỗi class chỉ còn một prompt, consistency loss giữa các prompt trong cùng class gần như không còn tác dụng. Khi `K = 1`, mô hình không còn cơ chế so khớp và làm ổn định nhiều prompt trong cùng class.

## 4. Kết Quả Sau Fine-Tune

### 4.1. Kết Quả Tổng Quan

Bảng dưới đây so sánh VadCLIP baseline với hai biến thể dùng một prompt đại diện cho mỗi class. Cả hai biến thể đều được train từ scratch trong 3 epoch và đánh giá theo cùng protocol với VadCLIP gốc.

| Mô hình | classifier AUC | classifier AP | alignment AUC | alignment AP | Ano-AUC | mAP@0.1 | mAP@0.2 | mAP@0.3 | mAP@0.4 | mAP@0.5 | avg mAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline VadCLIP | 88.02 | 33.56 | 85.69 | 26.50 | 70.23 | 11.72 | 7.83 | 6.40 | 4.53 | 2.93 | 6.68 |
| Selected prompt, class name + prompt | 86.87 | 32.02 | 85.36 | 26.36 | 68.40 | 14.59 | 10.37 | 7.10 | 6.03 | 3.51 | 8.32 |
| Selected prompt, prototype only | 86.25 | 28.50 | 84.69 | 24.04 | 66.99 | 13.08 | 9.76 | 6.63 | 5.70 | 3.00 | 7.64 |

So với baseline:

| Mô hình | Delta classifier AUC | Delta classifier AP | Delta alignment AUC | Delta alignment AP | Delta Ano-AUC | Delta avg mAP |
|---|---:|---:|---:|---:|---:|---:|
| Class name + prompt | -1.15 | -1.54 | -0.33 | -0.14 | -1.83 | +1.64 |
| Prototype only | -1.77 | -5.06 | -1.01 | -2.46 | -3.24 | +0.96 |

Kết quả cho thấy hướng một prompt/class tạo ra trade-off rõ ràng: khả năng định vị đoạn bất thường tăng lên qua `avg mAP`, nhưng khả năng phân tách frame bình thường và bất thường giảm qua `classifier AUC` và `Ano-AUC`.

### 4.2. Diễn Biến Theo Epoch

Với biến thể `class name + prompt`, kết quả tốt dần qua từng epoch. `classifier AUC` tăng từ `85.63` ở epoch 1 lên `86.87` ở epoch 3, trong khi `avg mAP` tăng từ `7.99` lên `8.32`. Điều này cho thấy việc thêm tên class trước prompt giúp quá trình học ổn định hơn, dù vẫn chưa khôi phục được AUC của baseline.

| Epoch | classifier AUC | classifier AP | alignment AUC | alignment AP | Ano-AUC | avg mAP |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 85.63 | 29.63 | 84.61 | 24.52 | 65.61 | 7.99 |
| 2 | 86.69 | 32.39 | 85.09 | 25.80 | 68.05 | 7.45 |
| 3 | 86.87 | 32.02 | 85.36 | 26.36 | 68.40 | 8.32 |

Với biến thể `prototype_only`, `avg mAP` cũng tăng so với baseline nhưng các chỉ số AUC thấp hơn rõ rệt. Điều này cho thấy nếu bỏ tên class, prompt đại diện dễ trở thành một mô tả quá cụ thể, không đủ đóng vai trò class embedding tổng quát.

| Epoch | classifier AUC | classifier AP | alignment AUC | alignment AP | Ano-AUC | avg mAP |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 86.16 | 29.47 | 84.36 | 23.90 | 66.47 | 5.95 |
| 2 | 86.05 | 28.51 | 84.81 | 24.35 | 66.87 | 7.26 |
| 3 | 86.25 | 28.50 | 84.69 | 24.04 | 66.99 | 7.64 |

### 4.3. Nhận Xét Theo Class

Ở cả hai biến thể, một số class được cải thiện khá rõ, đặc biệt là `Vandalism` và `Assault`. Với `class name + prompt`, classifier AUC của `Vandalism` tăng khoảng `+11.12` điểm và `Assault` tăng khoảng `+7.80` điểm. Với `prototype_only`, hai class này cũng tăng lần lượt khoảng `+10.97` và `+8.53` điểm.

Ngược lại, `Explosion` là class bị giảm mạnh nhất. Với `class name + prompt`, classifier AUC của `Explosion` giảm khoảng `-14.44` điểm; với `prototype_only`, mức giảm khoảng `-16.07` điểm. Điều này phù hợp với các quan sát trước đó: prompt mô tả `Explosion` rất dễ bị chồng lấn với các hiện tượng thị giác như lửa, khói, phá hủy, va chạm hoặc cháy lan, khiến class embedding không đủ tách biệt.

Một số class khác như `Fighting`, `Abuse`, `Arson`, `Burglary` cũng giảm ở biến thể `prototype_only`, cho thấy việc bỏ tên class làm mất semantic anchor và khiến prompt dễ bị hiểu lệch sang các class gần nghĩa.

### 4.4. Kết Luận Từ Kết Quả

Biến thể `class name + prompt` tốt hơn `prototype_only` ở hầu hết chỉ số tổng thể. Điều này cho thấy tên class vẫn là một anchor semantic quan trọng trong VadCLIP. Prompt đại diện có thể bổ sung ngữ cảnh, nhưng nếu dùng prompt một mình thì class embedding dễ bị hẹp hoặc lệch khỏi ý nghĩa ban đầu của class.

Tuy nhiên, cả hai biến thể đều chưa vượt baseline về AUC. Kết quả tốt nhất của hướng này là tăng `avg mAP` từ `6.68` lên `8.32`, nhưng đổi lại `classifier AUC` giảm từ `88.02` xuống `86.87` và `Ano-AUC` giảm từ `70.23` xuống `68.40`. Vì vậy, hướng một prompt/class có tín hiệu tích cực cho localization, nhưng chưa đủ tốt để thay thế hoàn toàn multi-prompt hoặc class prompt gốc.

## 5. Nhận Xét Kết Quả

Nguyên nhân chính khiến AUC giảm là một prompt duy nhất không đủ bao phủ toàn bộ biến thể thị giác của một class. Ví dụ:

- `Explosion` có thể xuất hiện dưới dạng ánh sáng chói, khói, lửa, mảnh vỡ hoặc công trình bị phá hủy.
- `Arson` có thể là hành động châm lửa, đổ chất cháy, hoặc chỉ còn cảnh lửa đang lan.
- `Robbery` có thể là đe dọa, giật đồ, khống chế nạn nhân hoặc lấy tài sản.
- `Assault` có thể là đấm, đá, xô đẩy, hoặc một hành động tấn công rất ngắn.

Khi chỉ giữ một prompt, model chỉ có một cách diễn đạt để đại diện cho class. Điều này làm giảm khả năng bao phủ các trường hợp khác nhau trong cùng class, đặc biệt với các class có biểu hiện thị giác đa dạng như `Explosion`, `Arson`, `Robbery` hoặc `Assault`.

Ngược lại, trong hướng multi-prompt, mỗi class có nhiều mô tả khác nhau. Khi model aggregate nhiều prompt, một frame chỉ cần khớp tốt với một trong các prompt phù hợp là vẫn có thể tạo tín hiệu class tốt. Đây là lý do multi-prompt giữ được semantic diversity tốt hơn hướng top-1 prompt.

So sánh hai biến thể cũng cho thấy `prototype_only` không phải lựa chọn tốt trong bối cảnh này. Khi không có tên class làm anchor, prompt đại diện phải tự gánh toàn bộ ý nghĩa của class. Nếu prompt hơi hẹp hoặc lệch semantic, toàn bộ class embedding sẽ bị lệch theo.

Trong khi đó, `class name + prompt` giữ được class anchor nên giảm drift tốt hơn. Kết quả của biến thể này vẫn thấp hơn baseline về AUC, nhưng tăng `avg mAP` mạnh hơn `prototype_only`.

Thí nghiệm một prompt/class cho thấy việc chọn prompt bằng CLIP similarity có thể giúp lọc nhiễu, nhưng nếu lọc quá mạnh thì thông tin semantic bị mất. Hướng tiếp theo vì vậy không nên chọn duy nhất một prompt, mà nên giữ một nhóm nhỏ prompt tốt và gán trọng số cho từng prompt.

# Nhật Ký Thí Nghiệm: Multi-Prompt

## 1. Mục Tiêu

Sau thí nghiệm chọn một prompt đại diện cho mỗi class, hướng tiếp theo là giữ lại nhiều prompt cho mỗi class. Mục tiêu của thí nghiệm multi-prompt là kiểm tra xem việc giữ nhiều mô tả ngắn có giúp model bao phủ tốt hơn các biến thể thị giác trong cùng một class hay không.

Câu hỏi chính của thí nghiệm:

> Nếu mỗi class có nhiều prompt ngắn, model có giữ được semantic diversity tốt hơn so với chỉ chọn một prompt đại diện không?

Động lực của hướng này đến từ hạn chế của top-1 prompt. Một prompt duy nhất thường quá hẹp, đặc biệt với các class có nhiều biểu hiện thị giác như `Explosion`, `Arson`, `Robbery`, `Assault`. Multi-prompt cho phép mỗi class có nhiều cách mô tả khác nhau, từ đó tăng khả năng một frame bất thường khớp với ít nhất một prompt phù hợp.

## 2. Cách Làm

### 2.1. Tạo Bộ Prompt 5x

Mỗi class được mô tả bằng 5 prompt ngắn theo phong cách gần với caption ảnh. Các prompt ưu tiên những tín hiệu có thể quan sát được trong frame, ví dụ actor, object, action, visible effect và scene context cơ bản.

So với các prototype dài trước đó, prompt 5x được viết ngắn hơn để giảm nguy cơ làm loãng text embedding của CLIP. Mỗi prompt cố gắng mô tả một khía cạnh khác nhau của class, thay vì viết lại cùng một ý nhiều lần.

Ví dụ với `Explosion`, các prompt có thể tập trung vào:

- một vụ nổ mạnh phá hủy vật thể hoặc công trình;
- lửa, khói và mảnh vỡ sau vụ nổ;
- tác động tức thời của vụ nổ lên môi trường xung quanh.

### 2.2. Đưa Multi-Prompt Vào VadCLIP

Kiến trúc chính của VadCLIP được giữ nguyên. Thay đổi nằm ở phần text prompt dùng để tạo class embedding.

Với mỗi class, thay vì chỉ có một text embedding, model encode 5 prompt riêng biệt:

```text
5 prompt / class
14 class
=> prompt logits [B, T, 14, 5]
```

Các prompt cùng class được aggregate mềm để tạo lại `logits2 [B, T, 14]`. Cách này giữ từng prompt riêng ở bước alignment, thay vì nén toàn bộ prompt thành một centroid ngay từ đầu.

Trong thí nghiệm này, prompt được dùng theo kiểu:

```text
class name + prompt
```

Ví dụ:

```text
explosion a powerful blast destroying an object or structure
```

Tên class được giữ lại để đóng vai trò semantic anchor, vì các thí nghiệm trước cho thấy `prototype_only` dễ làm class embedding lệch khỏi ý nghĩa ban đầu.

## 3. Cấu Hình Training

Thiết lập chính:

| Thành phần | Cấu hình |
|---|---|
| Dataset | UCF-Crime |
| Số class | 14 |
| Số prompt mỗi class | 5 |
| Prompt mode | `class name + prompt` |
| Text encoder | CLIP text encoder, frozen |
| Visual/model-specific layers | train từ scratch |
| Learning rate | 2e-5 |
| Số epoch | 3 |
| Scheduler milestone | epoch 2 |
| Inference | giống VadCLIP gốc, không dùng description |

Loss chính:

```text
L_total =
  L_cls
+ L_mil
+ L_text_reg
+ lambda_consistency * L_consistency
```

Khác với thí nghiệm một prompt/class, consistency loss có ý nghĩa hơn trong thí nghiệm này vì mỗi class có 5 prompt. Loss này giúp các prompt cùng class không tạo ra alignment quá lệch nhau trên các đoạn quan trọng của video.

## 4. Kết Quả Sau Fine-Tune

### 4.1. Kết Quả Tổng Quan

Bảng dưới đây so sánh VadCLIP baseline với multi-prompt 5x v5 sau 3 epoch.

| Mô hình | classifier AUC | classifier AP | alignment AUC | alignment AP | Ano-AUC | mAP@0.1 | mAP@0.2 | mAP@0.3 | mAP@0.4 | mAP@0.5 | avg mAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline VadCLIP | 88.02 | 33.56 | 85.69 | 26.50 | 70.23 | 11.72 | 7.83 | 6.40 | 4.53 | 2.93 | 6.68 |
| Multi-prompt 5x | 87.05 | 31.20 | 86.60 | 29.04 | 68.65 | 13.51 | 9.96 | 6.79 | 3.82 | 2.45 | 7.30 |

So với baseline:

| Mô hình | Delta classifier AUC | Delta classifier AP | Delta alignment AUC | Delta alignment AP | Delta Ano-AUC | Delta avg mAP |
|---|---:|---:|---:|---:|---:|---:|
| Multi-prompt 5x v5 | -0.97 | -2.35 | +0.91 | +2.53 | -1.58 | +0.62 |

Kết quả cho thấy multi-prompt 5x v5 giữ AUC tốt hơn hướng một prompt/class, đồng thời cải thiện rõ nhánh alignment so với baseline. `alignment AUC` tăng từ `85.69` lên `86.60`, và `alignment AP` tăng từ `26.50` lên `29.04`. Tuy nhiên, `classifier AUC`, `classifier AP` và `Ano-AUC` vẫn thấp hơn baseline.

### 4.2. Diễn Biến Theo Epoch

| Epoch | classifier AUC | classifier AP | alignment AUC | alignment AP | Ano-AUC | avg mAP |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 86.51 | 32.03 | 85.86 | 30.03 | 66.94 | 4.94 |
| 2 | 86.87 | 32.64 | 86.71 | 30.01 | 68.48 | 6.87 |
| 3 | 87.05 | 31.20 | 86.60 | 29.04 | 68.65 | 7.30 |

Theo epoch, `classifier AUC` và `Ano-AUC` tăng dần, cho thấy quá trình train ổn định hơn so với một số hướng prototype trước đó. `avg mAP` cũng tăng đều từ `4.94` lên `7.30`, vượt baseline ở epoch 2 và epoch 3.

Điểm đáng chú ý là `alignment AUC` đạt cao nhất ở epoch 2 với `86.71`, sau đó giảm nhẹ ở epoch 3 còn `86.60`. Điều này cho thấy multi-prompt giúp nhánh alignment tốt hơn, nhưng nếu train lâu hơn có thể cần theo dõi để tránh drift sang hướng tối ưu mAP cục bộ mà làm giảm AP/AUC.

### 4.3. Nhận Xét Theo Class

Một số class được cải thiện rõ ở nhánh classifier:

| Class | Baseline AUC | Final AUC | Delta |
|---|---:|---:|---:|
| Vandalism | 70.97 | 79.63 | +8.66 |
| Abuse | 56.98 | 62.39 | +5.41 |
| Shooting | 65.63 | 69.16 | +3.53 |
| Burglary | 70.44 | 72.52 | +2.08 |
| Assault | 84.34 | 85.92 | +1.58 |

Các class bị giảm mạnh nhất:

| Class | Baseline AUC | Final AUC | Delta |
|---|---:|---:|---:|
| Explosion | 59.18 | 45.81 | -13.37 |
| Fighting | 67.90 | 65.57 | -2.33 |
| Arrest | 68.57 | 67.54 | -1.03 |
| RoadAccidents | 56.83 | 55.90 | -0.93 |
| Robbery | 73.30 | 72.99 | -0.30 |

`Explosion` vẫn là class khó nhất trong nhóm prompt-based experiments. Dù đã dùng nhiều prompt, class này vẫn giảm mạnh. Nguyên nhân có thể đến từ việc các prompt của `Explosion` vẫn chồng lấn với các khái niệm như lửa, khói, phá hủy, cháy lan hoặc va chạm. Những tín hiệu này cũng xuất hiện trong `Arson` và `RoadAccidents`, khiến text embedding khó tách biệt hoàn toàn.

Ngược lại, các class như `Vandalism`, `Abuse`, `Shooting`, `Burglary` hưởng lợi từ multi-prompt. Đây là các class có những hành động hoặc đối tượng thị giác tương đối rõ hơn, nên việc cung cấp nhiều prompt giúp model bắt được nhiều biến thể mà không làm lệch class quá mạnh.

### 4.4. So Sánh Với Hướng Một Prompt Đại Diện

So với hướng chọn một prompt đại diện, multi-prompt 5x v5 có điểm mạnh là giữ AUC tốt hơn:

| Mô hình | classifier AUC | alignment AUC | Ano-AUC | avg mAP |
|---|---:|---:|---:|---:|
| Selected prompt, class name + prompt | 86.87 | 85.36 | 68.40 | 8.32 |
| Selected prompt, prototype only | 86.25 | 84.69 | 66.99 | 7.64 |
| Multi-prompt 5x v5 | 87.05 | 86.60 | 68.65 | 7.30 |

Multi-prompt 5x v5 có `classifier AUC`, `alignment AUC` và `Ano-AUC` tốt hơn cả hai biến thể một prompt/class. Điều này cho thấy giữ nhiều prompt giúp class embedding ổn định và tổng quát hơn.

Tuy nhiên, `avg mAP` của multi-prompt 5x v5 thấp hơn biến thể selected prompt `class name + prompt`. Điều này cho thấy một prompt đại diện có thể tạo tín hiệu localization mạnh hơn trong một số trường hợp, nhưng đổi lại kém ổn định hơn về AUC. Multi-prompt là hướng cân bằng hơn giữa localization và frame-level discrimination.

## 5. Nhận Xét Kết Quả

Multi-prompt 5x v5 là một bước tốt hơn so với việc chọn duy nhất một prompt nếu ưu tiên sự ổn định tổng thể. Nó không vượt baseline ở `classifier AUC` và `Ano-AUC`, nhưng giảm ít hơn so với `prototype_only`, đồng thời cải thiện nhánh alignment.

Kết quả này củng cố giả thuyết rằng prompt không nên bị lọc quá mạnh xuống còn một câu duy nhất. Một class bất thường trong UCF-Crime thường có nhiều biểu hiện khác nhau, nên cần nhiều prompt để giữ semantic diversity.

Tuy nhiên, việc dùng 5 prompt ngang trọng số vẫn còn hạn chế. Nếu một hoặc hai prompt trong class bị nhiễu, chúng vẫn có ảnh hưởng ngang với prompt tốt. Đây là lý do hướng tiếp theo hợp lý là **weighted multi-prompt**: vẫn giữ nhiều prompt, nhưng gán trọng số cao hơn cho prompt gần class anchor và ít chồng lấn với class khác.
