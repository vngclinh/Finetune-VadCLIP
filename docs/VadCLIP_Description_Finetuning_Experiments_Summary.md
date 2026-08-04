# Tổng kết các kiến trúc fine-tune VadCLIP với video description

Tài liệu này ghi lại các hướng fine-tune đã triển khai/thử nghiệm để tận dụng GPT video descriptions cho UCF-Crime. Mục tiêu chính là cải thiện VadCLIP mà không làm mất năng lực anomaly detection đã học từ checkpoint `model_ucf.pth`.
## Tập dữ liệu UCF-Crime và nguồn mô tả UCA
UCF-Crime là một tập dữ liệu lớn cho bài toán phát hiện bất thường trong video giám sát. Tập dữ liệu gồm 1.900 video thực tế, chưa được cắt ngắn, với tổng thời lượng khoảng 128 giờ. Trong đó có 950 video bình thường và 950 video bất thường. Các video được thu thập từ nhiều bối cảnh giám sát khác nhau như đường phố, cửa hàng, khu vực công cộng và không gian trong nhà, nên có độ đa dạng lớn về góc quay, ánh sáng, mật độ người và bối cảnh.

UCF-Crime bao gồm 13 loại bất thường có ảnh hưởng đến an toàn công cộng:

```text
Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting,
RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism
```

Khi huấn luyện và đánh giá bài toán anomaly detection, các video này thường được xem theo 14 nhóm: 13 nhóm bất thường ở trên và nhóm `Normal`. Theo protocol phổ biến của UCF-Crime, tập dữ liệu được chia thành 1.610 video huấn luyện và 290 video kiểm thử. Tập huấn luyện chủ yếu có nhãn cấp video, còn tập kiểm thử có ground truth cấp frame/segment để tính các chỉ số như frame-level AUC, Ano-AUC và detection mAP.

Trong nghiên cứu này, description không được viết thủ công từ đầu mà được xây dựng dựa trên UCA, một tập annotation mở rộng cho UCF-Crime. UCA cung cấp các câu mô tả theo từng khoảng thời gian trong video. Nói cách khác, thay vì chỉ biết một video thuộc class nào, ta có thêm các mô tả tự nhiên cho biết ở từng đoạn thời gian, nhân vật, hành động, đối tượng và tương tác nào đang xuất hiện.

Điểm quan trọng là UCA mô tả video theo nhiều timestamp rời rạc, trong khi hướng fine-tune cần một mô tả tổng quát cho toàn bộ video. Vì vậy, trước khi đưa description vào VadCLIP, cần có một bước tổng hợp các mô tả timestamp thành một global video description.

Nguồn tham khảo về UCF-Crime:

- Trang dự án UCF-Crime của UCF CRCV: https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/
- Trang dataset của UCF CRCV: https://www.crcv.ucf.edu/chenchen/datasets/

## Quy trình tạo global video description
Mục tiêu của bước tạo description là biến nhiều câu mô tả ngắn theo thời gian thành một mô tả duy nhất, ngắn gọn và trung tính cho toàn bộ video. Description này không dùng trực tiếp khi inference. Nó chỉ đóng vai trò là tín hiệu ngữ nghĩa bổ sung trong quá trình fine-tune.

Quy trình tổng quát gồm hai bước.

Đầu tiên, các câu mô tả timestamp của cùng một video được sắp xếp theo đúng thứ tự thời gian. Những câu bị lặp liền kề hoặc có nhiễu định dạng đơn giản được làm sạch. Kết quả của bước này là một chuỗi mô tả theo tiến trình thời gian của video, giúp mô hình ngôn ngữ nhìn được sự phát triển của sự kiện từ đầu đến cuối.

Tiếp theo, một mô hình ngôn ngữ lớn được dùng để tổng hợp chuỗi mô tả timestamp thành một global video description. Trong các thử nghiệm ban đầu, GPT cho kết quả tự nhiên và ổn định hơn so với Gemma/Ollama, nên pipeline chính sử dụng OpenAI GPT model `gpt-5.6-luna`. Mục tiêu của prompt là tạo description đủ ngắn để tránh nhiễu, nhưng vẫn giữ được các thông tin quan trọng như nhân vật, hành động chính, tương tác, đối tượng và diễn tiến thời gian.

### Prompt sử dụng để tạo description

System prompt được sử dụng nguyên văn như sau:

```text
You summarize timestamp-level video descriptions into one global video description for representation-learning supervision.
You must be faithful to the provided descriptions only.
Do not infer causes, identities, intentions, labels, or events that are not explicitly described.
Do not mention anomaly category names or dataset labels.
Prefer neutral wording such as person, people, individual, vehicle, object, and area.
Use plain ASCII punctuation only.
Return only valid JSON.
```

User prompt template được sử dụng nguyên văn như sau:

```text
Create one global description for the whole video.

Requirements:
- Use 40 to 70 words.
- Focus on visible actors, actions, interactions, objects, and temporal progression.
- Do not mention the class name: {class_name}.
- Do not use words from this forbidden list: abuse, arrest, arson, assault, burglary, explosion, fighting, road accident, roadaccident, robbery, shooting, shoplifting, stealing, vandalism, anomaly, anomalous, crime.
- Do not add any event that is not in the timestamp descriptions.
- Use plain ASCII punctuation only.
- Return exactly this JSON shape: {"description": "..."}

Video ID: {video_id}
Duration: {duration:.2f} seconds
Timestamp descriptions:
{timestamp_descriptions}
```

Trong đó `{timestamp_descriptions}` là danh sách các mô tả theo thời gian, được đưa vào theo dạng:

```text
1. start_time-end_time: sentence
2. start_time-end_time: sentence
3. start_time-end_time: sentence
...
```

Thiết kế prompt này có ba mục đích chính:

- Giữ description trung thành với annotation gốc, không tự bịa thêm sự kiện.
- Tránh để description lộ trực tiếp class label, vì nếu mô tả chứa các từ như `robbery`, `shooting` hoặc `shoplifting`, mô hình có thể học shortcut từ nhãn thay vì học ngữ nghĩa hành động.
- Giới hạn độ dài để description không trở thành một đoạn kể chuyện quá chi tiết, gây nhiễu cho quá trình fine-tune.

Sau khi sinh description, kết quả được kiểm tra ở mức nội dung: mô tả phải không rỗng, có độ dài hợp lý, không chứa trực tiếp tên class bất thường và không thêm sự kiện ngoài các câu timestamp đã cho. Các video có mô tả chưa đạt yêu cầu có thể được sinh lại để cải thiện chất lượng dữ liệu.

## Thống kê description đã tạo
Tập description hiện tại có 1.854 video theo phạm vi bao phủ của UCA. Trong đó, 1.850 video có global description không rỗng. Độ dài trung bình của description là khoảng 58 từ, median là 59 từ. Phần lớn description nằm trong khoảng 46-68 từ, tương đối sát với mục tiêu ban đầu là 40-70 từ.

Phân bố số video có description theo class:

| Class | Số video |
|---|---:|
| Normal | 910 |
| Abuse | 50 |
| Arrest | 50 |
| Arson | 50 |
| Assault | 48 |
| Burglary | 100 |
| Explosion | 50 |
| Fighting | 50 |
| RoadAccidents | 148 |
| Robbery | 149 |
| Shooting | 50 |
| Shoplifting | 50 |
| Stealing | 100 |
| Vandalism | 49 |

Nhìn chung, tập description đã tạo có độ bao phủ tốt trên hầu hết các class của UCF-Crime. Tuy nhiên, description hiện tại vẫn có một hạn chế quan trọng: độ dài trung bình khoảng 58 từ có thể còn quá dài đối với mục tiêu fine-tune VadCLIP. Một số description chứa nhiều chi tiết phụ như màu quần áo, vị trí đồ vật hoặc chuyển động nền. Các chi tiết này giúp mô tả video tự nhiên hơn, nhưng có thể khiến model học những tín hiệu không thật sự liên quan đến anomaly detection. Vì vậy, một hướng thử nghiệm tiếp theo là tạo thêm các phiên bản description ngắn hơn, tập trung hơn vào hành động chính và tương tác quan trọng.
## 1. Baseline VadCLIP
Baseline sử dụng checkpoint `model_ucf.pth`, không dùng description ở training hay inference trong pipeline đánh giá hiện tại.

Kết quả baseline khớp với paper VadCLIP trên UCF-Crime:

| Model | Classifier AUC | Alignment AUC | Ano-AUC | mAP@0.1 | mAP@0.2 | mAP@0.3 | mAP@0.4 | mAP@0.5 | AVG mAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 88.02 | 85.69 | 70.23 | 11.72 | 7.83 | 6.40 | 4.53 | 2.93 | 6.68 |

Nhận xét:

- Đây là mốc tham chiếu rất mạnh, nên các phương pháp fine-tune phải tránh làm lệch score distribution.
- Evaluator được xem là đáng tin cậy vì tái lập đúng các chỉ số paper.
## 2. V1: Positive-only cosine alignment
Ý tưởng ban đầu:

```text
visual_features -> mean pooling -> video_embedding
description -> frozen CLIP text encoder -> description_embedding
L_desc = 1 - cosine(video_embedding, description_embedding)
```

Loss tổng:

```text
L_total = L_cls + L_mil + L_text_reg + lambda_desc * L_desc
```

Kết quả diagnostic hiện có:

| Model | Classifier AUC | Alignment AUC | Ano-AUC | AVG mAP |
|---|---:|---:|---:|---:|
| Baseline | 88.02 | 85.69 | 70.23 | 6.68 |
| V1 epoch 1 | 85.79 | 81.72 | 65.44 | 3.46 |
| V1 final | 85.79 | 81.71 | 65.44 | 3.46 |

Nhận xét:

- Model tụt mạnh ngay từ epoch 1 và gần như không thay đổi sau đó.
- Nguyên nhân chính là description loss kéo toàn bộ visual embedding của chunk/video về description embedding.
- Với abnormal video, nhiều đoạn thật ra bình thường nhưng vẫn dùng cùng global video description có nội dung bất thường.
- Score của normal frames bị đẩy lên cao, làm giảm ranking normal/abnormal và gây tụt AUC, Ano-AUC, mAP.
- Hướng này không phù hợp vì nó tác động trực tiếp vào visual space chính của VadCLIP.
## 3. V2: Projection head + abnormal-aware pooling + distillation
Kiến trúc cải tiến:

```text
visual_features [B, T, 512]
-> top-k/weighted/mean pooling
-> desc_projection
-> projected_video_embedding [B, 512]
-> cosine với description_embedding
```

Trong đó `desc_projection` là MLP nhỏ:

```text
Linear(512, 512) -> LayerNorm -> GELU -> Linear(512, 512)
```

Thay đổi chính:

- Không còn ép trực tiếp visual embedding chính giống description embedding.
- Với abnormal sample, dùng top-k pooling theo anomaly score để chọn vùng nghi bất thường.
- Thêm frozen baseline teacher và distillation loss để hạn chế score drift.

Kết quả:

| Model | Classifier AUC | Alignment AUC | Ano-AUC | mAP@0.1 | mAP@0.2 | mAP@0.3 | mAP@0.4 | mAP@0.5 | AVG mAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 88.02 | 85.69 | 70.23 | 11.72 | 7.83 | 6.40 | 4.53 | 2.93 | 6.68 |
| V2 epoch 1 | 87.18 | 85.47 | 69.10 | 12.58 | 9.29 | 6.86 | 4.08 | 3.07 | 7.18 |
| V2 epoch 2 | 86.27 | 84.94 | 66.83 | 13.35 | 9.56 | 6.83 | 5.42 | 3.92 | 7.82 |
| V2 epoch 3 | 87.08 | 85.69 | 68.48 | 14.21 | 9.86 | 7.19 | 4.39 | 3.29 | 7.79 |
| V2 final | 87.03 | 85.89 | 68.29 | 14.21 | 9.77 | 6.66 | 4.83 | 1.68 | 7.43 |

Nhận xét:

- V2 sửa được lỗi nghiêm trọng của V1: normal score không còn bị đẩy lên bất thường.
- Score correlation với baseline cao hơn nhiều, cho thấy distillation/projection/top-k đã hạn chế drift.
- Tuy nhiên Classifier AUC và Ano-AUC vẫn thấp hơn baseline.
- AVG mAP tăng rõ, đặc biệt ở IoU thấp và trung bình.
- Điều này cho thấy description có giúp localization, nhưng vẫn làm thay đổi ranking frame-level ở một số class.
- Epoch 1 cân bằng nhất nếu ưu tiên AUC; epoch 2 hoặc 3 tốt hơn nếu ưu tiên AVG mAP.
## 4. Hướng contrastive: video-description InfoNCE
Hướng contrastive dùng InfoNCE giữa video embedding sau projection và description embedding:

```text
z_i = normalize(desc_projection(pool(visual_i)))
t_i = normalize(CLIP(description_i))
sim(i, j) = z_i @ t_j / temperature
L_contrastive = (CE(sim, diag) + CE(sim.T, diag)) / 2
```

Kết quả đánh giá của hướng contrastive:

| Model | Classifier AUC | Classifier AP | Alignment AUC | Alignment AP | Ano-AUC | AVG mAP |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 88.02 | 33.56 | 85.69 | 26.50 | 70.23 | 6.68 |
| Contrastive epoch 1 | 87.14 | 31.43 | 85.29 | 26.45 | 68.96 | 7.10 |
| Contrastive epoch 2 | 87.18 | 31.22 | 86.39 | 28.20 | 68.58 | 5.83 |
| Contrastive epoch 3/final | 87.18 | 31.57 | 85.97 | 27.84 | 68.34 | 7.16 |

Delta final so với baseline:

```text
Classifier AUC: -0.84
Classifier AP:  -1.99
Alignment AUC:  +0.27
Alignment AP:   +1.34
Ano-AUC:        -1.89
AVG mAP:        +0.48
```

Score drift final:

```text
classifier correlation với baseline: 0.96245
classifier mean absolute delta:       0.04868
normal mean score:                    0.28480 -> 0.24998
abnormal mean score:                  0.88005 -> 0.83384
```

Per-class final, tụt nhiều nhất:

| Class | Baseline AUC | Final AUC | Delta |
|---|---:|---:|---:|
| Explosion | 59.18 | 53.76 | -5.42 |
| Arson | 76.85 | 73.32 | -3.54 |
| Fighting | 67.90 | 66.04 | -1.86 |
| Stealing | 90.42 | 88.87 | -1.55 |
| Robbery | 73.30 | 71.85 | -1.45 |

Per-class final, tăng nhiều nhất:

| Class | Baseline AUC | Final AUC | Delta |
|---|---:|---:|---:|
| Abuse | 56.98 | 67.70 | +10.72 |
| RoadAccidents | 56.83 | 58.53 | +1.70 |
| Assault | 84.34 | 85.71 | +1.37 |
| Shooting | 65.63 | 66.84 | +1.21 |

Nhận xét:

- Contrastive tốt hơn V1 rõ rệt và có tăng AVG mAP.
- Tuy nhiên AUC, AP và Ano-AUC vẫn tụt so với baseline.
- Mặc dù có negatives trong batch, bản chất loss vẫn tác động vào video representation thông qua description branch.
- Kết quả phù hợp với lo ngại ban đầu: description embedding vẫn có thể làm lệch visual/class alignment đã học của VadCLIP.
## 5. Hướng class semantic regularization
Hướng class semantic không kéo visual embedding về description embedding. Thay vào đó, description được dùng để tạo soft class target cấp video trong class-logit space.

Ý tưởng:

```text
description -> frozen CLIP text encoder -> desc_embedding
VadCLIP class prompts -> class_embeddings
q_desc = softmax(cosine(desc_embedding, class_embeddings) / temperature)
q_final = (1 - alpha) * one_hot_label + alpha * q_desc

logits2 [B, T, 14]
-> temporal pooling
-> video_class_logits [B, 14]
L_sem = KLDivLoss(log_softmax(video_class_logits), q_final)
```

Loss tổng:

```text
L_total = L_cls + L_mil + L_text_reg + lambda_sem * L_sem
```

Thiết lập mặc định:

```text
lambda_sem = 0.05
semantic_alpha = 0.2
semantic_temperature = 0.07
video_logit_pooling = topk
top_k_ratio = 0.15
max_epoch = 3
use_amp = false
```

Kết quả đánh giá của hướng class semantic:

| Model | Classifier AUC | Classifier AP | Alignment AUC | Alignment AP | Ano-AUC | AVG mAP |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 88.02 | 33.56 | 85.69 | 26.50 | 70.23 | 6.68 |
| Class semantic epoch 1 | 87.44 | 32.72 | 85.66 | 26.56 | 69.18 | 6.55 |
| Class semantic epoch 2 | 87.17 | 32.92 | 85.87 | 27.72 | 68.72 | 7.32 |
| Class semantic epoch 3/final | 87.24 | 33.23 | 85.49 | 26.95 | 68.77 | 7.26 |

Delta final so với baseline:

```text
Classifier AUC: -0.78
Classifier AP:  -0.33
Alignment AUC:  -0.20
Alignment AP:   +0.45
Ano-AUC:        -1.46
AVG mAP:        +0.58
```

Score drift final:

```text
classifier correlation với baseline: 0.96994
classifier mean absolute delta:       0.04326
normal mean score:                    0.28480 -> 0.24957
abnormal mean score:                  0.88005 -> 0.83107
```

Per-class final, tụt nhiều nhất:

| Class | Baseline AUC | Final AUC | Delta |
|---|---:|---:|---:|
| Arson | 76.85 | 72.42 | -4.43 |
| Robbery | 73.30 | 70.36 | -2.94 |
| Fighting | 67.90 | 65.83 | -2.06 |
| Explosion | 59.18 | 57.68 | -1.49 |
| Vandalism | 70.97 | 69.57 | -1.41 |
| Shoplifting | 64.37 | 63.46 | -0.92 |

Per-class final, tăng nhiều nhất:

| Class | Baseline AUC | Final AUC | Delta |
|---|---:|---:|---:|
| Abuse | 56.98 | 63.16 | +6.17 |
| Assault | 84.34 | 87.97 | +3.62 |
| RoadAccidents | 56.83 | 58.23 | +1.40 |
| Shooting | 65.63 | 66.82 | +1.19 |

Nhận xét:

- Class semantic là hướng ổn định hơn contrastive trong lần chạy gần nhất.
- `Classifier AP` giảm rất ít so với baseline, tốt hơn contrastive rõ rệt.
- `AVG mAP` tăng từ 6.68 lên 7.26, tốt hơn contrastive final.
- `Classifier AUC` và `Ano-AUC` vẫn chưa đạt baseline, nên chưa thể kết luận là cải thiện tổng thể.
- Epoch 1 giữ AUC tốt nhất, còn epoch 2 cho AVG mAP cao nhất.
- Hướng này hợp lý hơn về mặt kiến trúc vì description chỉ làm mềm supervision trong class space, không ép visual embedding phải giống description embedding.
## 6. Nhận xét tổng hợp
Các hướng embedding alignment trực tiếp hoặc gián tiếp đều có một vấn đề chung: description được dùng như một target embedding. Điều này dễ xung đột với mục tiêu inference của VadCLIP, vốn dựa trên video-class anomaly detection.

Kết luận hiện tại:

- V1 không phù hợp vì gây drift mạnh và tụt toàn bộ metric.
- V2 là cải thiện đáng kể: mAP tăng, drift giảm, nhưng AUC vẫn tụt nhẹ.
- Contrastive có tính học thuật hơn V1 nhưng vẫn cần kiểm chứng vì có thể tiếp tục tác động vào visual representation.
- Hướng tiếp theo đáng cân nhắc hơn là dùng description để tạo supervision trong class-logit space, ví dụ video-level soft class semantic regularization, thay vì kéo visual embedding về description embedding.
## 7. Bài học rút ra
- Không nên dùng global video description để align toàn bộ frame/chunk của abnormal video.
- Cần tránh làm normal score tăng trên normal videos.
- Projection head giúp giảm drift nhưng không loại bỏ hoàn toàn xung đột mục tiêu.
- Distillation hữu ích như cơ chế kiểm soát drift, nhưng nếu quá mạnh sẽ làm model quay về baseline.
- Đánh giá cần xem đồng thời AUC, Ano-AUC, mAP, score distribution, per-class metrics và timeline, không chỉ nhìn loss.
## 8. Kết luận cập nhật sau các lần chạy mới
So sánh các hướng chính:

| Hướng | Tác động chính | Ưu điểm | Vấn đề |
|---|---|---|---|
| V1 cosine positive-only | Kéo video embedding gần description embedding | Ý tưởng đơn giản, trực tiếp | Drift rất mạnh, tụt toàn bộ metric |
| V2 projection + top-k + distill | Align qua projection, chọn vùng nghi bất thường | Tăng mAP, giảm drift so với V1 | AUC/Ano-AUC vẫn tụt, distill dễ làm hướng mới quay về baseline |
| Contrastive InfoNCE | Phân biệt video-description đúng/sai trong batch | Tăng mAP, alignment AP tăng | Vẫn tác động vào video representation, AUC/AP/Ano-AUC tụt |
| Class semantic | Dùng description tạo soft target trong class-logit space | Ổn định nhất hiện tại, mAP tăng, AP giữ tốt | AUC/Ano-AUC vẫn thấp hơn baseline |

Kết luận hiện tại:

- Nếu mục tiêu là bảo toàn anomaly ranking, baseline vẫn mạnh nhất.
- Nếu mục tiêu là khai thác description để cải thiện localization/mAP, class semantic và V2 có tín hiệu tích cực.
- Trong các hướng đã thử, class semantic là hướng đáng phát triển tiếp nhất vì ít xâm lấn visual embedding hơn.
- Vẫn cần giải quyết hiện tượng giảm abnormal mean score, vì cả contrastive và class semantic đều làm abnormal score trung bình thấp hơn baseline.
- Các thí nghiệm tiếp theo nên tập trung vào cách dùng description như regularization nhẹ trong class/logit space, thay vì alignment trực tiếp với visual embedding.
