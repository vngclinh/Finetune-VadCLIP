# Kết Quả Thử Nghiệm Fine-tune VadCLIP Với Class Prototypes

## 1. Quy trình tạo class prototype

Mục tiêu của bước này là tạo mô tả ngữ nghĩa cấp class cho 14 lớp của UCF-Crime, sau đó dùng các prototype này như tín hiệu phụ trong quá trình fine-tune VadCLIP.

Nguồn dữ liệu đầu vào là các video description đã được tạo từ UCA timestamp annotations. Để tránh lệch protocol, prototype chỉ được tạo từ các video thuộc train list của VadCLIP, tương ứng với file `ucf_CLIP_rgb_description.csv`. Các video không có description trong UCA không được dùng để tạo prototype.

Thống kê số video description dùng để tạo prototype:

| Class | Số description |
|---|---:|
| Abuse | 48 |
| Arrest | 44 |
| Arson | 41 |
| Assault | 45 |
| Burglary | 87 |
| Explosion | 29 |
| Fighting | 45 |
| Normal | 763 |
| RoadAccidents | 125 |
| Robbery | 143 |
| Shooting | 27 |
| Shoplifting | 29 |
| Stealing | 95 |
| Vandalism | 44 |

Quy trình tạo prototype gồm hai giai đoạn:

1. Encode các video description bằng `text-embedding-3-small`, sau đó gom cụm bằng KMeans theo từng class để phân cụm các video tương đồng giống nhau.
2. Dùng `gpt-5.6-luna` để tổng hợp prototype từ các cụm description.

Mỗi cụm sinh ra 3 candidate prototypes. Sau đó các candidate prototypes của cùng class được tổng hợp thành 5 final prototypes:

- `core_action`
- `actor_object_interaction`
- `temporal_progression`
- `scene_context`
- `hard_negative_distinction`

Ví dụ minh họa với class Robbery, một bộ final prototype có thể mang dạng như sau:

| Loại prototype | Ví dụ nội dung |
|---|---|
| `core_action` | One or more people approach others, create pressure, and take money or personal items through visible threat or force. |
| `actor_object_interaction` | A person confronts another person at close range, reaches toward bags, pockets, or carried items, and leaves with the taken property. |
| `temporal_progression` | People first move normally in the area, then a tense confrontation happens, property is taken, and the involved people quickly separate or leave. |
| `scene_context` | The interaction often happens in open public areas, sidewalks, storefronts, or roadside spaces where people are walking, waiting, or briefly stopping. |
| `hard_negative_distinction` | The scene differs from ordinary passing or conversation because one side visibly yields belongings under pressure instead of exchanging items voluntarily. |

### Prompt tạo candidate prototypes

System prompt:

```text
You create class-level visual-action prototypes for video anomaly detection.
You must summarize recurring patterns across multiple video descriptions.
Do not copy video-specific details unless they represent a repeated pattern.
Do not mention dataset labels, anomaly category names, or the word anomaly.
Use neutral, visual, action-focused language.
Return only valid JSON.
```

User prompt:

```text
Create candidate class-level prototypes from the following video descriptions.

Class name for internal reference only: {class_name}
Do not mention this class name in the output.

Requirements:
- Produce exactly {CANDIDATES_PER_CLUSTER} candidate prototypes.
- Each prototype must be 15 to 35 words.
- Focus on recurring actors, visible actions, interactions, objects, and temporal patterns.
- Ignore clothing colors, camera viewpoint, exact scene layout, and one-off background details.
- Do not use these forbidden words: {FORBIDDEN_LABELS}.
- Do not invent patterns that are not supported by the descriptions.
- Return exactly this JSON shape:
{
  "candidate_prototypes": [
    {"text": "...", "rationale": "..."}
  ]
}

Video descriptions:
{numbered_descriptions}
```

### Prompt tạo final prototypes

System prompt:

```text
You synthesize final class-level visual-action prototypes for video anomaly detection.
You must merge overlapping candidate patterns and preserve only general, recurring visual semantics.
Do not mention dataset labels, anomaly category names, or the word anomaly.
Return only valid JSON.
```

User prompt:

```text
Synthesize final class-level prototypes from candidate prototypes.

Class name for internal reference only: {class_name}
Do not mention this class name in the output.

Requirements:
- Produce exactly {FINAL_PROTOTYPES_PER_CLASS} final prototypes.
- Use these prototype types exactly:
  core_action
  actor_object_interaction
  temporal_progression
  scene_context
  hard_negative_distinction
- Each prototype must be 15 to 35 words.
- Keep prototypes general enough to represent the class, not a single video.
- Prefer visual actions and interactions over scene-specific details.
- Do not use these forbidden words: {FORBIDDEN_LABELS}.
- For Normal, describe routine behavior, ordinary interactions, and absence of visible disruption without using anomaly-related words.
- Return exactly this JSON shape:
{
  "final_prototypes": [
    {"type": "core_action", "text": "..."},
    {"type": "actor_object_interaction", "text": "..."},
    {"type": "temporal_progression", "text": "..."},
    {"type": "scene_context", "text": "..."},
    {"type": "hard_negative_distinction", "text": "..."}
  ]
}

Candidate prototypes:
{candidate_prototypes}
```

## 2. Compact prototype

Sau lần thử nghiệm đầu tiên, các prototype dài 15-35 từ cho thấy có thể làm loãng embedding ngữ nghĩa của class. Vì vậy, một bộ compact prototype được tạo thủ công từ 5 prototype dài, không gọi thêm LLM mà sử dụng trực tiếp chatGPT (model GPT 5.5).

Mỗi class có 3 prototype ngắn:

- `keyword_phrase`: 3-7 từ.
- `compact_action_phrase`: 8-12 từ.
- `short_visual_sentence`: 12-18 từ.

Ví dụ với class Robbery:

| Loại prototype | Nội dung |
|---|---|
| keyword_phrase | coerced taking of valuables |
| compact_action_phrase | threatening people to surrender cash or personal belongings |
| short_visual_sentence | A person uses threat or force to make others hand over property. |

Khi dùng cấu hình `compact_all`, cả 3 prototype của mỗi class được encode bằng CLIP text encoder, normalize, lấy trung bình, rồi normalize lại để tạo một centroid đại diện cho class.

## 3. Nguyên tắc chung khi tích hợp prototype

Ba thử nghiệm bên dưới đều can thiệp vào các representation đã có sẵn bên trong VadCLIP, thay vì thiết kế lại toàn bộ model. Điểm khác nhau nằm ở cách prototype được đưa vào quá trình fine-tune:

- Thử nghiệm 1 và 2 thêm prototype như một nhánh semantic supervision phụ trong lúc train.
- Thử nghiệm 3 không thêm nhánh phụ, mà thay trực tiếp class prompt trong nhánh `logits2`.
- Ở cả ba thử nghiệm, phần cần đọc kỹ là tín hiệu prototype đi vào đâu, tạo ra đầu ra phụ nào, và làm thay đổi loss fine-tune như thế nào.

## 4. Thử nghiệm 1: 5 prototype dài do LLM tạo ra

### Kiến trúc

Thử nghiệm đầu tiên không thay cách VadCLIP tạo `logits1` và `logits2`. Phần được thêm vào chỉ là một nhánh prototype phụ nhận `visual_features` từ model và so khớp chúng với prototype centroid của từng class.

```text
Đầu vào semantic mới:
5 prototype dài cho mỗi class
        |
        v
Frozen CLIP text encoder
        |
        v
5 prototype embeddings / class
        |
        v
normalize -> mean -> normalize
        |
        v
prototype_centroids [14, 512]


Điểm nối từ VadCLIP:
visual_features [B, T, 512]
        |
        v
normalize(visual_features)
        |
        v
matrix similarity với prototype_centroids
        |
        v
prototype_logits [B, T, 14]
        |
        v
L_mil_proto


Ràng buộc thêm trên text space:
learnable class text embeddings
        |
        v
cosine alignment với prototype_centroids
        |
        v
L_proto_text
```

Đầu vào mới của thử nghiệm này là 5 prototype dài cho mỗi class, tức 5 câu mô tả tổng quát đã được tổng hợp từ nhiều mô tả video huấn luyện. Các prototype này được encode bằng frozen CLIP text encoder, sau đó lấy trung bình trong từng class để tạo `prototype_centroids`. Từ phía video, ta chỉ lấy `visual_features` đã có sẵn bên trong VadCLIP làm điểm nối, rồi tính độ tương đồng giữa từng temporal snippet và từng prototype centroid để sinh `prototype_logits`. Đầu ra mới duy nhất của phần thêm vào là `prototype_logits`, và nó chỉ dùng để tạo thêm `L_mil_proto` trong huấn luyện.

Ngoài nhánh `prototype_logits`, thử nghiệm này còn thêm `L_proto_text` để kéo nhẹ learnable class text embeddings về gần prototype centroid tương ứng. Như vậy phần can thiệp gồm hai tín hiệu: một tín hiệu tác động lên visual representation qua `L_mil_proto`, và một tín hiệu tác động lên text representation qua `L_proto_text`.

### Hàm loss

Hàm loss của thử nghiệm 1 giữ nguyên các loss chính đang dùng để fine-tune VadCLIP, rồi cộng thêm hai thành phần prototype:

```text
L_total =
    L_cls
  + L_mil_class
  + lambda_proto_mil * L_mil_proto
  + L_text_reg
  + lambda_proto_text * L_proto_text
```

Ý nghĩa các thành phần liên quan trực tiếp đến phần thêm vào:

- `L_mil_proto`: MIL loss từ `prototype_logits`, dùng prototype centroid như nhãn ngữ nghĩa phụ cho từng class.
- `L_proto_text`: cosine loss nhẹ giữa learnable class text embedding và prototype centroid.
- `lambda_proto_mil` và `lambda_proto_text`: hai hệ số điều khiển mức độ prototype can thiệp vào quá trình fine-tune.

Prototype logits được tính theo:

```text
prototype_logits =
normalize(visual_features) @ normalize(prototype_centroids).T / temperature
```

Với thử nghiệm này:

```text
temperature = 0.07
```

### Cấu hình

Đầu vào semantic của thử nghiệm này là một bộ prototype dài, trong đó mỗi class được biểu diễn bởi 5 câu mô tả tổng quát:

- `core_action`
- `actor_object_interaction`
- `temporal_progression`
- `scene_context`
- `hard_negative_distinction`

Prototype centroid của mỗi class được tạo bằng cách encode 5 prototype bằng frozen CLIP text encoder, normalize từng embedding, lấy mean, rồi normalize lại.

Tham số training:

```text
pretrained_model_path = model_ucf.pth
use_pretrained_model = true
lambda_proto_mil = 0.005
lambda_proto_text = 0.005
prototype_temperature = 0.07
lr = 1e-5
max_epoch = 3
use_amp = false
```

### Kết quả

Baseline VadCLIP:

| Metric | Giá trị |
|---|---:|
| classifier_auc | 88.02 |
| classifier_ap | 33.56 |
| alignment_auc | 85.69 |
| alignment_ap | 26.50 |
| ano_auc | 70.23 |
| avg_mAP | 6.68 |

Kết quả final với 5 prototype dài:

| Metric | Giá trị | Chênh lệch so với baseline |
|---|---:|---:|
| classifier_auc | 86.40 | -1.62 |
| classifier_ap | 30.95 | -2.61 |
| alignment_auc | 84.42 | -1.28 |
| alignment_ap | 23.49 | -3.01 |
| ano_auc | 66.75 | -3.48 |
| avg_mAP | 4.74 | -1.94 |

Một số class tụt mạnh:

| Class | Chênh lệch classifier AUC |
|---|---:|
| Explosion | -19.50 |
| Robbery | -7.42 |
| Arson | -2.95 |
| Arrest | -2.01 |
| RoadAccidents | -1.60 |
| Burglary | -1.25 |

Một số class tăng:

| Class | Chênh lệch classifier AUC |
|---|---:|
| Shoplifting | +0.94 |
| Fighting | +1.88 |
| Shooting | +1.94 |
| Vandalism | +5.96 |
| Abuse | +8.24 |
| Assault | +8.92 |

### Nhận xét

Kết quả cho thấy prototype dài không phù hợp với cách VadCLIP đã học class embedding từ checkpoint UCF-Crime. Các prototype dài chứa nhiều chi tiết về actor, object, scene context và temporal progression, làm embedding semantic bị loãng. Khi thêm prototype loss, visual representation bị kéo khỏi không gian đã được checkpoint tối ưu, khiến AUC, Ano-AUC và mAP đều giảm.

## 5. Thử nghiệm 2: 3 loại compact prototype

### Kiến trúc

Thử nghiệm thứ hai giữ nguyên vị trí can thiệp của thử nghiệm 1, nhưng thay đầu vào semantic từ 5 prototype dài sang các prototype compact ngắn hơn. Vì vậy phần sửa nằm ở cách tạo `prototype_centroids`, còn điểm nối với `visual_features` và cách sinh `prototype_logits` vẫn giống thử nghiệm 1.

```text
Đầu vào semantic mới:
3 compact prototypes cho mỗi class
  - keyword_phrase
  - compact_action_phrase
  - short_visual_sentence
        |
        v
Frozen CLIP text encoder
        |
        v
3 compact embeddings / class
        |
        |------------------> dùng riêng từng loại prototype
        |
        |------------------> compact_all:
                             normalize -> mean 3 embeddings -> normalize
                                   |
                                   v
                             compact prototype_centroids [14, 512]


Điểm nối từ VadCLIP:
visual_features [B, T, 512]
        |
        v
normalize(visual_features)
        |
        v
similarity với compact prototype_centroids
        |
        v
prototype_logits [B, T, 14]
        |
        v
L_mil_proto


Ràng buộc text bổ sung:
learnable class text embeddings
        |
        v
L_proto_text với compact prototype_centroids
```

Đầu vào mới của thử nghiệm này là 3 prototype compact cho mỗi class, được viết ngắn hơn và tập trung hơn vào hành động cốt lõi. Với các cấu hình `keyword_phrase`, `compact_action_phrase` và `short_visual_sentence`, mỗi lần chạy chỉ dùng một loại prototype để tạo centroid. Với `compact_all`, cả ba embedding compact của cùng class được normalize, lấy trung bình, rồi normalize lại để tạo một centroid chung. Sau bước này, centroid compact được dùng giống thử nghiệm 1: so khớp với `visual_features` để tạo `prototype_logits`, rồi sinh thêm `L_mil_proto`.

So với prototype dài, thay đổi chính là chất lượng và độ cô đặc của tín hiệu semantic đưa vào nhánh phụ. Cơ chế fine-tune vẫn thêm hai lực kéo: `L_mil_proto` kéo visual representation về phía compact centroid, còn `L_proto_text` kéo nhẹ class text embedding về cùng không gian semantic đó.

### Hàm loss

Thử nghiệm 2 vẫn dùng cùng dạng loss prototype auxiliary với thử nghiệm 1:

```text
L_total =
    L_cls
  + L_mil_class
  + lambda_proto_mil * L_mil_proto
  + L_text_reg
  + lambda_proto_text * L_proto_text
```

Khác biệt chủ yếu nằm ở trọng số và chất lượng semantic của prototype:

- `L_mil_proto` vẫn ép visual representation khớp hơn với compact prototype centroids.
- `L_proto_text` vẫn giữ vai trò neo learnable class text embedding vào prototype centroid.
- Vì prototype ngắn hơn, tín hiệu semantic phụ đậm đặc hơn và ít kéo lệch representation hơn prototype dài.

### Cấu hình

Đầu vào semantic của thử nghiệm này là bộ compact prototype, trong đó mỗi class được mô tả bằng ba mức diễn đạt ngắn gọn khác nhau:

- `keyword_phrase`
- `compact_action_phrase`
- `short_visual_sentence`
- `compact_all`: lấy trung bình embedding của cả 3 loại.

Tham số training chính:

```text
pretrained_model_path = model_ucf.pth
use_pretrained_model = true
prototype_schema = compact
lambda_proto_mil = 0.3
lambda_proto_text = 0.01
prototype_temperature = 0.07
lr = 1e-5
max_epoch = 1
use_amp = false
```

Lý do chọn cấu hình này:

- Dùng checkpoint `model_ucf.pth` vì baseline đã mạnh và train từ đầu làm model dễ drift.
- Dùng compact prototype để giảm nhiễu semantic so với prototype dài.
- Chỉ train 1 epoch vì các lần trước cho thấy train lâu hơn làm AUC và Ano-AUC giảm dần.
- `lambda_proto_mil = 0.3` để prototype có ảnh hưởng đủ rõ nhưng không áp đảo loss gốc.
- `lambda_proto_text = 0.01` để hạn chế kéo class embedding gốc quá mạnh về prototype.

### Kết quả

| Cấu hình | classifier_auc | classifier_ap | alignment_auc | alignment_ap | ano_auc | avg_mAP |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 88.02 | 33.56 | 85.69 | 26.50 | 70.23 | 6.68 |
| keyword_phrase | 87.26 | 31.24 | 86.12 | 27.43 | 68.96 | 7.49 |
| compact_action_phrase | 87.25 | 31.58 | 85.55 | 26.42 | 69.04 | 7.26 |
| short_visual_sentence | 87.32 | 33.98 | 85.75 | 26.22 | 69.27 | 6.44 |
| compact_all | 87.32 | 33.49 | 85.83 | 26.49 | 69.20 | 7.24 |

Chênh lệch so với baseline:

| Cấu hình | classifier_auc | classifier_ap | alignment_auc | alignment_ap | ano_auc | avg_mAP |
|---|---:|---:|---:|---:|---:|---:|
| keyword_phrase | -0.76 | -2.32 | +0.43 | +0.93 | -1.27 | +0.80 |
| compact_action_phrase | -0.78 | -1.98 | -0.14 | -0.08 | -1.19 | +0.58 |
| short_visual_sentence | -0.70 | +0.43 | +0.06 | -0.28 | -0.96 | -0.24 |
| compact_all | -0.70 | -0.07 | +0.13 | -0.02 | -1.03 | +0.56 |

Per-class drift của `compact_all`:

| Class | Chênh lệch classifier AUC |
|---|---:|
| Explosion | -2.40 |
| Vandalism | -1.75 |
| Stealing | -0.91 |
| Arson | -0.81 |
| Arrest | -0.56 |
| RoadAccidents | -0.21 |
| Fighting | -0.08 |
| Shoplifting | +0.47 |
| Robbery | +0.64 |
| Burglary | +1.49 |
| Assault | +3.70 |
| Abuse | +3.83 |

### Nhận xét

Compact prototype cải thiện đáng kể so với prototype dài. Với `compact_all`, AUC chỉ giảm 0.70 điểm, AP gần như giữ nguyên, alignment AUC tăng nhẹ và avg_mAP tăng 0.56 điểm. Điều này cho thấy prototype ngắn ít làm lệch không gian feature hơn prototype dài.

Tuy nhiên, tất cả các cấu hình compact vẫn thấp hơn baseline về classifier AUC và Ano-AUC. Nguyên nhân chính là model baseline đã được train trên UCF-Crime, trong đó visual features và learnable class embeddings đã được tối ưu cùng nhau. Khi thêm nhánh prototype phụ, gradient từ `L_mil_proto` vẫn kéo visual representation về phía fixed CLIP prototype embeddings. Việc này có thể cải thiện localization ở một số class, nhưng cũng làm nhiễu alignment đã học của checkpoint.

## 6. Thử nghiệm 3: Prototype prompt replacement

### Mục tiêu

Sau hai thử nghiệm đầu, một vấn đề được đặt ra là: nhánh prototype phụ có thể làm giảm hiệu năng vì nó tạo thêm gradient kéo `visual_features` khỏi không gian đã được checkpoint VadCLIP tối ưu. Vì vậy, thử nghiệm thứ ba không thêm nhánh prototype riêng và không thêm prototype loss nữa. Thay vào đó, prototype được dùng để thay thế trực tiếp class prompt trong nhánh `logits2`.

Ý tưởng:

```text
VadCLIP gốc:
visual_features so với class prompt gốc -> logits2

Prompt replacement:
visual_features so với prototype/class+prototype prompt -> logits2
```

Như vậy, prototype không còn là một nhánh phụ nằm dưới visual branch. Nó trở thành text prompt được dùng trực tiếp trong class-text alignment branch của VadCLIP.

### Hai cấu hình thử nghiệm

Thử nghiệm này sử dụng lại bộ compact prototype đã được tạo sẵn ở phần trước. Với mỗi class, mô hình dùng đồng thời 3 prototype ngắn:

- `keyword_phrase`
- `compact_action_phrase`
- `short_visual_sentence`

Sau khi encode, embedding của 3 prompt được normalize, lấy trung bình, rồi normalize lại để tạo một class text embedding.

Trong VadCLIP gốc, class text được đưa vào giữa các token learnable prompt:

```text
[learnable prompt prefix] + class_name + [learnable prompt postfix]
```

Ví dụ:

```text
[learnable prompt prefix] + robbery + [learnable prompt postfix]
```

Trong thử nghiệm prompt replacement, cơ chế learnable prompt này vẫn được giữ nguyên. Điểm thay đổi chỉ là chuỗi text nằm ở giữa learnable prompt.

Hai cấu hình được thử:

```text
prototype_only:
[learnable prompt prefix] + prototype_text + [learnable prompt postfix]

class_plus_prototype:
[learnable prompt prefix] + class_name involving prototype_text + [learnable prompt postfix]
```

Vì vậy:

- `prototype_only` không có class name, nhưng vẫn có learnable prompt.
- `class_plus_prototype` có cả class name, có cả prototype text, và vẫn có learnable prompt.
- Cả hai cấu hình đều không phải CLIP text encoder thuần bên ngoài VadCLIP.

Nói cách khác, `prototype_only` là thay class name bằng prototype text trong learnable prompt. Còn `class_plus_prototype` là thay class name bằng một phrase dài hơn gồm class name và prototype text.

Trong đó, `class_plus_prototype` chính là cấu hình **prototype + learnable prompt**: prototype không được encode bằng CLIP text encoder thô ở ngoài model, mà được đưa vào vị trí text token bên trong learnable prompt của VadCLIP.

Ví dụ với class Robbery:

```text
prototype_only:
coerced taking of valuables
threatening people to surrender cash or personal belongings
A person uses threat or force to make others hand over property.

class_plus_prototype:
robbery involving coerced taking of valuables
robbery involving threatening people to surrender cash or personal belongings
robbery involving a person uses threat or force to make others hand over property.
```

Khi đi qua model, các ví dụ trên thực tế được đặt trong learnable prompt:

```text
prototype_only:
[learnable prompt prefix] + coerced taking of valuables + [learnable prompt postfix]

class_plus_prototype:
[learnable prompt prefix] + robbery involving coerced taking of valuables + [learnable prompt postfix]
```

### Kiến trúc

Thử nghiệm thứ ba không thêm `prototype_logits` và không thêm prototype loss. Phần được sửa nằm ở cách tạo text prompt cho nhánh alignment: thay vì dùng class name gốc ở giữa learnable prompt, ta đưa prototype text hoặc class name cộng prototype text vào vị trí đó.

```text
Đầu vào semantic mới:
3 compact prototypes cho mỗi class
        |
        v
tạo prompt replacement
        |
        |------------------> prototype_only
        |                    prototype_text
        |
        |------------------> class_plus_prototype
                             class_name involving prototype_text
        |
        v
đưa vào learnable prompt wrapper của VadCLIP
[learnable prefix] + replacement text + [learnable postfix]
        |
        v
text embeddings mới cho 14 class
        |
        v
thay class text embeddings gốc trong logits2


Điểm nối từ VadCLIP:
visual_features [B, T, 512]
        |
        v
so khớp với text embeddings mới
        |
        v
logits2 mới [B, T, 14]
        |
        v
L_mil_class
```

Đầu vào mới của thử nghiệm này là các compact prototype đã được sinh trước đó, nhưng chúng không được encode thành centroid để tạo nhánh phụ. Thay vào đó, mỗi prototype được biến thành một chuỗi text mới nằm giữa learnable prefix và learnable postfix của VadCLIP. Với `prototype_only`, chuỗi thay thế chỉ chứa prototype text. Với `class_plus_prototype`, chuỗi thay thế giữ lại class name rồi nối thêm prototype text để class name tiếp tục đóng vai trò semantic anchor.

Đầu ra bị thay đổi trực tiếp là `logits2`: nó không còn được tính từ class prompt gốc, mà từ text embeddings mới tạo bằng prototype prompt. Vì không có nhánh phụ, thử nghiệm này không tạo thêm `prototype_logits`; toàn bộ tác động của prototype đi thẳng vào class-text alignment branch thông qua prompt replacement.

Khác với thử nghiệm compact auxiliary, thử nghiệm này không có:

```text
prototype_logits
L_mil_proto
L_proto_text
```

### Hàm loss

Thử nghiệm này không thêm loss prototype. Loss fine-tune chỉ giữ các thành phần đang dùng cho VadCLIP, nhưng `L_mil_class` lúc này nhận `logits2` đã được tính từ prototype prompt:

```text
L_total = L_cls + L_mil_class + L_text_reg
```

Trong đó:

- `L_cls`: binary classification loss từ `logits1`.
- `L_mil_class`: MIL loss từ `logits2`, nhưng `logits2` lúc này được tính bằng prototype prompt replacement.
- `L_text_reg`: text regularization gốc của VadCLIP.

### Tham số training

```text
pretrained_model_path = model_ucf.pth
use_pretrained_model = true
compact_prototype_types = keyword_phrase + compact_action_phrase + short_visual_sentence
lr = 5e-6
max_epoch = 1
batch_size = 64
use_amp = false
eval_steps = 0
```

Lý do chọn cấu hình này:

- Dùng checkpoint `model_ucf.pth` để kiểm tra tác động của việc thay prompt trên một baseline đã mạnh.
- Dùng learning rate nhỏ `5e-6` vì thay prompt trực tiếp có thể làm nhánh alignment nhạy hơn fine-tune thông thường.
- Chỉ train 1 epoch để giảm rủi ro drift.
- Không dùng prototype loss vì mục tiêu là kiểm tra prototype như class prompt replacement, không phải auxiliary supervision.

### Kết quả

| Cấu hình | classifier_auc | classifier_ap | alignment_auc | alignment_ap | ano_auc | avg_mAP |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 88.02 | 33.56 | 85.69 | 26.50 | 70.23 | 6.68 |
| prototype_only | 86.71 | 30.32 | 83.20 | 22.03 | 67.23 | 3.68 |
| class_plus_prototype | 86.85 | 33.04 | 83.94 | 23.55 | 68.04 | 3.84 |

Chênh lệch so với baseline:

| Cấu hình | classifier_auc | classifier_ap | alignment_auc | alignment_ap | ano_auc | avg_mAP |
|---|---:|---:|---:|---:|---:|---:|
| prototype_only | -1.31 | -3.24 | -2.49 | -4.47 | -3.01 | -3.00 |
| class_plus_prototype | -1.17 | -0.52 | -1.75 | -2.95 | -2.19 | -2.85 |

Per-class drift của `class_plus_prototype`:

| Class | Chênh lệch classifier AUC |
|---|---:|
| Explosion | -7.21 |
| Arrest | -3.45 |
| Arson | -3.22 |
| Assault | -1.90 |
| Stealing | -1.84 |
| Shooting | -0.59 |
| Robbery | +0.01 |
| Fighting | +0.13 |
| Burglary | +1.34 |
| RoadAccidents | +1.38 |
| Shoplifting | +2.57 |
| Abuse | +4.73 |
| Vandalism | +4.92 |

### Nhận xét

Kết quả cho thấy `class_plus_prototype` tốt hơn `prototype_only`, nghĩa là class name vẫn có tác dụng giữ semantic anchor cho checkpoint. Cần nhấn mạnh rằng `prototype_only` không phải là bỏ learnable prompt; nó chỉ bỏ class name. `class_plus_prototype` cũng không phải thay class name hoàn toàn bằng prototype; nó giữ class name và bổ sung prototype trong cùng chuỗi text. Tuy nhiên, cả hai cấu hình prompt replacement đều thấp hơn baseline và thấp hơn compact auxiliary branch.

So với compact auxiliary `compact_all`:

| Cấu hình | classifier_auc | classifier_ap | alignment_auc | alignment_ap | ano_auc | avg_mAP |
|---|---:|---:|---:|---:|---:|---:|
| compact_all auxiliary | 87.32 | 33.49 | 85.83 | 26.49 | 69.20 | 7.24 |
| class_plus_prototype replacement | 86.85 | 33.04 | 83.94 | 23.55 | 68.04 | 3.84 |

Prompt replacement tụt mạnh hơn auxiliary vì nó thay trực tiếp class representation đã được checkpoint VadCLIP tối ưu. Trong checkpoint, class prompt gốc không chỉ là tên class ngôn ngữ tự nhiên, mà đã trở thành một representation được học cùng visual branch, `mlp1`, `logits2` và MIL loss. Khi thay prompt gốc bằng prototype prompt, toàn bộ phân phối softmax của `logits2` thay đổi, làm alignment branch và mAP giảm mạnh.

Nói cách khác:

```text
Auxiliary prototype:
giữ class prompt gốc, thêm tín hiệu phụ.

Prompt replacement:
thay trực tiếp hệ class-alignment đã học bằng một hệ text prompt mới.
```

Vì vậy, dù prompt replacement không kéo visual bằng một nhánh prototype phụ, nó vẫn phá nhánh alignment mạnh hơn do thay đổi trực tiếp text embedding dùng trong `logits2`.

## 7. Kết luận

Các thí nghiệm prototype cho thấy:

- Prototype dài do LLM tạo ra chứa quá nhiều chi tiết và làm giảm mạnh hiệu năng.
- Compact prototype giảm semantic dilution và ít gây drift hơn.
- `compact_all` là cấu hình cân bằng nhất hiện tại.
- Prototype hiện tại chưa nên được xem là thay thế hoàn toàn cho class embedding gốc của VadCLIP.
- Việc thêm prototype như một nhánh phụ vẫn có rủi ro kéo visual representation khỏi không gian đã được checkpoint tối ưu.
- Prompt replacement trực tiếp còn kém hơn auxiliary, kể cả khi dùng `class_plus_prototype` trong learnable prompt, vì nó thay đổi class-text alignment space mà checkpoint đã học.

Kết luận thực nghiệm:

```text
Compact prototype ổn định hơn prototype dài do LLM tạo ra và có thể cải thiện mAP trong một số cấu hình. Tuy nhiên, việc thay trực tiếp class prompt của VadCLIP bằng prototype prompt làm suy giảm không gian alignment mà checkpoint đã học. Cho đến hiện tại, chưa có biến thể prototype nào vượt được checkpoint VadCLIP đã huấn luyện trên UCF-Crime về frame-level AUC và Ano-AUC.
```

## 8. Bổ sung: Prompt replacement huấn luyện from scratch

Sau các thí nghiệm prompt replacement khởi tạo từ checkpoint `model_ucf.pth`, hai cấu hình tương tự được chạy lại theo hướng **from scratch**, tức là không load checkpoint VadCLIP đã huấn luyện trên UCF-Crime. Mục tiêu của lần chạy này là kiểm tra xem việc thay class prompt bằng prototype prompt có thể học tốt hơn nếu model không bị ràng buộc bởi class-text alignment cũ hay không.

Hai cấu hình được thử nghiệm:

- `prototype_only`: dùng compact prototype làm text prompt thay thế, không giữ class name.
- `class_plus_prototype`: dùng class name kết hợp compact prototype trong cùng prompt, theo dạng `{class_name} involving {prototype}`.

Trong cả hai cấu hình, learnable prefix/postfix của VadCLIP vẫn được giữ. Điểm khác biệt nằm ở phần text nằm giữa learnable prompt:

```text
prototype_only:
learnable prefix + prototype text + learnable postfix

class_plus_prototype:
learnable prefix + class name involving prototype text + learnable postfix
```

### Cấu hình huấn luyện

```text
use_pretrained_model = false
compact_prototype_types = keyword_phrase + compact_action_phrase + short_visual_sentence
lr = 2e-5
max_epoch = 10
scheduler_milestones = 4, 8
batch_size = 64
use_amp = false
eval_steps = 0
```

Loss vẫn giữ nguyên theo prompt replacement, không thêm prototype auxiliary loss:

```text
L_total = L_cls + L_mil_class + L_text_reg
```

Trong đó `L_mil_class` dùng `logits2` được tính từ prototype prompt replacement.

### Kết quả prototype_only from scratch

| Checkpoint tốt nhất | classifier_auc | classifier_ap | alignment_auc | alignment_ap | ano_auc | avg_mAP |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 88.02 | 33.56 | 85.69 | 26.50 | 70.23 | 6.68 |
| Best classifier AUC, epoch 10 | 86.73 | 33.15 | 85.75 | 28.99 | 67.61 | 6.22 |
| Best Ano-AUC, epoch 8 | 86.73 | 33.33 | 85.74 | 29.01 | 67.64 | 6.38 |
| Best avg_mAP, epoch 3 | 86.19 | 32.73 | 85.18 | 27.47 | 67.18 | 6.96 |

Nhận xét:

- `prototype_only` from scratch không vượt baseline về AUC và Ano-AUC.
- `avg_mAP` có tăng nhẹ ở epoch 3, từ `6.68` lên `6.96`, nhưng mức tăng nhỏ.
- AUC ổn định hơn `class_plus_prototype`, nhưng thiếu class anchor nên semantic class chưa đủ mạnh để thay thế class prompt gốc.
- Một số class bị tụt nặng, đặc biệt `Assault` và `Explosion`.

Per-class drift đáng chú ý ở checkpoint final:

| Class | Chênh lệch classifier AUC |
|---|---:|
| Assault | -21.46 |
| Explosion | -14.19 |
| Arrest | -5.62 |
| Arson | -5.47 |
| Burglary | -4.72 |
| Vandalism | +10.47 |
| Shooting | +4.66 |
| Shoplifting | +1.77 |

### Kết quả class_plus_prototype from scratch

| Checkpoint tốt nhất | classifier_auc | classifier_ap | alignment_auc | alignment_ap | ano_auc | avg_mAP |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 88.02 | 33.56 | 85.69 | 26.50 | 70.23 | 6.68 |
| Best classifier AUC, epoch 3 | 85.93 | 28.99 | 84.64 | 24.56 | 66.24 | 6.40 |
| Best Ano-AUC, epoch 3 | 85.93 | 28.99 | 84.64 | 24.56 | 66.24 | 6.40 |
| Best avg_mAP, epoch 9 | 85.84 | 29.18 | 85.33 | 25.20 | 65.45 | 8.25 |

Nhận xét:

- `class_plus_prototype` from scratch tăng `avg_mAP` rõ hơn, đạt tốt nhất `8.25`, cao hơn baseline `+1.57`.
- Tuy nhiên `classifier_auc` và `ano_auc` tụt mạnh hơn `prototype_only`.
- Điều này cho thấy cấu hình này có thể giúp localization tốt hơn, nhưng làm suy giảm khả năng phân biệt normal/abnormal tổng thể.
- Class name giúp giữ semantic anchor tốt hơn prototype-only ở một số class, nhưng vẫn chưa đủ để ổn định toàn bộ alignment space.

Per-class drift đáng chú ý ở checkpoint final:

| Class | Chênh lệch classifier AUC |
|---|---:|
| Assault | -41.79 |
| Explosion | -13.79 |
| Arson | -8.01 |
| Robbery | -3.73 |
| Stealing | -3.00 |
| Vandalism | +9.57 |
| Abuse | +6.31 |
| RoadAccidents | -0.02 |

### Nhận xét chung cho from scratch

Kết quả from scratch cho thấy việc bỏ checkpoint không giải quyết được vấn đề cốt lõi của prompt replacement. Khi không dùng checkpoint, model có nhiều tự do hơn để học text prompt mới, nhưng dữ liệu UCF-Crime và supervision MIL vẫn không đủ mạnh để học lại một không gian class-text alignment tốt hơn baseline.

Điểm tích cực là `class_plus_prototype` from scratch làm tăng mAP, cho thấy prototype có thể cung cấp tín hiệu hữu ích cho localization. Tuy nhiên, cái giá phải trả là AUC và Ano-AUC giảm mạnh. Điều này gợi ý rằng prototype prompt đang giúp model tập trung vào một số đoạn sự kiện rõ hơn, nhưng đồng thời làm phân phối anomaly score kém ổn định hơn trên toàn bộ test set.

Kết luận bổ sung:

```text
Huấn luyện prompt replacement from scratch không tốt hơn huấn luyện từ checkpoint. Prototype có tín hiệu hữu ích cho mAP, đặc biệt với class_plus_prototype, nhưng chưa đủ để thay thế class prompt gốc của VadCLIP. Hướng này chỉ nên được xem là bằng chứng rằng prototype có thể hỗ trợ localization, không phải là phương án thay trực tiếp class representation.
```
