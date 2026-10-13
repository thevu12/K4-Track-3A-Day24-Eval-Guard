# Đánh giá LLM Judge Phase B

Nhóm: Giang Thế Vũ — 02478; Nguyễn Hoàng Lê Nguyên — 02472.
Phụ trách đề xuất: Nguyễn Hoàng Lê Nguyên; cần xác nhận sau kiểm tra.

**Trạng thái: chưa có artifact judge live; chưa thể kết luận độ tin cậy của judge.**
Nguồn còn thiếu: `artifacts/judge_results.json`. Judge timestamp/model,
position bias, verbosity win rate và Cohen's κ: chưa đo.
Nguồn candidate A đã có trong `artifacts/answers_50q.json`: 50 answers
extractive fallback từ retrieval/rerank thật, `generation_model=null`.
Chưa có LLM generation hoặc judge qua API vì key còn trống.

## Năm cặp A/B và hai lượt

| Thành phần | Nguồn cố định |
|---|---|
| Candidate A | `artifacts/answers_50q.json:answer` |
| Candidate B | `fixtures/human_labels_10q.json:model_answer` |
| Reference | `fixtures/test_set_50q.json:ground_truth` |
| Question ids | 1, 5, 12, 21, 41 |

Lượt một gửi A trước B; lượt hai đổi vị trí rồi quy winner và scores về A/B
ban đầu. Final winner chỉ giữ winner khi hai lượt nhất quán; bất đồng thành
tie. Reasoning được giữ nguyên để review, kể cả khi pass2 mô tả vị trí swapped;
không lấy nhãn A trong văn bản reasoning pass2 để đếm winner của A gốc.

Chưa có năm rows judge hợp lệ để liệt kê winner, reasoning và scores.
Không gọi lại judge với nguồn khác sau khi đã xem kết quả. Renderer sẽ lấy
các trường gốc và hiển thị đủ candidate và reasoning khi artifact tồn tại.

Position bias count là số cặp hai lượt khác winner sau khi đã quy nguồn;
rate = count / 5. Đây là đo độ bất nhất theo thứ tự, có thể chịu bất định
của sampling; không tự chứng minh nguyên nhân chỉ là position bias.

Verbosity win rate = số lần answer thắng dài hơn đối thủ / số final winners
khác tie. Độ dài theo `len(text)` như source, không tự đổi sang token count.
Trường hợp không có decisive winner, JSON ghi 0 theo hợp đồng nhưng tỷ lệ này
không có bằng chứng thực nghiệm để diễn giải “không bias”. Answer dài hơn có
thể đúng/đủ hơn thật; cần đọc từng case trước khi kết luận ưu ái độ dài.

## Calibration đúng model_answer

Binary judge đánh giá đúng question và model_answer gốc trong
`fixtures/human_labels_10q.json`. Không dùng answer mới của pipeline thay thế.
Prompt chỉ có question, answer, reference; human_label/human_note được giữ
ngoài prompt và chỉ dùng sau khi judge đã trả nhãn.

Chưa có 10 labels/reasonings từ live judge nên không lập bảng agreement giả.
Khi có artifact, báo cáo sẽ ghi đủ question_id, model_answer, label, reasoning,
human_label và agree cùng thứ tự calibration.

Cách đối chiếu: p_o = số nhãn đồng ý / 10. Gọi J0/J1 và H0/H1 là số nhãn
0/1 của judge/human, p_e = (J0×H0 + J1×H1) / 100; κ = (p_o−p_e)/(1−p_e).
Nếu p_e=1, κ không xác định và phải báo lỗi thay vì gán 1. Mốc κ>0.6 là điều
kiện bonus lab, không phải chứng nhận judge dùng được ở mọi miền.

## Giới hạn và quyết định sử dụng

Chỉ có 5 cặp và 10 nhãn: một cặp thay đổi làm position rate thay 0.2; một
nhãn thay đổi tác động đáng kể κ. Nhãn human là bộ tham chiếu được cấp, không
phải gold bất biến. Nếu bất đồng, hai thành viên cần đọc câu gốc, reference,
answer và reasoning; lưu phân loại lỗi, không sửa fixture để làm đẹp agreement.

Nếu sinh lại answers bằng LLM, judge và generator có thể cùng họ model nên
lỗi tương quan; bản answers hiện tại chưa dùng generator LLM. Cần ghi model IDs,
prompt/version và hashes trong artifact; danh tính model do code khai báo không
tự chứng minh provider đã chạy. Đề xuất dùng judge như tín hiệu triage, giữ
review người cho câu lỗi nghiêm trọng hoặc có xung đột chính sách. Chưa có số
liệu thì không tuyên bố judge đã đạt/không đạt ngưỡng nào.

Để tách bias khỏi chất lượng: bổ sung bộ holdout có câu trả lời cùng nội dung
nhưng khác độ dài, randomized order và lặp nhiều lần với ngân sách được duyệt;
không thay năm cặp bắt buộc hoặc đưa human labels vào prompt.

Sau khi điền key cục bộ, chạy tất cả phần còn thiếu trên answers đã có:

```bash
.venv/bin/python scripts/complete_lab.py --live
```

Thêm `--regenerate` nếu muốn thay answers extractive bằng LLM generation
trước khi bắt đầu đánh giá; không đổi candidate A sau khi xem kết quả judge.
Lệnh và các tùy chọn xem [RUNBOOK.md](../RUNBOOK.md).

Answers thật đã có nhưng kết quả B còn thiếu. Có thể dùng `--output-dir` mới để preview và
giữ bản review hiện tại thay cho `--overwrite`. Renderer không gọi API và chỉ ghi Markdown
sau khi toàn bộ artifacts, schema, phép tính và source hashes hợp lệ; kết quả
unit/mock chỉ kiểm tra logic, không được dùng làm những observations còn thiếu.
