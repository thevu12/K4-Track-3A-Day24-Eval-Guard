# Phân tích lỗi Phase A

Nhóm: Giang Thế Vũ — 02478; Nguyễn Hoàng Lê Nguyên — 02472.
Phụ trách đề xuất: Giang Thế Vũ; cần thành viên xác nhận sau kiểm tra.

**Trạng thái: chưa có kết quả RAGAS live; báo cáo thực nghiệm chưa hoàn tất.**
Không suy điểm số từ test mock hoặc từ ground truth.
`artifacts/answers_50q.json` đã được tạo và validate; còn thiếu
`artifacts/ragas_50q.json`, nên chưa có evaluation timestamp/evaluator models.

## Bằng chứng pipeline đã chạy

50 answers đúng ids/distributions 20/20/10, tất cả có contexts;
50/50 `generation_mode=extractive_fallback`, `generation_model=null`.
BGE-M3 + Qdrant + cross-encoder đã chạy thật trên MPS; LLM generation chưa
chạy vì API key còn trống. Không gọi đây là answers do LLM sinh.
SHA-256 của bản answers hiện tại:
`35a3a69aef5c22b12dfcc6cb495eefef3b4e541d38ecb4137b67d9c5735a0cf3`.
Nếu regenerate, thay snapshot này cùng báo cáo theo hashes của lần chạy mới.

Corpus gồm 28 tài liệu, 94 chunks; 41 trang scan được OCR bằng Apple Vision,
sidecar giữ hash PDF và được loader đối chiếu. BCTC có nhiễu OCR và có thể
sai dấu/vị trí cột; chưa có chứng nhận transcribe thủ công. Vì vậy thiếu hoặc
sai context cần kiểm tra cả OCR/parse trước khi kết luận do ranking/generation.
Index mất 19.1 giây, 50 queries mất 106.4 giây wall time trong lần setup;
không dùng hai con số này làm latency percentile hoặc số đo Task 12.

## Kiểm tra sơ bộ nội dung answers đã lưu

Các quan sát sau do trợ lý AI đọc trực tiếp `answers_50q.json` theo SHA-256
trên; **không phải RAGAS bottom 10, không phải nhãn/điểm evaluator và không
khẳng định hai thành viên đã review**. Chọn năm ví dụ để tách lỗi thiếu tài
liệu khỏi lỗi không tổng hợp đủ thông tin. Mỗi ví dụ có ba contexts;
answer hiện tại trùng nguyên context thứ nhất, không trùng ground_truth.

| Id / điều cần trả lời | Answer và retrieved contexts thực tế | Đối chiếu ground_truth / kết luận có giới hạn |
|---|---|---|
| 1 — nghỉ khi kết hôn | Answer lấy toàn đoạn `nghi_phep_dac_biet.md`, có “Kết hôn: 3 ngày làm việc” và điều kiện có lương, không trừ phép năm. Contexts còn lại là nghỉ không lương và nghỉ phép năm 2024. | Nội dung cần trả lời đã nằm trong answer/context đầu và phù hợp reference. Tuy nhiên answer kèm nhiều chính sách không được hỏi; chưa có điểm relevancy/precision để định lượng ảnh hưởng. |
| 21 — Senior, 9 năm thâm niên: phép năm và khoảng lương | Contexts lần lượt là nghỉ phép 2024, nghỉ phép 2023, nghỉ không lương; không có `bang_luong_2024.md`. Answer chỉ có chính sách 2024 và ví dụ 9 năm = 18 ngày phép. | Reference yêu cầu cả 18 ngày phép lẫn lương Senior 20–35 triệu VNĐ/tháng. Phần lương thiếu trong answer và cả ba contexts; chỉ đổi generation prompt không bổ sung được evidence chưa retrieve. |
| 39 — so sánh mật khẩu v1/v2 | Context thứ nhất là `mat_khau_v2.md`: 12 ký tự, 120 ngày, MFA bắt buộc; context thứ hai là `mat_khau_v1.md`: 8 ký tự, 90 ngày và ghi đã thay thế. Answer chỉ xuất context đầu. | Retrieval đã mang về cả hai phiên bản nhưng answer không trình bày phần so sánh v1 mà reference yêu cầu. Đây là thiếu coverage ở đầu ra extractive trong ví dụ này; chưa thể quy lỗi cho việc không retrieve bản cũ. |
| 41 — phép năm hiện hành | Context đầu/answer là bản 2024: 15 ngày cơ bản, +1 ngày/3 năm; context thứ hai chứa bản 2023: 12 ngày, +1 ngày/5 năm. | Answer dùng đúng số và phiên bản hiện hành theo reference trong case này. Việc bản 2024 đứng đầu ở một query chưa chứng minh hệ thống nhận diện xung đột phiên bản nói chung. |
| 49 — phép năm cũ và bản hiện hành | Context đầu/answer là bản 2023: 12 ngày, +1 ngày/5 năm; bản 2024 nằm ở context thứ hai và có 15 ngày, +1 ngày/3 năm cùng câu thay thế bản cũ. | Answer trả lời phần lịch sử nhưng bỏ phần chính sách hiện hành; reference cần cả hai. Cả hai nguồn đã có, nên cần tổng hợp theo từng ý hỏi, không đơn giản xóa bản cũ hoặc ép luôn chỉ trả bản mới. |

Ưu tiên thử nghiệm từ các evidence này: với id21, tách query nhiều ý thành
truy hồi phép năm và bậc lương, giữ đa dạng nguồn rồi rerank; kiểm tra lại
contexts trước khi sinh câu trả lời. Với id39/id49, generator cần sử dụng
nhiều contexts và liệt kê đủ từng phiên bản/ý hỏi; không sửa answer hiện tại
thành reference. Với id1, thử câu trả lời ngắn bám đúng yêu cầu thay cho chép
toàn chunk. Chạy lại đủ 50 câu và RAGAS sau thay đổi, không chỉ tối ưu năm ví dụ.

## Điểm theo distribution

50 câu hỏi được cấp sẵn gồm factual 20, multi_hop 20, adversarial 10. Đây là
kích thước bộ thử, không phải kết quả hệ thống đã đạt.

| Metric | factual (20) | multi_hop (20) | adversarial (10) |
|---|---|---|---|
| faithfulness | chưa đo | chưa đo | chưa đo |
| answer_relevancy | chưa đo | chưa đo | chưa đo |
| context_precision | chưa đo | chưa đo | chưa đo |
| context_recall | chưa đo | chưa đo | chưa đo |
| avg_score | chưa đo | chưa đo | chưa đo |

Mỗi câu có bốn điểm hữu hạn trong [0,1]; avg_score là trung bình bốn điểm.
Generation mode/fallback đọc từ từng answer; bản hiện tại gồm 50 extractive
fallback và không có model generation. Nếu sinh lại bằng LLM, cập nhật số câu
và model theo kết quả mới, không giữ mô tả fallback cũ.
Evaluator models phải lấy từ metadata của m4 trong lần đánh giá, không lấy
JUDGE_MODEL thay thế. Thiếu điểm hoặc NaN làm lượt đánh giá không hợp lệ.

## Bottom 10 và cách xác định nguyên nhân

Chưa có bottom 10 đo được; không lựa mười câu tùy ý để điền vào báo cáo.
Sau khi đủ artifacts, `python scripts/render_reports.py --overwrite` lấy đúng rank, id, question,
distribution, avg_score, worst_metric, diagnosis và suggested_fix đã lưu; nó
giữ nguyên diagnosis/fix của Task 3 và kèm answer/contexts/reference để review.
Các gợi ý sinh theo metric chỉ là giả thuyết, không phải nguyên nhân đã chứng minh.

| Dấu hiệu cần kiểm tra | Đối chiếu trực tiếp | Thay đổi thử nghiệm có thể kiểm tra |
|---|---|---|
| context_recall thấp | Context có thiếu mệnh đề cần thiết trong reference, có thiếu tài liệu thứ hai của multi-hop không | So sánh hybrid retrieval và kích thước/top-k; giữ nguyên generator/evaluator |
| context_precision thấp | Top contexts có khác chủ đề hoặc sai phiên bản không | Rerank và metadata filter; ghi contexts trước/sau cùng một query |
| faithfulness thấp | Answer có khẳng định không nằm trong contexts không | Buộc bám nguồn, chỉ rõ thiếu bằng chứng; so sánh trên cùng contexts |
| answer_relevancy thấp | Answer có trả lời sai ý hỏi, thiếu điều kiện hoặc trả lời dài nhưng lệch ý không | Prompt cấu trúc trả lời theo từng ý; giữ nguyên retrieval để tách hiệu ứng |

Khi reference và context bất đồng, phải đọc văn bản nguồn trước khi quy lỗi cho
generator; faithfulness cao vẫn có thể đi kèm câu trả lời sai vì lấy bản cũ.
Với câu không đủ evidence, giữ khả năng từ chối thay vì tạo con số từ trí nhớ.

## Matrix và nhóm yếu nhất

Matrix worst_metric × distribution chưa được đo. Mỗi câu đóng góp đúng một
ô theo metric nhỏ nhất; hòa điểm dùng thứ tự metric cố định trong source.
Tổng ba cột luôn bằng 20/20/10 nên không dùng tổng cột để xác định nhóm yếu.
Nhóm yếu nhất phải có mean avg_score thấp nhất; dominant metric là metric được
chọn làm worst nhiều nhất, không chứng minh mọi câu trong ô đều thất bại.

Nhóm yếu theo mean thấp nhất: chưa đo. Mean và dominant metric: chưa đo.
Chưa có cơ sở kết luận retrieval hay generation là nút thắt thực tế.

## Bẫy adversarial: bằng chứng từ bộ đề, chưa phải kết quả chạy

- Câu 41–42 và 49 đặt hai phiên bản nghỉ phép cạnh nhau. Trong
  `data/nghi_phep_nam_v2023.md` là 12 ngày và +1 ngày/5 năm; bản 2024 thay thế
  bằng 15 ngày và +1 ngày/3 năm. Query hỏi lịch sử vẫn cần lấy bản cũ; không
  xóa mọi tài liệu cũ chỉ để tăng điểm câu hỏi hiện hành.
- Câu 43–45 kiểm tra mật khẩu: `data/mat_khau_v1.md` là 8 ký tự/90 ngày;
  `data/mat_khau_v2.md` là 12 ký tự/120 ngày và MFA bắt buộc. Cần giữ effective
  date và trạng thái thay thế trong metadata/context để generator phân biệt.
- Câu 46–48 và 50 kiểm tra phủ định/ngoại lệ: phép năm và PVI cho thử việc,
  tự xử lý malware, VPN cá nhân. Chỉ khớp từ khóa “nghỉ phép” hoặc “bảo mật”
  không chứng minh đã giữ được điều kiện loại trừ.

Những nhận xét này đến từ corpus/fixtures của lab, không phải kết luận hệ thống
đã mắc hoặc đã sửa lỗi. Sau khi có artifact, so sánh adversarial mean với factual
và multi_hop; điểm adversarial thấp hơn không tự chứng minh nhận ra version conflict.

## Chạy lại và giới hạn

Điền key vào `.env` cục bộ (đã ignore), không gửi qua chat hoặc commit.
Để chuyển bộ fallback hiện tại sang LLM generation rồi chạy đủ A/B/C:

```bash
.venv/bin/python scripts/preflight.py --live
.venv/bin/python scripts/complete_lab.py --live --regenerate
```

Nếu cố ý đánh giá pipeline extractive hiện tại, bỏ `--regenerate`; source
answers và metadata phải phản ánh đúng lựa chọn. `complete_lab.py` reuse
artifacts còn hợp lệ, không tự chạy lại quality-failing evidence để chọn điểm đẹp.

Renderer cần đủ kết quả A/B/C để đồng bộ ba báo cáo; không gọi API. Dùng
`--output-dir` để preview trước; `--overwrite` thay bản Markdown hiện tại nên
cần giữ lại nhận định review thủ công qua Git/diff. Các runner
live có chi phí và chỉ chạy sau khi chuẩn bị runtime/key. Checker hiện pass
answers nhưng báo thiếu ba artifacts A/B/C dù Markdown đã có nội dung.

Mỗi cải tiến cần lưu phiên bản mã/config, model IDs, corpus và source hashes;
đổi một nhóm yếu tố mỗi lần, xem cả mean lẫn từng câu và không dùng tập 50 câu
làm bằng chứng khái quát cho mọi yêu cầu. Rerun evaluator có bất định; cần
ngân sách chạy lặp trước khi coi thay đổi nhỏ là cải thiện đáng tin cậy.
