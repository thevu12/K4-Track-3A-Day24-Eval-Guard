# Blueprint CI/CD cho RAG và guardrails

Nhóm: Giang Thế Vũ — 02478; Nguyễn Hoàng Lê Nguyên — 02472.
Phụ trách đề xuất: cả hai thành viên; chưa có xác nhận review cá nhân.

Đây là **đề xuất kiến trúc serving, CI/CD và monitoring**, không khẳng định đã
triển khai endpoint, workflow, dashboard hay chạy full stack production.
Các runner lab A/B/C độc lập với endpoint serving đề xuất.

**Bằng chứng thực nghiệm chưa hoàn tất:** đã có `artifacts/answers_50q.json`
hợp lệ gồm 50 extractive fallback, đúng 20/20/10, `generation_model=null`.
Retrieval BGE-M3/Qdrant và cross-encoder chạy thật trên 28 tài liệu/94 chunks;
chưa có LLM generation vì key còn trống. Còn thiếu ba artifacts RAGAS/judge/guard,
nên chưa có evaluation timestamps/models, RAGAS, κ, guard pass rates và latency.
Các mục tiêu bên dưới là tiêu chí kiểm tra, không phải số đã đạt.

Runtime Python 3.11.16, 22 imports, spaCy model và Qdrant health đã pass preflight;
preflight chỉ còn lỗi API key. Đã tạo `.env` cục bộ bị ignore để điền key.
Sau đó chạy `.venv/bin/python scripts/complete_lab.py --live --regenerate` để
sinh LLM answers và thực hiện đánh giá; bỏ `--regenerate` nếu cố ý giữ bộ
extractive hiện tại. Chi tiết ở [RUNBOOK.md](../RUNBOOK.md).

## Kiến trúc và failure action

Luồng request đề xuất: Input → Presidio → NeMo input → RAG → NeMo output → response.
Request serving dừng sớm khi bị chặn; Task 12 vẫn đo đủ hai input layers.

| Tầng | Dữ liệu và trách nhiệm | Failure action đề xuất |
|---|---|---|
| Presidio | Phát hiện CCCD/CMND, phone VN, email trên input; chỉ dùng dữ liệu mẫu lab | Block hoặc redact trước external LLM; không ghi input thô vào log |
| NeMo input | Kiểm tra prompt injection/yêu cầu trái chính sách dựa quyết định rail | Block với thông báo an toàn; timeout/error phải phân biệt với quyết định allowed |
| RAG m1/m5 | Load, chunk và enrich; giữ source, parent/child, phiên bản, ngày hiệu lực | Tài liệu lỗi parse/index phải lộ lỗi; không âm thầm loại corpus rồi coi batch đầy đủ |
| RAG m2/m3 | Hybrid retrieval và reranking; giữ bằng chứng cho generation | Thiếu context thì báo thiếu evidence; query lịch sử được giữ bản cũ, query hiện hành ưu tiên bản thay thế |
| Generation | Chỉ trả lời từ retrieved contexts, giữ answer tách ground_truth | Không thay bằng reference khi API lỗi; fallback nếu dùng phải ghi đúng generation_mode/model |
| NeMo output | Kiểm tra câu trả lời, chặn/redact dữ liệu nhạy cảm; output sạch giữ nguyên | Không trả output chưa qua kiểm tra khi timeout; lưu category/error code, không lưu bí mật |

Secrets đặt trong `.env`/biến môi trường, không trong Git, prompt, logs hoặc
artifacts. Log đề xuất gồm request ID ngẫu nhiên, stage, model/config/corpus
version, decision, latency và error code; tuyệt đối không log API key, raw PII,
hay chain-of-thought. Debug contexts/answers chỉ dùng corpus lab hoặc bản đã
redact có kiểm soát quyền truy cập. Mọi fallback, circuit breaker và timeout
ở serving là đề xuất cần implement/test riêng, không suy từ runner offline.

Corpus có 41 trang OCR từ PDF scan; sidecar được so SHA-256 với PDF trước
khi load. Hash khớp không bảo đảm OCR đúng: BCTC còn nhiễu dấu/cột. Cần review
extracted text và nguồn khi câu trả lời sai; không âm thầm bỏ PDF khỏi corpus.

## Phạm vi và số đo latency

Task 12: scope **input_layers_sequential**, Presidio + NeMo input không short-circuit;
engine tạo trước đo, hai warm-up thật bị loại; ít nhất 20 timed samples cycling
8 clean inputs, dùng perf_counter và milliseconds. Percentile nội suy tuyến tính
tại (n−1)q; input_guard_ms của mỗi sample là tổng hai layer. Không cộng hai P95.

| Layer / scope | P50 ms | P95 ms | P99 ms | Tình trạng |
|---|---|---|---|---|
| Presidio | chưa đo | chưa đo | chưa đo | Cần runtime thật và raw samples Task 12 |
| NeMo input | chưa đo | chưa đo | chưa đo | Cần runtime thật và raw samples Task 12 |
| Input guard sequential | chưa đo | chưa đo | chưa đo | Cần ít nhất 20 samples sau 2 warm-up |
| RAG | chưa đo | chưa đo | chưa đo | Ngoài phạm vi bắt buộc Task 12 |
| NeMo output | chưa đo | chưa đo | chưa đo | Ngoài phạm vi bắt buộc Task 12 |
| Full guard / short-circuit request | chưa đo | chưa đo | chưa đo | Không suy từ input_guard_ms |

Input budget target: P95 <500ms; budget_ok chưa xác định. Không đổi số để đạt
target. Khi có kết quả, xác định layer đóng góp thời gian bằng raw samples,
đánh giá model/network/caching thay vì suy bottleneck trước đo.

Lượt setup đã ghi 19.1 giây indexing và 106.4 giây cho 50 queries. Đây chỉ là
wall time batch của pipeline, không có phân phối mẫu/warm-up Task 12; không
đưa các số này vào P50/P95/P99 của RAG, input guard hoặc full request.

Kế hoạch đo mở rộng: instrument từng layer và tổng request bằng monotonic
clock, dùng request ID để ghép spans; warm-up riêng từng engine rồi chạy ít
nhất 100 requests gồm allowed/blocked, tách cold/warm và concurrency. Ghi error
rates/timeouts bên cạnh latency để tránh “nhanh” do thất bại. Đo full guard
và full request có RAG riêng, giữ raw samples, config và dataset/version.

## CI gates đề xuất

CI chưa được triển khai hoặc chạy; các ngưỡng không tự có hiệu lực chỉ vì xuất
hiện trong Markdown. Job kiểm tra offline có thể dùng:

```bash
python -m pytest tests/ -q
python scripts/check_lab.py --with-tests
```

`check_lab.py` trả nonzero cho artifacts thiếu/sai schema/hash/phép tính,
attacks dưới 15/20, clean chưa allowed 8/8 hoặc output cases sai. Nó không
tự áp dụng những quality thresholds bổ sung bên dưới.

| Gate đề xuất thêm | Mốc | Hành động khi fail |
|---|---|---|
| RAGAS | Global faithfulness ≥0.75 và mean avg_score ≥0.65 trên 50 câu | Dừng promote; xem bottom 10 và mean từng distribution |
| Guard quality | Attacks ≥18/20, clean 8/8 allowed, output cases 4/4 | Dừng promote; sửa category lỗi rồi chạy lại cả controls |
| Judge | Theo dõi κ với baseline; review nếu ≤0.6 | Không dùng judge làm quyết định duy nhất; đối chiếu disagreements |
| Latency | Task 12 input P95 <500ms; full guard chỉ gate khi đã đo đúng scope | Giữ số đo thật, phân tích bottleneck; không coi input P95 là full guard |

Mốc bắt buộc lab vẫn attacks ≥15/20; bonus guard ≥18/20 kèm clean controls.
Để triển khai CI thật: protected job dùng secret store, bật batch live theo
manual dispatch/ngân sách, lưu artifacts bất biến kèm commit/model/hashes,
validate, rồi script threshold riêng phải `exit 1` khi một gate không đạt;
cấm PR tự sửa thresholds hoặc dùng fixture thay live results. Chưa có script
threshold bổ sung/workflow được tuyên bố triển khai trong blueprint này.

## Monitoring và cải tiến

Đề xuất Giang Thế Vũ tiếp nhận retrieval/generation; Nguyễn Hoàng Lê Nguyên
tiếp nhận judge/guard. Đây là phân công cần xác nhận, chưa phải lịch trực.

| Tín hiệu | Lấy mẫu/ngưỡng đề xuất | Xử lý |
|---|---|---|
| Retrieval/generation drift | Chạy holdout khi đổi corpus/model/prompt; alert nếu mean giảm >0.05 so baseline đã duyệt | So sánh theo distribution và source version; rollback index/config có version |
| Guard false positives | Review toàn bộ 8 controls mỗi thay đổi; alert khi có bất kỳ clean bị block | Đọc category/evidence đã redact; thêm holdout benign, không nới guard chỉ vì một case |
| Safety misses | Mỗi attack/output fail trong regression là ticket cần triage | Chặn promote; thêm test cùng loại nhưng khác câu chữ |
| Latency/errors | Dashboard P50/P95/P99, timeout rate theo stage, model và cold/warm | Alert P95 quá budget nhiều windows liên tiếp hoặc error rate >1%; xác nhận traffic/sample trước rollback |
| Judge disagreement | Review toàn bộ disagreements và sample approved answers trên holdout | Kiểm tra reference/prompt; không sửa human labels chỉ để tăng κ |

Các ngưỡng monitoring là đề xuất ban đầu, chưa hiệu chỉnh bằng traffic thật.
Raw logs không chứa PII; retention/access policy cần nhóm xác nhận trước production.
Nguồn fixture 50/20/8/4 quá nhỏ để suy safety/generalization rộng; phải giữ
test mới tách khỏi các câu đã dùng tuning.

Ưu tiên cải tiến có thể kiểm tra:

1. Từ answers fallback đã có, chốt có regenerate bằng LLM hay không rồi lấy
   đủ ba artifacts A/B/C và model/source provenance trước kết luận chất lượng.
2. Kiểm tra effective-date/version metadata bằng các câu 41–45/49; không bỏ
   bản cũ với câu hỏi lịch sử như id39/id49.
3. Ablation hybrid retrieval và reranking, giữ generator/evaluator cố định;
   so context_recall/precision và bottom 10, không chỉ global mean.
4. Dùng matched-content answers và repeated shuffled orders để đo judge bias
   trên holdout, vì năm cặp không đủ xác lập nguyên nhân.
5. Đo full request có short-circuit và lỗi/timeout; tối ưu sau khi biết bottleneck,
   luôn chạy lại clean controls khi đổi rails.

RAGAS mean/worst distribution: chưa đo. Cohen's κ: chưa đo.
Attacks passed và clean false-positive rate: chưa đo.
`python scripts/render_reports.py --overwrite` cập nhật các số này từ artifacts đã validate;
hash khớp chỉ xác minh nguồn nhất quán, không tự chứng minh đã có cuộc gọi API.
