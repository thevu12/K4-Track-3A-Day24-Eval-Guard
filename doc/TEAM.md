# Nhóm thực hiện K4 Track 3A

## Cập nhật thực nghiệm 13/10/2026

Mục này thay thế các mô tả trạng thái cũ ở dưới: đã sinh đủ 50/50 đáp án
`gemini-3.1-flash-lite`, không còn extractive fallback. Phase B đã lưu
`artifacts/judge_results.json` và qua checker. Phase A đã lưu 29/50 checkpoint
trong `.cache/ragas_rows`, nhưng chưa có artifact tổng hợp do Google trả quota
ngày 500 request/model/project (429, yêu cầu chờ khoảng 20 giờ 35 phút tại lúc lỗi).
Không tiếp tục gọi API sau lỗi quota ngày.

Phase C có kết quả thật nhưng chưa đạt: chặn nhầm 8/8 clean inputs; bản trước
sửa được giữ tại `artifacts/guard_results_before_token_fix.json`. Đã sửa NeMo
sang chat client và tăng max_tokens self-check từ mặc định 3 lên 1024; bản sửa
token chưa chạy live lại vì hết quota. Không coi kết quả guard hiện tại là đạt.

RAGAS Gemini dùng BGE-M3 local, Answer Relevancy strictness=1, chạy tuần tự;
timeout metric 300 giây, có checkpoint theo hash đầu vào/model. Chi tiết trong
RUNBOOK. 282 offline tests đạt ở lượt kiểm tra gần nhất. Dự án chưa đủ điều kiện
nộp bản hoàn chỉnh; cần hoàn tất Phase A, xác minh bản sửa Phase C và render lại
báo cáo. Không sửa nhãn, điểm hoặc threshold để vượt checker.

Tên nhóm làm việc: Vu-Nguyen (đề xuất theo hai thành viên; chưa xác nhận tên đăng ký).

Repo hiện tại: https://github.com/thevu12/K4-Track-3A-Day24-Eval-Guard.
Đây là remote của workspace, chưa xác nhận là link đã nộp trên vlearn.

Trưởng nhóm/người nộp: chưa được hai thành viên xác nhận.

Ngày lab/deadline áp dụng: chưa có thông báo lớp trong workspace.
Múi giờ áp dụng: Asia/Ho_Chi_Minh (UTC+07:00); không lấy ngày sửa file thay ngày lab.

Commit dùng để chấm: chưa chốt; chưa tạo commit hay nộp bài thay các thành viên.

## Thành viên và phân công

Danh tính dưới đây do người dùng cung cấp. Phân công là **đề xuất bàn giao**,
không phải xác nhận rằng một thành viên đã tự viết mã, chạy thí nghiệm hoặc
review. Không gán các thay đổi do AI hỗ trợ thành commit/công việc đã làm của
hai người. Mỗi người cần tự kiểm tra phần phụ trách và cập nhật bằng chứng thật.

| STT | Họ và tên | MSSV | Vai trò / công việc cần tiếp nhận | Bằng chứng hiện có |
|---|---|---|---|---|
| 1 | Giang Thế Vũ | 02478 | Đề xuất: tích hợp m1–m5, sinh answers, Phase A, môi trường và bản nộp | Danh tính đã cung cấp; chưa có commit/review cá nhân được xác nhận |
| 2 | Nguyễn Hoàng Lê Nguyên | 02472 | Đề xuất: Phase B/C, đối chiếu judge/guard, latency và blueprint | Danh tính đã cung cấp; chưa có commit/review cá nhân được xác nhận |

Người tích hợp/chạy batch/nộp link: chưa xác nhận. Mã được hoàn thiện với AI
hỗ trợ trong workspace dùng chung; review và xác nhận bài nộp thuộc nhóm.
README yêu cầu mỗi người có commit; không tạo commit giả danh hoặc backdate
để đáp ứng yêu cầu này.

## Môi trường và chuẩn bị

Môi trường được kiểm tra trong lượt hoàn thiện: macOS 26.5 arm64,
Python 3.11.16 trong `.venv`. Full runtime đã cài, `uv pip check` pass;
Annoy build bằng Xcode clang thành công. Pin PyArrow 17.0.0 và pandas 2.2.3
để tương thích NumPy 1.26.4; preflight đã pass cả 22 imports, Python, spaCy
`en_core_web_lg` 3.8.0 và Qdrant 1.9.7 health check. `python3` của hệ thống
vẫn là Python 3.13.0; dùng Python trong virtualenv để chạy bài.

Đã tải `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`; smoke test cross-encoder MPS
và lượt setup toàn bộ 50 câu chạy thật thành công. Corpus 28 tài liệu được
chia 94 chunks, gồm 41 trang PDF scan có sidecar OCR do
`scripts/ocr_pdfs.swift` tạo. Loader kiểm tra SHA-256 của PDF trước dùng OCR;
OCR có thể sai dấu hoặc cấu trúc cột, đặc biệt BCTC, chưa phải bản chép đã
được con người xác nhận. Xem [RUNBOOK.md](RUNBOOK.md) để tái lập môi trường.

`artifacts/answers_50q.json` đã có 50 records hợp lệ, đúng 20/20/10;
50/50 có contexts, `generation_mode=extractive_fallback`,
`generation_model=null`. Retrieval BGE-M3/Qdrant và reranking chạy thật,
nhưng không có LLM generation do chưa cấu hình OPENAI_API_KEY.
Index 94 chunks mất 19.1 giây; batch 50 queries mất 106.4 giây wall time.
Đây là thời gian batch setup, **không phải** P50/P95 hoặc Task 12 latency.

File `.env` cục bộ đã tạo và bị Git ignore, key còn trống; cấu hình host dùng
devices MPS và batch size 4. Không lưu API key, `.env`, virtualenv hoặc model
cache trong Git. Preflight chỉ còn fail ở API key; checker đã pass answers,
còn đúng ba artifacts A/B/C chưa có. Không dùng các kết quả runtime nói trên
để khẳng định RAGAS/judge/NeMo đã chạy qua API.

Các lệnh kiểm tra không gọi API:

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/preflight.py --live
.venv/bin/python scripts/check_lab.py --with-tests
```

Kết quả kiểm tra cuối ngày 12/10/2026 (Asia/Ho_Chi_Minh): full runtime
**279 passed, 3 deselected**; có 8 cảnh báo deprecation từ NeMo/Pydantic,
không phải lỗi test. Môi trường test tối thiểu Python 3.11 với lock riêng:
**277 passed, 2 skipped, 3 deselected**; hai tests bổ sung cần runtime ML/NeMo
được bỏ qua có chủ ý. Ba tests live không được chạy vì thiếu API key.
`git diff --check` sạch. Không suy điểm rubric từ các số lượng tests này.

Kiểm thử Presidio thật: cả 8 clean controls không có nhận diện PII; 4 PII
attacks đều được phát hiện. Registry được giới hạn về CCCD/CMND, phone/email
để tránh English NER coi câu tiếng Việt hay phiên bản policy là tên người/ID.
Đây chỉ là tầng Presidio, không phải kết quả 28 inputs toàn bộ guard stack.
NeMo được thử 4 luồng input/output với LLM giả lập và network bị cấm để xác
minh interface/log; chưa chứng minh chất lượng guard với LLM thật.

Chưa có bằng chứng RAGAS/judge/guard live ở thời điểm viết trang này. Việc có
source hoặc mock tests pass không chứng minh NeMo đã chạy thật. Preflight
`--live` chỉ kiểm tra runtime/models/service, không gọi provider API. Sau khi
điền key cục bộ, `.venv/bin/python scripts/complete_lab.py --live --regenerate`
sinh lại answers bằng LLM rồi chạy các phase cần thiết; có chi phí API.
Nếu muốn giữ nguyên answers extractive đã có, bỏ `--regenerate` và báo đúng
generation mode trong kết quả. Xem [REPORT_SCHEMAS.md](REPORT_SCHEMAS.md).
`scripts/render_reports.py` chỉ dựng báo cáo từ đủ bốn artifacts hợp lệ và
source hashes khớp; không tạo số liệu thay runner. Hai thành viên cần đọc lại
conclusions và xác nhận.

## INDIVIDUAL

### GiangTheVu-02478

- Phần được đề xuất: tích hợp chunking, hybrid search, reranking, enrichment,
  RAGAS adapter; sinh 50 answers và phân tích Phase A.
- Phần thực tế cá nhân đã làm: chưa được thành viên xác nhận; không suy từ tên
  tài khoản Git hoặc yêu cầu hỗ trợ của người dùng.
- Bằng chứng workspace đã có: `artifacts/answers_50q.json` từ retrieval thật với
  50 extractive fallback; không quy lượt chạy được AI hỗ trợ thành công việc
  cá nhân. Cần bổ sung commit của thành viên sau kiểm tra, `artifacts/ragas_50q.json`
  và nhận định đã review trong [failure_clusters.md](reports/failure_clusters.md).
  Chưa có evidence RAGAS live.
- Cách kiểm tra bàn giao: chạy tests Phase A/pipeline; đối chiếu 20/20/10,
  generation mode, model IDs, top contexts và bottom 10; chạy checker offline
  sau runner, không gọi API lại chỉ để xác nhận file.
- Nội dung cần tự giải thích: retrieval khác generation; context precision
  khác recall; faithfulness khác độ đúng theo reference; cách xử lý phiên bản
  cũ/hiện hành. Chưa có tự thuật “đã học” của thành viên.
- Giới hạn: mock pass không thay RAGAS thật; lỗi evaluator phải hiện rõ;
  extractive fallback phải mang mode/model đúng, không đổi ground truth thành answer.
- Nguồn/AI hỗ trợ: Codex hỗ trợ triển khai và kiểm tra source, viết bản phân tích
  có điều kiện. Cần tự đọc mã, thực hiện kiểm tra nói trên và tuân thủ chính sách
  AI của coach trước khi xác nhận phần đóng góp.

### NguyenHoangLeNguyen-02472

- Phần được đề xuất: pairwise judge và swap-and-average, binary calibration,
  Presidio/NeMo input-output rails, attack/clean/output suites và latency.
- Phần thực tế cá nhân đã làm: chưa được thành viên xác nhận; không có commit,
  buổi review hoặc lượt chạy cá nhân được khai thay.
- Bằng chứng cần bổ sung: commit do chính thành viên tạo sau kiểm tra;
  `artifacts/judge_results.json`, `artifacts/guard_results.json`,
  [bias_report.md](reports/bias_report.md) và [blueprint.md](reports/blueprint.md).
- Cách kiểm tra bàn giao: tests Phase B/C; đối chiếu nguồn A/B, nhãn binary trên
  đúng model_answer gốc, prompt không chứa human labels/notes; kiểm tra 20 attacks,
  8 clean controls, 4 outputs và raw latency sau hai warm-up.
- Nội dung cần tự giải thích: quy winner pass2 về A/B gốc; Cohen's kappa và
  chance agreement; false positives; input-layer P95 không phải full-guard P95.
  Chưa có tự thuật “đã học” của thành viên.
- Giới hạn: bộ thử nhỏ, judge/guard có bất định; guard đạt attacks mà chặn clean
  inputs vẫn không đạt; không sửa số hoặc labels để đạt ngưỡng.
- Nguồn/AI hỗ trợ: Codex hỗ trợ source, tests và báo cáo từ evidence hợp lệ.
  Thành viên cần tự thực hiện các bước kiểm chứng, xác nhận bài và chính sách AI.

## Cập nhật sau khi nộp

Chưa có phiên bản nộp được xác nhận. Không có lịch sử sửa sau deadline nào được
khai ở đây. Khi nộp thật, bổ sung link/commit và thời điểm; nếu sửa sau hạn,
ghi commit mới, lý do và chấp thuận của coach, không backdate.
