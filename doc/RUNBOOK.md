# Chạy bản triển khai của nhóm Vũ – Nguyên

Thành viên: Giang Thế Vũ (02478), Nguyễn Hoàng Lê Nguyên (02472).
Các thao tác dưới đây không tạo commit cá nhân hoặc tự nộp bài.

## Chuẩn bị runtime

Python 3.11 là runtime mục tiêu. Full stack cũ của lab dùng NumPy 1.26.4;
constraints đã pin thêm PyArrow 17.0.0 và pandas 2.2.3 để tránh lỗi import với
wheel mới đòi NumPy 2. Không suy từ `pip check` rằng mọi import/model đã chạy.

```bash
uv --cache-dir .cache/uv venv --python python3.11 .venv
uv --cache-dir .cache/uv pip install --python .venv/bin/python -r requirements/requirements.txt -c requirements/constraints-core.txt
uv --cache-dir .cache/uv pip install --python .venv/bin/python https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl
uv --cache-dir .cache/uv pip check --python .venv/bin/python
docker compose -f infra/docker-compose.yml up -d
```

Không chạy lại lệnh tạo virtualenv trên môi trường đang dùng nếu không chủ ý
thay môi trường. Docker chỉ bind localhost, dữ liệu Qdrant ở volume của lab;
không xóa volume/collection của người dùng khi khởi động lại.

Điền `OPENAI_API_KEY` trong `.env` cục bộ; không gửi key trong chat, commit hoặc
report. `.env.example` chỉ liệt kê tên biến. Trên Mac hỗ trợ MPS có thể dùng
`EMBEDDING_DEVICE=mps`, `RERANK_DEVICE=mps`; máy khác dùng `cpu` hoặc CUDA phù hợp.
`HF_HOME=.cache/huggingface` giữ model cache ở thư mục đã ignore.
`TLDEXTRACT_CACHE=.cache/tldextract` giữ cache phục vụ email recognition ở
đường dẫn ghi được, tránh thư mục cache mặc định bị chặn quyền.
`RAGAS_DO_NOT_TRACK=true` tắt telemetry RAGAS. Không gán kết quả offline thành live.

Với Gemini, đặt `LLM_PROVIDER=gemini` và `GEMINI_API_KEY` trong `.env`.
Cấu hình cục bộ được ưu tiên hơn biến shell. Model mặc định là
`gemini-3.1-flash-lite`; có thể cấu hình riêng `GENERATION_MODEL`, `JUDGE_MODEL`,
`EVALUATOR_LLM_MODEL`. RAGAS dùng BGE-M3 local cho embedding và Answer Relevancy
strictness=1 do endpoint Gemini không hỗ trợ nhiều candidates. Không so sánh
trực tiếp điểm này với cấu hình embedding/strictness khác mà không ghi chú.
Evaluator Gemini chạy tuần tự, chờ 4,25 giây trước mỗi request, timeout metric
300 giây. Checkpoint từng câu nằm trong `.cache/ragas_rows`, định danh bằng hash
đầu vào và model; chỉ lưu metric hữu hạn hợp lệ. Chạy lại Phase A sẽ reuse chúng.
`setup_answers.py --live --resume-fallbacks` chỉ sinh lại dòng fallback và giữ
nguyên các dòng LLM đã hợp lệ.

## Corpus, OCR và retrieval

Corpus gồm 25 Markdown và 3 PDF. `BCTC.pdf` (2 trang) và nghị định bảo vệ dữ
liệu cá nhân (39 trang) là ảnh scan. File `*.pdf.ocr.json` được tạo bằng Apple
Vision tiếng Việt/Anh, giữ hash PDF, số trang, text và confidence của OCR.
`m1` kiểm tra hash/số trang/thứ tự trước khi dùng sidecar cho trang không có
text layer. Bản OCR có thể sai dấu hoặc vị trí cột; chưa phải bản chép pháp lý
được xác nhận. Không bỏ tài liệu scan hay policy cũ để làm eval dễ hơn.

Nếu đổi PDF, phải OCR lại và review sidecar; loader từ chối sidecar stale.
Script macOS sau không ghi đè sidecar đã tồn tại:

```bash
swift scripts/ocr_pdfs.swift data/BCTC.pdf data/Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf
```

Pipeline: m1 chia parent/child theo ngân sách **từ**, giữ offsets/source/version;
m5 thêm tiêu đề/nguồn/section từ metadata, không sinh facts bằng LLM;
m2 fusion reciprocal-rank giữa BM25 và vector BGE-M3 (1024 chiều) trong Qdrant;
m3 chấm cặp question/document bằng `BAAI/bge-reranker-v2-m3`, giữ top 3.
Collection có fingerprint corpus/model; reindex không xóa collection trước.
Lần đầu tải hai model có thể mất nhiều phút và vài GB. Ground truth không đi
vào retrieval/generation; chỉ dùng ở evaluator/judge.

## Kiểm tra và hoàn tất thực nghiệm

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/preflight.py --live
.venv/bin/python scripts/complete_lab.py --live
```

`complete_lab.py` đòi key, kiểm tra runtime, tạo answers nếu thiếu, chạy phase
chưa có bằng chứng hợp lệ, dựng ba Markdown từ artifacts và chạy checker/tests.
Những phase có schema/source hashes còn hợp lệ được reuse, không trả phí lại.
Kết quả guard thật nhưng chưa đạt chất lượng vẫn được giữ và báo lỗi chất lượng;
không chạy lại tự động nhiều lần để chọn số đẹp.

`--regenerate` cho phép thay answers, làm stale reports A/B; `--rerun` yêu cầu
chạy lại phase. Kiểm tra/backup báo cáo thủ công trước khi chọn ghi đè.
Nếu chỉ cần từng bước:

```bash
.venv/bin/python scripts/setup_answers.py --live
.venv/bin/python scripts/run_experiments.py --phase a --live
.venv/bin/python scripts/run_experiments.py --phase b --live
.venv/bin/python scripts/run_experiments.py --phase c --live --latency-samples 20
.venv/bin/python scripts/render_reports.py --overwrite
.venv/bin/python scripts/check_lab.py --with-tests
```

Setup có thể dùng `extractive_fallback` khi không có key hoặc generation lỗi;
mode/model ghi rõ trong từng answer. Để chuyển answers fallback đã lưu sang
generation bằng LLM, dùng setup `--overwrite` hoặc complete `--regenerate`.
RAGAS không có evaluator heuristic thay thế: thiếu API hoặc metrics invalid
thì dừng, không điền điểm. Pairwise vẫn giữ `response_format=json_object` theo
contract starter và validate schema/types/ranges vì JSON mode không đảm bảo
schema ([OpenAI documentation](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat)).

## Trước khi nộp

Đọc lại các câu bottom 10 và bất đồng judge; renderer chỉ trình bày evidence,
không thay nhận định được kiểm chứng của nhóm. Đối chiếu 15/20 attacks và 8/8
clean, 4 outputs, ít nhất 20 latency samples sau 2 warm-up. Input-layer P95
không phải latency end-to-end. Không sửa số đo nếu vượt 500 ms.

Chốt trưởng nhóm, deadline theo thông báo lớp, đóng góp/review thực tế và
commit của từng người trong TEAM. Không mạo danh thành viên để tạo commit.
Chạy checker, rà secrets và commit bốn JSON thật cùng source/reports vào repo
chung; push/nộp sau khi nhóm xác nhận. Các báo cáo “chưa đo” không đủ để nộp.
