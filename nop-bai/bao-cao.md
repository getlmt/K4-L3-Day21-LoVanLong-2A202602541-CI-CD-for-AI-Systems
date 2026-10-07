# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Lò Văn Long |
| MSSV | 2A202602541 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/getlmt/K4-L3-Day21-LoVanLong-2A202602541-CI-CD-for-AI-Systems |
| Ngày nộp | 07/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** F1 cao nhất (0,7149), vượt ngưỡng 0,65. Accuracy cao nhất thuộc lần 1 (0,878): accuracy ba lần chỉ chênh 3 điểm % còn F1 chênh 11 điểm, tức accuracy bị lớp đa số chi phối. Lần 2 giảm cả learning_rate lẫn số cây (learning_rate nhỏ cần nhiều cây hơn) nên học chưa đủ.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Chỉ 24,8% mẫu có thu nhập trên 50K, nên mô hình luôn đoán "thu nhập thấp" vẫn đạt accuracy 0,752 dù vô dụng, còn F1 lớp dương (kết hợp precision và recall của lớp thiểu số) bằng 0. Không dùng `average="weighted"`/`"macro"` vì F1 lớp đa số (~0,92) sẽ kéo điểm lên. Ảnh `07-quality-gate-chan.png`: tham số yếu cho F1 0,5907, Quality Gate chặn, Release bị bỏ qua.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| MLflow lỗi `ImportError` với SQLite. | SQLAlchemy 2.1 không hợp mlflow 2.13. | Pin `sqlalchemy==2.0.54`. |
| VM không load được model. | VM dùng scikit-learn 1.7.2, CI dùng 1.4.2. | Pin 1.4.2 trên VM. |
| Fork không chạy Actions. | Fork tắt Actions, thiếu secrets. | Bật Actions, thêm secrets, chạy `workflow_dispatch`. |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2`) | 0.7354 | 0.8820 |

**Nhận xét:** F1 tăng 0,02 nhưng chỉ ứng với 3 người thu nhập cao đoán đúng thêm và 1 dự đoán nhầm ít đi trên 500 mẫu holdout, nằm trong dao động ngẫu nhiên vì hai batch cùng phân phối. Điều được kiểm chứng là commit dữ liệu tự kích hoạt pipeline và VM phục vụ model mới.

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

- [x] Bonus 2 - Điều chỉnh ngưỡng quyết định: quét 0,10–0,90, ngưỡng 0,30 cho F1 0,7537 (0,5: 0,7354), ghi vào `report.json`, MLflow.
- [x] Bonus 3 - Báo cáo precision / recall tự động: confusion matrix, precision/recall từng lớp ở `detail.txt`; bỏ sót người thu nhập cao tốn kém hơn (recall 0,66 < precision 0,83).
- [x] Bonus 4 - Hoàn trả về phiên bản trước: model mới lên `candidate/`, chỉ chép sang `current/` khi F1 mới ≥ F1 cũ.
- [x] Bonus 5 - Cảnh báo lệch lạc dữ liệu: `::warning::` khi tỷ lệ lớp dương lệch quá 5 điểm % so với 24,8%.
