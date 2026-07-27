# 🚁 Báo Cáo Tổng Quan Đồ Án: UAV Path Planning với Deep Q-Network & Curriculum Learning

Tài liệu này tổng hợp chi tiết các thành phần cốt lõi, kiến trúc thiết kế và đánh giá tính khoa học của đồ án, được cấu trúc sẵn để bạn dễ dàng đưa vào báo cáo môn học hoặc đồ án tốt nghiệp.

---

## 1. Giới thiệu & Đặt vấn đề

- **Mục tiêu cốt lõi**: Nghiên cứu và phát triển một hệ thống điều khiển tự động cho Thiết bị bay không người lái (UAV), có khả năng tự lập đường bay (Path Planning) và né tránh vật cản tĩnh trong không gian 2D.
- **Phương pháp tiếp cận**: Ứng dụng Học Tăng Cường Sâu (Deep Reinforcement Learning), cụ thể là thuật toán **Deep Q-Network (DQN)** kết hợp với chiến lược **Curriculum Learning** (Học theo giáo trình).
- **Môi trường giả lập**: Xây dựng một môi trường tùy chỉnh `UAVMapEnv` tương thích hoàn toàn với chuẩn API **Gymnasium**, sử dụng ảnh bản đồ nhị phân để mô phỏng không gian bay.

---

## 2. Thiết Kế Môi Trường Mô Phỏng (UAVMapEnv)

Môi trường mô phỏng là trái tim của hệ thống RL. Thay vì dùng ảnh dạng ma trận pixel thô, đồ án sử dụng phương pháp trích xuất đặc trưng (Feature Extraction) qua giả lập cảm biến.

> [!TIP]
> Việc sử dụng tia LiDAR giả lập thay cho ảnh pixel giúp mô hình giảm nhẹ số lượng tham số, hội tụ (converge) nhanh hơn và đặc biệt là UAV có thể bay ở các bản đồ mới mà không bị học vẹt (overfitting).

### 2.1. Không Gian Trạng Thái (Observation Space - 30D Vector)
Mô hình quan sát môi trường thông qua một vector liên tục gồm 30 chiều giá trị (đã chuẩn hóa về khoảng `[-1.0, 1.0]`):
1. **Dữ liệu LiDAR (24D)**: 24 tia cảm biến bắn ra xung quanh UAV (quét $360^\circ$, tầm xa 150px) để đo khoảng cách tới chướng ngại vật gần nhất.
2. **Định vị điểm đích (2D)**: Khoảng cách từ UAV tới đích.
3. **Góc hướng tới đích (2D)**: Biểu diễn dưới dạng $\sin$ và $\cos$ của góc lệch.
4. **Hướng bay của UAV (2D)**: Biểu diễn $\sin$ và $\cos$ của góc heading hiện tại.
5. **Cảnh báo va chạm (1D)**: Khoảng cách của tia LiDAR có giá trị ngắn nhất.

> [!NOTE]
> **Điểm sáng khoa học**: Việc mã hóa góc bằng hàm lượng giác ($\sin/\cos$) giúp giải quyết triệt để vấn đề gián đoạn số học của góc (ví dụ sự gián đoạn giữa $359^\circ$ và $1^\circ$), giúp Mạng nơ-ron học mượt mà hơn.

### 2.2. Không Gian Hành Động (Action Space - Discrete)
UAV điều hướng thông qua 5 quyết định hành động rời rạc. Tốc độ bay cố định là 4px/step.
- `0`: Tiến thẳng ($0^\circ$).
- `1` / `3`: Ngoặt nhẹ trái / phải ($5^\circ$).
- `2` / `4`: Ngoặt gắt trái / phải ($15^\circ$).

---

## 3. Thiết Kế Hàm Thưởng (Reward Shaping)

Để giải quyết vấn đề phần thưởng thưa thớt (Sparse Reward) đặc trưng trong bài toán tìm đường, hàm thưởng được thiết kế khéo léo để "dẫn dắt" UAV từng bước:

- **Thưởng/Phạt Cốt lõi**:
  - Tới đích (Goal): **$+300$** (Kết thúc thành công).
  - Đâm vào vật cản hoặc bay ra ngoài biên: **$-100$** (Kết thúc thất bại).
- **Phần thưởng Dẫn dắt (Progress Reward)**:
  - Thưởng một lượng dương nếu bước đi hiện tại làm giảm khoảng cách tới đích, và phạt âm nếu bay xa khỏi đích.
- **Hình phạt Cảnh cáo (Penalty)**:
  - Trừ **$-0.5$** điểm nếu UAV bay quá sát tường (LiDAR < 15px). Khuyến khích UAV chọn lộ trình an toàn giữa các khoảng trống.
  - Trừ **$-0.2$** điểm cho mỗi bước đi (Time Step Penalty). Khuyến khích UAV tìm lộ trình ngắn nhất, tốn ít thời gian nhất.

---

## 4. Chiến Lược Huấn Luyện (Training Strategy)

### 4.1. Curriculum Learning (Học theo giáo trình)
Thay vì huấn luyện ngay trên bản đồ khó, UAV được trải qua 3 giai đoạn (Stages) nâng dần độ khó:
1. **Stage 1 (Easy)**: Bản đồ chướng ngại vật đơn giản ở giữa, giúp UAV học cách đi vòng cơ bản.
2. **Stage 2 (Medium)**: Các trụ tròn phân tán. UAV học cách đưa ra quyết định chọn luồng lách qua các trụ.
3. **Stage 3 (Hard)**: Không gian chật hẹp, các dầm chữ nhật so le. UAV rèn luyện phản xạ bay sát mép và vượt địa hình phức tạp.
- **Điều kiện thăng cấp**: UAV được chuyển sang Stage tiếp theo khi Tỷ lệ thành công (Success Rate) đạt $\geq 70\%$ trong 200 episodes.

### 4.2. Kỹ Thuật Early Stopping
Hệ thống giám sát quá trình học và tự động dừng sớm (Early Stopping) nếu:
- Success Rate đạt ngưỡng xuất sắc ($\geq 85\%$).
- Sự cải thiện không đáng kể (tăng $< 2\%$) trong 500 episodes liên tiếp.
=> Giúp tiết kiệm tài nguyên tính toán và lấy được mô hình có độ cân bằng tốt nhất (tránh Overfitting vào tập Train).

---

## 5. Phương Pháp Kiểm Thử & Đánh Giá

Đồ án đảm bảo tính khách quan khoa học bằng cách phân tách tuyệt đối tập bản đồ Huấn luyện (Train Maps) và Kiểm thử (Eval Maps).

**Khả năng Zero-shot Generalization:**
UAV sau khi train sẽ được ném vào 3 bản đồ hoàn toàn xa lạ để đánh giá:
- **Heldout Map**: Các dầm ngang dài hoàn toàn khác biệt với cấu trúc khi Train.
- **Urban Map**: Mô phỏng khối nhà khu vực đô thị chật hẹp, bắt UAV phải luồn lách liên tục.
- **Dense Map**: Lưới chướng ngại vật mật độ dày đặc, thử thách phản xạ tránh né tức thời.
Việc UAV có thể hoàn thành xuất sắc các bản đồ này chứng minh mô hình thực sự đã học được "phản xạ điều hướng" thay vì ghi nhớ (memorize) lộ trình.

---

## 6. Tổng Kết

Đồ án có tính ứng dụng cao và phương pháp tiếp cận cực kỳ vững chắc về mặt lý thuyết Học Tăng Cường (Reinforcement Learning). Việc kết hợp giữa State biểu diễn bằng LiDAR, cơ chế Reward Shaping chi tiết và phương pháp Curriculum Learning giúp biến một bài toán điều hướng không gian phức tạp trở nên khả thi và tối ưu về tốc độ hội tụ mô hình.
