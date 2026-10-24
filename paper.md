Title: Analyzing Driver Visual Attention Behavior in Autonomous Driving Using Lightweight CNN with Temporal Shift Module
=> Lightweight CNN with Temporal Shift Module for Driver Visual Saliency Prediction in Autonomous Driving
Abstract: This paper presents a novel approach for predicting driver visual attention behavior in autonomous driving scenarios using a lightweight convolutional neural network (CNN) integrated with a Temporal Shift Module (TSM). The proposed model is trained on the DriverGaze360 dataset, which provides comprehensive gaze data for drivers in various driving conditions. By leveraging the TSM, our model effectively captures temporal dependencies in driver gaze patterns, enabling accurate prediction of visual attention. We evaluate the performance of our approach against existing methods and demonstrate its superiority in terms of accuracy and computational efficiency. The results highlight the potential of our model for real-time applications in autonomous driving systems, contributing to enhanced safety and driver assistance technologies.
------------------------------------------
project direction: 
a lightweight temporal visual-saliency model for driver attention. 
The screenshots show a MobileNetV3 backbone, Temporal Shift Modules (TSM) inside residual blocks, temporal pooling (mean/last/attention), and an FPN-lite decoder that upsamples multiscale features to a single sigmoid saliency map
 
Trong bối cảnh xe tự hành ngày càng phát triển, việc hiểu hành vi chú ý thị giác của tài xế là yếu tố quan trọng để đảm bảo an toàn giao thông, đặc biệt trong giai đoạn giao thoa giữa xe tự hành và xe do người lái. Các nghiên cứu gần đây tại CVPR, ICCV (2024–2026) đạt độ chính xác cao nhưng chủ yếu dùng kiến trúc nặng (Transformer), khó triển khai trên thiết bị tài nguyên hạn chế. Khoảng trống là thiếu một nghiên cứu phân tích hành vi chú ý dựa trên công cụ học sâu nhẹ, khả thi trên GPU tầm trung.

Mục tiêu: (1) Phân tích hành vi chú ý thị giác của tài xế trong các tình huống giao thông khác nhau bằng mô hình CNN nhẹ; (2) Xác định các vùng không gian tài xế thường tập trung nhìn và sự thay đổi theo ngữ cảnh; (3) So sánh hiệu năng MobileNetV2 với ResNet18 trên tập dữ liệu DriverGaze360.

Phương pháp: Sử dụng tập dữ liệu mở DriverGaze360 (CVPR 2026, ~1 triệu khung hình từ 19 tài xế). Video được trích frame, resize 224×1120. Mô hình dùng MobileNetV3 backbone + lớp tích chập 1×1 tạo bản đồ chú ý. Huấn luyện với loss KLD + MSE, optimizer Adam (lr=1e-4), batch size 8, 30 epochs, Mixed Precision trên RTX 2070. Phân tích hành vi bằng cách so sánh bản đồ chú ý dự đoán giữa các nhóm tình huống giao thông (giữ làn, chuyển làn, phanh).

Kết quả kỳ vọng: Mô hình đạt KLD < 3.0, SIM > 0.40. Phân tích cho thấy tài xế tập trung vào các vùng khác nhau tùy tình huống (ví dụ: nhìn xa hơn khi giữ làn, nhìn gương khi chuyển làn). Kết luận: CNN nhẹ vừa đủ tốt để phân tích hành vi chú ý, mở ra tiềm năng triển khai trên thiết bị nhúng.


Input: 16 RGB frames liên tiếp, kích thước \(224 \times 1120\).
Backbone: MobileNetV3-Large.
Temporal modeling: Temporal Shift Module (TSM).
Output: một saliency map dự đoán cho frame cuối của clip.
Dataset: DriverGaze360, với saliency maps làm ground truth.