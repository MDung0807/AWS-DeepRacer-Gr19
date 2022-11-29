# Chương 1. Giới thiệu về AWS DeepRacer
## 1.1. AWS DeepRacer
### 1.1.1. Khái niệm
AWS DeepRacer là phương pháp nhanh nhất (theo đúng nghĩa đen) để làm quen với công nghệ học tăng cường (RL), thông qua một chiếc xe đua tỷ lệ 1/18 tự hành hoàn toàn, được điều khiển bởi model học tăng cường, công cụ mô phỏng đua xe 3D và giải đua xe toàn cầu. Các nhà phát triển có thể huấn luyện, đánh giá và điều chỉnh các mô hình RL trong công cụ mô phỏng trực tuyến, triển khai các mô hình của họ lên AWS DeepRacer để có trải nghiệm xe tự hành trong thế giới thực và thi đấu trong Giải đua AWS DeepRacer để có cơ hội giành Cúp vô địch AWS.
### 1.1.2. Dịch vụ
AWS DeepRacer tích hợp với Amazon SageMaker để huấn luyện model học tăng cường, AWS RoboMaker để cung cấp bộ giả lập đua xe, Amazon Kinesis Video Streams để truyền phát video hậu trường mô phỏng ảo, Amazon S3 để lưu trữ model và Amazon CloudWatch để ghi nhật ký. Ngoài ra ở đề tài này còn sử dụng thêm AWS Lambda để khởi tạo và chạy các thuật toán “thưởng”.
Với những services ở trên, AWSDeepRacer giúp người dùng không cần phải bận tâm quá nhiều về việc lưu trữ, hay sử dụng những docker để tạo ra môi trường giả lập:
- Dựa trên các container Robomaker và Sagemaker, hỗ trợ trên rất nhiều nền tảng thiết lập CPU và GPU.
- Tập hợp rất nhiều kịch bản cho phép dễ dàng thiết kế model, không cần phải làm mọi thứ từ con số 0.
- Cho phép các bản mẫu AWS DeepRacer từ nguồn khác; Cho phép tải lên mô hình đã được đào tạo sẵn. Bên cạnh đó, AWSDeepRacer còn giúp đỡ trong việc xây dựng model kể cả mô phỏng lẫn vật lý:
- AWS DeepRacer giúp xây dựng một kết nối wifi dựa phương tiện AWS và AWS DeepRacer Console.
- AWS DeepRacer có thể chỉnh sửa một số thông tin của phương tiện (tốc độ, độ cân bằng, các góc bánh xe)
- AWS DeepRacer cũng đưa ra một số templates để xây dựng phương tiện cũng như đường đua vật lý để thử nghiệm ngoài đời thật AWSDeepRacer còn bảo dữ liệu người dùng ngăn chặn những xe mô phỏng của mình có thể bị rò rỉ ra ngoài:
- Dùng xác thực đa yếu tố(MFA) với mỗi người dùng mỗi lần đăng nhập.
- Dùng SSL/TLS để truy xuất với dữ liệu của AWS.
- Dùng API và lưu trữ thông tin người đăng nhập bằng AWS CloudTrail.
- Dùng các cách bảo mật mã hóa.
### 1.1.3. Hạn chế
### 1.1.4. AWS DeepRacer Evo
- AWS DeepRacer Evo là thế hệ tiếp theo trong đua xe tự hành. Nó được trang bị đầy đủ camera âm thanh nổi và cảm biến LiDAR để cho phép tránh vật thể và đua đối đầu, cung cấp cho bạn mọi thứ bạn cần để đưa cuộc đua của mình lên một tầm cao mới. Những cảm biến bổ sung này cho phép xe xử lý các môi trường phức tạp hơn và thực hiện các hành động cần thiết cho trải nghiệm đua xe mới. Trong các cuộc đua tránh vật thể, bạn sử dụng các cảm biến để phát hiện và tránh các chướng ngại vật trên đường đua. Đối đầu trực tiếp, bạn đua với một chiếc xe khác trên cùng đường đua và cố gắng tránh nó trong khi vẫn quay đầu trong thời gian vòng đua tốt nhất.
- Camera trái và phải hướng về phía trước tạo thành camera âm thanh nổi, giúp xe tìm hiểu thông tin độ sâu trong hình ảnh. Sau đó, nó có thể sử dụng thông tin này để cảm nhận và tránh các vật thể mà nó tiếp cận trên đường đua. Cảm biến LiDAR hướng về phía sau sẽ phát hiện các vật thể phía sau và bên cạnh xe.
- Xe AWS DeepRacer Evo, có sẵn trên Amazon.com , bao gồm xe AWS DeepRacer nguyên bản, mô-đun máy ảnh 4 megapixel bổ sung tạo hình ảnh âm thanh nổi với máy ảnh gốc, LiDAR quét, vỏ có thể vừa với máy ảnh âm thanh nổi và LiDAR, và một số phụ kiện và công cụ lắp đặt dễ sử dụng để lắp đặt nhanh chóng. Nếu đã sở hữu ô tô AWS DeepRacer, bạn có thể nâng cấp ô tô của mình để có khả năng tương tự như AWS DeepRacer Evo với Bộ cảm biến AWS DeepRacer .
## 1.2. Học tăng cường (Reinforment Learning)
### 1.2.1. Khái niệm
Reinforcement Learning là một lĩnh vực của ML,trong đó agent học cách cư xử trong môi trường bằng cách thực hiện các hành động và xem kết quả của các hành động. Đối với mỗi hành động tốt, agent nhận được phản hồi tích cực và đối với mỗi hành động xấu, agent nhận được phản hồi tiêu cực hoặc hình phạt. Do đó agent tự phát triển theo kinh nghiệm của nó để giảm thiểu rủi ro và tăng khả năng đến mục đích hơn.
### 1.2.2. Ứng dụng vào AWS DeepRacer
Trong reinforcement learning, các mô hình AWSDeepRacer dựa theo mục tiêu được định sẵn mà sẽ tác động với môi trường đường đua để tối đa hóa phần thưởng mà mình sẽ đạt.
Mục tiêu của RL trong AWSDeepRacer là tìm ra mô hình tối ưu nhất sau khi huấn luyện để có thể đem ra thực nghiệm ngoài thế giới vật lý.
AWSDeepRacer đưa ra rất nhiều lợi ích khi huấn luyện mô hình với một môi trường mô phỏng sử dụng RL:
+ Mô phỏng có thể ước tính mức độ tiến bộ mà mô hình đã đạt được và xác định thời điểm nó đi chệch hướng để tính toán phần thưởng.
+ Mô phỏng giải phóng người huấn luyện khỏi những công việc tẻ nhạt để thiết lập lại chiếc xe mỗi khi nó đi ra khỏi đường đua, như được thực hiện trong môi trường vật lý.
+ Mô phỏng có thể tăng tốc độ đào tạo. Mô phỏng cung cấp khả năng kiểm soát tốt hơn các điều kiện môi trường, ví dụ: chọn các tuyến đường, bối cảnh và tình trạng xe khác nhau.
