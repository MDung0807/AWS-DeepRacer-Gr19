# Chương 2. Xây dựng mô hình
## 2.1. Ga-ra
### 2.1.1. Phương tiện mặc định
- Định cấu hình phương tiện của bạn bằng một hoặc nhiều cảm biến, chọn cấu trúc liên kết mạng thần kinh và tùy chỉnh không gian hành động để đáp ứng tiêu chí đua xe của bạn. Tùy chỉnh diện mạo phương tiện của bạn để có hình ảnh cá nhân hóa trong quá trình đào tạo.
- Mặc định AWS cung cấp cho ta một phương tiện có cấu tạo gồm 1 camera mặt trước và cấu trúc mạng nơ ron phức hợp 3 lớp (3 layer CNN).
![alt](Images/1.jpg)
### 2.1.2.	Tùy chỉnh phương tiện
- **Để tạo mới một phương tiện ta chọn build new vehicle.**
![alt](Images/2.png)
- **Ta tiến hành đặt tên cho phương tiện**
![alt](Images/3.png)
- **Chúng ta có thể thay đổi diện mạo cho phương tiện của mình bằng cách chọn vào màu sắc mong muốn**
![alt](Images/4.png)
- **Tùy chọn loại camera để phù hợp với mục đích đào tạo.**
    - Nếu bạn muốn đua trên một chiếc xe duy nhất trên đường đua time-trial , hãy cân nhắc sử dụng camera đơn. Để đua quanh một đường đua mà không có xe hoặc chướng ngại vật khác, bạn không cần phải có đầu vào phức tạp, hơn nữa, bạn càng đi càng phức tạp thì quá trình đào tạo sẽ mất nhiều thời gian hơn.
    - Cân nhắc sử dụng cảm biến camera âm thanh nổi khi bạn muốn xây dựng mô hình tránh vật thể hoặc mô hình đua xe head-to-head. Bạn sẽ cần sử dụng chức năng phần thưởng theo cách để mô hình học được các đặc điểm chiều sâu từ hình ảnh của bạn, điều có thể làm được với máy ảnh âm thanh nổi. Lưu ý rằng trong các mô hình đua xe đối đầu, camera âm thanh nổi có thể không đủ để che các điểm mù.
    - Cân nhắc thêm LIDAR vào mô hình của bạn nếu bạn muốn tham gia vào các cuộc đua head-to-head. Cảm biến LIDAR hướng về phía sau và quét cách xe khoảng 0,5m. Nó sẽ phát hiện những chiếc xe đang tiến đến từ phía sau hoặc trong những điểm mù khi rẽ.
![alt](Images/5.png)
- **Sau khi hoàn tất xe mới sẽ xuất hiện bên trong ga-ra**
![alt](Images/6.png)
## 2.2. Xây dựng mô hình
### 2.2.1. Tên mô hình và loại đường đua
- **Nhấn chọn Create model bắt đầu xây dựng mô hình bao gồm môi trường và phương tiện cần phải đào tạo để có thể tham gia các cuộc đua sau này.**
![image](https://user-images.githubusercontent.com/96776355/204731248-38d37cec-7e58-448d-b5f4-525ebae1866a.png)
- **Đặt tên cho mô hình cần tạo**
![image](https://user-images.githubusercontent.com/96776355/204731482-bbb6c855-1fb2-454b-8ba4-e79525aca980.png)
- **Chọn loại đường đua: Có rất nhiều đường đua để bạn lựa chọn từ các cuộc đua đã diễn ra trước đó.**
![image](https://user-images.githubusercontent.com/96776355/204731611-81568e23-2202-4f04-83f2-5317eb966fb4.png)
### 2.2.1. Hình thức đua
Có 3 chế độ đua có thể lựa chọn:
- **Time trial**: chạy đua với thời gian mà không có vật cản hoặc đối thủ nào khác đang cạnh tranh
![image](https://user-images.githubusercontent.com/96776355/204731876-24cb0aa6-c3b1-408b-a298-f164e10a0cca.png)
- **Object avoidance**: Xe đua trên đường đua hai làn với một số chướng ngại vật cố định được đặt dọc theo đường đua. 
![image](https://user-images.githubusercontent.com/96776355/204731940-f0594e23-b43e-4ec0-b7a7-f36b799e8193.png)
    - Ở chế độ này ta có thể chọn tiêu chí để đào tạo, đồng thời chọn số vật cản trên đường đua:
       - Fixed location: vị trí các đối tượng (box) được phân bố đều trên đường đua.
       - Random location: vị trí các đối tượng(box) được phân bố một cách ngẫu nhiên trên đường đua. Do đó việc đào tạo 1 phương tiện sẽ mất nhiều thời gian hơn so với Fixed location.
-	**Head-to-head**: Ở chế độ này chiếc xe của chúng ta sẽ đua với các phương tiện khác trên đường đua có 2 làn. 
![image](https://user-images.githubusercontent.com/96776355/204732944-e29038e9-8cec-4c40-92fb-6cc9b8491f1f.png) <br>
Chúng ta có thể chọn số phương tiện trên đường đua kèm tốc độ được định sẵn, cũng như chọn thời gian nhỏ nhất và ngắn nhất để các xe này thay đổi làn đua một cách ngẫu nhiên.
### 2.2.3.	 Thuật toán huấn luyện và siêu tham số
**Có 2 thuật toán huấn luyện**: PPO và SAC
-	PPO: Thuật toán với policy gần
-	SAC: Thuật toán với policy ngẫu nhiên
![image](https://user-images.githubusercontent.com/96776355/204733562-dd6e8570-8cc9-45c2-9481-b7d82c4b8c32.png) <br>
**Sự khác biệt cơ bản giữa 2 thuật toán**:

| Proximal Policy Optimization (PPO)  | Soft Actor Critic (SAC) |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Hoạt động trong cả không gian hành động liên tục và rời rạc | Chỉ hoạt động trong không gian hành động liên tục  |
| Hoạt động trong policy  | Hoạt động ngoài policy  |
| Sử dụng chính quy hóa entrophy   | Thêm entrophy vào mục tiêu tối ưu hóa |

Chọn các siêu tham số:
</br>
- Thuật toán PPO: <br>
![image](https://user-images.githubusercontent.com/96776355/204806893-ca945376-c561-4065-b1c4-38a0db99a363.png) <br>
![image](https://user-images.githubusercontent.com/96776355/204806918-3ea5c883-e1dd-403e-8cee-714746bfda38.png)<br>
- Thuật toán SAC:<br>
![image](https://user-images.githubusercontent.com/96776355/204806963-a8389e66-7f50-4be9-8a48-e8c95c29f892.png)<br>
![image](https://user-images.githubusercontent.com/96776355/204807008-28975800-f3c5-4104-8bb1-7e797275b67e.png)<br>
**Định nghĩa các siêu tham số** 

| Tham số | Định nghĩa |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Batch size | Số lượng experience xe gần đây được lấy mẫu ngẫu nhiên từ bộ đệm trải nghiệm và được sử dụng để cập nhật trọng số mạng nơ-ron. Nếu bạn có 5120 experience trong bộ đệm và chỉ định kích thước batch là 512, sau đó bỏ qua lấy mẫu ngẫu nhiên, bạn sẽ nhận được 10 batch experience. Lần lượt, mỗi batch sẽ được sử dụng để cập nhật trọng số mạng thần kinh của bạn trong quá trình đào tạo. Sử dụng kích thước batch lớn hơn để thúc đẩy cập nhật ổn định và trơn tru hơn đối với trọng số mạng nơ-ron, nhưng lưu ý khả năng quá trình đào tạo có thể chậm hơn.  |
| Number of epochs | Một epoch đại diện cho một lần đi qua tất cả các batch, trong đó trọng số mạng nơ-ron được cập nhật sau mỗi batch được xử lý, trước khi tiếp tục với batch tiếp theo. 10 epochs ngụ ý rằng bạn sẽ cập nhật trọng số mạng nơ-ron, sử dụng tất cả các batch một lần, nhưng lặp lại quá trình này 10 lần. Sử dụng số lượng epoch lớn hơn để thúc đẩy các bản cập nhật ổn định hơn, nhưng mong đợi quá trình đào tạo chậm hơn. Khi kích thước batch nhỏ, bạn có thể sử dụng số lượng epoch nhỏ hơn.  |
| Learning rate | Tốc độ học tập kiểm soát mức độ cập nhật đối với trọng số mạng nơ-ron. Nói một cách đơn giản, khi bạn cần thay đổi trọng số của chính sách của mình để đạt được phần thưởng tích lũy tối đa, bạn nên thay đổi chính sách của mình bao nhiêu. Tỷ lệ học tập lớn hơn sẽ dẫn đến đào tạo nhanh hơn, nhưng nó có thể gặp khó khăn để hội tụ. Tỷ lệ học tập nhỏ hơn dẫn đến sự hội tụ ổn định, nhưng có thể mất nhiều thời gian để đào tạo. |
| Exploration  | Điều này đề cập đến phương pháp được sử dụng để xác định sự đánh đổi giữa thăm dò và khai thác. Nói cách khác, chúng ta nên sử dụng phương pháp nào để xác định khi nào chúng ta nên ngừng khám phá (lựa chọn ngẫu nhiên các hành động) và khi nào chúng ta nên khai thác kinh nghiệm mà chúng ta đã tích lũy được. Vì chúng tôi sẽ sử dụng không gian hành động rời rạc, bạn nên luôn chọn CategoricalParameters. |
| Entropy | Một mức độ không chắc chắn, hoặc ngẫu nhiên, được thêm vào phân phối xác suất của không gian hành động. Điều này giúp thúc đẩy việc lựa chọn các hành động ngẫu nhiên để khám phá trạng thái / không gian hành động một cách rộng rãi hơn. |
| Discount factor | Hệ số chỉ định mức độ đóng góp của phần thưởng trong tương lai vào phần thưởng tích lũy dự kiến. Hệ số chiết khấu càng lớn thì mô hình càng xa để xác định phần thưởng tích lũy mong đợi và đào tạo càng chậm. Với hệ số chiết khấu là 0,9, chiếc xe bao gồm phần thưởng từ thứ tự 10 bước trong tương lai để thực hiện một bước di chuyển. Với hệ số chiết khấu là 0,999, chiếc xe sẽ xem xét phần thưởng từ thứ tự 1000 bước trong tương lai để thực hiện một bước đi. Các giá trị hệ số chiết khấu được đề xuất là 0,99, 0,999 và 0,9999. |
| Loss type | Loại tổn thất chỉ định loại hàm mục tiêu (hàm chi phí) được sử dụng để cập nhật trọng số mạng. Các loại lỗi mất bình phương Huber và Trung bình hoạt động tương tự đối với các bản cập nhật nhỏ. Nhưng khi các bản cập nhật trở nên lớn hơn, tổn thất Huber có gia số nhỏ hơn so với tổn thất lỗi trung bình bình phương. Khi bạn gặp vấn đề về hội tụ, hãy sử dụng kiểu mất Huber. Khi độ hội tụ tốt và bạn muốn đào tạo nhanh hơn, hãy sử dụng kiểu mất lỗi bình phương trung bình. |
### 2.2.4.	Cách thức vận hành của mô hình
**Chọn không gian hành động:**
- **Không gian hành động liên tục**: Một không gian hành động liên tục cho phép tác nhân chọn một hành động từ một loạt các giá trị cho mỗi trạng thái.
- **Không gian hành động rời rạc**: Một không gian hành động rời rạc đại diện cho tất cả các hành động có thể có của tác nhân đối với từng trạng thái trong một tập hợp hữu hạn. <br>
![image](https://user-images.githubusercontent.com/96776355/204809768-1811c74d-9a99-48af-9a73-6ea173b27a19.png)
- **Steering angle(Góc lái)**: Các tham số của không gian liên tục: Góc lái xác định phạm vi góc lái mà bánh trước của mô hình của bạn có thể quay.
- **Speed(Tốc độ)**: Tốc độ mà mô hình có thể đạt được. Tốc độ tối đa/tối thiểu được xác định sẵn cho mô hình.

     Các tham số của không gian liên tục:
 ![image](https://user-images.githubusercontent.com/96776355/204811998-ee5519d1-714c-419f-bef2-142a8fe320a7.png)

    Các tham số của không gian rời rạc:
 ![image](https://user-images.githubusercontent.com/96776355/204812061-a68efda6-3d5a-41f5-86c5-c9f4b5c8e315.png)

    Chọn phương tiện cần đào tạo( có thể chọn phương tiện tự tạo hoặc mặc định)
![image](https://user-images.githubusercontent.com/96776355/204812148-1f580d4b-a33d-4b54-ac4f-fe59731aedb0.png)
## 2.3. Dánh giá hiệu suất
