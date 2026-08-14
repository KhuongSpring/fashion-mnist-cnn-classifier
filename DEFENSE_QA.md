# Hỏi–Đáp dự phòng cho buổi vấn đáp

Tài liệu này gom các câu hỏi hay gặp khi vấn đáp đề tài "Phân loại ảnh thời trang bằng CNN", kèm câu trả lời ngắn gọn, viết theo đúng những gì đề tài này thực sự làm (không phải lý thuyết chung chung) — để bạn trả lời tự tin dù chưa có nền tảng AI sâu.

Mẹo trả lời: nói ngắn gọn (2-3 câu), nếu giám khảo hỏi sâu hơn thì họ sẽ hỏi tiếp — không cần giải thích hết mọi thứ trong 1 câu.

---

## A. Khái niệm cơ bản

**1. CNN là gì, khác gì so với MLP?**
MLP nhìn ảnh như 1 dãy số rời rạc (784 pixel), không biết pixel nào nằm cạnh pixel nào. CNN dùng các bộ lọc (Conv) trượt qua ảnh để tìm các đặc trưng có tính không gian như cạnh, góc, hoa văn — giống cách mắt người nhận diện hình dạng. Vì vậy CNN phù hợp với ảnh hơn.

**2. Convolution (lớp Conv) làm gì?**
Một bộ lọc nhỏ (ví dụ 3×3) trượt qua toàn bộ ảnh, tại mỗi vị trí nhân — cộng để cho ra 1 con số thể hiện "vị trí đó có giống đặc trưng bộ lọc đang tìm không" (ví dụ cạnh dọc, cạnh ngang). Nhiều bộ lọc → nhiều loại đặc trưng được phát hiện cùng lúc.

**3. Pooling (MaxPool) để làm gì?**
Thu nhỏ ảnh lại (ví dụ 28×28 → 14×14) bằng cách lấy giá trị lớn nhất trong từng vùng nhỏ. Giúp giảm số lượng tính toán và làm mô hình ít nhạy cảm với việc vật thể bị lệch vị trí vài pixel.

**4. Overfitting là gì? Làm sao biết mô hình bị overfitting?**
Là hiện tượng mô hình "học thuộc lòng" dữ liệu train quá kỹ, đến mức làm bài mới (dữ liệu chưa từng thấy) lại kém hẳn. Dấu hiệu: accuracy trên tập train rất cao nhưng accuracy trên tập validation/test thấp hơn nhiều — khoảng cách này gọi là **overfitting gap** (xem mục 11 trong `Fashion_MNIST_Experiments.ipynb`, phần Ablation Study đã đo trực tiếp con số này).

**5. Underfitting là gì?**
Ngược lại với overfitting: mô hình quá đơn giản hoặc chưa học đủ, nên accuracy thấp ở cả tập train lẫn validation.

**6. Dropout hoạt động thế nào, tại sao giảm overfitting?**
Trong lúc train, Dropout ngẫu nhiên "tắt" một số neuron (ví dụ 30%) ở mỗi lượt học. Việc này buộc mô hình không được phụ thuộc quá nhiều vào 1 vài neuron cụ thể, phải học cách tổng quát hơn. Khi test/dự đoán thật, Dropout không hoạt động (dùng toàn bộ neuron).

**7. Batch Normalization là gì, tại sao giúp ích?**
Chuẩn hóa lại đầu ra của mỗi lớp (đưa về trung bình gần 0, phương sai gần 1) trước khi đưa vào lớp tiếp theo. Giúp quá trình học ổn định hơn, hội tụ nhanh hơn, và giảm phần nào overfitting.

**8. Data Augmentation là gì, tại sao giúp ích?**
Tạo thêm biến thể của ảnh train (xoay nhẹ, lật ngang, tịnh tiến nhẹ) để mô hình nhìn thấy nhiều "phiên bản" khác nhau của cùng 1 vật thể, thay vì học thuộc đúng những ảnh gốc → tổng quát hóa tốt hơn với ảnh mới.

**9. Tại sao chia tập Train / Validation / Test?**
- **Train**: dữ liệu mô hình học trực tiếp.
- **Validation**: dùng trong lúc train để theo dõi mô hình có đang overfitting không, và để điều chỉnh learning rate (không dùng để học trực tiếp).
- **Test**: chỉ dùng 1 lần duy nhất ở cuối, để đánh giá khách quan — mô hình chưa từng "nhìn thấy" tập này dù gián tiếp.

**10. Softmax là gì?**
Hàm biến đầu ra thô của mô hình (10 con số bất kỳ) thành 10 xác suất cộng lại bằng 100% — dùng để biết mô hình "tự tin" bao nhiêu % cho mỗi loại trang phục (xem demo mục 11.1 trong `Fashion_MNIST_Demo.ipynb`).

**11. ReLU là gì, tại sao dùng?**
Hàm kích hoạt đơn giản: giữ nguyên giá trị dương, đưa giá trị âm về 0. Giúp mô hình học được các quan hệ phi tuyến tính (không chỉ là phép cộng/nhân đơn giản), và tính toán nhanh hơn các hàm kích hoạt phức tạp khác.

---

## B. Về cách triển khai cụ thể trong đề tài

**12. Kiến trúc CNN trong đề tài gồm những gì?**
2 khối `Conv → BatchNorm → ReLU` lặp lại 2 lần, rồi `MaxPool → Dropout`, khối đầu 32 kênh, khối sau 64 kênh. Sau đó làm phẳng (Flatten) và qua 1 lớp fully-connected 256 neuron (có BatchNorm + Dropout) trước khi ra 10 lớp đầu ra (xem `src/models.py`).

**13. Kiến trúc MLP trong đề tài gồm những gì?**
3 lớp fully-connected: 784 → 256 → 128 → 10, mỗi lớp ẩn có BatchNorm + Dropout, dùng làm baseline so sánh với CNN.

**14. Tại sao dùng Adam làm optimizer thay vì SGD thường?**
Adam tự động điều chỉnh tốc độ học (learning rate) cho từng tham số dựa trên lịch sử gradient, nên thường hội tụ nhanh và ổn định hơn SGD thường mà không cần tinh chỉnh learning rate thủ công quá nhiều.

**15. Tại sao dùng CrossEntropyLoss?**
Đây là hàm mất mát chuẩn cho bài toán phân loại nhiều lớp (10 lớp). Nó đo mức "sai lệch" giữa xác suất mô hình dự đoán và nhãn thật — mô hình càng tự tin sai thì loss càng cao, thúc đẩy mô hình học đúng hướng.

**16. Tại sao chuẩn hóa (Normalize) dữ liệu ảnh trước khi đưa vào mô hình?**
Đưa giá trị pixel (0-255) về khoảng chuẩn (trừ mean, chia std) giúp quá trình học ổn định hơn, tránh gradient quá lớn/quá nhỏ, và mô hình hội tụ nhanh hơn.

**17. Data Augmentation cụ thể dùng những phép biến đổi nào, tại sao chọn chúng?**
Random horizontal flip, random rotation ±10°, random translate ±10%. Đây là các phép biến đổi nhẹ, hợp lý với quần áo/giày dép (lật ngang 1 chiếc áo vẫn là chiếc áo đó) — **không** dùng lật dọc hay xoay góc lớn vì sẽ tạo ra ảnh phi thực tế (áo lộn ngược, giày úp ngược).

---

## C. Về đánh giá và kết quả

**18. Accuracy, Precision, Recall, F1-score khác nhau thế nào?**
- **Accuracy**: tỉ lệ đoán đúng trên tổng số ảnh.
- **Precision** (của 1 lớp): trong số ảnh mô hình đoán là lớp đó, bao nhiêu % đoán đúng thật.
- **Recall** (của 1 lớp): trong số ảnh thật sự thuộc lớp đó, mô hình tìm ra được bao nhiêu %.
- **F1-score**: trung bình điều hòa của Precision và Recall, dùng khi cần cân bằng cả 2, đặc biệt hữu ích khi các lớp không cân bằng số lượng.

**19. Khi nào Accuracy cao nhưng vẫn cần xem thêm F1-score?**
Khi dữ liệu mất cân bằng lớp (một lớp có quá nhiều mẫu hơn lớp khác), Accuracy có thể cao giả tạo do mô hình đoán thiên về lớp nhiều mẫu. Fashion-MNIST thì các lớp đã cân bằng sẵn (mỗi lớp ~7000 ảnh train), nhưng F1-score vẫn hữu ích để xem từng lớp cụ thể có bị yếu không.

**20. Confusion Matrix đọc thế nào?**
Bảng 10×10, hàng là nhãn thật, cột là nhãn mô hình đoán. Đường chéo là số lượng đoán đúng; các ô ngoài đường chéo là số lượng nhầm lẫn — ô càng đậm màu/số càng lớn thì mô hình càng hay nhầm cặp nhãn đó với nhau.

**21. CNN hay nhầm lẫn giữa những lớp nào? Tại sao?**
Xem biểu đồ "Top 10 cặp nhãn hay nhầm lẫn" trong `Fashion_MNIST_Experiments.ipynb` (mục 12) — thường là các cặp có hình dáng giống nhau khi nhìn ở ảnh xám nhỏ 28×28, ví dụ Shirt / T-shirt/top / Pullover / Coat (đều là trang phục phần thân trên, khó phân biệt tay áo dài/ngắn ở độ phân giải thấp).

**22. Kết quả thực nghiệm ablation study cho thấy điều gì?**
So sánh CNN đầy đủ với 3 phiên bản bỏ từng kỹ thuật (không Augmentation / không Dropout / không BatchNorm) — phiên bản thiếu kỹ thuật có **overfitting gap** (train acc − val acc) cao hơn, chứng minh bằng số liệu thực tế rằng các kỹ thuật này thực sự giúp giảm overfitting, không chỉ là lý thuyết suông (xem mục 11 trong `Fashion_MNIST_Experiments.ipynb`).

**23. CNN "đắt" hơn MLP như thế nào?**
CNN có nhiều tham số hơn và thời gian train mỗi epoch lâu hơn MLP (số liệu cụ thể xem bảng so sánh mục 13 trong `Fashion_MNIST_Experiments.ipynb`) — đổi lại accuracy cao hơn. Đây là đánh đổi (trade-off) giữa độ chính xác và chi phí tính toán.

---

## D. Câu hỏi mở rộng / phản biện

**24. Nếu tăng số lớp Conv (mạng sâu hơn) thì kết quả có tốt hơn không?**
Có thể tốt hơn đến 1 mức nào đó vì mô hình học được đặc trưng phức tạp hơn, nhưng cũng dễ overfitting hơn (nhất là với bộ dữ liệu nhỏ, ảnh đơn giản như Fashion-MNIST) và tốn thời gian train hơn — cần cân bằng giữa độ phức tạp và lượng dữ liệu có sẵn.

**25. Mô hình có áp dụng được vào bài toán thực tế không?**
Về nguyên lý có, ví dụ hệ thống gợi ý thời trang, phân loại sản phẩm trong thương mại điện tử. Nhưng cần lưu ý mô hình hiện tại chỉ học trên ảnh 28×28 xám, nền đơn giản — với ảnh thật (màu, nền phức tạp, nhiều góc chụp) cần train lại với dữ liệu đa dạng hơn hoặc dùng mô hình pretrained lớn hơn (transfer learning).

**26. Hạn chế của đề tài hiện tại là gì?**
- Ảnh nhỏ, xám, nền đơn giản → không phản ánh ảnh thực tế phức tạp.
- Kiến trúc CNN còn đơn giản (2 khối Conv), có thể chưa khai thác hết tiềm năng của CNN.
- Chưa thử các kỹ thuật augmentation nâng cao hơn (CutMix, MixUp) hoặc kiến trúc hiện đại (ResNet, EfficientNet).

**27. Vì sao không dùng mô hình pretrained (transfer learning) luôn cho nhanh?**
Vì đề tài tập trung vào việc **tự xây dựng và hiểu** kiến trúc CNN cơ bản, so sánh với MLP, và phân tích tác dụng của từng kỹ thuật cải thiện — mục tiêu học thuật khác với mục tiêu đạt accuracy cao nhất có thể trong thực tế.

**28. Nếu ảnh test bị mờ/nhiễu thì mô hình có còn đoán đúng không?**
Data Augmentation hiện tại (xoay, lật, tịnh tiến) không bao gồm làm mờ/nhiễu, nên mô hình có thể nhạy cảm với ảnh chất lượng kém. Đây là hướng cải thiện tiềm năng: thêm augmentation dạng nhiễu (Gaussian noise, blur) nếu muốn mô hình chịu được ảnh thực tế kém chất lượng hơn.

---

## Checklist trước khi vào phòng vấn đáp

- [ ] Đã chạy xong `Fashion_MNIST_CNN.ipynb` và `Fashion_MNIST_Experiments.ipynb` trên Colab, có số liệu thật (không phải số liệu mẫu)
- [ ] Đã mở sẵn `Fashion_MNIST_Demo.ipynb`, chạy thử mục 11 (demo) ít nhất 1 lần để chắc chạy được
- [ ] Chuẩn bị sẵn 1-2 ảnh chụp thật (điện thoại) để demo mục 11.3 nếu được yêu cầu
- [ ] Đọc lại bảng so sánh MLP vs CNN và bảng ablation study, nhớ nằm lòng vài con số chính (accuracy, overfitting gap)
- [ ] Đọc qua tài liệu này 1 lượt, đặc biệt mục A (khái niệm cơ bản) — đây là nhóm câu hỏi dễ bị hỏi nhất
