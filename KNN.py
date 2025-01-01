import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Bước 1: Đọc dữ liệu (giả sử bạn có file CSV đã được lưu)
# Thay 'duong_dan_toi_file.csv' bằng đường dẫn thật của file CSV
data = pd.read_csv('C:\\Users\\ASUS VIVOBOOK\\Downloads\\heart_2020_cleaned.csv', low_memory=False)

# Bước 2: Tiền xử lý dữ liệu
# Các cột phân loại cần được mã hóa (chuyển đổi từ chuỗi thành số)
categorical_cols = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 
                    'Sex', 'AgeCategory', 'Race', 'Diabetic', 
                    'PhysicalActivity', 'GenHealth', 'Asthma', 
                    'KidneyDisease', 'SkinCancer']

# Sử dụng LabelEncoder để mã hóa các cột phân loại
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Bước 3: Chia dữ liệu thành features và label
X = data.drop(columns=['HeartDisease'])  # Bỏ cột dự đoán (HeartDisease)
y = data['HeartDisease']  # Cột nhãn cần dự đoán (HeartDisease)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bước 4: Tính khoảng cách Euclidean giữa hai điểm
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Bước 5: Hàm KNN tự viết
def knn_predict(X_train, y_train, X_test, k=5):
    predictions = []
    
    # Duyệt qua mỗi điểm cần dự đoán trong tập kiểm tra
    for test_point in X_test:
        distances = []
        
        # Tính khoảng cách từ test_point đến mỗi điểm trong tập huấn luyện
        for i in range(len(X_train)):
            distance = euclidean_distance(np.array(X_train.iloc[i]), np.array(test_point))
            distances.append((distance, y_train.iloc[i]))
        
        # Sắp xếp khoảng cách theo thứ tự tăng dần
        distances.sort(key=lambda x: x[0])
        
        # Lấy ra k hàng xóm gần nhất
        k_nearest_neighbors = distances[:k]
        
        # Dự đoán nhãn bằng cách chọn nhãn phổ biến nhất từ k hàng xóm gần nhất
        k_nearest_labels = [label for (dist, label) in k_nearest_neighbors]
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        
        # Thêm dự đoán vào danh sách
        predictions.append(most_common)
    
    return predictions

# Bước 6: Áp dụng mô hình KNN tự viết và dự đoán
k = 7  # Chọn k hàng xóm gần nhất
y_pred = knn_predict(X_train, y_train, X_test.values, k)

# Bước 7: Tính độ chính xác (Accuracy)
accuracy = np.sum(y_pred == y_test.values) / len(y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')