import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Bước 1: Nạp dữ liệu từ file CSV
df = pd.read_csv(r"C:\Users\ASUS VIVOBOOK\Downloads\heart_2020_cleaned.csv")

# Bước 2: Chỉ lấy các cột cần thiết cho phân cụm
columns_for_clustering = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
df_clustering = df[columns_for_clustering]

# Bước 3: Chuẩn hóa (standardize) các cột số
scaler = StandardScaler()
X = scaler.fit_transform(df_clustering)

# Bước 4: Khởi tạo tâm cụm ban đầu
np.random.seed(42)  # Đặt seed để tái tạo kết quả
k = 5  # Số cụm
centroids = X[np.random.choice(X.shape[0], k, replace=False)]  # Chọn ngẫu nhiên 3 điểm làm tâm cụm

# Hàm tính khoảng cách Euclidean
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Bước 5: Thuật toán K-Means
def kmeans(X, centroids, k, max_iters=100):
    for _ in range(max_iters):
        # Gán mỗi điểm dữ liệu vào cụm gần nhất
        clusters = [[] for _ in range(k)]
        for x in X:
            distances = [euclidean_distance(x, centroid) for centroid in centroids]
            closest_cluster = np.argmin(distances)
            clusters[closest_cluster].append(x)
        
        # Lưu lại tâm cụm trước khi cập nhật
        previous_centroids = centroids.copy()

        # Cập nhật lại các tâm cụm (centroids)
        for i in range(k):
            if clusters[i]:  # Kiểm tra nếu cụm không rỗng
                centroids[i] = np.mean(clusters[i], axis=0)
        
        # Kiểm tra nếu tâm cụm không thay đổi (hội tụ)
        if np.all(centroids == previous_centroids):
            print(f"K-Means hội tụ sau {_} lần lặp.")
            break
    
    return centroids, clusters

# Bước 6: Thực hiện K-Means
centroids, clusters = kmeans(X, centroids, k)

# Hiển thị kết quả
print("Tâm cụm cuối cùng:")
print(centroids)

print("Số lượng điểm trong mỗi cụm:")
for i, cluster in enumerate(clusters):
    print(f"Cụm {i+1}: {len(cluster)} điểm")