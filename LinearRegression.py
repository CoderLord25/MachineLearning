import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Đọc dữ liệu
heart_disease = pd.read_csv('C:\\Users\\ASUS VIVOBOOK\\Downloads\\heart_2020_cleaned.csv')

# Chuyển đổi cột từ 'Yes'/'No' thành 1/0
heart_disease['HeartDisease'] = heart_disease['HeartDisease'].map({'Yes': 1, 'No': 0})
heart_disease['Smoking'] = heart_disease['Smoking'].map({'Yes': 1, 'No': 0})
heart_disease['AlcoholDrinking'] = heart_disease['AlcoholDrinking'].map({'Yes': 1, 'No': 0})
heart_disease['PhysicalActivity'] = heart_disease['PhysicalActivity'].map({'Yes': 1, 'No': 0})
heart_disease['Asthma'] = heart_disease['Asthma'].map({'Yes': 1, 'No': 0})

# Chuẩn bị dữ liệu (sử dụng các biến quan trọng)
X = heart_disease[['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime', 'Smoking', 'AlcoholDrinking', 'PhysicalActivity', 'Asthma']]
y = heart_disease['HeartDisease']

# Thêm cột '1' vào X để đại diện cho hệ số chặn (intercept)
X = np.hstack([np.ones((X.shape[0], 1)), X])

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Hàm mất mát (loss function) theo công thức:
def loss_function(X, y, w):
    N = len(y)
    return (1 / (2 * N)) * np.sum((y - X @ w) ** 2)

# Gradient của hàm mất mát
def compute_gradient(X, y, w):
    N = len(y)
    return -(1 / N) * X.T @ (y - X @ w)

# Hàm huấn luyện mô hình bằng Gradient Descent
def gradient_descent(X, y, w_init, learning_rate=0.01, iterations=1000):
    w = w_init
    for i in range(iterations):
        gradient = compute_gradient(X, y, w)
        w -= learning_rate * gradient
        if i % 100 == 0:  # In giá trị hàm mất mát mỗi 100 iterations
            print(f"Iteration {i}: Loss = {loss_function(X, y, w)}")
    return w

# Khởi tạo trọng số ban đầu (hệ số w)
w_init = np.zeros(X_train.shape[1])

# Huấn luyện mô hình
learning_rate = 0.01
iterations = 1000
w_optimal = gradient_descent(X_train, y_train, w_init, learning_rate, iterations)

# Dự đoán kết quả trên tập kiểm tra
y_pred_manual = X_test @ w_optimal

# Tính Mean Squared Error (MSE) thủ công
def mse(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2) / len(y_true)

# Tính MSE
mse_value = mse(y_test, y_pred_manual)

# Tính giá trị trung bình của y_test
y_mean = np.mean(y_test)

# Tính R-squared (R²) thủ công
rss = np.sum((y_test - y_pred_manual) ** 2)  # Residual Sum of Squares (RSS)
tss = np.sum((y_test - y_mean) ** 2)         # Total Sum of Squares (TSS)
r2_manual = 1 - (rss / tss)

print(f"Optimal weights: {w_optimal}")
print(f"Mean Squared Error (MSE) : {mse_value}")
print(f"R-squared : {r2_manual}")
