import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Hàm tính log của xác suất của biến định lượng theo phân phối Gaussian
def gaussian_log_probability(x, mean, std):
    # Thêm kiểm tra để tránh chia cho 0
    if std == 0:
        std = 1e-6
    exponent = -((x - mean) ** 2) / (2 * std ** 2)
    return exponent - np.log(std) - 0.5 * np.log(2 * np.pi)

# Hàm dự đoán dựa trên các đặc trưng đã chọn
def predict_naive_bayes_selected(sample, prior_log_prob, likelihood_class_0, likelihood_class_1):
    # Danh sách các đặc trưng được sử dụng để dự đoán
    selected_features = ['BMI', 'SleepTime', 'PhysicalHealth', 'MentalHealth']
    
    # Khởi tạo log-xác suất cho mỗi lớp bằng log của xác suất tiên nghiệm
    log_prob_class_0 = prior_log_prob[0]
    log_prob_class_1 = prior_log_prob[1]
    
    # Tính toán log-xác suất có điều kiện cho các đặc trưng đã chọn
    for column in selected_features:
        value = sample[column]
        
        # Tính log-xác suất cho từng lớp
        log_prob_class_0 += gaussian_log_probability(value, likelihood_class_0[column]['mean'], likelihood_class_0[column]['std'])
        log_prob_class_1 += gaussian_log_probability(value, likelihood_class_1[column]['mean'], likelihood_class_1[column]['std'])

    # So sánh log-xác suất để đưa ra dự đoán
    if log_prob_class_0 > log_prob_class_1:
        return 0  # HeartDisease = 0
    else:
        return 1  # HeartDisease = 1

# Hàm tính độ chính xác của mô hình
def accuracy(predictions, actual):
    correct = (predictions == actual).sum()
    return correct / len(actual)

# Đọc dữ liệu từ file CSV
data = pd.read_csv(r"C:\Users\ASUS VIVOBOOK\Downloads\heart_2020_cleaned.csv")

# Tiền xử lý dữ liệu: Chuyển đổi các cột phân loại thành số
label_encoder = LabelEncoder()
encoded_data = data.copy()

# Áp dụng LabelEncoder cho các cột phân loại
for column in encoded_data.select_dtypes(include=['object']).columns:
    encoded_data[column] = label_encoder.fit_transform(encoded_data[column])

# Tính log của xác suất tiên nghiệm (prior probabilities)
prior_prob = encoded_data['HeartDisease'].value_counts(normalize=True)
prior_log_prob = np.log(prior_prob)

# Tách dữ liệu theo lớp
class_0 = encoded_data[encoded_data['HeartDisease'] == 0]
class_1 = encoded_data[encoded_data['HeartDisease'] == 1]

# Tính toán xác suất có điều kiện (mean và std) cho mỗi đặc trưng trong mỗi lớp
likelihood_class_0 = {}
likelihood_class_1 = {}

selected_features = ['BMI', 'SleepTime', 'PhysicalHealth', 'MentalHealth']

for column in selected_features:
    likelihood_class_0[column] = {
        'mean': class_0[column].mean(),
        'std': class_0[column].std()
    }
    likelihood_class_1[column] = {
        'mean': class_1[column].mean(),
        'std': class_1[column].std()
    }

# Dự đoán trên toàn bộ tập dữ liệu
predictions = []
for i in range(len(encoded_data)):
    sample_data = encoded_data.iloc[i]
    predicted_label = predict_naive_bayes_selected(sample_data, prior_log_prob, likelihood_class_0, likelihood_class_1)
    predictions.append(predicted_label)

# Chuyển danh sách dự đoán thành mảng numpy
predictions = np.array(predictions)

# Tính độ chính xác của mô hình
actual_labels = encoded_data['HeartDisease'].values
acc = accuracy(predictions, actual_labels)

print(f'Độ chính xác của mô hình: {acc * 100:.2f}%')

# In ra một số kết quả dự đoán và nhãn thực tế để kiểm tra
for i in range(12):
    print(f'Mẫu {i+1}: Nhãn dự đoán = {predictions[i]}, Nhãn thực tế = {actual_labels[i]}')
