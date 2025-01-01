import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer

# Đọc dữ liệu từ file CSV
df = pd.read_csv("C:\\Users\\ASUS VIVOBOOK\\Downloads\\heart_2020_cleaned.csv")

# Rescaling theo Min-Max 

 # Chỉ lấy hai cột cần scaling
columns_to_scale = ['BMI', 'PhysicalHealth']
data_to_scale = df[columns_to_scale]

 # Thực hiện scaling
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_to_scale)

 # Gán lại giá trị đã scaling vào DataFrame ban đầu
df[columns_to_scale] = data_scaled

 # In ra kết quả sau khi scaling hai cột
print(df)

# Rescaling theo Z-Score 

 # Chỉ lấy hai cột cần scaling
columns_to_scale = ['BMI', 'PhysicalHealth']
data_to_scale = df[columns_to_scale]

 # Thực hiện standardization
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_to_scale)

 # Gán lại giá trị đã standardization vào DataFrame ban đầu
df[columns_to_scale] = data_scaled

 # In ra kết quả sau khi standardization hai cột
print(df)

# Rescaling theo unit length

 # Chỉ lấy hai cột cần scaling
columns_to_scale = ['BMI', 'PhysicalHealth']
data_to_scale = df[columns_to_scale]

 # Thực hiện scaling theo unit length
scaler = Normalizer()
data_scaled = scaler.fit_transform(data_to_scale)

 # Gán lại giá trị đã scaling vào DataFrame ban đầu
df[columns_to_scale] = data_scaled

 # In ra kết quả sau khi scaling theo unit length cho hai cột
print(df)