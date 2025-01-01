import pandas as pd
import numpy as np
from graphviz import Digraph

# Tạo DataFrame với dữ liệu đã cho
data = {
    'age': ['<=30', '<=30', '31...40', '>40', '>40', '>40', '31...40', '<=30', '<=30', '>40', '<=30', '31...40', '31...40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'fair', 'fair', 'excellent', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}
df = pd.DataFrame(data)

# Hàm tính Entropy của thuộc tính đích
def calculate_entropy(data, target_column):
    values, counts = np.unique(data[target_column], return_counts=True)
    entropy = -np.sum([(counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(values))])
    return entropy

# Hàm tính Entropy có điều kiện của từng đặc trưng
def calculate_attribute_entropy(data, attribute, target_column):
    values, counts = np.unique(data[attribute], return_counts=True)
    attribute_entropy = np.sum([
        (counts[i] / np.sum(counts)) * calculate_entropy(data[data[attribute] == values[i]], target_column)
        for i in range(len(values))
    ])
    return attribute_entropy

# Hàm tính Information Gain cho từng đặc trưng
def calculate_information_gain(data, attribute, target_column):
    total_entropy = calculate_entropy(data, target_column)
    attribute_entropy = calculate_attribute_entropy(data, attribute, target_column)
    information_gain = total_entropy - attribute_entropy
    return information_gain

# Tính Entropy tổng của thuộc tính đích
target_column = 'buys_computer'
total_entropy = calculate_entropy(df, target_column)
print(f"Entropy của thuộc tính đích '{target_column}': {total_entropy:.4f}")

# Tính Information Gain cho từng đặc trưng
attributes = ['age', 'income', 'student', 'credit_rating']
information_gains = {attribute: calculate_information_gain(df, attribute, target_column) for attribute in attributes}

# In ra Entropy và Information Gain của từng đặc trưng
for attribute, ig in information_gains.items():
    attribute_entropy = calculate_attribute_entropy(df, attribute, target_column)
    print(f"Entropy của '{attribute}': {attribute_entropy:.4f}")
    print(f"Information Gain của '{attribute}': {ig:.4f}\n")

# Chọn đặc trưng có Information Gain cao nhất làm nút gốc
root_attribute = max(information_gains, key=information_gains.get)

# Khởi tạo đồ thị bằng graphviz
dot = Digraph()

# Hàm đệ quy để xây dựng cây quyết định
def build_tree(data, features, target_attribute, dot, parent=None, edge_label=""):
    # Nếu tất cả các mẫu đều có cùng giá trị của thuộc tính đích, trả về giá trị đó
    if len(np.unique(data[target_attribute])) == 1:
        decision = np.unique(data[target_attribute])[0]
        dot.node(name=decision, label=decision, shape='box')
        if parent:
            dot.edge(parent, decision, label=edge_label)
        return

    # Nếu không còn đặc trưng nào để phân chia, chọn nhãn phổ biến nhất
    if len(features) == 0:
        decision = data[target_attribute].value_counts().idxmax()
        dot.node(name=decision, label=decision, shape='box')
        if parent:
            dot.edge(parent, decision, label=edge_label)
        return

    # Tính toán Information Gain để chọn đặc trưng tốt nhất
    gains = {feature: calculate_information_gain(data, feature, target_attribute) for feature in features}
    best_feature = max(gains, key=gains.get)

    # Tạo nút cho đặc trưng tốt nhất
    dot.node(name=best_feature, label=best_feature)
    if parent:
        dot.edge(parent, best_feature, label=edge_label)

    # Phân chia dữ liệu và xây dựng các nhánh con
    for value in np.unique(data[best_feature]):
        subset = data[data[best_feature] == value]
        new_features = [f for f in features if f != best_feature]
        build_tree(subset, new_features, target_attribute, dot, parent=best_feature, edge_label=str(value))

# Gọi hàm để xây dựng cây từ nút gốc
build_tree(df, attributes, target_column, dot)

# Lưu và hiển thị cây
dot.render("decision_tree", format="png", cleanup=False)
dot.view()