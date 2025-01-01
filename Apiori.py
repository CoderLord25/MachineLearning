import pandas as pd
from collections import defaultdict
from itertools import combinations

# Đọc dữ liệu từ file CSV và chuyển đổi thành danh sách các giao dịch
df = pd.read_csv(r"C:\Users\ASUS VIVOBOOK\Downloads\test1.csv")
transactions = [{item for item in df.columns if row[item] == 't'} for _, row in df.iterrows()]

# Hàm tính độ hỗ trợ cho một tập hợp mục
def get_support(item_set):
    return sum(1 for transaction in transactions if item_set.issubset(transaction)) / len(transactions)

# Thuật toán Apriori
def apriori(transactions, min_support):
    frequent_item_sets = defaultdict(list)
    # Tìm các tập hợp mục đơn lẻ đạt ngưỡng hỗ trợ
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[frozenset([item])] += 1
    frequent_item_sets[1] = [(item, get_support(item)) for item in item_counts if get_support(item) >= min_support]

    # Sinh tập hợp mục kết hợp từ các tập hợp thường xuyên trước đó
    level = 2
    while True:
        current_level = [set1 | set2 for i, (set1, _) in enumerate(frequent_item_sets[level-1])
                         for set2, _ in frequent_item_sets[level-1][i+1:] if len(set1 | set2) == level]
        next_level = [(item, get_support(item)) for item in current_level if get_support(item) >= min_support]
        if not next_level:
            break
        frequent_item_sets[level] = next_level
        level += 1
    return frequent_item_sets

# Tạo luật kết hợp
def generate_rules(frequent_item_sets, min_confidence):
    rules = []
    for level in frequent_item_sets:
        for item_set, support in frequent_item_sets[level]:
            for i in range(1, len(item_set)):
                for antecedent in map(frozenset, combinations(item_set, i)):
                    consequent = item_set - antecedent
                    confidence = support / get_support(antecedent)
                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, confidence))
    return rules

# Thiết lập ngưỡng hỗ trợ và độ tin cậy
min_support = 0.2
min_confidence = 0.6

# Chạy thuật toán và tạo luật kết hợp
frequent_item_sets = apriori(transactions, min_support)
rules = generate_rules(frequent_item_sets, min_confidence)

# Hiển thị các tập hợp mục thường xuyên
for level, item_sets in frequent_item_sets.items():
    print(f"Tập hợp mục mức {level}:")
    for item_set, support in item_sets:
        print(f"  {set(item_set)} (độ hỗ trợ: {support:.2f})")

# Hiển thị các luật kết hợp
print("\nNumber of rules:", len(rules))
for antecedent, consequent, confidence in rules:
    print(f"{set(antecedent)} -> {set(consequent)} (confidence: {confidence:.2f})")