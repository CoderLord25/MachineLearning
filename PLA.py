import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(2)

# Tạo hai cụm dữ liệu trong không gian 2 chiều, mỗi cụm có 10 điểm dữ liệu
means = [[2, 2], [4, 4]]  # Tọa độ trung tâm của các cụm (phân biệt rõ hơn để dễ phân tách tuyến tính)
cov = [[0.3, 0.2], [0.2, 0.3]]  # Ma trận hiệp phương sai cho mỗi cụm
N = 10  # Số lượng điểm dữ liệu trong mỗi cụm
X0 = np.random.multivariate_normal(means[0], cov, N).T  # Cụm 1
X1 = np.random.multivariate_normal(means[1], cov, N).T  # Cụm 2

# Kết hợp dữ liệu từ hai cụm
X = np.concatenate((X0, X1), axis=1)
y = np.concatenate((np.ones((1, N)), -1 * np.ones((1, N))), axis=1)  # Nhãn: 1 cho cụm đầu tiên, -1 cho cụm thứ hai

# Thêm một hàng của các số 1 vào X để tính toán cho bias
X = np.concatenate((np.ones((1, 2 * N)), X), axis=0)

# Hàm để tính giá trị dự đoán dựa trên trọng số w và dữ liệu x
def h(w, x):    
    return np.sign(np.dot(w.T, x))

# Hàm kiểm tra xem mô hình đã hội tụ chưa
def has_converged(X, y, w):
    return np.array_equal(h(w, X), y)  # True nếu h(w, X) == y, ngược lại là False

# Thuật toán Perceptron
def perceptron(X, y, w_init):
    w = [w_init]
    N = X.shape[1]
    mis_points = []
    while True:
        # Trộn dữ liệu ngẫu nhiên
        mix_id = np.random.permutation(N)
        for i in range(N):
            xi = X[:, mix_id[i]].reshape(3, 1)
            yi = y[0, mix_id[i]]
            if h(w[-1], xi)[0] != yi:
                mis_points.append(mix_id[i])
                w_new = w[-1] + yi * xi  # Cập nhật w dựa trên xi và yi
                w.append(w_new)
                
        if has_converged(X, y, w[-1]):
            break
    return (w, mis_points)

# Khởi tạo trọng số ngẫu nhiên
d = X.shape[0]
w_init = np.random.randn(d, 1)
(w, m) = perceptron(X, y, w_init)

# Hàm để vẽ đường phân chia dựa trên trọng số w
def draw_line(w):
    w0, w1, w2 = w[0], w[1], w[2]
    if w2 != 0:
        x11, x12 = -100, 100
        return plt.plot([x11, x12], [-(w1 * x11 + w0) / w2, -(w1 * x12 + w0) / w2], 'k')
    else:
        x10 = -w0 / w1
        return plt.plot([x10, x10], [-100, 100], 'k')

# Visualize quá trình phân lớp
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation 

def viz_alg_1d_2(w):
    it = len(w)    
    fig, ax = plt.subplots(figsize=(5, 5))  
    
    def update(i):
        ani = plt.cla()
        # Vẽ các điểm dữ liệu
        ani = plt.plot(X0[0, :], X0[1, :], 'b^', markersize=8, alpha=0.8)
        ani = plt.plot(X1[0, :], X1[1, :], 'ro', markersize=8, alpha=0.8)
        ani = plt.axis([0, 6, 0, 6])  # Thiết lập phạm vi trục x và y để dễ nhìn hơn
        i2 = i if i < it else it - 1
        ani = draw_line(w[i2])  # Vẽ đường phân chia
        if i < it - 1:
            # Vẽ một điểm bị phân loại sai (misclassified point)
            circle = plt.Circle((X[1, m[i]], X[2, m[i]]), 0.15, color='k', fill=False)
            ax.add_artist(circle)
        # Ẩn các trục tọa độ
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_ticks([])
        cur_axes.axes.get_yaxis().set_ticks([])

        label = f'PLA: iter {i2}/{it - 1}'
        ax.set_xlabel(label)
        return ani, ax 
        
    anim = FuncAnimation(fig, update, frames=np.arange(0, it + 2), interval=1000)
    # Lưu ảnh động thành file GIF
    anim.save('pla_vis.gif', dpi=100, writer='imagemagick')
    plt.show()
    
viz_alg_1d_2(w)