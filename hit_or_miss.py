import cv2
import numpy as np
import matplotlib.pyplot as plt

def hit_or_miss_transform(image_path, kernel):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Kiểm tra xem ảnh có tồn tại không
    if img is None:
        print("Không thể đọc được ảnh.")
        return

    # Áp dụng phép biến đổi Hit-or-Miss
    result = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)

    # Hiển thị ảnh gốc và ảnh sau phép biến đổi Hit-or-Miss
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh Gốc')

    plt.subplot(1, 2, 2)
    plt.imshow(result, cmap='gray')
    plt.title('Kết Quả Phép Hit-or-Miss')

    plt.show()


image_path = 'image/test.png'  

# Định nghĩa kernel cho phép biến đổi Hit-or-Miss
kernel = np.array([[0, 1, 0],
                   [-1, 1, 1],
                   [0, 1, 0]], dtype=np.int8)

# Gọi hàm để thực hiện phép biến đổi Hit-or-Miss
hit_or_miss_transform(image_path, kernel)
