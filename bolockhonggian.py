import cv2
import numpy as np
import matplotlib.pyplot as plt

def spatial_filtering(image_path, kernel):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Kiểm tra xem ảnh có tồn tại không
    if img is None:
        print("Không thể đọc được ảnh.")
        return

    # Áp dụng bộ lọc không gian
    filtered_img = cv2.filter2D(img, -1, kernel)

    # Hiển thị ảnh gốc và ảnh đã được xử lý
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh Gốc')

    plt.subplot(1, 2, 2)
    plt.imshow(filtered_img, cmap='gray')
    plt.title('Ảnh Sau Bộ Lọc')

    plt.show()


image_path = 'image/bolockhonggian.png' 

# Kernel là một ma trận, ví dụ: làm mịn bằng bộ lọc trung bình 3x3
kernel = np.ones((3, 3), np.float32) / 9.0
kernel1 = np.ones((6, 6), np.float32) / 12.0

spatial_filtering(image_path, kernel)
spatial_filtering(image_path,kernel1)
