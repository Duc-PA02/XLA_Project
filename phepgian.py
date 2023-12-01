import cv2
import numpy as np
import matplotlib.pyplot as plt

def contrast_stretching(image_path, min_output, max_output):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Kiểm tra xem ảnh có tồn tại không
    if img is None:
        print("Không thể đọc được ảnh.")
        return

    # Normalization
    img_normalized = cv2.normalize(img, None, min_output, max_output, cv2.NORM_MINMAX)

    # Hiển thị ảnh gốc và ảnh sau khi biến đổi giãn ảnh
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh Gốc')

    plt.subplot(1, 2, 2)
    plt.imshow(img_normalized, cmap='gray')
    plt.title('Ảnh Sau Biến Đổi Giãn Ảnh')

    plt.show()


image_path = 'image/test.png'  
min_output = 0  # Giá trị đầu ra tối thiểu
max_output = 255  # Giá trị đầu ra tối đa
contrast_stretching(image_path, min_output, max_output)
