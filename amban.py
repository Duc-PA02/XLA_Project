import cv2
import numpy as np
import matplotlib.pyplot as plt

def basic_histogram_equalization(image_path):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Kiểm tra xem ảnh có tồn tại không
    if img is None:
        print("Không thể đọc được ảnh.")
        return

    # Biến đổi âm bản
    equalized_img = cv2.equalizeHist(img)
    

    # Hiển thị ảnh gốc và ảnh đã được biến đổi âm bản
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh Gốc')

    plt.subplot(1, 2, 2)
    plt.imshow(equalized_img, cmap='gray')
    plt.title('Biến Đổi Âm Bản')

    plt.show()

image_path = 'image/amban.png'
basic_histogram_equalization(image_path)

