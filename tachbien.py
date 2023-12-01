import cv2
import numpy as np
import matplotlib.pyplot as plt

def edge_detection(image_path):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Kiểm tra xem ảnh có tồn tại không
    if img is None:
        print("Không thể đọc được ảnh.")
        return

    # Áp dụng bộ lọc Sobel để tách biên
    edges = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=9)

    # Hiển thị ảnh gốc và ảnh sau tách biên
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh Gốc')

    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Ảnh Sau Tách Biên (Sobel)')

    plt.show()


image_path = 'image/test.png'  
edge_detection(image_path)
