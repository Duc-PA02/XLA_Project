import cv2
import numpy as np
import matplotlib.pyplot as plt

def prewitt_edge_detection(image_path):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Kiểm tra xem ảnh có tồn tại không
    if img is None:
        print("Không thể đọc được ảnh.")
        return

    # Áp dụng bộ lọc Prewitt theo hướng ngang và hướng dọc
    prewitt_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    prewitt_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Tính toán biên bằng cách kết hợp biên từ cả hai hướng
    prewitt_edges = np.sqrt(prewitt_x**2 + prewitt_y**2)

    # Hiển thị ảnh gốc và ảnh sau khi áp dụng Kỹ thuật Prewitt
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh Gốc')

    plt.subplot(1, 2, 2)
    plt.imshow(prewitt_edges, cmap='gray')
    plt.title('Ảnh Sau Kỹ thuật Prewitt')

    plt.show()


image_path = 'image/test.png'  
prewitt_edge_detection(image_path)
