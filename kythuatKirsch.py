import cv2
import numpy as np
import matplotlib.pyplot as plt

def kirsch_edge_detection(image_path):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Kiểm tra xem ảnh có tồn tại không
    if img is None:
        print("Không thể đọc được ảnh.")
        return

    # Bộ lọc Kirsch theo 8 hướng khác nhau
    kirsch_filters = [
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])
    ]

    # Áp dụng bộ lọc Kirsch
    kirsch_edges = np.zeros_like(img, dtype=np.float32)
    for kernel in kirsch_filters:
        filtered = cv2.filter2D(img, cv2.CV_64F, kernel)
        kirsch_edges = np.maximum(kirsch_edges, filtered)

    # Chuyển đổi giá trị về 8-bit unsigned integer
    kirsch_edges = np.uint8(np.abs(kirsch_edges))

    # Hiển thị ảnh gốc và ảnh sau khi áp dụng Kỹ thuật Kirsch
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh Gốc')

    plt.subplot(1, 2, 2)
    plt.imshow(kirsch_edges, cmap='gray')
    plt.title('Ảnh Sau Kỹ thuật Kirsch')

    plt.show()


image_path = 'image/test.png'  
kirsch_edge_detection(image_path)
