import cv2
import numpy as np
import matplotlib.pyplot as plt

def morphological_operation(image_path):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Kiểm tra xem ảnh có tồn tại không
    if img is None:
        print("Không thể đọc được ảnh.")
        return

    # Áp dụng phép co (dilation) để làm nổi bật các cạnh
    kernel = np.ones((5, 5), np.uint8)  # Kích thước kernel
    dilated_img = cv2.dilate(img, kernel, iterations=1)

    # Hiển thị ảnh gốc và ảnh sau phép co
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh Gốc')

    plt.subplot(1, 2, 2)
    plt.imshow(dilated_img, cmap='gray')
    plt.title('Ảnh Sau Phép Co')

    plt.show()


image_path = 'image/test.png'  
morphological_operation(image_path)
