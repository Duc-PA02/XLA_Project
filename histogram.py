import cv2
import numpy as np
import matplotlib.pyplot as plt

def threshold_segmentation(image_path, threshold_value):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Kiểm tra xem ảnh có tồn tại không
    if img is None:
        print("Không thể đọc được ảnh.")
        return

    # Áp dụng phân đoạn theo ngưỡng
    _, segmented_image = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)

    # Hiển thị ảnh gốc và ảnh sau phân đoạn
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh Gốc')

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image, cmap='gray')
    plt.title(f'Phân đoạn theo ngưỡng {threshold_value}')

    plt.show()


image_path = 'image/test.png'  
threshold_value = 128  # Thay đổi giá trị ngưỡng tùy ý
threshold_segmentation(image_path, threshold_value)
