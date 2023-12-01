import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_log_transform(image_path, c=1):
    # Đọc ảnh
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Kiểm tra xem ảnh có hợp lệ không
    if image is None:
        print("Không thể đọc ảnh.")
        return

    # Áp dụng biến đổi logarith
    log_transformed = c * np.log1p(image)

    # Chuyển đổi kiểu dữ liệu về uint8 để có thể hiển thị bằng matplotlib
    log_transformed = np.uint8(log_transformed)

    # Hiển thị ảnh gốc và ảnh sau biến đổi logarith
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Ảnh Gốc')

    plt.subplot(1, 2, 2)
    plt.imshow(log_transformed, cmap='gray')
    plt.title('Biến Đổi Logarith')

    plt.show()

# Đường dẫn đến ảnh
image_path = 'image/logarith.png'

# Hệ số c có thể điều chỉnh theo nhu cầu, thường là giá trị dương
c_value = 1

# Áp dụng biến đổi logarith
apply_log_transform(image_path, c_value)
