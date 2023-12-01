import cv2
import numpy as np
import matplotlib.pyplot as plt

def power_transform(image_path, gamma=1.0):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Kiểm tra xem ảnh có tồn tại không
    if img is None:
        print("Không thể đọc được ảnh.")
        return

    # Normalize giá trị pixel về đoạn [0, 1]
    normalized_img = img / 255.0

    # Áp dụng phép biến đổi hàm mũ
    power_transformed_img = np.power(normalized_img, gamma)

    # Scale lại giá trị pixel về đoạn [0, 255]
    power_transformed_img = (power_transformed_img * 255).astype(np.uint8)

    # Hiển thị ảnh gốc và ảnh đã được biến đổi
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh Gốc')

    plt.subplot(1, 2, 2)
    plt.imshow(power_transformed_img, cmap='gray')
    plt.title('Biến Đổi Hàm Mũ')

    plt.show()


image_path = 'image/hammu.png'  
power_transform(image_path, gamma=1.5)
