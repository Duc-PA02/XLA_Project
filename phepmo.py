import cv2
import numpy as np
import matplotlib.pyplot as plt

def image_opening(image_path, kernel_size):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Kiểm tra xem ảnh có tồn tại không
    if img is None:
        print("Không thể đọc được ảnh.")
        return

    # Tạo kernel cho phép mở ảnh
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Áp dụng phép mở ảnh
    img_opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # Hiển thị ảnh gốc và ảnh sau phép mở ảnh
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh Gốc')

    plt.subplot(1, 2, 2)
    plt.imshow(img_opened, cmap='gray')
    plt.title(f'Ảnh Sau Phép Mở Ảnh (kernel_size={kernel_size})')

    plt.show()


image_path = 'image/test.png'  
kernel_size = 50  # Thay đổi kích thước kernel để thử nghiệm
image_opening(image_path, kernel_size)
