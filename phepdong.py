import cv2
import numpy as np
import matplotlib.pyplot as plt

def image_closing(image_path, kernel_size):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Kiểm tra xem ảnh có tồn tại không
    if img is None:
        print("Không thể đọc được ảnh.")
        return

    # Tạo kernel cho phép đóng ảnh
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Áp dụng phép đóng ảnh
    img_closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Hiển thị ảnh gốc và ảnh sau phép đóng ảnh
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh Gốc')

    plt.subplot(1, 2, 2)
    plt.imshow(img_closed, cmap='gray')
    plt.title(f'Ảnh Sau Phép Đóng Ảnh (kernel_size={kernel_size})')

    plt.show()


image_path = 'image/test.png'  
kernel_size = 100  # Thay đổi kích thước kernel để thử nghiệm
image_closing(image_path, kernel_size)
