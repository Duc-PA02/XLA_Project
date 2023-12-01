import cv2
import numpy as np
import matplotlib.pyplot as plt

def smoothing_and_sharpening(image_path):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Kiểm tra xem ảnh có tồn tại không
    if img is None:
        print("Không thể đọc được ảnh.")
        return

    # Làm mịn ảnh
    smoothed_img = cv2.GaussianBlur(img, (5, 5), 0)

    # Làm nét ảnh
    sharpened_img = cv2.filter2D(img, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))

    # Hiển thị ảnh gốc, ảnh đã làm mịn và ảnh đã làm nét
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh Gốc')

    plt.subplot(1, 3, 2)
    plt.imshow(smoothed_img, cmap='gray')
    plt.title('Ảnh Làm Mịn')

    plt.subplot(1, 3, 3)
    plt.imshow(sharpened_img, cmap='gray')
    plt.title('Ảnh Làm Nét')

    plt.show()


image_path = 'image/test.png'  
smoothing_and_sharpening(image_path)
