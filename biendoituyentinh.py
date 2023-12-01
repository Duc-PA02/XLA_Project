import cv2
import numpy as np
import matplotlib.pyplot as plt

def piecewise_linear_transform(image_path, points):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Kiểm tra xem ảnh có tồn tại không
    if img is None:
        print("Không thể đọc được ảnh.")
        return

    # Normalize giá trị pixel về đoạn [0, 1]
    normalized_img = img / 255.0

    # Áp dụng phép biến đổi tuyến tính phân đoạn
    piecewise_linear_transformed_img = np.piecewise(normalized_img,
                                                    [normalized_img < points[0][0],
                                                     (normalized_img >= points[0][0]) & (normalized_img < points[1][0]),
                                                     (normalized_img >= points[1][0]) & (normalized_img < points[2][0]),
                                                     normalized_img >= points[2][0]],
                                                    [lambda x: points[0][1] * x,
                                                     lambda x: (points[1][1] - points[0][1]) / (points[1][0] - points[0][0]) * (x - points[0][0]) + points[0][1],
                                                     lambda x: (points[2][1] - points[1][1]) / (points[2][0] - points[1][0]) * (x - points[1][0]) + points[1][1],
                                                     lambda x: points[2][1] * x])

    # Scale lại giá trị pixel về đoạn [0, 255]
    piecewise_linear_transformed_img = (piecewise_linear_transformed_img * 255).astype(np.uint8)

    # Hiển thị ảnh gốc và ảnh đã được biến đổi
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh Gốc')

    plt.subplot(1, 2, 2)
    plt.imshow(piecewise_linear_transformed_img, cmap='gray')
    plt.title('Biến Đổi Tuyến Tính Phân Đoạn')

    plt.show()


image_path = 'image/test.png'  
# Đặt các điểm cho phép biến đổi tuyến tính
# Ví dụ: [(0, 0), (0.5, 0.7), (0.8, 1)]
points = [(0, 0), (0.5, 0.7), (0.8, 1)]
piecewise_linear_transform(image_path, points)
