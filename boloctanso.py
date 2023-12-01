import cv2
import numpy as np
import matplotlib.pyplot as plt

def frequency_filtering(image_path, cutoff_frequency):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Kiểm tra xem ảnh có tồn tại không
    if img is None:
        print("Không thể đọc được ảnh.")
        return

    # Thực hiện biến đổi Fourier
    f_transform = np.fft.fft2(img)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Tạo bộ lọc thông thấp
    rows, cols = img.shape
    crow, ccol = rows // 2 , cols // 2  # Tìm tâm của hình ảnh
    mask = np.ones((rows, cols), np.uint8)
    r = cutoff_frequency
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0

    # Áp dụng bộ lọc
    f_transform_shifted = f_transform_shifted * mask

    # Thực hiện ngược biến đổi Fourier
    f_ishift = np.fft.ifftshift(f_transform_shifted)
    img_filtered = np.fft.ifft2(f_ishift)
    img_filtered = np.abs(img_filtered)

    # Hiển thị ảnh gốc và ảnh đã được lọc
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh Gốc')

    plt.subplot(1, 2, 2)
    plt.imshow(img_filtered, cmap='gray')
    plt.title(f'Ảnh Sau Bộ Lọc Thông Thấp (cutoff={cutoff_frequency})')

    plt.show()


image_path = 'image/test.png'  
cutoff_frequency = 30  # Thay đổi giá trị để thử nghiệm
frequency_filtering(image_path, cutoff_frequency)
