import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def generate_large_grayscale_matrix(size=500):
    """
    Create a synthetic grayscale image with various shapes and gradients
    to demonstrate edge detection.
    """
    matrix = np.zeros((size, size))
    
    # Add a rectangle
    matrix[100:200, 100:300] = 100
    
    # Diagonal gradient
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    matrix += ((x + y) / (2 * size)) * 255
    
    # Circle
    center_x, center_y = size // 2, size // 2
    mask = (x - center_x) ** 2 + (y - center_y) ** 2
    matrix[(mask < 10000) & (mask > 9000)] = 200  # Fixed ** operator for square
    
    return matrix

def create_edge_detection_filters():
    """Create 4 different edge detection filter matrices."""
    return {
        'Sobel X': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        'Sobel Y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        'Prewitt X': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        'Laplacian': np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    }

def convolve2d(image, kernel):
    """2D convolution of image with a given kernel."""
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    output = np.zeros((image_h - kernel_h + 1, image_w - kernel_w + 1))
    
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = np.sum(image[i:i+kernel_h, j:j+kernel_w] * kernel)
    
    return output

def main():
    original_image = generate_large_grayscale_matrix()
    filters = create_edge_detection_filters()
    
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.title('Original Image')
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')
    
    for i, (name, filter_matrix) in enumerate(filters.items(), start=2):
        edge_image = convolve2d(original_image, filter_matrix)
        edge_image_scaled = MinMaxScaler().fit_transform(edge_image)
        
        plt.subplot(2, 3, i)
        plt.title(f'{name} Edge Detection')
        plt.imshow(edge_image_scaled, cmap='gray')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
