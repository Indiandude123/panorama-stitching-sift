import numpy as np

def gaussian_blur(image, sigmaX=1, sigmaY=1):
    """Apply Gaussian blur to the image"""
    kernel_size_x = calculate_kernel_size(sigmaX)
    kernel_size_y = calculate_kernel_size(sigmaY)

    kernel_x = gaussian_kernel(kernel_size_x, sigmaX)
    kernel_y = gaussian_kernel(kernel_size_y, sigmaY)

    blurred_image = convolution(convolution(image, kernel_x.T), kernel_y)

    return blurred_image

def calculate_kernel_size(sigma=1):
    """Calculate the kernel size based on the given sigma value"""
    
    return int(2 * np.ceil(3 * sigma) + 1)

def gaussian_kernel(kernel_size, sigma=1):
    """Generate a Gaussian kernel"""
    m, n = [(ss - 1.0) / 2.0 for ss in (kernel_size, kernel_size)]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    kernel = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    kernel /= 2.0 * np.pi * sigma * sigma
    return kernel / kernel.sum()

def convolution(image, kernel):
    """Apply convolution to the image"""
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    result = np.zeros_like(image)

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    for i in range(image_height):
        for j in range(image_width):
            result[i, j] = np.sum(padded_image[i:i+kernel_height, j:j+kernel_width] * kernel)

    return result


