from gaussian_blur import gaussian_blur
import numpy as np
import math

def custom_bgr_to_gray(image):
    blue_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    red_channel = image[:, :, 2]
    gray_image = (0.2989 * red_channel + 0.5870 * green_channel + 0.1140 * blue_channel).astype(np.uint8)
    return gray_image


def gray_to_bgr(gray_image):
    bgr_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)
    bgr_image[:, :, 0] = gray_image
    bgr_image[:, :, 1] = gray_image
    bgr_image[:, :, 2] = gray_image
    return bgr_image


def reduce_noise(image):
    blurred_image = gaussian_blur(image)
    return blurred_image


def adjust_contrast(image, alpha=1.25, beta=0):
    """Adjust the contrast of an image"""
    adjusted_image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
    return adjusted_image


def bl_resize(original_img, new_h, new_w):
	old_h, old_w, c = original_img.shape
	resized = np.zeros((new_h, new_w, c))
	w_scale_factor = (old_w ) / (new_w ) if new_h != 0 else 0
	h_scale_factor = (old_h ) / (new_h ) if new_w != 0 else 0
	for i in range(new_h):
		for j in range(new_w):
			x = i * h_scale_factor
			y = j * w_scale_factor
			x_floor = math.floor(x)
			x_ceil = min( old_h - 1, math.ceil(x))
			y_floor = math.floor(y)
			y_ceil = min(old_w - 1, math.ceil(y))

			if (x_ceil == x_floor) and (y_ceil == y_floor):
				q = original_img[int(x), int(y), :]
			elif (x_ceil == x_floor):
				q1 = original_img[int(x), int(y_floor), :]
				q2 = original_img[int(x), int(y_ceil), :]
				q = q1 * (y_ceil - y) + q2 * (y - y_floor)
			elif (y_ceil == y_floor):
				q1 = original_img[int(x_floor), int(y), :]
				q2 = original_img[int(x_ceil), int(y), :]
				q = (q1 * (x_ceil - x)) + (q2	 * (x - x_floor))
			else:
				v1 = original_img[x_floor, y_floor, :]
				v2 = original_img[x_ceil, y_floor, :]
				v3 = original_img[x_floor, y_ceil, :]
				v4 = original_img[x_ceil, y_ceil, :]

				q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
				q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
				q = q1 * (y_ceil - y) + q2 * (y - y_floor)

			resized[i,j,:] = q
	return resized.astype(np.uint8)

def bl_resize_bw(original_img, new_h, new_w):
    old_h, old_w = original_img.shape
    resized = np.zeros((new_h, new_w))
    w_scale_factor = (old_w) / (new_w) if new_h != 0 else 0
    h_scale_factor = (old_h) / (new_h) if new_w != 0 else 0
    for i in range(new_h):
        for j in range(new_w):
            x = i * h_scale_factor
            y = j * w_scale_factor
            x_floor = math.floor(x)
            x_ceil = min(old_h - 1, math.ceil(x))
            y_floor = math.floor(y)
            y_ceil = min(old_w - 1, math.ceil(y))

            if (x_ceil == x_floor) and (y_ceil == y_floor):
                q = original_img[int(x), int(y)]
            elif (x_ceil == x_floor):
                q1 = original_img[int(x), int(y_floor)]
                q2 = original_img[int(x), int(y_ceil)]
                q = q1 * (y_ceil - y) + q2 * (y - y_floor)
            elif (y_ceil == y_floor):
                q1 = original_img[int(x_floor), int(y)]
                q2 = original_img[int(x_ceil), int(y)]
                q = (q1 * (x_ceil - x)) + (q2 * (x - x_floor))
            else:
                v1 = original_img[x_floor, y_floor]
                v2 = original_img[x_ceil, y_floor]
                v3 = original_img[x_floor, y_ceil]
                v4 = original_img[x_ceil, y_ceil]

                q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
                q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
                q = q1 * (y_ceil - y) + q2 * (y - y_floor)

            resized[i, j] = q
    return resized.astype(np.uint8)


import numpy as np

def resize_nearest_neighbor(image, output_shape):
    scale_x = output_shape[1] / image.shape[1]
    scale_y = output_shape[0] / image.shape[0]

    resized_image = np.zeros(output_shape, dtype=image.dtype)

    rows = np.arange(output_shape[0]) / scale_y
    cols = np.arange(output_shape[1]) / scale_x

    rows_nearest = np.round(rows).astype(int)
    cols_nearest = np.round(cols).astype(int)

    rows_clamped = np.clip(rows_nearest, 0, image.shape[0] - 1)
    cols_clamped = np.clip(cols_nearest, 0, image.shape[1] - 1)

    row_indices, col_indices = np.meshgrid(rows_clamped, cols_clamped, indexing='ij')

    resized_image = image[row_indices, col_indices]

    return resized_image.astype(image.dtype)


def warp_perspective_custom(img1, h, img2):
    height = max(img1.shape[0], img2.shape[0])
    width = img1.shape[1] + img2.shape[1]

    warped_img = np.zeros((height, width, img2.shape[2]), dtype=np.uint8)

    hinv = np.linalg.inv(h)
    
    for y in range(height):
        for x in range(width):

            src_pos = np.dot(hinv, np.array([x, y, 1]))
            src_pos /= src_pos[2] 
            src_x, src_y = int(src_pos[0]), int(src_pos[1])

            if 0 <= src_x < img1.shape[1] and 0 <= src_y < img1.shape[0]:
                warped_img[y, x] = img1[src_y, src_x]

    warped_img[0:img2.shape[0], 0:img2.shape[1]] = img2

    return warped_img