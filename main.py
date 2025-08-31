from utils import *
from knn_matcher import custom_knn_matcher
from sift import computeKeypointsAndDescriptors
from homography import find_homography_ransac
import os
import cv2 as cv
import sys


def custom_stitcher(img1, img2):
    resized_im1 = bl_resize(img1, img1.shape[0]//2, img1.shape[1]//2)
    resized_im2 = bl_resize(img2, img2.shape[0]//2, img2.shape[1]//2)
    print("Resized images successfully")
    
    gray_im1 = custom_bgr_to_gray(resized_im1)
    gray_im2 = custom_bgr_to_gray(resized_im2)
    print("Converted to grayscale successfully")
    
    adjusted_gray_im1 = adjust_contrast(gray_im1)
    adjusted_gray_im2 = adjust_contrast(gray_im2)
    print("Adjusted contrast successfully")


    blurred_gray_im1 = reduce_noise(adjusted_gray_im1)
    blurred_gray_im2 = reduce_noise(adjusted_gray_im2)
    print("Blurred images successfully")

    kp1, des1 = computeKeypointsAndDescriptors(blurred_gray_im1)
    print("Computed keypoints and descriptor for image 1 successfully")
    kp2, des2 = computeKeypointsAndDescriptors(blurred_gray_im2)
    print("Computed keypoints and descriptor for image 2 successfully")

    matches_knn = custom_knn_matcher(des1, des2, k=4, ratio=0.6)
    matches_knn = sorted(matches_knn, key=lambda x: x[2])
    print("Performed KNN matching of keypoints successfully")
    points1_knn = np.float32([kp1[m[0]].pt for m in matches_knn]).reshape(-1, 1, 2)
    points2_knn = np.float32([kp2[m[1]].pt for m in matches_knn]).reshape(-1, 1, 2)

    homography = find_homography_ransac(points1_knn, points2_knn)
    print("Computed homography matrix successfully")
    warped_img_custom = warp_perspective_custom(resized_im1, homography, resized_im2)
    print("Perspective transform successfull")
    resized_image_panorama_ = bl_resize(warped_img_custom, warped_img_custom.shape[0]*2, warped_img_custom.shape[1]*2)
    print("Panorama generated")
    
    print("\n")
    return resized_image_panorama_

def get_last_name(filename):
    return int(filename.split('/')[-1].split('.')[0])

def process_images(input_path, output_path):
    data_dir = input_path
    image_paths = []
    for file_name in os.listdir(data_dir):
        image_path = os.path.join(data_dir, file_name)
        image_paths.append(image_path)
    print(image_paths)

    image_paths = sorted(image_paths, key=get_last_name)
    print("\n")
    print(image_paths)
    for i, image_path in enumerate(image_paths):
        if i==0:
            im1 = cv.imread(image_path)
            im2 = cv.imread(image_paths[i+1])
            panorama_img = custom_stitcher(im1, im2)
        elif i==1:
            continue
        else:
            im1 = cv.imread(image_path)
            im2 = panorama_img
            panorama_img = custom_stitcher(im1, im2) 
    
    #### Write to output path #### 
    cv.imwrite(output_path, panorama_img)
    

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 main.py <part_id> <input_path> <output_path>")
        return
    
    part_id = int(sys.argv[1])
    input_path = sys.argv[2]
    output_path = sys.argv[3]
    
    if part_id == 1:
        process_images(input_path, output_path)
    elif part_id == 2:
        # process_video(input_path, output_path)
        print("Not done")
        pass
    else:
        print("Invalid part ID. Use 1 or 2.")


if __name__ == "__main__":
    main()
            
            

