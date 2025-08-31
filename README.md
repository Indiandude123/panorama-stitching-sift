# SIFT-Based Panorama Stitching

This project implements a **custom panorama stitching pipeline** using **SIFT (Scale-Invariant Feature Transform)** features, custom KNN matching, and RANSAC-based homography estimation.  
It processes a set of overlapping images and stitches them into a seamless panorama.

---

## Features
- Custom **image preprocessing** (grayscale conversion, contrast adjustment, denoising).  
- **SIFT-based keypoint detection** and descriptor extraction.  
- **Custom KNN matcher** with ratio test for robust keypoint matching.  
- **RANSAC-based homography estimation** for alignment.  
- **Perspective warping** to generate the stitched panorama.  

---

## Project Structure

├── main.py # Main entry point
├── sift.py # Custom SIFT keypoint + descriptor extraction
├── knn_matcher.py # Custom KNN matching
├── homography.py # RANSAC-based homography estimation
├── utils.py # Helper functions (resize, grayscale, blur, etc.)
└── README.md


---

## Usage

### Run Panorama Stitching

``` bash
python3 main.py 1 <input_folder> <output_image>
```

-   `<input_folder>` → Directory containing sequential images (named
    numerically, e.g., `1.jpg`, `2.jpg`, `3.jpg`).\
-   `<output_image>` → File path where the final panorama will be saved.

Example:

``` bash
python3 main.py 1 ./input_images ./output/panorama.jpg
```

------------------------------------------------------------------------

## Dependencies

Make sure you have the following installed:\
- Python **3.10+**\
- OpenCV (`cv2`)\
- NumPy

Install with:

``` bash
pip install opencv-python numpy
```

------------------------------------------------------------------------

## Example Workflow

**Input overlapping images:**

    ./input_images/
    ├── 1.jpg
    ├── 2.jpg
    └── 3.jpg

**Run:**

``` bash
python3 main.py 1 ./input_images ./panorama.jpg
```

**Output:**

    Stitched panorama saved at ./panorama.jpg

------------------------------------------------------------------------
