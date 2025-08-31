import numpy as np



def find_homography_ransac(points1, points2, num_iterations=8000, tolerance=1, num_correspondences=4):
    best_homography = None
    best_inliers = 0

    for _ in range(num_iterations):
        indices = np.random.choice(len(points1), num_correspondences, replace=False)
        src_points = points1[indices]
        dst_points = points2[indices]

        homography = estimate_homography(src_points, dst_points)
        inliers = calculate_inliers(homography, points1, points2, tolerance)
        if len(inliers) > best_inliers:
            best_homography = homography
            best_inliers = len(inliers)

    return best_homography


def estimate_homography(src_points, dst_points):
    num_points = len(src_points)
    if num_points < 4:
        raise ValueError("At least 4 point correspondences are required to compute homography.")

    A = np.zeros((2 * num_points, 9))

    for i in range(num_points):
        x, y = tuple(src_points[i][0])
        u, v = tuple(dst_points[i][0])
        A[2*i] = [-x, -y, -1, 0, 0, 0, u*x, u*y, u]
        A[2*i+1] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]

    _, _, V = np.linalg.svd(A)
    homography = V[-1].reshape(3, 3)

    homography /= homography[2, 2]

    return homography


def calculate_inliers(homography, points1, points2, tolerance):
    inliers = []
    for src, dst in zip(points1, points2):
        src_homogeneous = np.array([src[0][0], src[0][1], 1])
        dst_homogeneous = np.array([dst[0][0], dst[0][1], 1])

        projected_dst = np.dot(homography, src_homogeneous)
        projected_dst /= projected_dst[2] 

        distance = np.linalg.norm(dst_homogeneous - projected_dst)
        
        if distance < tolerance:
            inliers.append((src, dst))

    return inliers

