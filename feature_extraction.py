import cv2
import numpy as np
import scipy.stats as stats
from math import sqrt
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, blob_log
from skimage.filters import frangi

def extract_super_features(img_gray):
    img_resized = cv2.resize(img_gray, (256, 256))
    _, mask_binary = cv2.threshold(img_resized, 15, 255, cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(img_resized, (35, 35), 0)
    blurred = cv2.bitwise_and(blurred, blurred, mask=mask_binary)
    _, _, _, maxLoc = cv2.minMaxLoc(blurred)
    cv2.circle(mask_binary, maxLoc, 35, 0, -1)

    mask_boolean = mask_binary > 0
    features = []

    # 1. FRANGI
    vessel_map = frangi(img_resized)
    vessels_eye = vessel_map[mask_boolean]
    if len(vessels_eye) == 0: vessels_eye = np.array([0.0])
    features.extend([np.mean(vessels_eye), np.std(vessels_eye), stats.skew(vessels_eye), stats.kurtosis(vessels_eye)])

    # 2. TORTUOSITY
    vessel_norm = cv2.normalize(vessel_map, None, 0, 255, cv2.NORM_MINMAX)
    edges = cv2.Canny(np.uint8(vessel_norm), 50, 150)
    edges_eye = edges[mask_boolean]
    features.append(np.sum(edges_eye > 0) / (len(edges_eye) + 1e-6))

    # 3. BLOB
    inverted_img = cv2.bitwise_not(img_resized)
    inverted_img = cv2.bitwise_and(inverted_img, inverted_img, mask=mask_binary)
    blobs = blob_log(inverted_img, max_sigma=15, num_sigma=10, threshold=0.15)
    spot_count = len(blobs)
    total_spot_area = sum([np.pi * (r * sqrt(2))**2 for _, _, r in blobs]) if spot_count > 0 else 0
    features.extend([spot_count, total_spot_area])

    # 4. GLCM
    glcm = graycomatrix(img_resized, distances=[1, 3], angles=[0, np.pi/4], levels=256, symmetric=True, normed=True)
    features.extend([graycoprops(glcm, 'contrast').mean(), graycoprops(glcm, 'energy').mean(),
                     graycoprops(glcm, 'homogeneity').mean(), graycoprops(glcm, 'correlation').mean(),
                     graycoprops(glcm, 'ASM').mean()])

    # 5. LBP
    lbp = local_binary_pattern(img_resized, 8, 1, method='uniform')
    lbp_eye = lbp[mask_boolean]
    if len(lbp_eye) == 0: lbp_eye = np.array([0.0])
    hist, _ = np.histogram(lbp_eye.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype("float") / (hist.sum() + 1e-6)
    features.extend(hist.tolist())

    return features
