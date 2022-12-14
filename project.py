import cv2
import numpy as np
from scipy import signal
from sklearn.feature_extraction import image
from matplotlib import pyplot as plt

from cp_hw2 import lRGB2XYZ

def NCC(a, b):
    a = a.flatten()
    b = b.flatten()
    a = (a - np.mean(a)) / np.std(a - np.mean(a))
    b = (b - np.mean(b)) / np.std(b - np.mean(b))
    # c = np.sum(a*b) / np.sqrt(np.sum(a**2)*np.sum(b**2))
    c = np.correlate(a,b)
    # c = np.max(signal.correlate2d(a, b, mode="same"))
    return c

def read_rgb(path):
    I = cv2.imread(path)
    I = I[:,:,::-1] / 255
    return I

def read_lum(path):
    I = read_rgb(path)
    xyz = lRGB2XYZ(I)
    I_lum = xyz[:,:,1]
    return I_lum

def save_image(path, I, normalize=True):
    if normalize:
        I = (I - np.min(I)) / (np.max(I) - np.min(I))

    if len(I.shape) == 3:
        cv2.imwrite(path, 255 * I[:,:,::-1])
    else:
        cv2.imwrite(path, 255 * I)

if __name__ == "__main__":
    N = 1
    num_corres = 100
    patch_size = 50
    trial = "trial3"

    I_tar = read_lum(f"data/{trial}/img_1.tiff")[::N,::N]
    I_ref = read_lum(f"data/{trial}/img_3.tiff")[::N,::N]
    save_image(f"data/{trial}/I_tar.png", I_tar)
    save_image(f"data/{trial}/I_ref.png", I_ref)

    pts_src = np.zeros((num_corres,2))
    pts_dst = np.zeros((num_corres,2))
    for i in range(num_corres):
        y1 = np.random.randint(I_ref.shape[0] - patch_size)
        x1 = np.random.randint(I_ref.shape[1] - patch_size)
        patch = I_ref[y1:y1+patch_size, x1:x1+patch_size]

        img = I_tar.copy().astype(np.float32)
        template = patch.astype(np.float32)
        res = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        x2, y2 = top_left[0], top_left[1]

        # print((x1, x2), (y1, y2))
        pts_src[i] = (x1, y1)
        pts_dst[i] = (x2, y2)

    h, status = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC)

    I_ref_w = cv2.warpPerspective(I_ref.astype(np.float32), h, (I_ref.shape[1], I_ref.shape[0]))
    save_image(f"data/{trial}/I_ref_w.png", I_ref_w)

    d = 10
    S = np.zeros(I_tar.shape)
    for i in range(d, I_tar.shape[0] - (d + 1)):
        for j in range(d, I_tar.shape[1] - (d + 1)):
            tar = I_tar[i - d: i + d + 1,
                        j - d: j + d + 1]
            ref = I_ref_w[i - d: i + d + 1,
                        j - d: j + d + 1]
            ncc = NCC(tar, ref)
            S[i,j] = ncc
    S = S[d:S.shape[0]-(d+1), d:S.shape[1]-(d+1)]
    save_image(f"data/{trial}/S.png", S)
