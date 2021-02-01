import cv2 as cv
import numpy as np


def overlay_vertical_seam(img, seam):
    img_seam_overlay = np.copy(img)
    x_coords, y_coords = np.transpose([(i,int(j)) for i,j in enumerate(seam)])
    img_seam_overlay[x_coords, y_coords] = (0,255,0)
    return img_seam_overlay


def compute_energy_matrix(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sobel_x = cv.Sobel(gray,cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(gray,cv.CV_64F, 0, 1, ksize=3)
    abs_sobel_x = cv.convertScaleAbs(sobel_x)
    abs_sobel_y = cv.convertScaleAbs(sobel_y)
    return cv.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)


def find_vertical_seam(img, energy):
    rows, cols = img.shape[:2]
    seam = np.zeros(img.shape[0])
    dist_to = np.zeros(img.shape[:2]) + float('inf')
    dist_to[0,:] = np.zeros(img.shape[1])
    edge_to = np.zeros(img.shape[:2])
    for row in range(rows-1):
        for col in range(cols):
            if col != 0 and dist_to[row+1, col-1] > dist_to[row, col] + energy[row+1, col-1]:
                dist_to[row+1, col-1] = dist_to[row, col] + energy[row+1, col-1]
                edge_to[row+1, col-1] = 1
            if dist_to[row+1, col] > dist_to[row, col] + energy[row+1, col]:
                dist_to[row+1, col] = dist_to[row, col] + energy[row+1, col]
                edge_to[row+1, col] = 0
            if col != cols-1 and dist_to[row+1, col+1] > dist_to[row, col] + energy[row+1, col+1]:
                dist_to[row+1, col+1] = dist_to[row, col] + energy[row+1, col+1]
                edge_to[row+1, col+1] = -1
    seam[rows-1] = np.argmin(dist_to[rows-1, :])
    for i in (x for x in reversed(range(rows)) if x > 0):
        seam[i-1] = seam[i] + edge_to[i, int(seam[i])]
    return seam


def remove_vertical_seam(img, seam):
    rows, cols = img.shape[:2]
    for row in range(rows):
        for col in range(int(seam[row]), cols-1):
            img[row, col] = img[row, col+1]
    img = img[:, 0:cols-1]
    return img


def get_img(path,args,percent = 25,gray=False):
    img = cv.imread(path,0) if gray else cv.imread(path)
    dim = (int(img.shape[1] * percent / 100), int(img.shape[0] * percent / 100))
    img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    img = cv.GaussianBlur(img, (3, 3), 0)
    kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv.filter2D(img, -1, kernel_sharpen_1)


def add_vertical_seam(img, seam, num_iter):
    seam = seam + num_iter
    rows, cols = img.shape[:2]
    zero_col_mat = np.zeros((rows,1,3), dtype=np.uint8)
    img_extended = np.hstack((img, zero_col_mat))
    for row in range(rows):
        for col in range(cols, int(seam[row]), -1):
            img_extended[row, col] = img[row, col-1]
        for i in range(3):
            v1 = img_extended[row, int(seam[row])-1, i]
            v2 = img_extended[row, int(seam[row])+1, i]
            img_extended[row, int(seam[row]), i] = (int(v1)+int(v2))/2
    return img_extended


def seam(path):
    img_input = cv.imread(path)
    num_seams = int(10)
    img = np.copy(img_input)
    img_overlay_seam = np.copy(img)
    energy = compute_energy_matrix(img)
    for i in range(num_seams):
        seam = find_vertical_seam(img, energy)
        img_overlay_seam = overlay_vertical_seam(img_overlay_seam, seam)
        img = remove_vertical_seam(img, seam)
        energy = compute_energy_matrix(img)
        if i % 5 == 0:
            print('Number of seams removed = ', i+1)
    cv.imshow('Input', img_input)
    cv.imshow('Seams', img_overlay_seam)
    cv.imshow('Output', img)
    cv.waitKey()
