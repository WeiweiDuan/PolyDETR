import os
import sys
import cv2
import numpy as np

def end_pt(img, x, y):
    # check 8-dir, if only one connection, it's an end pt
    subreg = img[max(0, x-1):min(x+2, img.shape[0]), \
                 max(0, y-1):min(y+2, img.shape[1])]
    if np.sum(subreg) <= 2: 
        return True
    return False

def conn4dir(img, x, y):
    # check the connections in four directions
    count = 0
    if img[x-1, y] == 1:
        count += 1
    if img[x+1, y] == 1:
        count += 1
    if img[x, y+1] == 1:
        count += 1
    if img[x, y-1] == 1:
        count += 1
    # if < 2, needs buffering  
    if count < 2 and not end_pt(img, x, y):
        return False
    # else, no needs buffering
    return True

def buff_pt(img, x, y):
    # check four diagonal directions
    # buffer in the 4 connections
    # return (x, y), which needs for buffering
    if img[x-1, y-1] == 1:
        if img[x-1, y] == 0:
            return (x-1, y)
        if img[x, y-1] == 0:
            return (x, y-1)
    if img[x-1, y+1] == 1:
        if img[x-1, y] == 0:
            return (x-1, y)
        if img[x, y+1] == 0:
            return (x, y+1)
    if img[x+1, y-1] == 1:
        if img[x, y-1] == 0:
            return (x, y-1)
        if img[x+1, y] == 0:
            return (x+1, y)
    if img[x+1, y+1] == 1:
        if img[x, y+1] == 0:
            return (x, y+1)
        if img[x+1, y] == 0:
            return (x+1, y)
    return (x, y)

def buff(img):
    buf_img = np.zeros_like(img)
    nz_indices = np.where(img!=0)
    for i in range(nz_indices[0].size):
        x, y = nz_indices[0][i], nz_indices[1][i]
        if not conn4dir(img, x, y):
            bf_x, bf_y = buff_pt(img, x, y)
            buf_img[bf_x, bf_y] = 1
        buf_img[x, y] = 1
    return buf_img

def buff_main(pred_image):
    pred_img = pred_image/255
    buf_pred_img = buff(pred_img)
    return buf_pred_img*255