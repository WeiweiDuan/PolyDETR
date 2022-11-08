import cv2
import math
import numpy as np

def rotate(image, angle=90):
    row,col = image.shape[:2]
    center = tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

def rotate_pt(point, cent, angle_degree=90):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle in [0, 360].
    """
    angle = math.radians(angle_degree)
    s, c = math.sin(angle), math.cos(angle)
#      translate point back to origin:
    px, py = point
    cx, cy = cent
    px -= cx
    py -= cy
#    rotate point
    xnew = px * c - py * s
    ynew = px * s + py * c
#    translate point back:
    px = xnew + cx
    py = ynew + cy
    return int(abs(px)), int(abs(py))

def rotate_nodes_dict(nodes_dict, angle_degree, cent=[128,128]):
    rot_nodes_dict = {}
    for key, value in nodes_dict.items():
        rot_node = rotate_pt(value, cent, angle_degree)
        rot_nodes_dict[key] = rot_node
    return rot_nodes_dict
    

def rotate_line(line, angle_degree=0, cent=[128,128]):
    # input: line[[x_s,y_s],[x_e,y_e]]
    # return a rotated line [[r_x_s,r_y_s],[r_x_e,r_y_e]]
    x_s, y_s = line[0]
    x_e, y_e = line[1]
    rot_line_list = []

    rot_x_s, rot_y_s = rotate_pt([x_s,y_s], cent, angle_degree)
    rot_x_e, rot_y_e = rotate_pt([x_e,y_e], cent, angle_degree)
    return [[rot_x_s, rot_y_s], [rot_x_e, rot_y_e]]