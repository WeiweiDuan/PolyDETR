import cv2
import os
import numpy as np
from itertools import product
import itertools


MAP_RANGE = { # height range, width range
    'CO_SanLuis': [(0, 7500), (0, 6500)],
    'CO_SanchezRes': [(0, 8000), (0, 7500)],
    'CO_Granite': [(0, 8000), (0, 6000)],
    'CO_Eagle': [(0, 7500), (0, 6000)],
    'CO_BigCostilla': [(0, 13000), (0, 12000)],
    'CA_Sage': [(0, 7500), (0, 6500)],
}



############ Data generation ##########
def standarization(x_train):
    mean, std = x_train.mean(), x_train.std()
    # global standardization of pixels
    pixels = (x_train - mean) / std
    # clip pixel values to [-1,1]
    pixels = np.clip(pixels, -1.0, 1.0)
    # shift from [-1,1] to [0,1] with 0.5 mean
    pixels = (pixels + 1.0) / 2.0
    return pixels


def array2img(x, data_format='channels_last'):
    mean = [103.939, 116.779, 123.68]

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] += mean[0]
            x[1, :, :] += mean[1]
            x[2, :, :] += mean[2]
        else:
            x[:, 0, :, :] += mean[0]
            x[:, 1, :, :] += mean[1]
            x[:, 2, :, :] += mean[2]
    else:
        x[..., 0] += mean[0]
        x[..., 1] += mean[1]
        x[..., 2] += mean[2]

    if data_format == 'channels_first':
        # 'BGR'->'RGB'
        if x.ndim == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'BGR'->'RGB'
        x = x[..., ::-1]

    return x

def adjust_gamma(image, gamma=0.5):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

    
def square_from_center(image, center_y, center_x, window_size):
    origin_y = int(center_y - (window_size - 1) / 2)
    origin_x = int(center_x - (window_size - 1) / 2)
    return np.array(image[origin_y:origin_y + window_size, origin_x:origin_x + window_size])

def square_from_center_label(image, center_y, center_x, window_size):
    origin_y = int(center_y - (window_size - 1) / 2)
    origin_x = int(center_x - (window_size - 1) / 2)
    return np.array(image[origin_y:origin_y + window_size, origin_x:origin_x + window_size]).astype(np.float64)

def generate_data_from_center_coords(image, coordinates, window_size, gamma=1.0):
    data = []
    idx = []
    for y_coord, x_coord in coordinates:
        cropped_image = square_from_center(image, y_coord, x_coord, window_size)
        if cropped_image.shape != (window_size, window_size, 3):
            continue
        else:
            if gamma == 1.0:
                data.append(cropped_image) 
            else:
                data.append(adjust_gamma(cropped_image, gamma))
            idx.append([y_coord, x_coord])
    return np.array(data), np.array(idx)

def generate_data_from_center_coords_label(image, coordinates, window_size):
    data = []

    for y_coord, x_coord in coordinates:
        cropped_image = square_from_center_label(image, y_coord, x_coord, window_size)
#         print('single label size: ', cropped_image.shape)
        if cropped_image.shape != (window_size, window_size, 3):
            continue
        else:
            data.append(cropped_image)
    return np.array(data)


def points_generator(DATA_PATH, OBJECT_LIST, OBJECT_NUMS, MAP_PATH, label_map, random=True, win_size=256):
    obj_list = []
    obj_length = len(OBJECT_LIST)
        
    for obj_index in range(obj_length):
        obj_name = OBJECT_LIST[obj_index]
        print('read points from file: ', os.path.join(DATA_PATH, obj_name+'.txt'))
        obj_points = np.loadtxt(os.path.join(DATA_PATH, obj_name+'.txt'), dtype=np.int32, delimiter=",")
        if obj_points.size == 2: # if txt only has one point, extend the dim
            obj_points = np.expand_dims(obj_points, 0)
        np.random.shuffle(obj_points)
        obj_list.append(obj_points)
    ##### load the positive coord in the first txt 
    x_train_coor_pos = obj_list[0][:OBJECT_NUMS[0]]

    x_train_coor_neg = []
    if obj_length > 2: # if length=2, 1st num is #positive samples, 2nd num is #negative samples
        for i in range(1, obj_length):
            if i == 1:
                x_train_coor_neg = obj_list[i][:OBJECT_NUMS[i]]
            else:
                x_train_coor_neg = np.concatenate((x_train_coor_neg, obj_list[i][:OBJECT_NUMS[i]]), axis=0)
        print('negative points from other categories: ', x_train_coor_neg.shape)
    
    if random and OBJECT_NUMS[-1]>0:
        p_rand = []
        height, width = label_map.shape[:2]
#         x_rand = np.random.randint(0, height, size=OBJECT_NUMS[-1])
#         y_rand = np.random.randint(0, width, size=OBJECT_NUMS[-1])
        h_min, h_max = np.min(obj_points[:, 0]), np.max(obj_points[:, 0])
        w_min, w_max = np.min(obj_points[:, 1]), np.max(obj_points[:, 1])
        print('random negative are sampled from [%d:%d, %d:%d]'%(h_min, h_max, w_min, w_max))
        x_rand = np.random.randint(h_min, h_max, size=OBJECT_NUMS[-1])
        y_rand = np.random.randint(w_min, w_max, size=OBJECT_NUMS[-1])        
        for i in range(OBJECT_NUMS[-1]):
            cropped_label_img \
                = label_map[x_rand[i]-win_size//2:x_rand[i]+win_size//2, y_rand[i]-win_size//2:y_rand[i]+win_size//2]
            if np.any(cropped_label_img):
                continue
            p_rand.append([x_rand[i], y_rand[i]])
        p_rand = np.array(p_rand)
        
        #### bbox gives 0 negative, randonly select negative across the map
        if p_rand.shape[0] < 100:
            print('randomly select negative across the map')
            x_rand = np.random.randint(0, height, size=OBJECT_NUMS[-1])
            y_rand = np.random.randint(0, width, size=OBJECT_NUMS[-1])       
        p_rand = []
        for i in range(OBJECT_NUMS[-1]):
            cropped_label_img \
                = label_map[x_rand[i]-win_size//2:x_rand[i]+win_size//2, y_rand[i]-win_size//2:y_rand[i]+win_size//2]
            if np.any(cropped_label_img):
                continue
            p_rand.append([x_rand[i], y_rand[i]])
        p_rand = np.array(p_rand)
        print('random points: ', p_rand.shape)
        if x_train_coor_neg != []:
            x_train_coor_neg = np.concatenate((x_train_coor_neg, p_rand), axis=0)
        else:
            x_train_coor_neg = p_rand
    return x_train_coor_pos, x_train_coor_neg


def data_generator(DATA_PATH, MAP_PATH, LABEL_PATH, OBJECT_LIST, OBJECT_NUMS, WIN_SIZE, NB_CLASSES,\
                   shift_augment=False, shift_range=(0, 0), num_shift=2, times4multi=1,\
                   check_data=False, gamma=1.0, random=True):
    img = cv2.imread(MAP_PATH)
    label = cv2.imread(LABEL_PATH)*255
    pos_coor, neg_coor =  points_generator(DATA_PATH, OBJECT_LIST, OBJECT_NUMS, MAP_PATH, label, random=random)
    print('Before augmentation, total positive, negative coor: ', pos_coor.shape, neg_coor.shape)
    if shift_augment == False:
        coor = np.vstack((pos_coor, neg_coor))
        np.random.shuffle(coor)
    else:
        aug_pos_coor = gen_shifted_coor(pos_coor, shift_range, num_shift=num_shift,\
                                        times4multi=times4multi, label_img=label) # num_shift=2 means doubling the data
        print('after shifting positive coor shape: ', aug_pos_coor.shape)
        coor = np.vstack((aug_pos_coor, neg_coor))
        np.random.shuffle(coor)
        print('after shiting all coor shape: ', coor.shape)
    x_train, x_indices = generate_data_from_center_coords(img, coor, WIN_SIZE, gamma)
    print('x_train shape: ', x_train.shape)

    y_train = generate_data_from_center_coords_label(label, coor, WIN_SIZE)
    print('y_train shape: ', y_train.shape)

    y_train = np.array(y_train)
    x_train = x_train.astype(np.float64)
    return x_train, y_train, x_indices
    
def gen_shifted_coor(coor_arr, shift_range, num_shift=2, times4multi=1, label_img=None):
    ##### coor_arr: (#coords, 2), a list of coordinates
    ##### shift range: (dx, dy), the shift is (-dx, dx), (-dy, dy)
    ##### num_shift: randomly select how many (dx, dy) pairs
    ##### return augmented coor_arr (#aug_coords, 2)
    aug_coor_arr = []
    for x, y in coor_arr:
        dx = np.random.randint(-shift_range[0], shift_range[0], (num_shift))
        dy = np.random.randint(-shift_range[1], shift_range[1], (num_shift))
        aug_coor_arr.append([x, y])        
        for i in range(num_shift):
#             print('dx, dy: ', dx[i], dy[i])
            new_x, new_y = x+dx[i], y+dy[i]
            aug_coor_arr.append([new_x, new_y])
#         if times4multi != 1:
#             label_list = generate_data_from_center_coords_label(label_img, [[new_x,new_y]], 256)
#             if len(label_list) == 0:
#                 continue
#             label = label_list[0]
#             if np.sum(label[:,:,1]/255) > 270:
#                 more_dx = np.random.randint(-shift_range[0], shift_range[0], (num_shift*times4multi))
#                 more_dy = np.random.randint(-shift_range[1], shift_range[1], (num_shift*times4multi))
#                 for j in range(num_shift*times4multi):
#                     more_new_x, more_new_y = x+more_dx[j], y+more_dy[j]
#                     aug_coor_arr.append([more_new_x, more_new_y])
    aug_coor_arr = np.array(aug_coor_arr)
    return aug_coor_arr
