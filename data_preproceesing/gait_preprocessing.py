from cgi import FieldStorage
import pathlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
root_directory = '/media/aaron/963AA6803AA65D4D/ITRI_SSTC/S100/gait/OU_MVLP/OUMVLP_raw'
angles = [i for i in range(0, 91, 15)] + [i for i in range(180, 271, 15)]
phase = '01'  # or '01'

new_root_directory = '/media/aaron/963AA6803AA65D4D/ITRI_SSTC/S100/gait/OU_MVLP/OUMVLP_process'
angles = [i for i in range(0, 91, 15)] + [i for i in range(180, 271, 15)]
# angles = [270]
phase = '01'  # or '01'

# 向量中的第一個非零idx


def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

# 找每列中第一個非全黑色列和最後一個非全黑色列的idx


def find_first_and_end_nonzero(img):
    first = False
    end = False
    start_end = []
    check = False
    for i, row in enumerate(img):
        tmp_value = first_nonzero(row, axis=0)
        if tmp_value > 0 and not first:
            start_end.append(i)
            first = True
        if tmp_value < 0 and first == True:
            start_end.append(i)
            first = False
    if len(start_end) >= 2:
        return start_end[0], start_end[1]
    else:
        return [-1, -1]


# 找中心線 (依照矩陣每行的pixel總和，最大者為中線)
def findMedianLine(img):
    sum_ = -1
    medianLine_idx = -1
    for i, column in enumerate(img.transpose()):
        col_sum = np.sum(column)
        if col_sum > sum_:
            sum_ = col_sum
            medianLine_idx = i
    return medianLine_idx


# 確定每個資料夾下的檔案名稱
OUMVLP = {}
for ang in angles:
    angle_file_dir = f'{root_directory}/Silhouette_{str(ang).zfill(3)}-{phase}'
    OUMVLP[f'Silhouette_{str(ang).zfill(3)}-{phase}'] = os.listdir(angle_file_dir)

# 檢測檔案是否存在(資料夾也可)


def file_exist(path):
    file = pathlib.Path(path)
    return file.exists()


# new folder to save
if not file_exist(new_root_directory):
    os.mkdir(new_root_directory)
for folder, subjects in OUMVLP.items():
    angle_dir = f'{new_root_directory}/{folder}'
    if not file_exist(angle_dir):
        os.mkdir(angle_dir)
        for sub in subjects:
            sub_dir = f'{new_root_directory}/{folder}/{sub}'
            if not file_exist(sub_dir):
                os.mkdir(sub_dir)

# 對OUMVLP的每個檔案做處理
for folder, subjects in OUMVLP.items():
    print(f'{folder}')
    # 迭代每個subject
    for sub in tqdm(subjects):
        imgs_filename = sorted(os.listdir(
            f'{root_directory}/{folder}/{sub}'))  # list
        # 每個subject有好幾個圖 （走路步態）
        for img_f in imgs_filename:
            if file_exist(f'{new_root_directory}/{folder}/{sub}/{img_f}'):
                continue
            img = cv2.imread(f'{root_directory}/{folder}/{sub}/{img_f}', 0)
            # 例外確保
            if img is None:
                print(f'{root_directory}/{folder}/{sub}/{img_f}')
                continue
            # 找出人物上下邊界
            first, end = find_first_and_end_nonzero(img)
            # 如果原圖全黑
            if first == -1:
                cv2.imwrite(
                    f'{new_root_directory}/{folder}/{sub}/{img_f}', cv2.resize(img, (64, 64)))
                continue
            # 切除上下多餘區域
            crop_img = img[first:end, :]
            # 將高resize成64, 寬則等比變化
            height = crop_img.shape[0]
            width = crop_img.shape[1]
            img_new = cv2.resize(crop_img, (int(width * 64 / height), 64))

            bound_y = img_new.shape[1]
            # 依照中線為基礎左右延伸32pixel，不夠的補零
            padding = np.zeros((64, 32))
            img_new_padding = np.concatenate(
                (padding, img_new, padding), axis=1)
            # 求出中線
            median_idx = findMedianLine(img_new_padding)
            new_new_crop = img_new_padding[:, median_idx-32:median_idx+32]
            # 存檔

            ret = cv2.imwrite(
                f'{new_root_directory}/{folder}/{sub}/{img_f}', new_new_crop)
            if not ret:
                print(f'{new_root_directory}/{folder}/{sub}/{img_f}')
                print('save file have some issue!!!')
