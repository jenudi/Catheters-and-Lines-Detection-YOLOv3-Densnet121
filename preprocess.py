import pandas as pd
import numpy as np
import random
import ast
from PIL import Image, ImageFile
import albumentations as A
from albumentations.pytorch import ToTensor
import math


def split_data(df):
    val_indexes = [index for index in range(len(df)) if index % 10 == 0]
    train = df.drop(index=val_indexes)
    train.reset_index(drop=True,inplace=True)
    val = df.iloc[val_indexes,:]
    val.reset_index(drop=True,inplace=True)
    return train.copy(),val.copy()


def check_for_leakage(df1, df2,args):
    patients_in_both_groups = np.intersect1d(df1[args.patients].values, df2[args.patients].values)
    patients_in_both_groups = set(patients_in_both_groups)
    leakage = (len(patients_in_both_groups) > 0)
    if leakage:
        leakage_index = [index for index, value in enumerate(df1[args.patients]) if value == list(patients_in_both_groups)[0]]
        df1.drop(leakage_index, inplace=True)
        df1.reset_index(drop=True,inplace=True)
    return df1,df2


def convert_to_yolo(df,index):
    rows = 1500
    cols = Image.open(df['path'][index]).size[0]
    annot = df['data'][index]
    cls = df['cls'][index]
    if cls == 0:
        y_list = list(annot[:,1])
        x_list = list(annot[:,0])
        left = True if (sum(x_list) / len(x_list)) < (cols // 2) else False # check the insertion direction of the CVC
        x_sorted = sorted(x_list,reverse=left)[:len(x_list) // 2]
        ax = x_sorted[0]
        first_angle = 0.0
        stop_inedx = len(x_sorted)
        for i in range(1,len(x_sorted)):
            dx = x_sorted[i] - ax
            dy = y_list[x_list.index(x_sorted[i])] - y_list[x_list.index(ax)]
            angle= abs(-math.degrees(math.atan2(dy, dx)))
            if i == 1:
                first_angle = angle
            elif first_angle == 0.0:
                first_angle = angle
                continue
            if i > 1 and (abs(first_angle - angle) > 2.5):
                stop_inedx = i + 1
                break
        annot = np.array([[i,y_list[x_list.index(i)]] for i in x_sorted[:stop_inedx]])
    bbox = (np.min(annot, axis=0)[0],np.min(annot, axis=0)[1],np.max(annot, axis=0)[0],np.max(annot, axis=0)[1])
    x_min, y_min, x_max, y_max = bbox[:4]
    y_min = 0 if y_min < 0 else y_min
    x_min = 0 if x_min < 0 else x_min
    y_max = rows if y_max > rows else y_max
    x_max = cols if x_max > cols else x_max
    bbox = (x_min,y_min,x_max,y_max,cls)
    album = A.augmentations.bbox_utils.convert_bbox_to_albumentations(bbox, 'pascal_voc', rows, cols, check_validity=False)
    yolo = A.augmentations.bbox_utils.convert_bbox_from_albumentations(album, 'yolo', rows, cols, check_validity=False)
    try:
        album = A.augmentations.bbox_utils.convert_bbox_to_albumentations(yolo, 'yolo', rows, cols, check_validity=True)
    except ValueError:
        yolo = []
    return yolo


def start():
    name2id = {'CVC': 0, 'ETT': 1, 'NGT': 2}
    df = pd.read_csv('train_annotations.csv')
    df['data'] = [np.array(ast.literal_eval(i)) for i in df['data']]
    df.rename(columns={'StudyInstanceUID': 'path'}, inplace=True)
    df['path'] = ['train/train/' + i + '.jpg' for i in df['path']]
    df['cls'] = [name2id[i[:3]] if i[:3] in name2id.keys() else 3 for i in df['label']]
    df.drop(df[df['cls'] == 3].index, inplace = True)
    df.drop(df[df['cls'] == 2].index, inplace = True)
    df['yolo'] = [convert_to_yolo(df,index) for index in range(len(df))]
    index_to_drop = [index for index,value in enumerate(df['yolo']) if len(value) == 0]
    df.drop(index_to_drop, inplace=True)
    print(f"{len(index_to_drop)} were removed")
    df.reset_index(drop=True,inplace=True)
    df.sort_values(by='cls',inplace=True)
    return df


df = start()

train_set, val_set = split_data(df)
cls0 = list(np.where(np.array(train_set['cls']) == 0))[0].tolist()
random.shuffle(cls0)
train_set.drop(cls0[3200:], inplace = True)
train_set.reset_index(drop=True,inplace=True)
print(train_set['cls'].value_counts())
train_set = train_set.groupby('path',as_index=False).agg({'data': lambda x: tuple(x), 'yolo': lambda x: tuple(x), 'cls': lambda x: list(x)})
val_set = val_set.groupby('path',as_index=False).agg({'data': lambda x: tuple(x), 'yolo': lambda x: tuple(x)})
print(len(train_set)/len(val_set))