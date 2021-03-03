import cv2 as cv
import torch
import pandas as pd
import numpy as np
import imgaug.augmenters as iaa
import random
import ast
import matplotlib.pyplot as plt


def show(img):
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])
    plt.show()


def get_indices(series):
    cls0 = list(np.where(np.array(series) == 0))[0].tolist()
    cls1 = list(np.where(np.array(series) == 1))[0].tolist()
    cls2 = list(np.where(np.array(series) == 2))[0].tolist()
    cls3 = list(np.where(np.array(series) == 3))[0].tolist()
    random.shuffle(cls3)
    indices = cls0 * 90 + cls1 * 6 + cls2 + cls3[:7300]
    random.shuffle(indices)
    return indices


def split_data(df):
    val_indexes = [index for index in range(len(df)) if index % 5 == 0]
    test_indexes = [index for index in range(len(df)) if index % 6 == 0]
    train = df.drop(index=val_indexes+test_indexes)
    train.reset_index(drop=True,inplace=True)
    val = df.iloc[val_indexes,:]
    val.reset_index(drop=True,inplace=True)
    test = df.iloc[test_indexes,:]
    test.reset_index(drop=True,inplace=True)
    return train.copy(),val.copy(),test.copy()


def check_for_leakage(df1, df2,args):
    patients_in_both_groups = np.intersect1d(df1[args.patients].values, df2[args.patients].values)
    patients_in_both_groups = set(patients_in_both_groups)
    leakage = (len(patients_in_both_groups) > 0)
    if leakage:
        print('Found leakage, fixing leakage')
        leakage_index = [index for index, value in enumerate(df1[args.patients]) if value == list(patients_in_both_groups)[0]]
        df1.drop(leakage_index, inplace=True)
        df1.reset_index(drop=True,inplace=True)
    return df1,df2


def normed_weights(df):
    w = df['class'].value_counts().tolist()
    return torch.FloatTensor([1 - (x / sum(w)) for x in w][::-1])


def make_ett_data(args):
    df = pd.read_csv(args.csv)
    df[args.labels] = df[args.ett].values.tolist()
    df.set_index(args.uid, inplace=True)
    df_annot = pd.read_csv(args.csv_annotations)
    df_annot = df_annot[df_annot[args.labels[:-1]].isin(args.ett)]
    df_annot.set_index(args.uid, inplace=True)
    df = pd.concat([df, df_annot], axis=1)
    df.reset_index(inplace=True)
    df[args.cls] = [i.index(1) if sum(i) == 1 else 3 for i in df[args.labels]]
    df[args.annotations] = [np.array(ast.literal_eval(i))
                                  if type(i) != float else [] for i in df[args.data]]
    df.rename(columns={'index': 'path'}, inplace=True)
    df.drop(args.all_labels + [args.swan_ganz] + ['label', 'labels','data'], axis=1, inplace=True)
    df['has_bool'] = [1 if i in [0,1,2] else 0 for i in df[args.cls]]
    df['path'] = [args.image_path + i + '.jpg' for i in df['path']]
    assert df.notnull().all().all()
    df.sort_values(args.cls).reset_index(drop=True,inplace=True)
    return df


def aug_img(img):
    img = np.expand_dims(img, axis=0)
    one = iaa.OneOf([
                     iaa.Affine(rotate=(-10, 10),mode='constant'),
                     iaa.ScaleX((0.7, 1.3)),
                     iaa.ScaleY((0.9, 1.1)),
                     iaa.PerspectiveTransform(scale=(0.01, 0.1)),
                     ])
    two = iaa.OneOf([
                       iaa.GaussianBlur(sigma=(0.0, 3.0)),
                        iaa.LinearContrast((0.4, 1.6)),
                       ])
    simetimes1 = iaa.Sometimes(0.25, iaa.Fliplr(1))
    simetimes2 = iaa.Sometimes(0.5,two)
    seq = iaa.Sequential([one,simetimes1,simetimes2],random_order=True)
    images_aug = seq(images=img)
    return images_aug[0]


def start(args):
    df = make_ett_data(args)
    train,val,test = split_data(df)
    train,val = check_for_leakage(train, val,args)
    train,test = check_for_leakage(train, test,args)
    val,test = check_for_leakage(val, test,args)
    train.drop(args.patients,inplace=True,axis=1)
    val.drop(args.patients,inplace=True,axis=1)
    test.drop(args.patients,inplace=True,axis=1)
    return train.values, val.values, test.values, normed_weights(train)


def fill(path,aug=False):
    img = cv.imread(path)
    if aug:
        img = aug_img(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    bin_img = img[0:2000, 700:-700]
    _, bin_img = cv.threshold(bin_img, 120, 255, cv.THRESH_BINARY)
    bin_img = cv.resize(bin_img,
                    (int(bin_img.shape[1] * 0.20), int(bin_img.shape[0] * 0.20)),
                    interpolation=cv.INTER_AREA)
    temp_list = [sum(bin_img[:,j]) for j in range(bin_img.shape[1])]
    temp_index = (5 * temp_list.index(max(temp_list))) + 700
    img = img[100:1200,temp_index-550:temp_index+550]
    img[:, :200] = 0
    img[:, 900:] = 0
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    return img


