import cv2 as cv
import torch
import pandas as pd
import numpy as np
import imgaug.augmenters as iaa
import random
import ast


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
    val_indexes = [index for index in range(len(df)) if index % 9 == 0]
    test_indexes = [index for index in range(len(df)) if index % 10 == 0]
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
    df[args.annotations] = [np.array(ast.literal_eval(i)[0])
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
    one = iaa.OneOf([iaa.Affine(scale=(0.9,1.1),mode='constant'),
                     iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},mode='constant'),
                     iaa.Affine(rotate=(-5, 5),mode='constant'),
                     iaa.Affine(shear=(-6, 6),mode='constant'),
                     iaa.ScaleX((0.9, 1.1)),
                     iaa.ScaleY((0.9, 1.1)),
                     iaa.PerspectiveTransform(scale=(0.01, 0.05))])
    two = iaa.OneOf([iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)),
                     iaa.AdditiveLaplaceNoise(scale=(0, 0.1 * 255)),
                    iaa.Salt(0.05)])
    three = iaa.OneOf([iaa.GaussianBlur(sigma=1.0),
                        iaa.imgcorruptlike.Fog(severity=1),
                        iaa.imgcorruptlike.Spatter(severity=1)])
    simetimes2 = iaa.Sometimes(0.5, two)
    simetimes3 = iaa.Sometimes(0.05,three)
    seq = iaa.Sequential([one,simetimes2,simetimes3],random_order=True)
    images_aug = seq(images=img)
    return images_aug[0]


def fill(path,aug=False):
    img = cv.imread(path,0)
    _, img = cv.threshold(img,50,255,cv.THRESH_TOZERO)
    img = cv.equalizeHist(img)
    img = cv.filter2D(img, -1, np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]))
    middle = img.shape[1] // 2
    img = img[:1400,middle-600:middle+600]
    img = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
    if aug:
        img = aug_img(img)
    return img


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