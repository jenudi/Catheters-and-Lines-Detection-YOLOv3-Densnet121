import cv2 as cv
import torch
import pandas as pd
import numpy as np
import random
import imgaug.augmenters as iaa


def get_indices(series):
    cls0 = list(np.where(np.array(series) == 0))[0].tolist()
    cls1 = list(np.where(np.array(series) == 1))[0].tolist()
    cls2 = list(np.where(np.array(series) == 2))[0].tolist()
    cls3 = list(np.where(np.array(series) == 3))[0].tolist()
    random.shuffle(cls3)
    indices = cls0 * 90 + cls1 * 6 + cls2 + cls3[:7300]
    random.shuffle(indices)
    return indices


def make_weights_for_balanced_classes(df, nclasses): # make a row with len similar to dataset where each value is the weight of the class of the specific row (the weight gives more power to the less dominant class)
    count = [0] * nclasses
    for i in df['class']:
        count[i] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(df)
    for idx, val in enumerate(df['class']):
        weight[idx] = weight_per_class[val]
    return torch.DoubleTensor(weight)


def check_for_none_imgs(df,args):
    temp_list = list()
    for index,path in enumerate(df[args.img_col]):
        img_path = args.images + path + '.jpg'
        img = cv.imread(img_path)
        if img is None:
            print('Found a None type')
            temp_list.append(img_path)
        if index % 50 == 0:
            print(f'Index number {index}')
    if len(temp_list) != 0:
        print(temp_list)
    else:
        print('Did not find any None Type')


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
    w = df.labels.value_counts().tolist()
    return torch.FloatTensor([1 - (x / sum(w)) for x in w][::-1])


def start(args):
    df = make_ett(args)
    train,val,test = split_data(df)
    train,val = check_for_leakage(train, val,args)
    train,test = check_for_leakage(train, test,args)
    val,test = check_for_leakage(val, test,args)
    train.drop(args.patients,inplace=True,axis=1)
    val.drop(args.patients,inplace=True,axis=1)
    test.drop(args.patients,inplace=True,axis=1)
    return train.values, val.values, test.values, normed_weights(train)


def aug_img(img):
    img = np.expand_dims(img, axis=0)
    seq = iaa.Sequential([iaa.Fliplr(0.5), # horizontal flips
                          iaa.Crop(percent=(0, 0.1)),
                          iaa.Sometimes(0.5,iaa.GaussianBlur(sigma=(0, 0.5))),
                          iaa.LinearContrast((0.75, 1.5)),
                          iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                          iaa.Multiply((0.8, 1.2), per_channel=0.2),
                          iaa.Affine(scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                                     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                                     rotate=(-5, 5),
                                     #shear=(-8, 8)
                                     )],
                         random_order=True)
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
    #img1 = cv.resize(img, (int(img.shape[1] * 0.5),int(img.shape[0] * 0.5)),interpolation = cv.INTER_AREA)
    #cv.imshow('Input', img1)
    #cv.waitKey()
    return img


def make_ett(args):
    df = pd.read_csv(args.csv)
    df[args.labels] = df[args.ett].values.tolist()
    df.drop(args.all_labels+[args.swan_ganz],axis=1,inplace=True)
    df[args.labels] = [i + [1] if sum(i) == 0 else i + [0] for i in df[args.labels]]
    df[args.cls] = [i.index(1) for i in df[args.labels]]
    df.rename(columns={args.uid:'path'},inplace=True)
    df['path'] = [args.image_path + i + '.jpg' for i in df['path']]
    assert df.notnull().all().all()
    df.sort_values(args.labels).reset_index(drop=True,inplace=True)
    return df


def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0,0,0
    for data, _,_ in loader:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches +=1
        print(num_batches)
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5
    return mean, std

#m = tensor([0.3165])
#s = tensor([0.2770])