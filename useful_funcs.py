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