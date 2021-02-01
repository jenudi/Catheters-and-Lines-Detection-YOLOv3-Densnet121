# Imports
from preprocess import *
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.optim as optim
import torchvision.transforms as transforms
import datetime
import time
from torch.utils.tensorboard import SummaryWriter
import os
from base_CNN import ProModel
from args import ProArgs
# import torch.cuda

# %%
args = ProArgs(img_size=320,learning_rate = 1e-3,batch_size = 20,num_epochs=5)


METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3


# Dataset
class ProDataset(Dataset):
    def __init__(self, values, transform=None, train_set=True):
        self.values = values
        self.transform = transform
        self.train_set = train_set
        self.indices_to_aug = list()
        if train_set:
            self.indices = get_indices(values[:, 2])

    def __getitem__(self, index):  # 0: path, 1: labels, 2: cls
        if self.train_set:
            augmentations = False
            index = random.sample(self.indices, 1)[0]
            self.indices.remove(index)
            if index not in self.indices_to_aug and\
                    (self.values[index][2] == 0 or self.values[index][2] == 1):
                self.indices_to_aug.append(index)
            else:
                augmentations = True
            img = fill(path=self.values[index][0], aug=augmentations)
        else:
            img = fill(self.values[index][0])
        if img is None:
            print(index)
        if self.transform:
            img = self.transform(img)
        y = torch.tensor(self.values[index][1])
        z = torch.tensor([self.values[index][2]])
        return img, y, z

    def __len__(self):
        return len(self.indices) if self.train_set else len(self.values)


class Ranzcr:
    def __init__(self, args):

        self.args = args
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = self.model()
        self.optimizer = self.optimizer()
        self.class_weights = None
        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

    def model(self):
        model = ProModel()
        if self.use_cuda:
            print(f"Using CUDA; {torch.cuda.device_count()} devices.")
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def optimizer(self):
        return optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.99)

    def init_dls(self):
        batch_size = self.args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        train_set, val_set, test_set, self.class_weights = start(self.args)
        train_data_set = ProDataset(values=train_set,
                                    transform=transforms.Compose([transforms.ToPILImage(),
                                                                  transforms.Resize(
                                                                      size=(self.args.img_size, self.args.img_size)),
                                                                  transforms.Grayscale(num_output_channels=1),
                                                                  transforms.ToTensor(), ]))
        # transforms.Normalize([0.3165], [0.2770])]))

        train_data_loader = DataLoader(train_data_set,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=self.args.num_workers,
                                       pin_memory=self.use_cuda)

        val_data_set = ProDataset(values=val_set,
                                  train_set=False,
                                  transform=transforms.Compose([transforms.ToPILImage(),
                                                                transforms.Resize(
                                                                    size=(self.args.img_size, self.args.img_size)),
                                                                transforms.Grayscale(num_output_channels=1),
                                                                transforms.ToTensor(), ]))
        # transforms.Normalize([0.3165], [0.2770])]))

        val_data_loader = DataLoader(val_data_set,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=self.args.num_workers,
                                     pin_memory=self.use_cuda)

        return train_data_loader, val_data_loader

    def main(self):
        print('Starting Ranzcr.main()')
        self.initTensorboardWriters()
        train_dl, val_dl = self.init_dls()
        for epoch_ndx in range(1, self.args.num_epochs + 1):
            trnMetrics_t = self.training(epoch_ndx, train_dl)
            self.log_metrics(epoch_ndx, 'trn', trnMetrics_t)
            valMetrics_t = self.validation(epoch_ndx, val_dl)
            self.log_metrics(epoch_ndx, 'val', valMetrics_t)
        print("Finished: Ranzcr.main()")

    def training(self, epoch_ndx, train_dl):
        self.model.train()
        trnMetrics_g = torch.zeros(METRICS_SIZE,
                                   len(train_dl.dataset),
                                   device=self.device)
        batch_iter = self.estimation(train_dl,
                                     desc_str=f"{epoch_ndx} Training",
                                     start_ndx=train_dl.num_workers)

        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()
            loss = self.compute_batch_loss(batch_ndx,
                                           batch_tup,
                                           train_dl.batch_size,
                                           trnMetrics_g)

            print(f"batch: {batch_ndx}, loss: {loss}")
            loss.backward()
            self.optimizer.step()
        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics_g.to('cpu')

    def validation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(METRICS_SIZE,
                                       len(val_dl.dataset),
                                       device=self.device)

            batch_iter = self.estimation(val_dl,
                                         desc_str=f"{epoch_ndx} Validation ",
                                         start_ndx=val_dl.num_workers)

            for batch_ndx, batch_tup in batch_iter:
                self.compute_batch_loss(batch_ndx,
                                        batch_tup,
                                        val_dl.batch_size,
                                        valMetrics_g)

        return valMetrics_g.to('cpu')

    def compute_batch_loss(self, batch_ndx,
                           batch_tup,
                           batch_size,
                           metrics_g):

        input_t, label_t, cls0 = batch_tup
        cls1 = torch.reshape(cls0, (-1,))
        input_g = input_t.to(self.device,
                             non_blocking=True)
        cls1 = cls1.to(self.device,
                       non_blocking=True)
        logits_g, probability_g = self.model(input_g)
        loss_func = nn.CrossEntropyLoss(reduction='none')  # weight=self.class_weights,

        loss_g = loss_func(logits_g, cls1)

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)
        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = cls1.detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = torch.max(probability_g, 1)[1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g.detach()

        return loss_g.mean()

    def log_metrics(self, epoch_ndx, mode_str, metrics_t):

        cls0_label_mask = metrics_t[METRICS_LABEL_NDX] == 0
        cls1_label_mask = metrics_t[METRICS_LABEL_NDX] == 1
        cls2_label_mask = metrics_t[METRICS_LABEL_NDX] == 2
        cls3_label_mask = metrics_t[METRICS_LABEL_NDX] == 3
        cls0_pred_mask = metrics_t[METRICS_PRED_NDX] == 0
        cls1_pred_mask = metrics_t[METRICS_PRED_NDX] == 1
        cls2_pred_mask = metrics_t[METRICS_PRED_NDX] == 2
        cls3_pred_mask = metrics_t[METRICS_PRED_NDX] == 3
        cls0_count = int(cls0_label_mask.sum())
        cls1_count = int(cls1_label_mask.sum())
        cls2_count = int(cls2_label_mask.sum())
        cls3_count = int(cls3_label_mask.sum())

        cls0_true_p = int((cls0_label_mask & cls0_pred_mask).sum())
        cls1_true_p = int((cls1_label_mask & cls1_pred_mask).sum())
        cls2_true_p = int((cls2_label_mask & cls2_pred_mask).sum())
        cls3_true_p = int((cls3_label_mask & cls3_pred_mask).sum())

        true_p = cls0_true_p + cls1_true_p + cls2_true_p + cls3_true_p

        cls0_false_n = int((cls0_label_mask & cls3_pred_mask).sum()) + \
                       int((cls0_label_mask & cls1_pred_mask).sum()) + \
                       int((cls0_label_mask & cls2_pred_mask).sum())
        cls1_false_n = int((cls1_label_mask & cls0_pred_mask).sum()) + \
                       int((cls1_label_mask & cls2_pred_mask).sum()) + \
                       int((cls1_label_mask & cls3_pred_mask).sum())
        cls2_false_n = int((cls2_label_mask & cls0_pred_mask).sum()) + \
                       int((cls2_label_mask & cls1_pred_mask).sum()) + \
                       int((cls2_label_mask & cls3_pred_mask).sum())
        cls3_false_n = int((cls3_label_mask & cls0_pred_mask).sum()) + \
                       int((cls3_label_mask & cls1_pred_mask).sum()) + \
                       int((cls3_label_mask & cls2_pred_mask).sum())
        false_n = cls0_false_n + cls1_false_n + cls2_false_n + cls3_false_n

        cls0_false_p = int((cls3_label_mask & cls0_pred_mask).sum()) + \
                       int((cls1_label_mask & cls0_pred_mask).sum()) + \
                       int((cls2_label_mask & cls0_pred_mask).sum())
        cls1_false_p = int((cls3_label_mask & cls1_pred_mask).sum()) + \
                       int((cls0_label_mask & cls1_pred_mask).sum()) + \
                       int((cls2_label_mask & cls1_pred_mask).sum())
        cls2_false_p = int((cls3_label_mask & cls2_pred_mask).sum()) + \
                       int((cls1_label_mask & cls2_pred_mask).sum()) + \
                       int((cls0_label_mask & cls2_pred_mask).sum())
        cls3_false_p = int((cls0_label_mask & cls3_pred_mask).sum()) + \
                       int((cls1_label_mask & cls3_pred_mask).sum()) + \
                       int((cls2_label_mask & cls3_pred_mask).sum())
        false_p = cls0_false_p + cls1_false_p + cls2_false_p + cls3_false_p

        metrics_dict = dict()
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/cls0'] = metrics_t[METRICS_LOSS_NDX, cls0_label_mask].mean()
        metrics_dict['loss/cls1'] = metrics_t[METRICS_LOSS_NDX, cls1_label_mask].mean()
        metrics_dict['loss/cls2'] = metrics_t[METRICS_LOSS_NDX, cls2_label_mask].mean()
        metrics_dict['loss/cls3'] = metrics_t[METRICS_LOSS_NDX, cls3_label_mask].mean()
        metrics_dict['correct/all'] = (cls0_true_p + cls1_true_p + cls2_true_p + cls3_true_p) / np.float32(
            metrics_t.shape[1]) * 100
        metrics_dict['correct/cls0'] = cls0_true_p / np.float32(cls0_count) * 100
        metrics_dict['correct/cls1'] = cls1_true_p / np.float32(cls1_count) * 100
        metrics_dict['correct/cls2'] = cls2_true_p / np.float32(cls2_count) * 100
        metrics_dict['correct/cls3'] = cls3_true_p / np.float32(cls3_count) * 100
        metrics_dict['pr/precision'] = true_p / np.float32(true_p + false_p)
        metrics_dict['pr/recall'] = true_p / np.float32(true_p + false_n)
        metrics_dict['pr/f1_score'] = 2 * (metrics_dict['pr/precision'] * metrics_dict['pr/recall']) / \
                                      (metrics_dict['pr/precision'] + metrics_dict['pr/recall'])

        for key, value in metrics_dict.items():
            if mode_str == 'trn':
                self.trn_writer.add_scalar(key, value, self.totalTrainingSamples_count)
            else:
                self.val_writer.add_scalar(key, value, self.totalTrainingSamples_count)

    def estimation(self, iter, desc_str, start_ndx=0, print_ndx=4, backoff=None, iter_len=None):
        if iter_len is None:
            iter_len = len(iter)
        if backoff is None:
            backoff = 2
            while backoff ** 7 < iter_len:
                backoff *= 2
        assert backoff >= 2
        while print_ndx < start_ndx * backoff:
            print_ndx *= backoff
        print(f"{desc_str} ----/{iter_len}, starting")
        start_ts = time.time()
        for (current_ndx, item) in enumerate(iter):
            yield (current_ndx, item)
            if current_ndx == print_ndx:
                duration_sec = ((time.time() - start_ts) / (current_ndx - start_ndx + 1) * (iter_len - start_ndx))
                done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)
                done_td = datetime.timedelta(seconds=duration_sec)
                print(f"{desc_str} {current_ndx:-4}/{iter_len}, "
                      f"done at {str(done_dt).rsplit('.', 1)[0]}, "
                      f"{str(done_td).rsplit('.', 1)[0]}")
                print_ndx *= backoff
            if current_ndx + 1 == start_ndx:
                start_ts = time.time()
        print(f"{desc_str} ----/{iter_len}, "
              f"done at {str(datetime.datetime.now()).rsplit('.', 1)[0]}")

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.time_str)
            self.trn_writer = SummaryWriter(log_dir=log_dir + '-trn_cls-')
            self.val_writer = SummaryWriter(log_dir=log_dir + '-val_cls-')


training = Ranzcr(args).main()
# tensorboard --logdir=runs
# %%
