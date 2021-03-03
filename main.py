# Imports
from preprocess import *
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import datetime
from torch.utils.tensorboard import SummaryWriter
from Densenet121 import Densenet, clac_param
from args import *
import os
import torch.cuda
clac_param(Densenet())
args = ProArgs(img_size=224,batch_size = 150,num_epochs=30,model=Densenet(),learning_rate=0.0001, momentum=0.9)
#python -m tensorboard.main --logdir=runs
# %%


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

    def __getitem__(self, index):  # 0: path, 1: cls, 2: annot, 3: has_bool 4: aug
        if self.train_set and self.values[index][4] == 1:
            img = fill(self.values[index][0], aug=True)
        else:
            img = fill(self.values[index][0], aug=False)
        if img is None:
            print(index)
        if self.transform:
            img = self.transform(img)
        cls = torch.tensor(self.values[index][1])
        has_bool = torch.tensor([self.values[index][3]])
        return img, cls, has_bool

    def __len__(self):
        return len(self.values)


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
        self.training_loss = None
        self.val_loss = None
        self.training_sampels = 0
        self.num_workers = 0

    def model(self):
        model = args.model
        if self.use_cuda:
            print(f"Using CUDA; {torch.cuda.device_count()} devices.")
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def lr_schedule(self,epoch):
      if 4 < epoch <= 7:
        self.args.learning_rate = 0.00005
      elif 7 < epoch <= 11:
        self.args.learning_rate = 0.000025
      elif epoch > 11:
        self.args.learning_rate = 0.0000125

    def optimizer(self):
        return optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.99,weight_decay=0.1)

    def init_dls(self):
        batch_size = self.args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        train_set, val_set, test_set, self.class_weights = start(self.args)
        train_data_set = ProDataset(values=train_set,
                                    transform=transforms.Compose([transforms.ToPILImage(),
                                                                  transforms.Resize(
                                                                      size=(self.args.img_size, self.args.img_size)),
                                                                  #transforms.Grayscale(num_output_channels=1),
                                                                  transforms.ToTensor(), ]))
        # transforms.Normalize([0.3165], [0.2770])]))
        train_data_loader = DataLoader(train_data_set,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=self.num_workers,
                                       pin_memory=self.use_cuda)
        val_data_set = ProDataset(values=val_set,train_set=False,
                                  transform=transforms.Compose([transforms.ToPILImage(),
                                                                transforms.Resize(
                                                                    size=(self.args.img_size, self.args.img_size)),
                                                                #transforms.Grayscale(num_output_channels=1),
                                                                transforms.ToTensor(), ]))
        # transforms.Normalize([0.3165], [0.2770])]))
        val_data_loader = DataLoader(val_data_set,batch_size=batch_size,shuffle=False,num_workers=self.num_workers,pin_memory=self.use_cuda)
        return train_data_loader, val_data_loader

    def main(self,continue_training=False,decay_learning=False):
        if continue_training:
            try:
                self.model.load_state_dict(torch.load('cnn_model.pth'))
                print('Loaded model for training')
            except FileNotFoundError:
                print('File Not Found')
                return
        else:
            print('Starting Ranzcr.main()')
        self.initTensorboardWriters()
        train_dl, val_dl = self.init_dls()
        for epoch_ndx in range(1, self.args.num_epochs + 1):
            if decay_learning:
              self.lr_schedule(epoch_ndx)
            if epoch_ndx % 4 == 3:
                valMetrics_t = self.validation(epoch_ndx, val_dl)
                self.log_metrics(epoch_ndx, 'val', valMetrics_t)
            else:
                trnMetrics_t = self.training(epoch_ndx, train_dl)
                self.log_metrics(epoch_ndx, 'trn', trnMetrics_t)
        print("Finished: Ranzcr.main()")

    def training(self, epoch_ndx, train_dl):
        self.model.train()
        trnMetrics_g = torch.zeros(METRICS_SIZE,len(train_dl.dataset),device=self.device)
        self.training_loss = 0.0
        for batch_ndx, batch_tup in enumerate(train_dl, 0):
            self.optimizer.zero_grad()
            loss = self.compute_batch_loss(batch_ndx,batch_tup,train_dl.batch_size,trnMetrics_g)
            if batch_ndx % 10 == 9:
              print(f"Train: epoch: {epoch_ndx}, batch: {batch_ndx}, loss: {loss:.4f}")
            self.training_loss += loss
            loss.backward()
            self.optimizer.step()
        torch.save(self.model.state_dict(), 'drive/MyDrive/my_model/cnn_model.pth')
        print('model was saved')
        print(f"Train: epoch: {epoch_ndx}, loss: {self.training_loss/len(train_dl):.4f}\n")
        self.totalTrainingSamples_count += len(train_dl.dataset)
        return trnMetrics_g.to('cpu')

    def validation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(METRICS_SIZE,len(val_dl.dataset),device=self.device)
            self.val_loss = 0.0
            for batch_ndx, batch_tup in enumerate(val_dl, 0):
                loss = self.compute_batch_loss(batch_ndx,batch_tup,val_dl.batch_size,valMetrics_g)
                if batch_ndx % 10 == 9:
                  print(f"Validation: epoch: {epoch_ndx}, batch: {batch_ndx}, loss: {loss:.4f}")
                self.val_loss += loss
            print(f"Validation: epoch: {epoch_ndx}, loss: {self.val_loss/len(val_dl):.4f}\n")
        return valMetrics_g.to('cpu')

    def compute_batch_loss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, cls_t, has_bool = batch_tup
        cls_t = torch.reshape(cls_t, (-1,))
        input_g = input_t.to(self.device,non_blocking=True)
        cls_g = cls_t.to(self.device,non_blocking=True)
        logits_g, probability_g = self.model(input_g)
        loss_func = nn.CrossEntropyLoss(reduction='none') # ,weight=self.class_weights,
        loss_g = loss_func(logits_g, cls_g)
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + cls_g.size(0)
        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = cls_g.detach()
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


    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.time_str)
            self.trn_writer = SummaryWriter(log_dir=log_dir + '-trn_cls-')
            self.val_writer = SummaryWriter(log_dir=log_dir + '-val_cls-')

# %%
if __name__ == '__main__':
    training = Ranzcr(args).main()
# tensorboard --logdir=runs
# %%
