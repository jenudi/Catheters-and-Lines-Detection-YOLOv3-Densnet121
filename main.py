class Ranzcr:
  def __init__(self,img_size,num_classes,lr,batch_size,num_epochs,weight_decay,conf_t = 0.05,nms_iou_t=0.5):
    self.use_cuda = torch.cuda.is_available()
    self.device = torch.device("cuda" if self.use_cuda else "cpu")
    self.img_size = img_size
    self.num_classes = num_classes
    self.lr = lr
    self.batch_size = batch_size
    self.num_epochs = num_epochs
    self.weight_decay = weight_decay
    self.num_workers = 0
    self.S = [self.img_size // 32, self.img_size // 16, self.img_size // 8]
    self.conf_t = conf_t
    self.map_iou_t = np.arange(0.5, 1.0, 0.05)
    self.nms_iou_t = nms_iou_t
    self.model = YOLOv3(num_classes=self.num_classes).to(self.device)
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=8,gamma=0.05)

  def save_checkpoint(self, filename="yolo_model.pth.tar"):
    checkpoint = {"state_dict": self.model.state_dict(),"optimizer": self.optimizer.state_dict(),}
    torch.save(checkpoint, filename)

  def load_checkpoint(self, checkpoint_file="yolo_model.pth.tar",):
    checkpoint = torch.load(checkpoint_file, map_location=self.device)
    self.model.load_state_dict(checkpoint["state_dict"])
    self.optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in self.optimizer.param_groups:
        param_group["lr"] = self.lr

  def val_loss(self, val_loader, loss_fn):
    self.model.eval()
    loop = tqdm(val_loader, position=0,leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(self.device)
        y0, y1, y2 = (y[0].to(self.device),y[1].to(self.device),y[2].to(self.device))
        with torch.no_grad():
            out = self.model(x)
            loss = (loss_fn(out[0], y0, self.scaled_anchors[0])+ loss_fn(out[1], y1, self.scaled_anchors[1])+ loss_fn(out[2], y2, self.scaled_anchors[2]))
        losses.append(loss.item())
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)
    self.model.train()

  def train_fn(self, train_loader, loss_fn, scaler):
    loop = tqdm(train_loader, position=0,leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(self.device)
        y0, y1, y2 = (y[0].to(self.device),y[1].to(self.device),y[2].to(self.device))
        with torch.cuda.amp.autocast():
          out = self.model(x)
          loss = (loss_fn(out[0], y0, self.scaled_anchors[0])+ loss_fn(out[1], y1, self.scaled_anchors[1])+ loss_fn(out[2], y2, self.scaled_anchors[2]))
        losses.append(loss.item())
        self.optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)

  def init_dls(self):
    train_data_set = YOLODataset(values=train_set.values,anchors= self.anchors,transform=new_transforms)
    train_data_loader = DataLoader(train_data_set,batch_size=self.batch_size,num_workers=self.num_workers,pin_memory=self.use_cuda, sampler=sampler)
    val_data_set = YOLODataset(values=val_set.values,anchors= self.anchors,transform=test_transforms)
    val_data_loader = DataLoader(val_data_set,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,pin_memory=self.use_cuda)
    return train_data_loader, val_data_loader

  def main(self):
    loss_fn = YOLOLoss()
    scaler = torch.cuda.amp.GradScaler()
    train_dl, val_dl = self.init_dls()
    self.model.train()
    for epoch in range(self.num_epochs):
        self.train_fn(train_dl, loss_fn, scaler,) 
        self.lr_scheduler.step()   
        if (1+ epoch) % 5 == 0:
            self.val_loss(val_dl, loss_fn)
            self.save_checkpoint()
            self.model.train()
