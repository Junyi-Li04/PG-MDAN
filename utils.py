import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def smape_loss_func(preds, labels):

  preds = np.asarray(preds)
  labels = np.asarray(labels)
  mask= labels > 0
  return np.mean(np.fabs(labels[mask]-preds[mask])/(np.fabs(labels[mask])+np.fabs(preds[mask])))

def mae_loss_func(preds, labels):
  preds = np.asarray(preds)
  labels = np.asarray(labels)
  mask= labels > 0
  return np.fabs((labels[mask]-preds[mask])).mean()

def nrmse_func2(preds, labels):

  preds = np.asarray(preds)
  labels = np.asarray(labels)
  mask = labels > 0
  return np.sqrt(np.sum((labels[mask]-preds[mask])**2)/np.sum(np.square(labels[mask])))

def tensorconstruct (time_slide1, time_slide2, flow, near_road, k = 6, t_p = 31, t_input = 12, t_pre = 6, num_links = 40):
  image = []
  for i in range(np.shape(near_road)[0]):
    road_id = []
    for j in range(k):
      road_id.append(near_road[i][j])
    image.append(flow[road_id, :])
  image1 = np.reshape(image, [-1, k, len(flow[0,:])])
  image2 = np.transpose(image1,(1,2,0))
  image3 = []
  label = []

  for i in range(0,t_p):
    for j in range(144-t_input-t_pre):
      image3.append(image2[:, i*144+j:i*144+j+t_input, :][:])
      label.append(flow[:, i*144+j+t_input:i*144+j+t_input+t_pre][:])

  image3 = np.asarray(image3)
  label = np.asarray(label)

  image_train = image3[math.floor(np.shape(image3)[0]*time_slide1) : math.ceil(np.shape(image3)[0]*time_slide2)]
  image_test = image3[math.floor(np.shape(image3)[0]*time_slide2):]
  label_train = label[math.floor(np.shape(image3)[0]*time_slide1) : math.ceil(np.shape(image3)[0]*time_slide2)]
  label_test = label[math.floor(np.shape(label)[0]*time_slide2):]


  return image_train, image_test, label_train, label_test


def series_impute (small_train, small_label, long_size):
  train_tmp = small_train
  label_tmp = small_label
  for i in range (math.ceil(long_size/len(small_train))):
    train_tmp = np.concatenate((train_tmp,small_train), axis = 0)
    label_tmp = np.concatenate((label_tmp,small_label), axis = 0)

  train_tmp = train_tmp[:long_size,:,:,:]
  label_tmp = label_tmp[:long_size,:,:]

  return train_tmp, label_tmp

def time_slide(time_slide1s, time_slide1t, time_slide2):

  image_train_s1, image_test_s1, label_train_s1, label_test_s1 = tensorconstruct(time_slide1 = time_slide1s, time_slide2=time_slide2,
  flow = flow_source1, near_road=near_road_source1, k = 6, t_p = 31, t_input = 12, t_pre = 6, num_links = 40)

  image_train_s2, image_test_s2, label_train_s2, label_test_s2= tensorconstruct(time_slide1 = time_slide1s, time_slide2=time_slide2,
  flow =flow_source2, near_road=near_road_source2, k = 6, t_p = 31, t_input = 12, t_pre = 6, num_links = 40)

  image_train_s3, image_test_s3, label_train_s3, label_test_s3 = tensorconstruct(time_slide1 = time_slide1s, time_slide2=time_slide2,
  flow =flow_source3, near_road=near_road_source3, k = 6, t_p = 31, t_input = 12, t_pre = 6, num_links = 40)

  image_train_s4, image_test_s4, label_train_s4, label_test_s4 = tensorconstruct(time_slide1 = time_slide1s, time_slide2=time_slide2,
  flow =flow_source4, near_road=near_road_source4, k = 6, t_p = 31, t_input = 12, t_pre = 6, num_links = 40)

  image_train_t, image_test_t, label_train_t, label_test_t = tensorconstruct(time_slide1 = time_slide1t, time_slide2=time_slide2,
  flow =flow_target, near_road=near_road_target, k = 6, t_p = 31, t_input = 12, t_pre = 6, num_links = 40)

  len_t = len(image_train_t)
  len_s = len(image_train_s1)
  long_size = max(len_t,len_s)
  if len_t <= len_s:
    image_train_t,label_train_t= series_impute (image_train_t, label_train_t, long_size)

  image_train_s1 = image_train_s1.reshape(-1, 40, 6, 12)
  image_train_s2 = image_train_s2.reshape(-1, 40, 6, 12)
  image_train_s3 = image_train_s3.reshape(-1, 40, 6, 12)
  image_train_s4 = image_train_s4.reshape(-1, 40, 6, 12)
  image_train_t = image_train_t.reshape(-1, 40, 6, 12)

  image_test_s1 = image_test_s1.reshape(-1, 40, 6, 12)
  image_test_s2 = image_test_s2.reshape(-1, 40, 6, 12)
  image_test_s3 = image_test_s3.reshape(-1, 40, 6, 12)
  image_test_s4 = image_test_s4.reshape(-1, 40, 6, 12)
  image_test_t = image_test_t.reshape(-1, 40, 6, 12)

  test = image_test_t
  label_test = label_test_t

  return image_train_s1, image_test_s1, label_train_s1, label_test_s1, image_train_s2, image_test_s2, label_train_s2, label_test_s2, image_train_s3, image_test_s3, label_train_s3, label_test_s3, image_train_s4, image_test_s4, label_train_s4, label_test_s4, image_train_t, image_test_t, label_train_t, label_test_t, test, label_test


def get_train_loader(data,label,batch_size,shuffle=False):
  """
  Get train dataloader of source domain or target domain
  :return: dataloader
  """
  tensor_x = torch.Tensor(data) # transform to torch tensor
  tensor_y = torch.Tensor(label)

  my_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y) # create your datset
  my_dataloader = torch.utils.data.DataLoader(my_dataset,batch_size=batch_size,shuffle=shuffle,drop_last=True,pin_memory=True) # create your dataloader

  return my_dataloader

def get_test_loader(data,label,batch_size,shuffle=False):
  """
  Get test dataloader of source domain or target domain
  :return: dataloader
  """
  tensor_x = torch.Tensor(data) # transform to torch tensor
  tensor_y = torch.Tensor(label)

  my_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y) # create your datset
  my_dataloader = torch.utils.data.DataLoader(my_dataset,batch_size=batch_size,shuffle=shuffle,drop_last=True,pin_memory=True) # create your dataloader

  return my_dataloader


def optimizer_scheduler(optimizer, p):
  """
  Adjust the learning rate of optimizer
  :param optimizer: optimizer for updating parameters
  :param p: a variable for adjusting learning rate
  :return: optimizer
  """
  for param_group in optimizer.param_groups:
    param_group['lr'] = 0.01 / (1. + 10 * p) ** 0.75

  return optimizer