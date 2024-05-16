from __future__ import absolute_import, division, print_function, unicode_literals
import torch.optim as optim
import pandas as pd
import numpy as np
import math
import random
import argparse, sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from torch.utils.data import DataLoader
from torch.nn import Parameter
import time
from collections import Counter
import matplotlib.pyplot as plt
from model import *
from utils import *

path = '/content/drive/MyDrive/Colab Notebooks/others/PG-MDAN'
os.chdir(path)
from google.colab import drive
drive.mount('/content/drive')

def train(training_mode, feature_extractor, class_classifier, domain_classifier, class_criterion, domain_criterion,
        source_dataloader1, source_dataloader2, source_dataloader3, source_dataloader4, target_dataloader,val_dataloader, optimizer, epoches):
  for epoch in range(epoches):
    early_stopping = EarlyStopping(patience=10, verbose=True)
    # track the losses as the model trains
    train_losses,valid_losses,avg_train_losses,avg_valid_losses = [],[],[],[]
    # steps
    start_steps = epoch * len(source_dataloader1)
    total_steps = 10 * len(source_dataloader1)

    if training_mode == 'Pre-train':
      print('Pre-train Epoch: {}'.format(epoch))
      # setup models
      feature_extractor.train()
      class_classifier.train()
      domain_classifier.train()
      for batch_idx, (sdata1, sdata2, sdata3, sdata4, tdata) in enumerate(zip(source_dataloader1, source_dataloader2,
      source_dataloader3, source_dataloader4, target_dataloader)):

        # setup hyperparameters
        p = float(batch_idx + start_steps) / total_steps
        constant = 2. / (1. + np.exp(-gamma * p)) - 1

        # prepare the data
        input1s, label1s= sdata1
        input2s, label2s= sdata2
        input3s, label3s= sdata3
        input4s, label4s= sdata4
        input2, label2= tdata

        size = min((input1s.shape[0], input2.shape[0]))
        input1s, label1s = input1s[0:size, :, :], label1s[0:size]
        input2s, label2s = input2s[0:size, :, :], label2s[0:size]
        input3s, label3s = input3s[0:size, :, :], label3s[0:size]
        input4s, label4s = input4s[0:size, :, :], label4s[0:size]
        input2, label2 = input2[0:size, :, :], label2[0:size]

        input1s, label1s= Variable(input1s.to(device)), Variable(label1s.to(device).float())
        input2s, label2s= Variable(input2s.to(device)), Variable(label2s.to(device).float())
        input3s, label3s= Variable(input3s.to(device)), Variable(label3s.to(device).float())
        input4s, label4s= Variable(input4s.to(device)), Variable(label4s.to(device).float())
        input2, label2= Variable(input2.to(device)), Variable(label2.to(device).float())

        # setup optimizer
        optimizer = optimizer_scheduler(optimizer, p)
        optimizer.zero_grad()

        # prepare domain labels
        source_label1s = Variable(torch.zeros((input1s.size()[0])).type(torch.FloatTensor).to(device))
        source_label2s = Variable(torch.zeros((input2s.size()[0])).type(torch.FloatTensor).to(device))
        source_label3s = Variable(torch.zeros((input3s.size()[0])).type(torch.FloatTensor).to(device))
        source_label4s = Variable(torch.zeros((input4s.size()[0])).type(torch.FloatTensor).to(device))

        target_labels = Variable(torch.ones((input2.size()[0])).type(torch.FloatTensor).to(device))

        # compute the output of source domain and target domain
        src_feature1 = feature_extractor(input1s)
        src_feature2 = feature_extractor(input2s)
        src_feature3 = feature_extractor(input3s)
        src_feature4 = feature_extractor(input4s)
        tgt_feature = feature_extractor(input2)

        # compute the loss of source & target preds
        class_pred1s = class_classifier(src_feature1)
        class_pred2s = class_classifier(src_feature2)
        class_pred3s = class_classifier(src_feature3)
        class_pred4s = class_classifier(src_feature4)
        class_predt = class_classifier(tgt_feature)

        class_loss1s = class_criterion(class_pred1s, label1s)
        class_loss2s = class_criterion(class_pred2s, label2s)
        class_loss3s = class_criterion(class_pred3s, label3s)
        class_loss4s = class_criterion(class_pred4s, label4s)
        class_losst = class_criterion(class_predt, label2)

        # compute the domain loss of src_feature and target_feature
        tgt_preds = domain_classifier(tgt_feature, constant)
        src_preds1 = domain_classifier(src_feature1, constant)
        src_preds2 = domain_classifier(src_feature2, constant)
        src_preds3 = domain_classifier(src_feature3, constant)
        src_preds4 = domain_classifier(src_feature4, constant)

        tgt_loss = domain_criterion(tgt_preds.float(), target_labels.long())
        src_loss1 = domain_criterion(src_preds1.float(), source_label1s.long())
        src_loss2 = domain_criterion(src_preds2.float(), source_label2s.long())
        src_loss3 = domain_criterion(src_preds3.float(), source_label3s.long())
        src_loss4 = domain_criterion(src_preds4.float(), source_label4s.long())

        # static weights
        domain_loss = 1*tgt_loss + 0.37*src_loss1 + 0.06*src_loss2 + 0.31*src_loss3 + 0.26*src_loss4
        class_loss = class_losst + 0.37*class_loss1s + 0.06*class_loss2s + 0.31*class_loss3s + 0.26*class_loss4s

        #dinamic weights
        # domain_loss = MFD_weights[epoch][0]*tgt_loss + MFD_weights[epoch][1]*src_loss1 + MFD_weights[epoch][2]*src_loss2 + MFD_weights[epoch][3]*src_loss3 + MFD_weights[epoch][4]*src_loss4
        # class_loss = MFD_weights[epoch][0]*class_losst + MFD_weights[epoch][1]*class_loss1s + MFD_weights[epoch][2]*class_loss2s + MFD_weights[epoch][3]*class_loss3s + MFD_weights[epoch][4]*class_loss4s

        class_loss = class_loss/(5*batch_size*40*6)

        loss = (1-theta)*class_loss + theta * domain_loss

        loss=loss.float()
  #             print(loss.dtype)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # print loss
        if (batch_idx + 1) % 10 == 0:
            print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tPred Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
              batch_idx * len(input2), len(target_dataloader.dataset),
              100. * batch_idx / len(target_dataloader), loss.item(), class_loss.item(),
              domain_loss.item()
            ))

            total_loss.append(loss.item())
            c_loss.append(class_loss.item())
            d_loss.append(domain_loss.item())


    elif training_mode == 'Fine-tune':
      print('FT Epoch: {}'.format(epoch))
      # setup models
      feature_extractor.train()
      class_classifier.train()
      domain_classifier.train()
      for batch_idx,tdata in enumerate(target_dataloader):
        p = float(batch_idx + start_steps) / total_steps
        constant = 2. / (1. + np.exp(-gamma * p)) - 1
        #prepare the target data
        input2, label2, = tdata

        # print(input2.shape)
        # print(input2.type)
        input2, label2, = Variable(input2.to(device)), Variable(label2.to(device).float())

        # setup optimizer
        optimizer = optimizer_scheduler(optimizer, p)
        optimizer.zero_grad()

        # compute the output of target domain
        tgt_feature = feature_extractor(input2)

        # compute the loss of target preds
        class_predt = class_classifier(tgt_feature)

        class_losst = class_criterion(class_predt, label2)

        loss = class_losst/(batch_size*40*6)
        loss=loss.float()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # print loss
        if (batch_idx + 1) % 10 == 0:
          print('[({:.0f}%)]\tPreds Loss: {:.6f}'.format(
            100. * batch_idx / len(target_dataloader), loss.item()))
      # val
      feature_extractor.eval()
      class_classifier.eval()
      for batch_idx,vdata in enumerate(val_dataloader):
        input_val, label_val, = vdata
        input_val, label_val, = Variable(input_val.to(device)), Variable(label_val.to(device).float())
        # forward pass: compute predicted outputs by passing inputs to the model
        tgt_feature_val = feature_extractor(input_val)
        # compute the loss of target preds
        class_pred_val = class_classifier(tgt_feature_val)

        # calculate the loss
        loss_val = class_criterion(class_pred_val, label_val)
        # record validation loss
        valid_losses.append(loss_val.item())

      train_loss = np.average(train_losses)
      valid_loss = np.average(valid_losses)
      avg_train_losses.append(train_loss)
      avg_valid_losses.append(valid_loss)

      train_losses = []
      valid_losses = []
      # early_stopping needs the validation loss to check if it has decresed,
      # and if it has, it will make a checkpoint of the current model
      early_stopping(valid_loss, class_classifier)

      if early_stopping.early_stop:
        print("Early stopping")
        break


def test1(feature_extractor, class_classifier, domain_classifier,target_dataloader):
  """
  Test the performance of the model
  :param feature_extractor: network used to extract feature from target samples
  :param class_classifier: network used to predict labels
  :param domain_classifier: network used to predict domain
  :param source_dataloader: test dataloader of source domain
  :param target_dataloader: test dataloader of target domain
  :return: None
  """
  feature_extractor.eval()
  class_classifier.eval()
  domain_classifier.eval()
  target_correct = 0.0
  domain_correct = 0.0
  tgt_correct = 0.0

  pred_list_t = []
  label_list_t = []
  for batch_idx, tdata in enumerate(target_dataloader):
    # setup hyperparameters
    p = float(batch_idx) / len(target_dataloader)
    constant = 2. / (1. + np.exp(-10 * p)) - 1

    inputt, labelt= tdata
    inputt, labelt= Variable(inputt.to(device)), Variable(labelt.to(device).float())
    tgt_labels = Variable(torch.ones((inputt.size()[0])).type(torch.FloatTensor).to(device))

    outputt = class_classifier(feature_extractor(inputt))
    predt = outputt
    pred_list_t.append(predt.tolist())
    label_list_t.append(labelt.tolist())

    tgt_preds = domain_classifier(feature_extractor(inputt), constant)
    tgt_preds = tgt_preds.data.max(1, keepdim=True)[1]
    tgt_correct += tgt_preds.eq(tgt_labels.data.view_as(tgt_preds)).cuda().sum()



  target_nrmse = nrmse_func2(pred_list_t, label_list_t)
  target_smape = smape_loss_func(pred_list_t, label_list_t)
  target_mae = mae_loss_func(pred_list_t, label_list_t)

  print('\nTarget error: {}/{}/{} \n'.format(target_nrmse, target_smape, target_mae))
  t_nrmse_list.append(target_nrmse)
  t_smape_list.append(target_smape)
  t_mae_list.append(target_mae)
  return t_mae_list,t_smape_list,t_nrmse_list
  # np.save(file = r'.\preds\nrmse=%.4f, mae=%.4f, smape=%.4f' %(target_nrmse, target_mae, target_smape), arr = pred_list_t)

def main():
  # prepare the source data and target data
  image_train_s1, image_test_s1, label_train_s1, label_test_s1, image_train_s2, image_test_s2, label_train_s2, label_test_s2, image_train_s3, image_test_s3, label_train_s3, label_test_s3, image_train_s4, image_test_s4, label_train_s4, label_test_s4, image_train_t,image_test_t,label_train_t,label_test_t,test,label_test= time_slide(time_slide1s=0/31, time_slide1t=22/31, time_slide2=25/31)

  print(label_train_s1.shape)
  time01 = time.time()
  val_num=144
  src_train_dataloader1 = get_train_loader(image_train_s1,label_train_s1,batch_size=batch_size,shuffle=True)
  src_test_dataloader1 = get_test_loader(image_test_s1,label_test_s1,batch_size=batch_size,shuffle=True)
  src_train_dataloader2 = get_train_loader(image_train_s2,label_train_s2,batch_size=batch_size,shuffle=True)
  src_test_dataloader2 = get_test_loader(image_test_s2,label_test_s2,batch_size=batch_size,shuffle=True)
  src_train_dataloader3 = get_train_loader(image_train_s3,label_train_s3,batch_size=batch_size,shuffle=True)
  src_test_dataloader3 = get_test_loader(image_test_s3,label_test_s3,batch_size=batch_size,shuffle=True)
  src_train_dataloader4 = get_train_loader(image_train_s4,label_train_s4,batch_size=batch_size,shuffle=True)
  src_test_dataloader4 = get_test_loader(image_test_s4,label_test_s4,batch_size=batch_size,shuffle=True)

  tgt_train_dataloader = get_train_loader(image_train_t,label_train_t,batch_size=batch_size,shuffle=True)
  tgt_test_dataloader = get_test_loader(image_test_t,label_test_t,batch_size=batch_size,shuffle=True)
  val_train_dataloader = get_train_loader(image_test_t[:val_num,:,:,:],label_test_t[:val_num,:,:],batch_size=batch_size,shuffle=True)

  time02 = time.time()

  print('data_process_time: '+ str (time02-time01))

  # init models
  feature_extractor = Extractor().to(device)
  class_classifier = Predictor().to(device)
  domain_classifier = Domain_classifier().to(device)

  # init criterions
  class_criterion = nn.MSELoss().to(device)
  # domain_criterion = torch.nn.CrossEntropyLoss().to(device)
  domain_criterion = nn.NLLLoss()

  # init optimizer
  optimizer = optim.Adam([
          {'params': feature_extractor.parameters()},
                          {'params': class_classifier.parameters()},
                          {'params': domain_classifier.parameters()}
  ], lr= 0.001)

  time1 = time.time()

  train('Pre-train', feature_extractor, class_classifier, domain_classifier, class_criterion, domain_criterion,
    src_train_dataloader1, src_train_dataloader2, src_train_dataloader3, src_train_dataloader4, tgt_train_dataloader,val_train_dataloader, optimizer, epoches=50)
  test1(feature_extractor, class_classifier, domain_classifier,tgt_test_dataloader)

  time2 = time.time()
  train('Fine-tune', feature_extractor, class_classifier, domain_classifier, class_criterion, domain_criterion,
    src_train_dataloader1, src_train_dataloader2, src_train_dataloader3, src_train_dataloader4, tgt_train_dataloader,val_train_dataloader, optimizer, epoches=40)
  test1(feature_extractor, class_classifier, domain_classifier,tgt_test_dataloader)

  time3 = time.time()
  print('pretraining time: ' + str(time2-time1))
  print('finetuning time: ' + str(time3-time2))

device = torch.device("cuda:0")
total_loss, d_loss, c_loss = [],[],[]
s1_nrmse_list, s2_nrmse_list, s3_nrmse_list, s4_nrmse_list, t_nrmse_list, domain_loss_list = [],[],[],[],[],[]
s1_smape_list, s2_smape_list, s3_smape_list, s4_smape_list, t_smape_list = [],[],[],[],[]
s1_mae_list, s2_mae_list, s3_mae_list, s4_mae_list, t_mae_list = [],[],[],[],[]
near_road_target = np.argsort(np.array(pd.read_csv('./data/dis_blue.csv',header = None)))
flow_target = np.array(pd.read_csv('./data/flow_blue.csv', header= 0))

near_road_source1 = np.argsort(np.array(pd.read_csv('./data/dis_green.csv',header = None)))
flow_source1 = np.array(pd.read_csv('./data/flow_green.csv', header= 0))

near_road_source2 = np.argsort(np.array(pd.read_csv('./data/dis_yellow.csv',header = None)))
flow_source2 = np.array(pd.read_csv('./data/flow_yellow.csv', header= 0))

near_road_source3 = np.argsort(np.array(pd.read_csv('./data/dis_purple.csv',header = None)))
flow_source3 = np.array(pd.read_csv('./data/flow_purple.csv', header= 0))

near_road_source4 = np.argsort(np.array(pd.read_csv('./data/dis_red.csv',header = None)))
flow_source4 = np.array(pd.read_csv('./data/flow_red.csv', header= 0))
# pred_list_s1, pred_list_s2, pred_list_s3, pred_list_s4, pred_list_t = [],[],[],[],[]
# label_list_s1, label_list_s2, label_list_s3, label_list_s4, label_list_t = [],[],[],[],[]

if __name__ == '__main__':
  setup_seed(0)
  target_links=40
  gamma = 5
  theta = 0.05
  batch_size = 64
  time_start=time.time()
  main()
  time_end=time.time()
  print('total run time: (min)',(time_end-time_start)/60.)
