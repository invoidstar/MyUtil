import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from dataset import *
from myNet import *
import time 
import datetime
import wandb




def train():
  config = {
        "epochs": 200,
        "batch_size": 8,
        "learning_rate": 1e-3
    }
  # 日志打印
  wandb.init(config = config, project = 'MyNetDemo', name = 'GoPro', job_type = 'training')

  # 数据读取
  train_dataset = GoPro_TrainDataset()
  dataloader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle = True)
  print("%s 数据集训练集一共有图片 %d " % (train_dataset.dataset_name, len(train_dataset)))

  # 模型加载
  device = torch.device('cuda:3')
  device_ids = [1, 2]
  net = UNet()
  net.to(device)

  # 优化器
  # MSELoss的损失函数可参考该博客： https://blog.csdn.net/qq_40968179/article/details/128260036
  optimizer = optim.Adam(net.parameters(), lr=config['learning_rate'])

  # 损失函数
  criterion = nn.MSELoss(reduction = 'mean').to(device)
  
  for epoch in range(config['epochs']):
    if epoch < config['epochs'] * 0.6:
      current_lr = config['learning_rate']
    else:
      current_lr = config['learning_rate']/10

    for param_group in optimizer.param_groups:
      param_group["lr"] = current_lr
      # print('learning rate %f' % current_lr)

    start_time = time.perf_counter()

    for i, data in enumerate(dataloader, 0):
      net.train()
      net.zero_grad()
      optimizer.zero_grad()

      ori_img = data[0]
      noise_data = data[1]

      ori_img = ori_img.to(device)
      noise_data = noise_data.to(device)

      out = net(noise_data)

      loss = criterion(out, ori_img)
      loss.backward()
      optimizer.step()
      
    
    last_time = int(time.perf_counter() - start_time)
    last_time = datetime.timedelta(seconds = last_time)


    print("epoch %d: loss = %f, time = %s" % (epoch, 100*loss.item(), str(last_time)))
    wandb.log({'epoch': epoch, 'loss': 100*loss, 'time': str(last_time)})

  
  torch.save(net.state_dict(), 'weights_500.pth')



def test():
  pass


def ceshi():
  pass  




if __name__ == '__main__':
  train()
  # test()
  # ceshi()


