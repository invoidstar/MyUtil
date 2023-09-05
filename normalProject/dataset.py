import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image
from skimage import util, img_as_float, io
from torchvision import transforms
from utils.measure import Measure




class MyDataset(Dataset):
  def __init__(self):
    self.dataset_name = "DnCNN"
    self.imgsz = (160, 160)
    self.img_dir = 'data/Train400/'
    self.img_namelist = os.listdir(self.img_dir)

  # need to overload
  def __len__(self):
    return len(self.img_namelist)

  # need to overload
  def __getitem__(self, idx):
    img = Image.open(self.img_dir+self.img_namelist[idx])
    img = img.crop((10,10,170,170))

    img_a = img_as_float(img)	

    noiseimg = util.random_noise(img_a, mode = "gaussian")
    noiseimg = transforms.ToPILImage()(np.float32(noiseimg))

    img = transforms.ToTensor()(img)
    noiseimg = transforms.ToTensor()(noiseimg)

    return img, noiseimg

class GoPro_TrainDataset(Dataset):
  def __init__(self):
    self.dataset_name = "GoPro_9G"
    self.data_dir = '/home/chuq/Project/datasets/GoPro_9G/train/'
    self.folder_namelist = os.listdir(self.data_dir)
    self.imgsize = (1280, 720)

    


  def __len__(self):
    sum = 0
    for folder_name in self.folder_namelist:
      img_namelist = os.listdir(self.data_dir + folder_name + '/sharp')
      sum += len(img_namelist)
    return sum
    

  def __getitem__(self, idx):
    for folder_name in self.folder_namelist:
      img_namelist = os.listdir(self.data_dir + folder_name + '/sharp/')
      for img_name in img_namelist:
        gt = Image.open(self.data_dir+folder_name+'/sharp/'+img_name)
        img = Image.open(self.data_dir+folder_name+'/blur/'+img_name)
        # torchvision.transforms.ToTensor把图像转化为[0,1]的torch.float32类型，并改为channel first，将[width,height,channel]->[channel,height,width]
        gt = transforms.ToTensor()(gt)  # 感觉和这个是一个意思 gt = torch.tensor(np.array(gt)/255)
        img = transforms.ToTensor()(img)
        # measure = Measure()
        # print("ssim:",measure.get_ssim(self.data_dir+folder_name+'/sharp/'+img_name,self.data_dir+folder_name+'/blur/'+img_name).item())
        # print("psnr:",measure.get_psnr(self.data_dir+folder_name+'/sharp/'+img_name,self.data_dir+folder_name+'/blur/'+img_name).item())
        return gt, img
    


def test1():
  img_dir = 'data/Train400/'
  dataset = MyDataset(img_dir)
  dataloader = DataLoader(dataset=dataset, batch_size=8)
  for img_batch, noiseimg_batch in dataloader:
    print(img_batch.shape)
    print(noiseimg_batch.shape)
    break

def test2():
  dataset = GoPro_TrainDataset()
  dataloader = DataLoader(dataset=dataset, batch_size=12)
  idx = 0

  # 正好读完一遍数据集
  for gt_batch, img_batch in dataloader:
    print(idx)
    idx += 1
    print(gt_batch.shape)
    print(img_batch.shape)
    print("=======================")





if __name__ =='__main__':
  print("hello dataset")
  test2()
