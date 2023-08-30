import pyiqa
import torch
import numpy as np
import os

if __name__ == '__main__':
  FR = 1 # 是否进行有参考指标测评
  NR = 0 # 是否进行无参考指标测评
  
  # metric可以自行增删(具体指标可以参考 https://github.com/chaofengc/Awesome-Image-Quality-Assessment )
  # 有参考指标FR(Full Reference)
  psnr_metric = pyiqa.create_metric('psnr').cuda()
  ssim_metric = pyiqa.create_metric('ssim').cuda()
  lpips_metric = pyiqa.create_metric('lpips').cuda()
  
  # 无参考指标NR(No Reference)
  niqe_metric = pyiqa.create_metric('niqe').cuda()
  pi_metric = pyiqa.create_metric('pi').cuda()

  if FR == 1:
    print("下面开始计算有参考指标")
    folder_name = "NPE"
    # GT图片文件夹
    gtfolder_path = "dataset/" + folder_name
    # 待测试图片文件夹
    srcfolder_path = "dataset/dataset/" + folder_name
    
    imglist = os.listdir(gtfolder_path)
    print("GT图片文件夹路径为:" + gtfolder_path)
    print("待测试图片文件夹路径为：" + srcfolder_path)
    imglist.sort()

    file = open('result_FR_'+folder_name+'.txt', mode='w')

    for imgname in imglist:
        gt = gtfolder_path + "/" + imgname
        src = srcfolder_path + "/" +imgname
        print("正在处理文件:" + imgname)
        psnr_score = niqe_metric(gt,src) 
        ssim_score = pi_metric(gt,src) 
        lpips_score = lpips_metric(gt,src)
        psnr_score = psnr_score.cpu().numpy().tolist()
        ssim_score = ssim_score.cpu().numpy().tolist()
        lpips_score = lpips_score.cpu().numpy().tolist()
      
        # TODO 格式化字符串
        res = imgname + "\t psnr:" + str(psnr_score) + ";" + "\t ssim:" + str(ssim_score) + ";" + "\t lpips:" + str(lpips_score) + ";"
        file.write(res+"\n")

    file.close()


  if NR == 1:
    print("下面开始计算无参考指标")

    folder_name = "NPE"
    folder_path = "dataset/dataset/" + folder_name
    
    imglist = os.listdir(folder_path)
    print("读取的图片文件夹路径为:"+folder_path)
    imglist.sort()

    file = open('result_NF_'+folder_name+'.txt', mode='w')

    for imgname in imglist:
        src = folder_path + "/" + imgname
        print("正在处理文件:"+src)
        niqe_score = niqe_metric(src) #评估niqe
        pi_score = pi_metric(src) #评估pi
        niqe_score = niqe_score.cpu().numpy().tolist()
        pi_score = pi_score.cpu().numpy().tolist()

        # TODO 格式化字符串
        res = imgname + "\t niqe:" + str(niqe_score) + ";" + "\t pi:" + str(pi_score) + ";"
        file.write(res+"\n")

    file.close()
