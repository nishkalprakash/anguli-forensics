import fingerprint_enhancer 
import cv2 as cv
# import matplotlib.pyplot as plt
import numpy as np
# from numpy import array
# from skimage.morphology import skeletonize
# from collections import deque as queue
import cv2
# import os
# import math
import argparse
from sklearn.cluster import AgglomerativeClustering


def enhancer(img):
  enhanced = fingerprint_enhancer.enhance_Fingerprint(img)		# enhance the fingerprint image
  # plt.imshow(enhanced)
  return enhanced

def binarize(img):
  _,bin = cv.threshold(img, 100, 255, cv.THRESH_BINARY)
  # plt.imshow(cv.cvtColor(res, cv.COLOR_RGB2BGR))
  return bin

def skeletonize(img):
  skeleton = cv.ximgproc.thinning(img, thinningType = cv.ximgproc.THINNING_GUOHALL)
  # plt.imshow(cv.cvtColor(skeleton, cv.COLOR_RGB2BGR))
  return skeleton


def grad(img):
  gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  kernelx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
  kernely = np.array([[1, 2, 1],[0, 0, 0],[-1,-2,-1]])
  img_x = cv2.filter2D(gray_img, -1, kernelx)
  img_y = cv2.filter2D(gray_img, -1, kernely)

  new=np.hypot(img_x,img_y)
  return new

def orientation(img,flag = True):
  if(flag):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  gx, gy = cv.Sobel(img, cv.CV_32F, 1, 0), cv.Sobel(img, cv.CV_32F, 0, 1)
  gx2, gy2 = gx**2, gy**2
  W = (23, 23)
  gxx = cv.boxFilter(gx2, -1, W, normalize = False)
  gyy = cv.boxFilter(gy2, -1, W, normalize = False)
  gxy = cv.boxFilter(gx * gy, -1, W, normalize = False)
  gxx_gyy = gxx - gyy
  gxy2 = 2 * gxy

  orientations = (cv.phase(gxx_gyy, -gxy2) + np.pi) / 2 # '-' to adjust for y axis direction
  return orientations

def frequency(img,n,flag = True):
  if(flag):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  l, b = img.shape
  bl = l // n
  bb = b // n

  freq = np.zeros((bl,bb),dtype=float)

  for i in range(bl):
    for j in range(bb):
      region = img[i*n:(i+1)*n -1 , j*n:(j+1)*n -1 ]
      smoothed = cv.blur(region, (5,5), -1)
      xs = np.sum(smoothed, 1)
      local_maxima = np.nonzero(np.r_[False, xs[1:] > xs[:-1]] & np.r_[xs[:-1] >= xs[1:], False])[0]
      distances = local_maxima[1:] - local_maxima[:-1]
      if len(distances) > 0:
        ridge_period = np.average(distances)
        if(ridge_period>0):
          freq[i][j] = ridge_period
      else:
        freq[i][j] = np.nan_to_num(0)
      
  return freq

def hessian(img,flag = True):
  kernelx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
  kernely = np.array([[1, 2, 1],[0, 0, 0],[-1,-2,-1]])
  if(flag):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  gx = cv2.filter2D(gray_img, -1, kernelx)
  gy = cv2.filter2D(gray_img, -1, kernely)
  gxx = cv2.filter2D(gx, -1, kernelx)
  gyy = cv2.filter2D(gy, -1, kernely)
  gxy = cv2.filter2D(gx, -1, kernely)
  gyx = cv2.filter2D(gy, -1, kernelx)
  hess = (gxx*gyy-gxy*gyx)
  k = hess / ( (1+(gx**2)+(gy**2))**2 )
  intensity = gray_img 
  return intensity,k

def block(img,n): #n-->grid size
  l, b = img.shape
  num_blocks_l = l // n
  num_blocks_b = b // n

  mean_matrix = np.zeros((num_blocks_l, num_blocks_b))
  median_matrix = np.zeros((num_blocks_l, num_blocks_b))
  var_matrix = np.zeros((num_blocks_l, num_blocks_b))

  # loop through each block and calculate the mean, median, and variance
  for i in range(num_blocks_l):
      for j in range(num_blocks_b):
          # get the current block
          block = img[i*n:(i+1)*n, j*n:(j+1)*n]
          
          # calculate mean, median, and variance
          block_mean = np.mean(block)
          block_median = np.median(block)
          # block_var = np.var(block)

          block_var = 0
          for k in range(n):
              for l in range(n):
                  block_var += (block[k,l] - block_mean)**2
          block_var /= n**2
          
          # store the results in the matrices
          mean_matrix[i, j] = block_mean
          median_matrix[i, j] = block_median
          var_matrix[i, j] = block_var

  return mean_matrix,median_matrix,var_matrix



def fvect(img,n=23,normalize=False,flag = True):
  f1a = grad(img)
  if(flag):
    f2a = orientation(img, flag)
    f3a = frequency(img,n, flag)
    f4a,f5a = hessian(img, flag)
  else:
    img = enhancer(img) 
    f2a = orientation(img, flag)
    f3a = frequency(img,n, flag)
    f4a,f5a = hessian(img, flag)

  f1 = block(f1a,n)
  f2 = block(f2a,n)
  f3 = f3a
  f4 = block(f4a,n)
  f5 = block(f5a,n)
  f1=np.asarray(f1)
  f2=np.asarray(f2)
  f3=np.asarray(f3)
  f4=np.asarray(f4)
  f5=np.asarray(f5)

  if(normalize==True):
    f1[0] = (f1[0]-np.min(f1[0])) / (np.max(f1[0])-np.min(f1[0]))
    f1[1] = (f1[1]-np.min(f1[1])) / (np.max(f1[1])-np.min(f1[1]))
    f1[2] = (f1[2]-np.min(f1[2])) / (np.max(f1[2])-np.min(f1[2]))
    f2[0] = (f2[0]-np.min(f2[0])) / (np.max(f2[0])-np.min(f2[0]))
    f2[1] = (f2[1]-np.min(f2[1])) / (np.max(f2[1])-np.min(f2[1]))
    f2[2] = (f2[2]-np.min(f2[2])) / (np.max(f2[2])-np.min(f2[2]))
    f4[0] = (f4[0]-np.min(f4[0])) / (np.max(f4[0])-np.min(f4[0]))
    f4[1] = (f4[1]-np.min(f4[1])) / (np.max(f4[1])-np.min(f4[1]))
    f4[2] = (f4[2]-np.min(f4[2])) / (np.max(f4[2])-np.min(f4[2]))
    f5[0] = (f5[0]-np.min(f5[0])) / (np.max(f5[0])-np.min(f5[0]))
    f5[1] = (f5[1]-np.min(f5[1])) / (np.max(f5[1])-np.min(f5[1]))
    f5[2] = (f5[2]-np.min(f5[2])) / (np.max(f5[2])-np.min(f5[2]))
    f3 = (f3-np.min(f3)) / (np.max(f3)-np.min(f3))

  l, b = f1a.shape
  bl = l // n
  bb = b // n

  fv =[[[]]*bb]*bl # Mean feature vector with length 5
  fv1=[[[]]*bb]*bl # Median feature vector with length 5
  fv2=[[[]]*bb]*bl  # Variance feature vector with length 5
  fv=np.array(fv)
  fv1=np.array(fv1)
  fv2=np.array(fv2)
  fv=fv.tolist()
  fv1=fv1.tolist()
  fv2=fv2.tolist()
  for i in range(bl):
    for j in range(bb):
      fv[i][j].append(f1[0][i][j])
      fv[i][j].append(f2[0][i][j])
      fv[i][j].append(f3[i][j])
      fv[i][j].append(f4[0][i][j])
      fv[i][j].append(f5[0][i][j])
      fv1[i][j].append(f1[1][i][j])
      fv1[i][j].append(f2[1][i][j])
      fv1[i][j].append(f3[i][j])
      fv1[i][j].append(f4[1][i][j])
      fv1[i][j].append(f5[1][i][j])
      fv2[i][j].append(f1[2][i][j])
      fv2[i][j].append(f2[2][i][j])
      fv2[i][j].append(f3[i][j])
      fv2[i][j].append(f4[2][i][j])
      fv2[i][j].append(f5[2][i][j])
  fv=np.array(fv)
  fv1=np.array(fv1)
  fv2=np.array(fv2)
  X1 = fv.reshape(bl*bb,5) #Mean
  X2 = fv1.reshape(bl*bb,5) #Median
  X3 = fv2.reshape(bl*bb,5) #Variance
  return X1,X2,X3


def cluster_image(image,n):
  l, b = image.shape[0],image.shape[1]
  bl = l // n
  bb = b // n
  X1,X2,X3 = fvect(image,n, False)
  clustering = AgglomerativeClustering(n_clusters = 3, metric ="euclidean" ,linkage="ward" ).fit(X1)
  clq = clustering.labels_ 
  clq = clq.reshape(bl,bb) #16*16 cluster labels
  # plt.imshow(clq)
  return clq,X1


def Final_cluster_Method_1(clq,n):  
  temp = [100,0,100,0,-1]
  lab = [temp]*3
  lab = np.asarray(lab)
  bl = clq.shape[0]
  bb = clq.shape[1]
  for i in range(bl):
    for j in range(bb):
      z = clq[i][j]
      lab[z][0]=min(lab[z][0],i)
      lab[z][1]=max(lab[z][1],i)
      lab[z][2]=min(lab[z][2],j)
      lab[z][3]=max(lab[z][3],j)
  # print(lab)
  for x in lab:
    x[4]=x[1]-x[0]+x[3]-x[2]
  # print(lab)
  res = -1
  min_ = 100
  for i in range(len(lab)):
    if(lab[i][4]<min_):
      res = i
      min_=lab[i][4]
  # print(res,min_)
  temp1 = np.zeros((bl,bb))
  temp1=np.asarray(temp1)
  for i in range(bl):
    for j in range(bb):
      if(clq[i][j]==res):
        temp1[i][j]=1
  lts = [0,0,0,0]
  lts[0]=(lab[res][0]+1)*n
  lts[1]=(lab[res][1]-1)*n
  lts[2]=(lab[res][2]+1)*n
  lts[3]=(lab[res][3]-1)*n
  # temp2 = cv2.rectangle(images1[i1],(lts[2],lts[0]),(lts[3],lts[1]),(255,0,0), 2) 
  return temp1,lts

def simsum(clust):
  sim1 = []
  for i in range(len(clust)):
    temp = []
    for j in range(len(clust)):
      kk = dot(clust[i], clust[j]) /(norm(clust[i])*norm(clust[j]))
      # sim1[i][j] =kk
      temp.append(kk)
      # print(kk)
    sim1.append(temp)

  sim1=np.asarray(sim1)
  sum = np.sum(sim1)
  return sum,np.min(sim1) 

def Final_cluster_Method_2(clq, X_, n):  
  temp = [100,0,100,0,-1]
  lab = [temp]*3
  lab = np.asarray(lab)
  bl = clq.shape[0]
  bb = clq.shape[1]
  for i in range(bl):
    for j in range(bb):
      z = clq[i][j]
      lab[z][0]=min(lab[z][0],i)
      lab[z][1]=max(lab[z][1],i)
      lab[z][2]=min(lab[z][2],j)
      lab[z][3]=max(lab[z][3],j) 

  print(clq.shape)
  clq1 = clq.reshape(bl*bb)
  clust = [[]]*3
  clust=np.asarray(clust)
  clust=clust.tolist()
  for i in range(len(clq1)):
    clust[clq1[i]].append(X_[i]) 
  clust=np.asarray(clust)

  sum1,m1= simsum(clust[0])
  sum2,m2= simsum(clust[1])
  sum3,m3= simsum(clust[2])
  temp2= [sum1/(len(clust[0])**2),sum2/(len(clust[1])**2),sum3/(len(clust[2])**2)]
  temp2 = np.asarray(temp2) 
  res = np.argmax(temp2)

  temp1 = np.zeros((bl,bb))
  temp1=np.asarray(temp1) 
  for i in range(bl):
    for j in range(bb):
      if(clq[i][j]==res):
        temp1[i][j]=1
  lts = [0,0,0,0]
  lts[0]=(lab[res][0]+1)*n
  lts[1]=(lab[res][1]-1)*n
  lts[2]=(lab[res][2]+1)*n
  lts[3]=(lab[res][3]-1)*n
  # temp2 = cv2.rectangle(images1[i1],(lts[2],lts[0]),(lts[3],lts[1]),(255,0,0), 2) 
  return temp1,lts


def cmd_args():
    parser = argparse.ArgumentParser(description ='Image path name')
    parser.add_argument('img_path', type=str, help='Image path to be read')
    args = parser.parse_args()
    return args

def main_using_arguments():
  args = cmd_args()
  img = cv2.imread(args.img_path)

  n = 23 # block size
  clq,X_ = cluster_image(img,n)
  fclq,bbox = Final_cluster_Method_1(clq,n)
  # print(bbox)
  # cv2.imshow('samp',img)
  # cv2.waitKey(0)

  crop_img = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
  # cv2.imshow('samp1',crop_img)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
  return crop_img

def ROI_extractor(path):
  original_img = cv2.imread(path.as_posix())
  n = 23 # block size
  clq,X_ = cluster_image(original_img,n)
  fclq,bbox = Final_cluster_Method_1(clq,n)

  ROI = original_img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
  return ROI,path


if __name__=="__main__":
    # main_using_arguments()
    pass