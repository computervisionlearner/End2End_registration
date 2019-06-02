#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 20:06:11 2019

@author: no1
"""

import argparse
import cv2
import numpy as np
import os

ratio = 0.8
reprojThresh =4.0

def get_the_kps_and_features(image):
  '''
  输入：一张图片
  输出：图片的特征点坐标集和特征点对应的特征向量集
  '''
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  descriptor = cv2.xfeatures2d.SIFT_create()
  raw_kps,features = descriptor.detectAndCompute(gray, None)
  
  return raw_kps, features
  

def match_keypoints(features1, features2):
  '''
  输入：两张图片的特征向量集
  功能：用knn算法进行特征匹配
  输出：找出在features2中与features1中每个特征向量最接近的两个特征向量
  '''
  bf = cv2.BFMatcher()
  matches = bf.knnMatch(features1, features2, k = 2)
  return matches

def find_H(kps1, kps2, raw_matches):
  '''
  输入：两幅图像的特征点对象和特征点匹配集
  功能：通过计算某特征向量的最接近的特征向量是否小于一定比例下的次接近的特征向量，
  来判断是否是最佳匹配点对
  输出：最佳匹配的关键点对、计算单应矩阵H、配对状态
  
  ''' 
  good_matches = []
  for m0,m1 in raw_matches:
    if m0.distance <m1.distance * ratio:
      good_matches.append(m0)
      
  if len(good_matches) > 4:
    pts1 = np.float32([kps1[i.queryIdx] for i in good_matches])
    pts2 = np.float32([kps2[i.trainIdx] for i in good_matches])
    H, status = cv2.findHomography(pts2, pts1, cv2.RANSAC, reprojThresh)

#    H = cv2.getPerspectiveTransform(pts2[:4],pts1[:4])
    return good_matches, H ,status
  return None

def register(warp, des):


  registered = cv2.add(0.5*warp, 0.5*des)
  
  return registered


def qipantu(image1, image2, num):
  """
  num:每行画几个格子
  """
  H, W = image1.shape[:2]
  size = W//num
  index1 = [i for i in range(num) if i%2==0]
  index2 = [i for i in range(num) if i%2!=0]
  mask = np.zeros((H, W, 3), dtype=np.uint8)
  for i in index1:
    for j in index1:
      mask[i*size:(i+1)*size, j*size:min((j+1)*size, W), :] = 1
  for i in index2:
    for j in index2:
      mask[i*size:(i+1)*size, j*size:min((j+1)*size, W), :] = 1      
  merge = mask*image1 + (1-mask)*image2
  return merge
  

def drew_matches(img1, raw_kps1, img2, raw_kps2, goods ,status):
  '''
  功能：特征点对的可视化（选取了前20个对）
  '''
  draw_params = dict(
  singlePointColor = None,
  matchesMask = status.flatten().tolist(),       # draw only inliers dots
  flags = 2)
  
  result = cv2.drawMatches(img1,raw_kps1,
                           img2, raw_kps2, goods, None, **draw_params)

  return result
  

def drawMatches(left_picture, right_picture,  kps1, kps2, matches, status):
  (hB, wB) = right_picture.shape[:2]
  (hA, wA) = left_picture.shape[:2]
  vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
  vis[0:hB, wA:] = right_picture
  vis[0:hA, :wA] = left_picture
  for (match, s) in zip(matches, status):
    queryIdx, trainIdx = match.queryIdx, match.trainIdx
    if s == 1:
      ptA = (int(kps1[queryIdx][0]), int(kps1[queryIdx][1]))
      ptB = (int(kps2[trainIdx][0]) + wA, int(kps2[trainIdx][1]))
      cv2.line(vis,ptA,ptB, (0, 255, 0), 2)
  return vis

#def evaluation(kps1, kps2, homo):
#  """
#  kps2->kps1_1
#  """
#  kps1_1 = cv2.perspectiveTransform(kps2.astype(np.float32).reshape(-1,1,2), homo)
#  kps1_1 = np.squeeze(kps1_1)
#  err = np.mean(np.linalg.norm(kps1_1-kps1, axis=1))
#  return err

def evaluation(kps1, kps2, homo):
  """
  kps2->kps1_1
  """
  kps1_1 = cv2.perspectiveTransform(kps2.astype(np.float32).reshape(-1,1,2), homo)
  kps1_1 = np.squeeze(kps1_1)
  err = np.sqrt(np.mean(np.square(np.linalg.norm(kps1_1-kps1, axis=1))))
  return err
  
if __name__ =="__main__":    
  parser = argparse.ArgumentParser()
  parser.add_argument('--image1', default = '05_058.tif', 
                      help = 'path to the left image')
  
  parser.add_argument('--image2', default = '05_058L_8.png',
                      help = 'path to the right image')
  
  args = vars(parser.parse_args())
  model_dir = "sift"
  image1 = cv2.imread(args['image1'])
  image2 = cv2.imread(args['image2'])
  imagename = os.path.basename(args['image1']).split("_")[0]
#  coords_and_homo = np.load("test_trans/3_coords_before_after_homo.npz")
#  kps2_dt, kps1_dt, homo = coords_and_homo["arr_0"],coords_and_homo["arr_1"],coords_and_homo["arr_2"]
  
  
  kps1, features1 = get_the_kps_and_features(image1)
  kps2, features2 = get_the_kps_and_features(image2)  
  
  kps1 = np.float32([kp.pt for kp in kps1])
  kps2 = np.float32([kp.pt for kp in kps2])  
  
  raw_matches = match_keypoints(features1, features2)
  
  good_matches, H, status = find_H(kps1, kps2, raw_matches)
  
  match_result = drawMatches(image1, image2, kps1, kps2, good_matches, status)  

  warp = cv2.warpPerspective(image2, H, (image1.shape[1],
                            image1.shape[0]))
  registered = register(warp, image1)
  qipan = qipantu(warp, image1, 6)
#  error = evaluation(kps1_dt, kps2_dt, H)
#  print(error)
#  np.save(os.path.join(model_dir, "err{}.png".format(imagename)), error)
  cv2.imwrite(os.path.join(model_dir, "matches_{}.png".format(imagename)), match_result)
  cv2.imwrite(os.path.join(model_dir, "register_{}.png".format(imagename)), registered)
  cv2.imwrite(os.path.join(model_dir, "qipantu_{}.png".format(imagename)), qipan)