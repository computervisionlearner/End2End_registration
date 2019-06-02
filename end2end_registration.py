#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 11:09:28 2019

@author: no1
输入图像块的尺寸是64×64
"""

import argparse
import cv2
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
import glob
import time
import logging
from scipy import misc
from matplotlib import pyplot as plt

ratio = 0.96
reprojThresh =4.0

def evaluation(kps1, kps2, homo):
  """
  kps2->kps1_1
  """
  kps1_1 = cv2.perspectiveTransform(kps2.astype(np.float32).reshape(-1,1,2), homo)
  kps1_1 = np.squeeze(kps1_1)
  err = np.mean(np.linalg.norm(kps1_1-kps1, axis=1))

  return err

def initLogging(logFilename='record.log'):
  """Init for logging
  """
  logging.basicConfig(
                    level    = logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w');
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
  console.setFormatter(formatter)
  logging.getLogger('').addHandler(console)


#%%    特征提取
def get_unique_kps(image, name):
  '''
  输入：一张图片
  输出：图片的特征点坐标集和图像块
  '''
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  rows, cols = gray.shape
  descriptor = cv2.xfeatures2d.SIFT_create()
  kps, features = descriptor.detectAndCompute(gray, None)
  kps_img = cv2.drawKeypoints(gray, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  cv2.imwrite(name, kps_img)
  kps = np.float32([kp.pt for kp in kps])
  coords_origin = np.unique(kps, axis=0)
  
  filtered_idx = np.where((coords_origin[:,0]<cols-32)&(coords_origin[:,0]>32) & \
                          (coords_origin[:,1]<rows-32)&(coords_origin[:,1]>32)
                          )[0]  
  coords_filtered = np.asarray(coords_origin[filtered_idx], dtype=np.int64)

  return coords_filtered

def get_patches(image, coords):
  patches = []
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  for coord in coords:
    patch = gray[coord[1]-32:coord[1]+32, coord[0]-32:coord[0]+32]
    patches.append(patch)
  patches = np.expand_dims(np.asarray(patches), axis=3)
  
  return patches

#%%    特征描述
def get_descriptors(patches, image_name, sess):
        
  
  features = sess.graph.get_tensor_by_name('output:0')
  images_pl = sess.graph.get_tensor_by_name('input_image:0')
  steps_per_epoch = patches.shape[0] // batch_size
  descriptors = []
  iterator = tqdm(range(steps_per_epoch))
  for step in iterator:
    batch_patch = patches[step*batch_size:(step+1)*batch_size]
    batch_feature = sess.run(features, feed_dict={images_pl: batch_patch})
    descriptors.extend(batch_feature)
    
  rest_num = patches.shape[0] - steps_per_epoch*batch_size
  if rest_num>0:
    rest_patch = patches[-1*rest_num:]
    rest_feature = sess.run(features, feed_dict={images_pl: rest_patch})
    descriptors.extend(rest_feature)
  savepath = os.path.join(descriptors_dir, os.path.basename(image_name).split(".")[0])
  np.save(savepath, np.asarray(descriptors))  
    
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
  size1 = W//num
  size2 = H//num
  index1 = [i for i in range(num) if i%2==0]
  index2 = [i for i in range(num) if i%2!=0]
  mask = np.zeros((H, W, 3), dtype=np.uint8)
  for i in index1:
    for j in index1:
      mask[i*size2:(i+1)*size2, j*size1:min((j+1)*size1, W), :] = 1
  for i in index2:
    for j in index2:
      mask[i*size2:(i+1)*size2, j*size1:min((j+1)*size1, W), :] = 1      
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

if __name__ =="__main__":    
  parser = argparse.ArgumentParser()
  parser.add_argument('--image1', default = '5_optical_qd_tiny.png', 
                      help = 'path to the left image')
  
  parser.add_argument('--image2', default = '5_SAR_qd_tiny.png',
                      help = 'path to the right image')
  
  parser.add_argument('--descriptors_dir', default = '5_SAR_OPT',
                      help = 'path to the right image')
  
  parser.add_argument('--kps_path', default = 'aaa.npz',
                      help = 'path to the right image')
  
  parser.add_argument('--batch_size', type = int, default = 128)
  args = vars(parser.parse_args())
  initLogging(args['descriptors_dir'] + ".log")
  
  image1 = cv2.imread(args['image1'])   #参考图像
  image2 = cv2.imread(args['image2'])   #待配准图像 
  batch_size = args['batch_size']
  descriptors_dir = args['descriptors_dir']
  kps_path = args['kps_path']
  if not os.path.exists(descriptors_dir):
    os.mkdir(descriptors_dir)    
    
  t1 = time.time()
  if os.path.exists(kps_path):
    logging.info("已检测到生成的特征点，跳过特征点检测步骤")
    coords = np.load(kps_path)
    kps2, kps1, H_gt = coords["arr_0"], coords["arr_1"], coords["arr_2"]
  else:
    logging.info("特征点检测...")
    kps_name1 = os.path.basename(args['image1']).split(".")[0]+"_kps.png"
    kps_name2 = os.path.basename(args['image2']).split(".")[0]+"_kps.png"
    kps1 = get_unique_kps(image1, kps_name1)
    kps2 = get_unique_kps(image2, kps_name2)
  logging.info("光学图像上的特征点数目为: {}, SAR图像上的特征点数目为: {}".format(len(kps1), len(kps2)))
  patches1 = get_patches(image1, kps1)
  patches2 = get_patches(image2, kps2)
  t2 = time.time()
  dur_kps_detect = t2 - t1
  logging.info("特征点检测耗时:{}s".format(dur_kps_detect))
  imagename1 = os.path.basename(args['image1'])
  imagename2 = os.path.basename(args['image2']) 
  files = glob.glob(os.path.join(descriptors_dir, "*.npy"))
  
  if len(files)<2:
    logging.info("特征描述...")
    graph = tf.Graph()
    with graph.as_default():
      with tf.gfile.FastGFile("SoftRegNet.pb", 'rb') as model_file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(model_file.read())
        tf.import_graph_def(graph_def,name='')
    sess = tf.Session(graph=graph) 
    get_descriptors(patches1, args['image1'], sess)
    get_descriptors(patches2, args['image2'], sess)
  else:
    logging.info("已检测到生成的描述子，跳过特征描述过程")
  t3 = time.time()
  dur_descriptor = t3 - t2
  logging.info("特征描述耗时:{}s".format(dur_descriptor))
  
  features1 = np.load(os.path.join(descriptors_dir, imagename1.split(".")[0]+".npy"))
  features2 = np.load(os.path.join(descriptors_dir, imagename2.split(".")[0]+".npy"))
  
  raw_matches = match_keypoints(features1, features2)
  t4 = time.time()
  dur_matching = t4 - t3
  logging.info("特征匹配耗时:{}s".format(dur_matching))
  good_matches, H, status = find_H(kps1, kps2, raw_matches)
  t5 = time.time()
  dur_homo = t5 - t4
  logging.info("模型参数估计耗时:{}s".format(dur_homo))
  
  match_result = drawMatches(image1, image2, kps1, kps2, good_matches, status)  
  t6 = time.time()
  dur_drawmatches = t6 - t5
  logging.info("画匹配的关键点耗时:{}s".format(dur_drawmatches))
  warp = cv2.warpPerspective(image2, H, (image1.shape[1],
                            image1.shape[0]))
  registered = register(warp, image1)
  t7 = time.time()
  dur_warp = t7 - t6
  logging.info("空间变换和插值耗时:{}s".format(dur_warp))
  qipan = qipantu(warp, image1, 6)
  
  if os.path.exists(kps_path):
    err = evaluation(kps1, kps2, H)
  dur_all = dur_kps_detect + dur_descriptor + dur_matching + dur_warp
  logging.info("总耗时:{}s".format(dur_all))
  cv2.imwrite("matches.png", match_result)
  cv2.imwrite("register.png", registered)
  cv2.imwrite("qipantu.png", qipan)
  plt.figure()
  plt.imshow(match_result)
  plt.figure()
  plt.imshow(np.uint8(registered))
  plt.figure()
  plt.imshow(qipan)
  plt.show()
