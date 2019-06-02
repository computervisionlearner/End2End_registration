#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:58:19 2017

@author: fs

用回归　的方法计算损失
"""

import tensorflow as tf
import cv2
import pdb
import numpy as np

def _conv(name, inputs, size, input_channels, output_channels, reuse = False):
  """ 　纯卷积   """
  with tf.variable_scope(name, reuse = reuse):

    kernel = _weight_variable('weights', shape=[size, size ,input_channels, output_channels])
    biases = _bias_variable('biases',[output_channels])
    out = tf.nn.bias_add(_conv2d(inputs, kernel),biases)

  return out

def _conv2d(value, weight):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(value, weight, strides=[1, 1, 1, 1], padding='SAME')

def _conv_s2(name, inputs, size, input_channels, output_channels, reuse = False):
  """ 　纯卷积   """
  with tf.variable_scope(name, reuse = reuse):

    kernel = _weight_variable('weights', shape=[size, size ,input_channels, output_channels])
    biases = _bias_variable('biases',[output_channels])
    out = tf.nn.bias_add(_conv2d_s2(inputs, kernel),biases)

  return out

def _conv2d_s2(value, weight):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(value, weight, strides=[1, 2, 2, 1], padding='SAME')

def _max_pool_2x2(value, name):
  """max_pool_2x2 downsamples a feature map by 2X."""
  with tf.variable_scope(name):
    return tf.nn.max_pool(value, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name=name)


def _weight_variable(name, shape, mean=0):
  """weight_variable generates a weight variable of a given shape."""
  initializer = tf.truncated_normal_initializer(mean=mean,stddev=0.1)
  var = tf.get_variable(name,shape,initializer=initializer, dtype=tf.float32)
  return var


def _bias_variable(name, shape):
  """bias_variable generates a bias variable of a given shape."""
  initializer = tf.constant_initializer(0.1)
  var = tf.get_variable(name, shape, initializer=initializer,dtype=tf.float32)
  return var

def _batch_norm(name, inputs, reuse = False):
  """ batch Normalization
  """
  #equals to tf.nn.batch_normalization() just when batch=1
  with tf.variable_scope(name, reuse = reuse):
    scale = 1   #shape=(64,)
    offset = 0   #shape=(64,)
    mean, variance = tf.nn.moments(inputs, axes=[0,1,2], keep_dims=True)  # (1, 1, 1, 64) nomornize in axis=0&1
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon) #rsqrt(x)=1./sqrt(x)
    normalized = (inputs-mean)*inv
    return scale*normalized + offset

def _instance_norm(name, inputs, reuse=False):
  """ Instance Normalization
  """
  #equals to tf.nn.batch_normalization() just when batch=1
  with tf.variable_scope(name, reuse=reuse):
    depth = inputs.get_shape()[3]   #dimension(64)      input.shape=(1, 256, 256, 64)
    scale = _weight_variable("scale", [depth], mean=1.0)   #shape=(64,)
    offset = _bias_variable("offset", [depth])   #shape=(64,)
    
    mean, variance = tf.nn.moments(inputs, axes=[1,2], keep_dims=True)  # (1, 1, 1, 64) nomornize in axis=0&1
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon) #rsqrt(x)=1./sqrt(x)
    normalized = (inputs-mean)*inv
    return scale*normalized + offset


def _new_batch_norm(name, x, phase_train, reuse=False):

    with tf.variable_scope(name, reuse=reuse):
        beta = 0
        gamma = 1 
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)
    return normed

def bn_layer(x, scope, is_training, epsilon=0.001, decay=0.99, reuse=None):
    """
    Performs a batch normalization layer
    Args:
        x: input tensor
        scope: scope name
        is_training: python boolean value
        epsilon: the variance epsilon - a small float number to avoid dividing by 0
        decay: the moving average decay
    Returns:
        The ops of a batch normalization layer
    """
    with tf.variable_scope(scope, reuse=reuse):
        shape = x.get_shape().as_list()
        # gamma: a trainable scale factor
        gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0), trainable=True)
        # beta: a trainable shift value
        beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0), trainable=True)
        moving_avg = tf.get_variable("moving_avg", shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
        moving_var = tf.get_variable("moving_var", shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
        if is_training:
            # tf.nn.moments == Calculate the mean and the variance of the tensor x
            avg, var = tf.nn.moments(x, np.arange(len(shape)-1), keep_dims=True)
            avg=tf.reshape(avg, [avg.shape.as_list()[-1]])
            var=tf.reshape(var, [var.shape.as_list()[-1]])
            #update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
            update_moving_avg=tf.assign(moving_avg, moving_avg*decay+avg*(1-decay))
            #update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
            update_moving_var=tf.assign(moving_var, moving_var*decay+var*(1-decay))
            control_inputs = [update_moving_avg, update_moving_var]
        else:
            avg = moving_avg
            var = moving_var
            control_inputs = []
        with tf.control_dependencies(control_inputs):
            output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)
 
    return output
 
 
def _new_batch_norm1(x, scope, is_training, epsilon=0.001, decay=0.99):
    """
    Returns a batch normalization layer that automatically switch between train and test phases based on the 
    tensor is_training
    Args:
        x: input tensor
        scope: scope name
        is_training: boolean tensor or variable
        epsilon: epsilon parameter - see batch_norm_layer
        decay: epsilon parameter - see batch_norm_layer
    Returns:
        The correct batch normalization layer based on the value of is_training
    """
    #assert isinstance(is_training, (ops.Tensor, variables.Variable)) and is_training.dtype == tf.bool
 
    return tf.cond(
        is_training,
        lambda: bn_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=True, reuse=None),
        lambda: bn_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=False, reuse=True),
    )

def global_CNN(name, inputs, reuse = False, output_channels = None):
  """ 　全卷积   """
  with tf.variable_scope(name, reuse = reuse):
    
    size1, size2, input_channels = inputs.shape.as_list()[1:]
    if output_channels is None:
      output_channels = input_channels
      
    kernel = _weight_variable('weights', shape=[size1, size2 ,input_channels, output_channels])
    biases = _bias_variable('biases',[output_channels])
    out = tf.nn.conv2d(inputs, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, biases)

  return out

def get_features_base(images, descriptor_lens, reuse=False):
  images = tf.image.resize_bilinear(images,(32,32))
  conv1 = _conv('conv1', images, 3, 1, 32, reuse = reuse) #(batch,32,32,32)
  conv1 = _batch_norm('norm1', conv1, reuse = reuse)
  conv1 = tf.nn.relu(conv1, name = 'relu1')


  conv2 = _conv('conv2', conv1, 3,32,32,reuse = reuse)#(batch,32,32,32)
  conv2 = _batch_norm('norm2',conv2,reuse = reuse)
  conv2 = tf.nn.relu(conv2, name = 'relu2')


  conv3 = _conv_s2('conv3', conv2, 3,32,64,reuse = reuse)#(batch,16,16,64)
  conv3 = _batch_norm('norm3',conv3,reuse = reuse)
  conv3 = tf.nn.relu(conv3, name = 'relu3')
  
  conv4 = _conv('conv4', conv3, 3,64,64,reuse = reuse)#(batch,16,16,64)
  conv4 = _batch_norm('norm4',conv4,reuse = reuse)
  conv4 = tf.nn.relu(conv4, name = 'relu4')
  
  conv5 = _conv_s2('conv5', conv4, 3,64,128,reuse = reuse)#(batch,8,8,128)
  conv5 = _batch_norm('norm5',conv5,reuse = reuse)
  conv5 = tf.nn.relu(conv5, name = 'relu5')
  
  conv6 = _conv('conv6', conv5, 3,128,128,reuse = reuse)#(batch,8,8,128)
  conv6 = _batch_norm('norm6',conv6,reuse = reuse)
  conv6 = tf.nn.relu(conv6, name = 'relu6')

  conv7 = global_CNN('conv7', conv6, reuse = reuse, output_channels = descriptor_lens)#(batch,1,1,128)
  conv7 = _batch_norm('norm7',conv7,reuse = reuse)
  conv7 = tf.squeeze(conv7, name="squeeze")
  out = tf.nn.l2_normalize(conv7, axis=-1,name="lrn")
  return conv7, out

def get_features(images, descriptor_lens, flag_pl, reuse=False):
  """
  phase_train:新的bn会用到
  """
  #images = tf.image.resize_bilinear(images,(32,32))
  conv1 = _conv('conv1', images, 3, 1, 32, reuse = reuse) #(batch,32,32,32)
  conv1 = _batch_norm('norm1', conv1, reuse = reuse)
  #conv1 = _instance_norm('norm1', conv1, reuse=reuse)
  conv1 = tf.nn.leaky_relu(conv1, alpha=0.01, name="leaky_relu1")

  conv2 = _conv('conv2', conv1, 3,32,32,reuse = reuse)#(batch,32,32,32)
  conv2 = _batch_norm('norm2',conv2,reuse = reuse)
  #conv2 = _instance_norm('norm2', conv2, reuse=reuse)
  conv2 = tf.nn.leaky_relu(conv2, alpha=0.01, name="leaky_relu2")

  conv3 = _conv_s2('conv3', conv2, 3,32,64,reuse = reuse)#(batch,16,16,64)
  conv3 = _batch_norm('norm3',conv3,reuse = reuse)
  #conv3 = _instance_norm('norm3', conv3, reuse=reuse)
  conv3 = tf.nn.leaky_relu(conv3, alpha=0.01, name="leaky_relu3")
  
  conv4 = _conv('conv4', conv3, 3,64,64,reuse = reuse)#(batch,16,16,64)
  conv4 = _batch_norm('norm4',conv4,reuse = reuse)
  #conv4 = _instance_norm('norm4', conv4, reuse=reuse)
  conv4 = tf.nn.leaky_relu(conv4, alpha=0.01, name="leaky_relu4")
  
  conv5 = _conv_s2('conv5', conv4, 3,64,128,reuse = reuse)#(batch,8,8,128)
  conv5 = _batch_norm('norm5',conv5,reuse = reuse)
  #conv5 = _instance_norm('norm5', conv5, reuse=reuse)
  conv5 = tf.nn.leaky_relu(conv5, alpha=0.01, name="leaky_relu5")
  
  conv6 = _conv('conv6', conv5, 3,128,128,reuse = reuse)#(batch,8,8,128)
  conv6 = _batch_norm('norm6',conv6,reuse = reuse)
  #conv6 = _instance_norm('norm6', conv6, reuse=reuse)
  conv6 = tf.nn.leaky_relu(conv6, alpha=0.01, name="leaky_relu6")
  keep_prob = tf.cond(flag_pl, lambda:0.75, lambda:1.)
  conv6 = tf.nn.dropout(conv6, keep_prob)  

  conv7 = global_CNN('conv7', conv6, reuse = reuse, output_channels = descriptor_lens)#(batch,1,1,128)
  conv7 = _batch_norm('norm7', conv7, reuse = reuse)
  #conv7 = _instance_norm('norm7', conv7, reuse=reuse)
  conv7 = tf.squeeze(conv7, name="squeeze")
  out = tf.nn.l2_normalize(conv7, -1,name="lrn")
  return conv7, out

def get_distance(f1, f2):
  """ 
  features1 / features2  p*q  
  (p = the number of the 3Dpoints in a batch) 
  """
  result = tf.matmul(f1, f2, transpose_b=True)   
  D = tf.sqrt(2.*tf.subtract(1.,result))
  return D
  
def loss1(feature_D, coord_D, mask, cur_epoch, logging):
  """ 
  输入一个矩阵,矩阵的每个元素是两两特征向量的欧式距离　
  返回第一个损失函数   
  """
  #theta = np.maximum(32 - cur_epoch, 1)
  logging.info(">> loss1:common regression loss")
  theta = 1
  Y = 2*tf.exp(tf.divide(coord_D,-1*theta))
  mask = tf.cast(mask, tf.float32)
  Y = tf.multiply(Y, mask)   #对称矩阵
  num = tf.cast(Y.shape.as_list()[0], tf.float32) 
  E1 = tf.divide(tf.reduce_sum(tf.squared_difference(2-feature_D, Y)), num)
  tf.summary.scalar('E1', E1)
  return E1

def loss1_hard(feature_D, coord_D, mask, cur_epoch, logging):
  """
  挖掘每行每列的最困难的样本
  """
  logging.info(" loss1:hard negative mining + regression loss")
  similarity = tf.subtract(2., feature_D)
  theta = tf.get_variable("theta", dtype=tf.float32, initializer=1.)
  Y = 2*tf.exp(tf.divide(coord_D,-1*theta))
  mask = tf.cast(mask, tf.float32)
  Y = tf.multiply(Y, mask)   #对称矩阵
  error = tf.squared_difference(similarity, Y)
  error = tf.matrix_band_part(error, 0, -1)   #取上半角矩阵
  error_flatten = tf.reshape(error, (feature_D.shape.as_list()[0]*feature_D.shape.as_list()[1],))
  harderror = tf.nn.top_k(error_flatten, k=feature_D.shape.as_list()[0])
  E1 = tf.reduce_mean(harderror.values)
  tf.summary.scalar('E1', E1)
  return E1

def loss1_hard1(feature_D, coord_D, mask, cur_epoch, logging):
  """
  挖掘正负困难样本（各占一半）
  """
  logging.info(">> loss1:hard pos and neg mining + regression loss")
  similarity = tf.subtract(2., feature_D)
  #theta = tf.get_variable("theta", dtype=tf.float32, initializer=1.)
  theta = 1.
  tf.summary.scalar("theta", theta)
  Y = 2*tf.exp(tf.divide(coord_D,-1*theta))
  #Y = 2*tf.exp(-1*(coord_D//theta))
  mask = tf.cast(mask, tf.float32)
  Y = tf.multiply(Y, mask)   #对称矩阵
  error = tf.squared_difference(similarity, Y)
  error_pos = tf.diag_part(error)   #取出对角矩阵，　此时是个向量
  error_upper = tf.matrix_band_part(error, 0, -1)   #取上半角矩阵
  error_upper = tf.matrix_set_diag(error_upper, tf.zeros(error.shape.as_list()[0], dtype=tf.float32))   #把对角线所有元素置0
  error_neg = tf.reshape(error_upper, (feature_D.shape.as_list()[0]*feature_D.shape.as_list()[1],))
  #困难负样本损失
  hard_error_neg = tf.nn.top_k(error_neg, k=feature_D.shape.as_list()[0]//2)#取64个负样本
  hard_error_pos = tf.nn.top_k(error_pos, k=feature_D.shape.as_list()[0]//2)#取64个正样本
  #mean, var = tf.nn.moments(error_pos, axes=0)
  E1 = tf.reduce_sum(tf.add(hard_error_neg.values, hard_error_pos.values))*0.1
  #E1 = tf.reduce_mean(tf.add_n([hard_error_neg.values, hard_error_pos.values]))*2
  tf.summary.scalar('E1', E1)
  return E1

def loss2(f1,f2,logging):
  """
  输入的特征向量为　　最后一层BN的输出，　即在LRN之前
  此函数中的f1/f2与get_distance函数中的l1/l2不同
  """
  logging.info(">> loss2:correlation matrix constraint")
  f_len = tf.cast(f1.shape.as_list()[1], tf.float32)
  num = tf.cast(f1.shape.as_list()[0], tf.float32)
  R1 = tf.divide(tf.matmul(f1,f1,transpose_a=True), f_len)
  R2 = tf.divide(tf.matmul(f2,f2,transpose_a=True), f_len)
  diag1 = tf.diag_part(R1)
  diag2 = tf.diag_part(R2)
  E2 = (0.5*(tf.reduce_sum(tf.square(R1))+tf.reduce_sum(tf.square(R2))-\
       tf.reduce_sum(tf.square(diag1))-tf.reduce_sum(tf.square(diag2))))
  E2 = tf.divide(E2, num)
  
  tf.summary.scalar('E2', E2)
  return E2
  

def loss3(D, D_T, logging):
  """
  """
  logging.info(">> loss3:symmetric loss")
  num = tf.cast(D.shape.as_list()[0], dtype=tf.float32)
  E3 = tf.divide(tf.reduce_sum(tf.squared_difference(D, D_T)), num)
  tf.summary.scalar('E3', E3)
  return E3
  

def get_coord_distance(coords_pl):
  """
  计算各关键点之间的距离
  """
  rows = coords_pl.shape.as_list()[0]
  X1 = tf.tile(coords_pl, (1,rows))
  X1 = tf.reshape(X1, (-1,2))
  X2 = tf.tile(coords_pl, (rows,1))
  coord_D = tf.norm((X1-X2), axis=1)
  coord_D = tf.reshape(coord_D, (rows, rows))
  return coord_D

def get_mask(ids):
  """
  ids: 每个batch下各patch 所属的id
  """
  rows = ids.shape.as_list()[0]
  ids = tf.reshape(ids,(rows,1))
  ids1 = tf.tile(ids,(1,rows))
  ids1 = tf.reshape(ids1,(-1,1))
  
  ids2 = tf.tile(ids,(rows,1))
  diff = tf.abs(ids1-ids2)
  diff = tf.reshape(diff,(rows,rows))
  mask = tf.equal(diff,0)
  return mask

def caculate_loss(map1, map2, out1, out2, ids_pl, coords_pl, cur_epoch, logging):
  with tf.variable_scope('caculate_loss') :
    feature_D = get_distance(out1, out2)
    feature_D_T = tf.transpose(feature_D)
    coord_D = get_coord_distance(coords_pl)
    mask = get_mask(ids_pl) 
    E1 = loss1_hard1(feature_D, coord_D, mask, cur_epoch, logging)  #做了改动
    #E2 = loss2(map1, map2, logging)
    E3 = loss3(feature_D, feature_D_T, logging)
    return tf.add_n([E1,E3], name = "total_loss")
    
def evaluation(features1,features2,labels,thresh):
  with tf.variable_scope('evaluation') :
    distance = tf.sqrt(tf.reduce_sum(tf.square(features1-features2),axis=1))
    predicts = tf.cast(tf.greater(distance,thresh),dtype=tf.float32)
    batch_precision = tf.reduce_sum(tf.cast(tf.equal(predicts,labels),dtype=tf.float32))
    tp = tf.reduce_sum(tf.cast(tf.equal(tf.add(predicts,labels),2,name='tp'),dtype=tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.equal(tf.add(predicts,labels),0,name='fn'),dtype=tf.float32))
    fp = tf.subtract(tf.reduce_sum(tf.cast(tf.equal(labels,0),dtype=tf.float32)),fn)
    tn = tf.subtract(tf.reduce_sum(tf.cast(tf.equal(labels,1),dtype=tf.float32)),tp)    
    
    eval_all = {'precision':batch_precision,'tp':tp,'tn':tn,'fp':fp,'fn':fn}
  return eval_all  

def training(loss):
  with tf.variable_scope('training') :
    optimizer = tf.train.AdamOptimizer(1e-4, 0.5)

    gen_grads_and_vars = optimizer.compute_gradients(loss)
    gen_train = optimizer.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([loss])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

  return tf.group(update_losses, incr_global_step, gen_train)



def test_get_features():

  image1 = cv2.resize(cv2.imread("1.png"),(32,32))
  image1 = np.expand_dims(cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY),axis=0)
   
  image2 = cv2.resize(cv2.imread("2.png"),(32,32))
  image2 = np.expand_dims(cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY),axis=0)
 
  images = np.expand_dims(np.concatenate((image1,image2),axis=0),4)
  out = get_features(tf.constant(images,dtype=tf.float32))
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  value = sess.run(out)
  aaa = np.square(value)
  bbb = np.sum(aaa,axis=-1)
  return bbb

if __name__ == "__main__":
  a=tf.constant([[1,2],[3,4],[5,6]],dtype=tf.float32)
  a=tf.divide(a,tf.norm(a,axis=0))
  b=tf.constant([[7,8],[9,10],[11,12]],dtype=tf.float32)
  b=tf.divide(b,tf.norm(b,axis=0))
  
  distance = get_distance(a,b)
  
  
  distance1= []
  
  for i in range(2):
    for j in range(2):
      distance1.append(tf.norm(a[:,i]-b[:,j]))
      
  sess = tf.Session()    
  d1,d2 = sess.run([distance,distance1])
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
