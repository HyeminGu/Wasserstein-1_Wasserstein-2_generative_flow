#!/usr/bin/env python
# coding: utf-8
# Proximal OT Flow GAN with Wasserstein-p proximals with p>1

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # avoid tensorflow warning
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from numpy import concatenate

def load_result(path, cnn=True, losses=False):
    import pickle
    result_path = path + "/result.pickle"

    with open(result_path, "rb") as fr:
        if cnn:
            G_W, G_b, G_conv_W, D_W, D_b, D_conv_W, T, dt, D_loss_evals, loss_OT_L2_evals, loss_OT_evals, loss_terminal_evals, D_loss_test_evals, alpha1,_,_ = pickle.load(fr)
            
            if losses == False:
                return G_W, G_b, G_conv_W, D_W, D_b, D_conv_W, T, dt, alpha1
            else:
                return D_loss_evals, loss_OT_L2_evals, loss_OT_evals, loss_terminal_evals, D_loss_test_evals
        else:
            G_W, G_b, D_W, D_b, T, dt, D_loss_evals, loss_OT_L2_evals, loss_OT_evals, loss_terminal_evals, D_loss_test_evals, alpha1,_,_ = pickle.load(fr)
            if losses == False:
                return G_W, G_b, D_W, D_b, T, dt, alpha1
            else:
                return D_loss_evals, loss_OT_L2_evals, loss_OT_evals, loss_terminal_evals, D_loss_test_evals


# neural network structure
def feed_conv_NN(X, W, act = tf.nn.tanh):
    A = X
    L = len(W)
    for i in range(L):
        A = tf.nn.max_pool2d(input = act( tf.nn.conv2d(input = A, filters=W[i], strides=(1,1), padding='SAME') ), ksize=(2,2), strides=(2,2), padding='SAME')
    return A
    
def feed_NN(X, W, b, act = tf.nn.tanh):
    A = X
    L = len(W)
    for i in range(L-1):
        A = act(tf.add(tf.matmul(A, W[i]), b[i]))
    return tf.add(tf.matmul(A, W[-1]), b[-1])

def fun_U_cnn(x, t, conv_W,  W, b):
    N_samples = tf.shape(x)[0]
            
    filtered_x = feed_conv_NN(x, conv_W, act=tf.nn.relu)
    h = tf.transpose(filtered_x, perm=[3,2,1,0])
    h = tf.transpose(tf.reshape(h, [-1, N_samples]))
    
          
    inputs = tf.concat([h,t], axis = 1)
    U = feed_NN(inputs, W, b, act=tf.nn.relu)
    return U
    
def fun_U(x, t, W, b):
    inputs = tf.concat([x,t], axis = 1)
    U = feed_NN(inputs, W, b)

    return U
    
    
def generator_cnn(x, steps, dt, conv_W, W, b, alpha1):
    currentx = x
    x_shape = tf.shape(x)
    tt = tf.fill([x_shape[0], 1], 1.0)
    
    for i in range(steps):
        currentt = tt * i * dt
        currentU = fun_U_cnn(currentx, currentt, conv_W, W, b)
        grad_U = tf.gradients(currentU, currentx)[0]
        
        currentv = -grad_U/alpha1
        #if p.power > 2.0 or p.power < 2.0:
        #    currentv *= tf.pow(tf.reduce_sum(tf.square(grad_U), axis = [1,2,3], keepdims = True), (2.0-p.power)/(p.power -1.0)/2.0)
        currentx += dt * currentv
        #if p.sigma>0:
        #    currentx += p.sigma* tf.sqrt(dt)*tf.random_normal(x_shape)
        
    return currentx
    
    
    
def generator(x, steps, dt, W, b, alpha1):
    tt = tf.fill(tf.shape(x), 1.0)[:,:1]

    currentx = x
    x_shape = tf.shape(x)
    
    for i in range(steps):
        currentt = tt * i * dt
        currentU = fun_U(currentx, currentt, W, b)
        grad_U = tf.gradients(currentU, currentx)[0]
        
        currentv = -grad_U/alpha1
        #if p.power > 2.0 or p.power < 2.0:
        #    currentv *= tf.pow(tf.norm(grad_U, axis = 1, keepdims = True), (2.0-p.power)/(p.power -1.0))
        currentx = currentx + dt * currentv
        #if p.sigma>0:
        #    currentx += p.sigma* tf.sqrt(dt)*tf.random_normal(x_shape)
   
    return currentx

def generate_fnn(path, N_samples, total_dim, dt = 0.0, bs=200):
    G_W, G_b, D_W, D_b, T, old_dt, alpha1 = load_result(path, cnn=False)
    if old_dt != dt:
        dt = old_dt
    
    G_W = [tf.Variable(w, trainable=False) for w in G_W]
    G_b = [tf.Variable(b, trainable=False) for b in G_b]
    #G_conv_W = [tf.Variable(conv_w, trainable=False) for conv_w in G_conv_W]

    D_W = [tf.Variable(w) for w in D_W]
    D_b = [tf.Variable(b) for b in D_b]
    #D_conv_W = [tf.Variable(conv_w) for conv_w in D_conv_W]
    
    steps = int(T/dt)
    Pdata = tf.placeholder(tf.float32, [None,total_dim])
    GPdata = generator(Pdata, steps, dt, G_W, G_b, alpha1)
    
    # initial data
    from data_generator import dataGenerator
    if 'Checkerboard' in path:
        initial_dist = 'Uniform'
    else:
        initial_dist = 'Gaussian'
    misc_P = {'submnfld_dim': total_dim, 'label':None, 'pretrained_ae':None, 'random_seed': 100}
    P = dataGenerator(N_samples, initial_dist, total_dim, misc_P)
    
    # configuration / initialization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    out = []
    for i in range(int(N_samples/bs)):
        thisP = P.nextbatch(bs)
        out.append( sess.run(GPdata, feed_dict={Pdata: thisP}) )
    
    return concatenate(out)
    
    
def generate_cnn(path, N_samples, total_dim, dt = 0.0, bs=200):
    G_W, G_b, G_conv_W, D_W, D_b, D_conv_W, T, old_dt, alpha1 = load_result(path)
    if old_dt != dt:
        dt = old_dt
    
    G_W = [tf.Variable(w, trainable=False) for w in G_W]
    G_b = [tf.Variable(b, trainable=False) for b in G_b]
    G_conv_W = [tf.Variable(conv_w, trainable=False) for conv_w in G_conv_W]

    D_W = [tf.Variable(w) for w in D_W]
    D_b = [tf.Variable(b) for b in D_b]
    D_conv_W = [tf.Variable(conv_w) for conv_w in D_conv_W]
    
    steps = int(T/dt)
    Pdata = tf.placeholder(tf.float32, [None] + total_dim)
    GPdata = generator_cnn(Pdata, steps, dt, G_conv_W, G_W, G_b, alpha1)

    # initial data
    from data_generator import dataGenerator
    initial_dist = 'Image_uniform'
    misc_P = {'submnfld_dim': total_dim, 'label':None, 'pretrained_ae':None, 'random_seed': 100}
    P = dataGenerator(N_samples, initial_dist, total_dim, misc_P)
    
    # configuration / initialization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    out = []
    for i in range(int(N_samples/bs)):
        thisP = P.nextbatch(bs)
        out.append( sess.run(GPdata, feed_dict={Pdata: thisP}) )
    
    return concatenate(out)
    


def discriminator(X, conv_W, W, b):
    N_samples = tf.shape(X)[0]
    
    filtered_X = feed_conv_NN(X, conv_W, act= tf.nn.relu)
    h = tf.transpose(filtered_X, perm=[3,2,1,0])
    h = tf.transpose(tf.reshape(h, [-1, N_samples]))
    y = feed_NN(h, W,b, act= tf.nn.relu)
    return y
    

'''
# GAN loss
steps = int(T/dt)
Pdata = tf.placeholder(tf.float32, [None] + p.total_dim)
GPdata, loss_OT, loss_OT_L2 = generator(Pdata, steps, dt, G_conv_W, G_W, G_b)

Qdata = tf.placeholder(tf.float32, [None]+ p.total_dim)

fake_score = discriminator(GPdata,D_conv_W, D_W,D_b)
real_score = discriminator(Qdata,D_conv_W, D_W,D_b)
GP = gradient_penalty(GPdata, Qdata, D_conv_W, D_W, D_b, L=p.L)

# discriminator loss
D_loss_1 = tf.reduce_mean(fake_score)
if p.f == 'W1':
    D_loss_2 = tf.reduce_mean(real_score)
elif p.f == 'reverseKL':
    D_loss_2 = -1 - tf.reduce_mean(tf.log(tf.nn.relu(-real_score)+1e-6))
else:
    D_loss_2 = tf.log(tf.reduce_mean(tf.exp(real_score)))
D_loss_intermediate = -D_loss_1 + D_loss_2
D_loss =  D_loss_intermediate + lamda * GP

# test discriminator loss
test_dt = dt
test_steps = int(T/test_dt)
test_GPdata, _, _ = generator(Pdata, test_steps, test_dt, G_conv_W, G_W, G_b, calc_reg=False)

test_fake_score = discriminator(test_GPdata, D_conv_W,D_W,D_b)
test_real_score = discriminator(Qdata, D_conv_W,D_W,D_b)

D_loss_1_test = tf.reduce_mean(test_fake_score)
if p.f == 'W1':
    D_loss_2_test = tf.reduce_mean(test_real_score)
elif p.f == 'reverseKL':
    D_loss_2_test = -1 - tf.reduce_mean(tf.log(tf.nn.relu(-test_real_score)+1e-6))
else:
    D_loss_2_test = tf.log(tf.reduce_mean(tf.exp(test_real_score)))
D_loss_intermediate_test = -D_loss_1_test + D_loss_2_test


'''


