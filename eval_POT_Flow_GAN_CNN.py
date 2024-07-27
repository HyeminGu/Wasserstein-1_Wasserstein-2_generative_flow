#!/usr/bin/env python
# coding: utf-8
# Proximal OT Flow GAN with Wasserstein-p proximals with p>1

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # avoid tensorflow warning
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pickle
import sys

from data_generator import dataGenerator

import argparse
parser = argparse.ArgumentParser()

# Data configuration
parser.add_argument(
    '--N_samples_Q', type=int, help='total number of target samples',  default=6000,
)
parser.add_argument(
    '--N_samples_P', type=int, help='total number of initial samples',  default=50000,
)
parser.add_argument(
    '--Rep', type=int, help='random seed for data generator', default=0,
)
parser.add_argument(
    '--example', type=str, #choices=['Mixture_of_8_gaussians','Checkerboard', 'Swissroll', 'Circles', 'Moons', 'Pinwheel', 'Spirals2', 'MNIST',],
)
parser.add_argument(
    '--total_dim', type=int, help='total dimension of dataset',
)
parser.add_argument(
    '-submnfld_dim', type=int, help='submanifold dimension of dataset',
)
parser.add_argument(
    '-label', type=int, help='pick certain label of dataset',
)
parser.add_argument(
    '-pretrained_ae', type=str, help='path to pretrained autoencoder',
)

# Loss/flow configuration
parser.add_argument(
    '--power', type=float, help='power p for W_p distance',  default=2.0,
)
parser.add_argument(
    '--alpha1', type=float, help='coefficient for L2 regularizer', default=1.0#0.5
)
parser.add_argument(
    '--alpha2', type=float, help='coefficient for HJB regularizer', default=0.1
)
parser.add_argument(
    '--alpha3', type=float, help='coefficient for terminal condition regularizer', default=0.1
)
parser.add_argument(
    '--loss_case', type=str, help='OT: OT regularization only, PG: potential generator, OC: optimality conditions, No_OT: no OT regularization', default='OT',
)
parser.add_argument(
    '--f', type=str, choices=['KL', 'alpha', 'reverse_KL', 'W1'], default='reverse_KL',
)
parser.add_argument(
    '-L', type=float, help='Lipschitz constant for discriminator',
)
parser.add_argument(
    '--T', type=float, help='terminal time', default=1.0
)
parser.add_argument(
    '--dt', type=float, help='time step size', default=0.2
)
parser.add_argument(
    '--sigma', type=float, help='noise level for SDE', default=0.0
)

# Neural network/training configuration
parser.add_argument(
    '--lamda', type=float, help='coefficient for Gradient Penalty (discriminator)', default=0.1
)

# Miscellaneous parameters
parser.add_argument(
    '--CUDA', type=str, help='number of CUDA devices', default='0',
)
parser.add_argument(
    '-savedir', type=str, help='specific save directory',
)
parser.add_argument(
    '--time_rescale', type=int, help='refine dt for test dataset', default = 2,
)

p = parser.parse_args()

if p.power <= 1.0:
    raise ValueError

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= p.CUDA

tf.reset_default_graph()
tf.set_random_seed(p.Rep)
np.random.seed(p.Rep)

# data generation
if p.example == 'MNIST':
    initial_dist = 'Image_uniform'
    p.total_dim = [28, 28, 1]
elif p.example == 'CIFAR10':
    initial_dist = 'Image_uniform'
    p.total_dim = [32, 32, 3]
else:
    initial_dist = 'Gaussian'
misc_Q = {'submnfld_dim': p.submnfld_dim, 'label':p.label, 'pretrained_ae':p.pretrained_ae, 'random_seed': p.Rep}
misc_P = {'submnfld_dim': p.total_dim, 'label':p.label, 'pretrained_ae':p.pretrained_ae, 'random_seed': p.Rep}
Q = dataGenerator(p.N_samples_Q, p.example, p.total_dim, misc_Q)
P = dataGenerator(p.N_samples_P, initial_dist, p.total_dim, misc_P)

if p.pretrained_ae != None:
    from autoencoder_mnist import load_autoencoder
    _, decoder = load_autoencoder(p.pretrained_ae)#'pretrained_ae/mnist_64d'


if p.savedir == None:
    savedir = f"assets/{p.example}/"
    savedir += "D1" if p.L != None else ""
    savedir += "" if p.loss_case == "No_OT" else "P2"
    savedir += f"{p.loss_case}" if "OT" not in p.loss_case else ""
    savedir += f"/Flow-GAN-Rep{p.Rep}/" if p.submnfld_dim == None else f"/Flow-GAN-dim{p.total_dim}-Rep{p.Rep}/"


# Load trained flow
restart_point = savedir
restart_path = restart_point + "/" if restart_point[-1] != "/" else restart_point
restart_path = restart_path + "result.pickle"
try:
    with open(restart_path, "rb") as fr:
        G_W, G_b, G_conv_W, D_W, D_b, D_conv_W, T, dt, old_D_loss_evals, old_loss_OT_L2_evals, _, _, _, _,_,_ = pickle.load(fr)
except:
    with open(restart_path, "rb") as fr:
        G_W, G_b, G_conv_W, D_W, D_b, D_conv_W, T, dt, old_D_loss_evals, old_loss_OT_L2_evals, _, _, _,_,_ = pickle.load(fr)
G_W = [tf.Variable(w, trainable=False) for w in G_W]
G_b = [tf.Variable(b, trainable=False) for b in G_b]
G_conv_W = [tf.Variable(conv_w, trainable=False) for conv_w in G_conv_W]

D_W = [tf.Variable(w) for w in D_W]
D_b = [tf.Variable(b) for b in D_b]
D_conv_W = [tf.Variable(conv_w) for conv_w in D_conv_W]

lamda = p.lamda
bs = min([200, p.N_samples_Q])
alpha1 = p.alpha1
alpha2 = p.alpha2
alpha3 = p.alpha3
T = p.T
dt = p.dt


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
    

def discriminator(X, conv_W, W, b):
    N_samples = tf.shape(X)[0]
    
    filtered_X = feed_conv_NN(X, conv_W, act= tf.nn.relu)
    h = tf.transpose(filtered_X, perm=[3,2,1,0])
    h = tf.transpose(tf.reshape(h, [-1, N_samples]))
    y = feed_NN(h, W,b, act= tf.nn.relu)
    return y
    

def gradient_penalty(Gen, Tar, conv_W, W, b, L=1.0, batch_size = bs):
    if L == None:
        return 0
    zu = tf.random_uniform([batch_size,1,1,1], minval=0, maxval=1, dtype=tf.float32)
    D_interpolates = zu * Tar + (1 - zu) * Gen
    D_disc_interpolates = discriminator(D_interpolates, conv_W, W, b)
    D_gradients = tf.gradients(D_disc_interpolates, [D_interpolates])[0]
    D_slopes = tf.reduce_sum(tf.square(D_gradients), reduction_indices=[1,2,3])
    D_gradient_penalty = tf.reduce_mean(tf.nn.relu(D_slopes-L**2))

    return D_gradient_penalty


def fun_U(x, t, conv_W,  W, b):
    N_samples = tf.shape(x)[0]
            
    filtered_x = feed_conv_NN(x, conv_W, act=tf.nn.relu)
    h = tf.transpose(filtered_x, perm=[3,2,1,0])
    h = tf.transpose(tf.reshape(h, [-1, N_samples]))
    
          
    inputs = tf.concat([h,t], axis = 1)
    U = feed_NN(inputs, W, b, act=tf.nn.relu)
    return U

def running_cost(x, t, G_conv_W, G_W, G_b):
    N_samples = tf.cast(tf.shape(x)[0], dtype=tf.float32)
    thisx = tf.stop_gradient(x)
    thist = t
    U = fun_U(thisx, thist, G_conv_W, G_W, G_b)
    grad_U = tf.gradients(U,thisx)[0]
    lapl_U = 0
    if p.sigma > 0.:
        lapl_U = 1.
        #diag_hess_U = [tf.gradients(grad_U[:,j,k,l],thisx)[0][:,j, k, l] for j in range(p.total_dim[0]) for k in range(p.total_dim[1]) for l in range(p.total_dim[2])]
        #lapl_U = sum(diag_hess_U)
    res = tf.gradients(U, thist)[0][:,0] + (1.0 -1.0/p.power)/alpha1 * tf.pow(tf.math.reduce_sum(tf.math.square(grad_U), axis = [1,2,3]), p.power/(p.power -1.0)/2.0) - 0.5* p.sigma**2 * lapl_U
    HJB = tf.reduce_mean(tf.abs(res))
    Lp = 1.0/p.power * tf.pow(tf.reduce_sum(tf.math.square(grad_U/alpha1)), p.power/2.0)/N_samples
        
    return Lp, HJB

def generator(x, steps, dt, conv_W, W, b, calc_reg = True):
    currentx = x
    x_shape = tf.shape(x)
    tt = tf.fill([x_shape[0], 1], 1.0)
    
    # running cost
    loss_OT = [0.0 for i in range(steps)]
    loss_OT_Lp = [0.0 for i in range(steps)]
    
    for i in range(steps):
        currentt = tt * i * dt
        currentU = fun_U(currentx, currentt, conv_W, W, b)
        grad_U = tf.gradients(currentU, currentx)[0]
        
        currentv = -grad_U/alpha1
        if p.power > 2.0 or p.power < 2.0:
            currentv *= tf.pow(tf.reduce_sum(tf.square(grad_U), axis = [1,2,3], keepdims = True), (2.0-p.power)/(p.power -1.0)/2.0)
        currentx += dt * currentv
        if p.sigma>0:
            currentx += p.sigma* tf.sqrt(dt)*tf.random_normal(x_shape)
        
        # running cost
        if calc_reg == True:
            loss_OT_Lp[i], loss_OT[i] = running_cost(currentx, currentt, conv_W, W, b)
        
    loss_OT = tf.reduce_sum(loss_OT) * dt
    loss_OT_Lp = tf.reduce_sum(loss_OT_Lp) * dt
        
    return currentx, loss_OT, loss_OT_Lp
    

# GAN loss
steps = int(T/dt) * p.time_rescale
Pdata = tf.placeholder(tf.float32, [None] + p.total_dim)
GPdata, loss_OT, loss_OT_L2 = generator(Pdata, steps, dt/p.time_rescale, G_conv_W, G_W, G_b)

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
test_dt = dt/p.time_rescale
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


# Comparing potential and discriminator
#height_shift = tf.Variable(0.0, dtype=tf.float32)
x_T = tf.stop_gradient(GPdata)
U_T = fun_U(x_T, tf.fill([tf.shape(x_T)[0],1], T), G_conv_W, G_W, G_b)
disc = discriminator(x_T, D_conv_W, D_W, D_b)
v1 = tf.gradients(disc, x_T)[0]
v2 = tf.gradients(U_T, x_T)[0]

#loss_terminal = tf.reduce_mean(tf.square(tf.subtract(U_T/alpha1, disc) + height_shift))
#loss_terminal = tf.reduce_mean(tf.square(tf.subtract(U_T, disc) + height_shift))
N_samples_ = tf.cast(tf.shape(x_T)[0], dtype=tf.float32)
loss_terminal = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(v1-v2)))/N_samples_


if p.loss_case == 'OT':
    G_loss = D_loss_1 + alpha1 * loss_OT_L2
elif p.loss_case == 'No_OT':
    G_loss = D_loss_1
elif p.loss_case == 'PG':
    G_loss = D_loss_1 + alpha2 * loss_OT + alpha1 * loss_OT_L2
elif p.loss_case == 'OC':
    G_loss = D_loss_1 + alpha2 * loss_OT + alpha1 * loss_OT_L2 + alpha3 * loss_terminal
elif p.loss_case == 'HJB':
    G_loss = D_loss_1 + alpha2 * loss_OT + alpha1 * loss_OT_L2



# configuration / initialization
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# training
from plot_result import plot_2d, plot_orth_axes_saturation, plot_image

    

'''
# train loss
thisP = P.nextbatch(5000)
thisQ = Q.nextbatch(Q.allsize)
out = sess.run(GPdata, feed_dict={Pdata: thisP})
D_loss_eval = np.abs(-sess.run(D_loss_intermediate, feed_dict= {Pdata: thisP, Qdata: thisQ}))
loss_OT_eval = sess.run(loss_OT, feed_dict= {Pdata: thisP, Qdata: thisQ})

print("Train dataset : divergence %.4f, kinetic energy %.4f" % (D_loss_eval, loss_OT_eval))

# plot result for train data
if p.pretrained_ae != None:
    thisP = decoder.predict(thisP)
    thisQ = decoder.predict(thisQ)
    out = decoder.predict(out)
    
if type(p.total_dim) == type([]):
#if p.example in ['MNIST', 'CIFAR10']:
    plot_image(out, savedir+"/eval_scale%d" % p.time_rescale, 5)
else:
    if p.submnfld_dim != None:
        plot_orth_axes_saturation(out, savedir+"/eval_scale%d" % p.time_rescale)
    
    plot_count = min([Q.allsize, 1000])
    thisQ = thisQ[:plot_count]
    plot_2d(out, thisP, thisQ, savedir+"/eval_scale%d" % p.time_rescale, Q.scale)
'''

# test loss
thisP = dataGenerator(5000, initial_dist, p.total_dim, misc_P).nextbatch(5000)
thisQ = Q.nextbatch(Q.allsize)
out = sess.run(test_GPdata, feed_dict={Pdata: thisP})
D_loss_test_eval = np.abs(-sess.run(D_loss_intermediate_test, feed_dict= {Pdata: thisP, Qdata: thisQ}))

print("Test dataset : divergence %.4f" % (D_loss_test_eval))

# plot result for test data
if p.pretrained_ae != None:
    thisP = decoder.predict(thisP)
    thisQ = decoder.predict(thisQ)
    out = decoder.predict(out)
   
if type(p.total_dim) == type([]):
#if p.example in ['MNIST', 'CIFAR10']:
    plot_image(out, savedir+"/eval_scale%d_test_" % p.time_rescale, 5)
else:
    if p.submnfld_dim != None:
        plot_orth_axes_saturation(out, savedir+"/eval_scale%d_test_" % p.time_rescale)
    
    plot_count = min([Q.allsize, 1000])
    thisQ = thisQ[:plot_count]
    plot_2d(out, thisP, thisQ, savedir+"/eval_scale%d_test_" % p.time_rescale, Q.scale)
    
    






