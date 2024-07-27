#!/usr/bin/env python
# coding: utf-8
# Proximal OT Flow GAN with Wasserstein-p proximals with p>1

import numpy as np
import random
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # avoid tensorflow warning
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
import pickle


folder = 'assets/MNIST/D1P2/Flow-GAN-Rep'
restart_points = [str(x) for x in [11,22,23, 24]]  #range(501, 504+1)

    
# Load trained flow
#trained_G_Ws, trained_G_bs, trained_G_conv_Ws = {}, {}, {}
D_losses, loss_OT_L2s = [], []
alpha1_s = []
for i, restart_point in enumerate(restart_points):
    restart_path = folder + restart_point + "/" if restart_point[-1] != "/" else folder + restart_point
    restart_path = restart_path + "result.pickle"
    try:
        with open(restart_path, "rb") as fr:
            j, k, D_W, D_b, _, _, D_loss_evals, loss_OT_L2_evals, loss_OT_evals, loss_terminal_evals, D_loss_test_evals, alpha1s, alpha2, alpha3 = pickle.load(fr)
            #j, k, l, D_W, D_b, D_conv_W, _, _, D_loss_evals, loss_OT_L2_evals, loss_OT_evals, loss_terminal_evals, D_loss_test_evals = pickle.load(fr)
    except:
        with open(restart_path, "rb") as fr:
            j, k, l, D_W, D_b, D_conv_W, _, _, D_loss_evals, loss_OT_L2_evals, loss_OT_evals, loss_terminal_evals, D_loss_test_evals, alpha1s, alpha2, alpha3 = pickle.load(fr)
        
    D_losses.append(D_loss_evals[-1])
    loss_OT_L2s.append(loss_OT_L2_evals[-1])
    if type(alpha1s) != type([]):
        alpha1s = [alpha1s]*(int(5000/100)+2)

    alpha1_s.extend(alpha1s[1:-1])
    
    #trained_G_Ws[i] = [tf.Variable(w, trainable=False) for w in trained_G_Ws[i]]
    #trained_G_bs[i] = [tf.Variable(b, trainable=False) for b in trained_G_bs[i]]
    #trained_G_conv_Ws[i] = [tf.Variable(conv_w, trainable=False) for conv_w in trained_G_conv_Ws[i]]

 
# create data
x = [f'$prox_{n+1}$' for n in range(len(restart_points))]
#[r'$prox_1$', r'$prox_2$', r'$prox_3$', r'$prox_4$', r'$prox_5$', r'$prox_6$', r'$prox_7$']

# plot bars in stack manner
plt.bar(x, D_losses, color='sandybrown', label = r'$D_{KL}$')
loss_OT_L2s_= [alpha1s[-1]*loss_OT_L2 for loss_OT_L2 in loss_OT_L2s]
plt.bar(x, loss_OT_L2s, bottom=D_losses, color='royalblue', label = r'$W_2^2$')
#plt.ylim([0, 20])
plt.legend(fontsize=17)
plt.tight_layout()
plt.show()

n = len(x)
fig, axes = plt.subplots(1, n, figsize=(12, 2))

for i in range(n):
    axes[i].pie([D_losses[i], loss_OT_L2s[i]], labels = [r'$D_{KL}$', r'$W_2^2$'], autopct='%1.0f%%')
    axes[i].set_title(x[i], fontsize = 12)
#plt.legend(loc='best', fontsize=17)
plt.tight_layout()
plt.show()

plt.plot(range(1, 5000*n+1, 100), alpha1_s, label=r'$\alpha_1$')
plt.legend()
plt.tight_layout()
plt.show()

