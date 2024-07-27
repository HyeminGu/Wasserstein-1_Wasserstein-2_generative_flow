#!/usr/bin/env python
# coding: utf-8
import numpy as np
import random
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle


class dataGenerator:
    def __init__(self, allsize, example, total_dim, misc={}):
        self.allsize = allsize
        self.example = example
        self.total_dim = total_dim
        self.random_seed = misc['random_seed'] if misc['random_seed'] != None else None
        self.submnfld_dim = misc['submnfld_dim'] if misc['submnfld_dim'] != None else total_dim
        if self.example == 'Gaussian':
            self.scale = 1.
            self.alldata = self.gaussian(allsize).astype(np.float32)
        elif self.example == 'Uniform':
            self.scale = 1.
            self.alldata = self.uniform(allsize).astype(np.float32)
        elif self.example == 'Logistic':
            self.scale = 1.
            self.alldata = self.logistic(allsize).astype(np.float32)
        elif self.example == 'Mixture_of_8_gaussians':
            self.scale = 2.
            self.alldata = self.mixture_of_8_gaussians(allsize).astype(np.float32)
        elif self.example == 'Checkerboard':
            self.scale = 1.
            self.alldata = self.checkerboard(allsize).astype(np.float32)
        elif self.example == 'Swissroll':
            self.scale = 5.
            self.alldata = self.swissroll(allsize).astype(np.float32)
        elif self.example == 'Circles':
            self.scale = 2.
            self.alldata = self.circles(allsize).astype(np.float32)
        elif self.example == 'Moons':
            self.scale = 7.
            self.alldata = self.moons(allsize).astype(np.float32)
        elif self.example == 'Pinwheel':
            self.scale = 2.
            self.alldata = self.pinwheel(allsize).astype(np.float32)
        elif self.example == 'Spirals2':
            self.scale = 2.
            self.alldata = self.spirals2(allsize).astype(np.float32)
        elif self.example == 'Student-t':
            self.scale = 40.
            df = misc['df']
            self.alldata = self.student_t(allsize, df).astype(np.float32)
 
        elif self.example == 'MNIST':
            self.scale = 1.
            label = misc['label']
            pretrained_ae = misc['pretrained_ae']
            self.alldata = self.mnist(allsize, label, pretrained_ae).astype(np.float32)
        elif self.example == 'CIFAR10':
            self.scale = 1.
            label = misc['label']
            pretrained_ae = misc['pretrained_ae']
            self.alldata = self.cifar10(allsize, label, pretrained_ae).astype(np.float32)
        elif self.example == 'Image_uniform':
            self.scale = 0.5
            pretrained_ae = misc['pretrained_ae']
            self.alldata = self.image_uniform(allsize, pretrained_ae).astype(np.float32)
        '''elif self.example == 'Image_gaussian':
            self.scale = 1.
            pretrained_ae = misc['pretrained_ae']
            self.alldata = self.image_gaussian(allsize, pretrained_ae).astype(np.float32)'''
            
    def nextbatch(self, bs):
        indices = np.random.choice(self.allsize, bs, replace = False)
        return self.alldata[indices,:]

    
        
    def gaussian(self, sample_size):
        if self.random_seed != None:
            np.random.seed(self.random_seed)
        embedded_dataset = np.zeros((self.allsize,self.total_dim))
        dataset = np.random.normal(0,1,[self.allsize, self.submnfld_dim])*self.scale
        embedded_dataset[:,np.arange(self.submnfld_dim)] = dataset
        return embedded_dataset
        
    def uniform(self, sample_size):
        if self.random_seed != None:
            np.random.seed(self.random_seed)
        embedded_dataset = np.zeros((self.allsize,self.total_dim))
        dataset = np.random.uniform(-1, 1, [self.allsize, self.submnfld_dim])*self.scale
        embedded_dataset[:,np.arange(self.submnfld_dim)] = dataset
        return embedded_dataset
        
    def logistic(self, sample_size):
        if self.random_seed != None:
            np.random.seed(self.random_seed)
        embedded_dataset = np.zeros((self.allsize,self.total_dim))
        dataset = np.random.logistic(2, 1, [self.allsize, self.submnfld_dim])*self.scale
        embedded_dataset[:,np.arange(self.submnfld_dim)] = dataset
        return embedded_dataset

    def mixture_of_8_gaussians(self, sample_size):
        if self.random_seed != None:
            np.random.seed(self.random_seed)
        embedded_dataset = np.zeros((self.allsize,self.total_dim))
        centers = [
            (1,0),
            (-1,0),
            (0,1),
            (0,-1),
            (1./np.sqrt(2), 1./np.sqrt(2)),
            (1./np.sqrt(2), -1./np.sqrt(2)),
            (-1./np.sqrt(2), 1./np.sqrt(2)),
            (-1./np.sqrt(2), -1./np.sqrt(2))
        ]
        centers = [(self.scale*x,self.scale*y) for x,y in centers]
        dataset = []
        for i in range(sample_size):
            point = np.random.randn(2)*.2
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        embedded_dataset[:,np.arange(self.submnfld_dim)] = dataset
        return embedded_dataset
        
    def checkerboard(self, sample_size):
        if self.random_seed != None:
            np.random.seed(self.random_seed)
        embedded_dataset = np.zeros((self.allsize,self.total_dim))
        dataset = []
        for i in range(4):
            for j in range(4):
                if (i + j)%2 == 0:
                    center = np.array([i * 0.5 - 0.75, j * 0.5 - 0.75])
                    dataset.append(center + np.random.uniform(-0.25,0.25,[self.allsize//8,2]))
        dataset = np.concatenate(dataset)
        embedded_dataset[:,np.arange(self.submnfld_dim)] = dataset
        return embedded_dataset
        
    def swissroll(self, sample_size):
        if self.random_seed != None:
            np.random.seed(self.random_seed)
        embedded_dataset = np.zeros((self.allsize,self.total_dim))
        dataset = sklearn.datasets.make_swiss_roll(n_samples=sample_size, noise=1.0)[0]
        dataset = dataset[:, [0, 2]] * 0.07 * self.scale
        embedded_dataset[:,np.arange(self.submnfld_dim)] = dataset
        return embedded_dataset
        
    def circles(self, sample_size):
        if self.random_seed != None:
            np.random.seed(self.random_seed)
        embedded_dataset = np.zeros((self.allsize,self.total_dim))
        dataset = sklearn.datasets.make_circles(n_samples=sample_size, factor=.5, noise=0.08)[0]
        dataset = dataset * self.scale
        embedded_dataset[:,np.arange(self.submnfld_dim)] = dataset
        return embedded_dataset
        
    def moons(self, sample_size):
        if self.random_seed != None:
            np.random.seed(self.random_seed)
        embedded_dataset = np.zeros((self.allsize,self.total_dim))
        dataset = sklearn.datasets.make_moons(n_samples=sample_size, noise=0.05)[0]
        dataset = (dataset  - np.array([0.5, 0.2]))* self.scale * 0.6
        embedded_dataset[:,np.arange(self.submnfld_dim)] = dataset
        return embedded_dataset
 
    def pinwheel(self, sample_size):
        if self.random_seed != None:
            np.random.seed(self.random_seed)
        embedded_dataset = np.zeros((self.allsize,self.total_dim))
        
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = sample_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
                
        rng = np.random.RandomState()
        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        dataset = self.scale * 0.7 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))
        embedded_dataset[:,np.arange(self.submnfld_dim)] = dataset
        return embedded_dataset
        
    def spirals2(self, sample_size):
        if self.random_seed != None:
            np.random.seed(self.random_seed)
        embedded_dataset = np.zeros((self.allsize,self.total_dim))
        
        n = np.sqrt(np.random.rand(sample_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(sample_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(sample_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        
        dataset = x * self.scale * 0.32
        embedded_dataset[:,np.arange(self.submnfld_dim)] = dataset
        return embedded_dataset
        
    def student_t(self, sample_size, df=1.5):
        embedded_dataset = np.zeros((self.allsize,self.total_dim))
        
        from scipy.stats import multivariate_t
        
        d = self.submnfld_dim
        P_ = multivariate_t(np.zeros(d), np.eye(d), df=df)
        dataset = P_.rvs(size=self.allsize, random_state=0)
        embedded_dataset[:,np.arange(self.submnfld_dim)] = dataset
        return embedded_dataset
        
        
    def mnist(self, sample_size, label=None, pretrained_ae=None):
        if self.random_seed != None:
            np.random.seed(self.random_seed)
        import tensorflow as tf
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
        x_train = x_train/255.0
        if label != None:
            label_idx = np.ndarray.flatten(y_train == label)
            x_train = x_train[label_idx]
        
        idx = np.random.permutation(x_train.shape[0])
        dataset = x_train[idx[:sample_size]]
        if dataset.ndim == 3:
            dataset = np.expand_dims(dataset, 3)
            
        if pretrained_ae != None:
            from autoencoder_mnist import load_autoencoder
            encoder, _ = load_autoencoder(pretrained_ae)
            dataset = encoder.predict(dataset)
            
        print(np.max(dataset), np.min(dataset))
                
        return dataset
        
    def mnist_logistic(self, sample_size, pretrained_ae=None):
        if self.random_seed != None:
            np.random.seed(self.random_seed)
        import tensorflow as tf
        
        dataset = np.random.logistic(2, 1, [self.allsize, 28,28,1])*self.scale
        
        if pretrained_ae != None:
            from autoencoder_mnist import load_autoencoder
            encoder, _ = load_autoencoder(pretrained_ae)
            dataset = encoder.predict(dataset)
            
        #print(np.max(dataset), np.min(dataset))
        
        return dataset
        
    def cifar10(self, sample_size, label=None, pretrained_ae=None):
        if self.random_seed != None:
            np.random.seed(self.random_seed)
        import tensorflow as tf
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()#path="cifar10.npz")
        x_train = x_train/255.0
        if label != None:
            label_idx = np.ndarray.flatten(y_train == label)
            x_train = x_train[label_idx]
        
        idx = np.random.permutation(x_train.shape[0])
        dataset = x_train[idx[:sample_size]]
        if dataset.ndim == 3:
            dataset = np.expand_dims(dataset, 3)
            
        if pretrained_ae != None:
            from autoencoder_cifar10 import load_autoencoder
            encoder, _ = load_autoencoder(pretrained_ae)
            dataset = encoder.predict(dataset)
            
        print(np.max(dataset), np.min(dataset))
                
        return dataset
        
    def image_uniform(self, sample_size, pretrained_ae=None):
        if self.random_seed != None:
            np.random.seed(self.random_seed)
            
        import tensorflow as tf
        
        dataset = np.random.uniform(0, 1.0, [self.allsize]+ self.total_dim)*self.scale
        
        if pretrained_ae != None:
            from autoencoder_mnist import load_autoencoder
            encoder, _ = load_autoencoder(pretrained_ae)
            dataset = encoder.predict(dataset)
            
        #print(np.max(dataset), np.min(dataset))
        
        return dataset

