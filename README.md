# Wasserstein-proximal generative flows
Code submission for the paper **Well-posed generative flows via composition of Wasserstein-1 and Wasserstein-2 proximals** in Neurips 2024

Tensorflow version 1 implementation of CNN potential flow adversarial model which is adapted from the code for [Potential flow GAN](https://arxiv.org/abs/1908.11462).

## Required library
* tensorflow>=1.15,<2.0
* keras


## Training Wasserstein-proximal generative flows
MNIST
```
# No regularization
python3 POT_Flow_GAN_CNN.py --example MNIST --total_dim 784 --Rep 3 --f reverse_KL --alpha1 0.05 --T 5.0 --dt 1.0 --loss_case No_OT --iterations 200
# Wasserstein-1 proximal flow
python3 POT_Flow_GAN_CNN.py --example MNIST --total_dim 784 --Rep 3 --f reverse_KL -L 1.0 --alpha1 0.05 --T 5.0 --dt 1.0 --loss_case No_OT
# Wasserstein-2 proximal flow
python3 POT_Flow_GAN_CNN.py --example MNIST --total_dim 784 --Rep 3 --f reverse_KL --alpha1 0.05 --T 5.0 --dt 1.0 --loss_case OT --iterations 200
# Wasserstein-1/Wasserstein-2 proximal flow
python3 POT_Flow_GAN_CNN.py --example MNIST --total_dim 784 --Rep 3 --f reverse_KL -L 1.0 --alpha1 0.05 --T 5.0 --dt 1.0 --loss_case OT
```

## Plotting results
Run all blocks of ```assets/MNIST/MNIST_result_plots.ipynb``` to plot figures in the paper.
