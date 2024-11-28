# Wasserstein-proximal generative flows
Code submission for the paper **Well-posed generative flows via composition of Wasserstein-1 and Wasserstein-2 proximals** 

Tensorflow version 1 implementation of CNN potential flow adversarial model which is adapted from the code for [Potential flow GAN](https://arxiv.org/abs/1908.11462).

## Required library
* tensorflow>=1.15,<2.0
* keras


## Directory structure
data\_generator.py : function generating true samples from target distribution or initial distribution
POT\_Flow\_GAN.py: script training flow GANs for low dimensional examples
POT\_Flow\_GAN\_CNN.py: script training flow GANs for Image examples
plot\_result.py: functions plotting results
sample\_generation.py: functions loading trained flow GAN and generating additional samples
low_dimensional_result.ipynb: script plotting results from low dimensional examples
MNIST_result.ipynb: script plotting results from image example (MNIST)
assets/ : results and parameters of training flow GAN
figures/ : figures generated from low_dimensional_result.ipynb or MNIST_result.ipynb



## Step 1: Training Wasserstein-proximal generative flows
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

Pinwheel
```
python3 POT_Flow_GAN.py --total_dim 2 --example Pinwheel --Rep 3 --alpha1 0.5 --iterations 25000 --lamda 0.1 --gen_dims 64 64 64 64 64 --disc_dims 64 64 64
python3 POT_Flow_GAN.py --total_dim 7 -submnfld_dim 2 --example Pinwheel --Rep 3 --alpha1 0.1 --iterations 25000 --lamda 0.01 --gen_dims 64 64 64 64 64 --disc_dims 64 64 64
python3 POT_Flow_GAN.py --total_dim 12 -submnfld_dim 2 --example Pinwheel --Rep 3 --alpha1 0.1 --iterations 25000 --lamda 0.01 --gen_dims 64 64 64 64 64 --disc_dims 64 64 64

python3 POT_Flow_GAN.py --total_dim 2 --example Pinwheel --Rep 3 --alpha1 0.5 --iterations 25000 --lamda 0.1 -L 1.0 --gen_dims 64 64 64 64 64 --disc_dims 64 64 64
python3 POT_Flow_GAN.py --total_dim 7 -submnfld_dim 2 --example Pinwheel --Rep 3 --alpha1 0.1 --iterations 25000 --lamda 0.01 -L 1.0 --gen_dims 64 64 64 64 64 --disc_dims 64 64 64
python3 POT_Flow_GAN.py --total_dim 12 -submnfld_dim 2 --example Pinwheel --Rep 3 --alpha1 0.1 --iterations 25000 --lamda 0.01 -L 1.0 --gen_dims 64 64 64 64 64 --disc_dims 64 64 64

# Comparison PF generator
python3 POT_Flow_GAN.py --total_dim 2 --example Pinwheel --Rep 3 --alpha1 0.5 --iterations 25000 --lamda 0.1 -L 1.0 --gen_dims 64 64 64 64 64 --disc_dims 64 64 64 --f W1 --loss_case PG
python3 POT_Flow_GAN.py --total_dim 7 -submnfld_dim 2 --example Pinwheel --Rep 3 --alpha1 0.1 --iterations 25000 --lamda 0.01 -L 1.0 --gen_dims 64 64 64 64 64 --disc_dims 64 64 64 --f W1 --loss_case PG
python3 POT_Flow_GAN.py --total_dim 12 -submnfld_dim 2 --example Pinwheel --Rep 3 --alpha1 0.1 --iterations 25000 --lamda 0.01 -L 1.0 --gen_dims 64 64 64 64 64 --disc_dims 64 64 64 --f W1 --loss_case PG

```

Moons
```
python3 POT_Flow_GAN.py --total_dim 2 --example Moons --Rep 3 --alpha1 0.5 --iterations 25000 --lamda 0.1
python3 POT_Flow_GAN.py --total_dim 7 -submnfld_dim 2 --example Moons --Rep 3 --alpha1 0.1 --iterations 25000 --lamda 0.01
python3 POT_Flow_GAN.py --total_dim 12 -submnfld_dim 2 --example Moons --Rep 3 --alpha1 0.1 --iterations 25000 --lamda 0.01

python3 POT_Flow_GAN.py --total_dim 2 --example Moons --Rep 3 --alpha1 0.5 --iterations 25000 --lamda 0.1 -L 1.0
python3 POT_Flow_GAN.py --total_dim 7 -submnfld_dim 2 --example Moons --Rep 3 --alpha1 0.1 --iterations 25000 --lamda 0.01 -L 1.0
python3 POT_Flow_GAN.py --total_dim 12 -submnfld_dim 2 --example Moons --Rep 3 --alpha1 0.1 --iterations 25000 --lamda 0.01 -L 1.0


#Comparison PF generator
python3 POT_Flow_GAN.py --total_dim 2 --example Moons --Rep 3 --alpha1 0.5 --iterations 25000 --lamda 0.1 -L 1.0 --f W1 --loss_case PG
python3 POT_Flow_GAN.py --total_dim 7 -submnfld_dim 2 --example Moons --Rep 3 --alpha1 0.1 --iterations 25000 --lamda 0.01 -L 1.0 --f W1 --loss_case PG
python3 POT_Flow_GAN.py --total_dim 12 -submnfld_dim 2 --example Moons --Rep 3 --alpha1 0.1 --iterations 25000 --lamda 0.01 -L 1.0 --f W1 --loss_case PG

```


## Step 2: Plotting results
Run all blocks of ```low_dimensional_result.ipynb``` or ```MNIST_result.ipynb``` to plot figures in the paper.
