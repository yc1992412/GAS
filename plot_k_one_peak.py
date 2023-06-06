# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 12:23:27 2023

@author: Fate
"""

from cmath import nan
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
# from sklearn.metrics import mean_squared_error # 均方误差
import torch.nn as nn
from torch.autograd import Variable
from torch import autograd, Tensor
from typing import Callable
import argparse
import numpy as np
import torch
import random

torch.manual_seed(1) # 为CPU设置随机种子
torch.cuda.manual_seed_all(1) # 为所有GPU设置随机种子
np.random.seed(1)
random.seed(1)

## epoch 是adaptive, epoch_P是 Pinns的
# epoch = 1
# epoch = 5
epoch = 9
epoch_P = 2999


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs_adaptive", type=int, default=10, help="adaptive训练的epochs总数")
parser.add_argument("--epoch_P", type=float, default=3000, help="PINNs训练时的epoch次数")
parser.add_argument("--n_sample_in", type=int, default=500, help="内部点需要的总样本数")
parser.add_argument("--n_sample_bound", type=int, default=50, help="每条边上需要的样本数,n_sample_bound*4=n_sample_in/4")

parser.add_argument("--complete_batches", type=int, default=100, help="用来学习PINNs的总batch数")
parser.add_argument("--batch_size_in", type=int, default=500, help="内部点每个batch的大小")
parser.add_argument("--batch_size_bound", type=int, default=50, help="边界点每个batch的大小, 应小于n_sample_bound/ndim")

parser.add_argument("--num_max_index", type=int, default=20, help="对残差最大的前opt.num_max_index个点构造高斯分布")
parser.add_argument("--Gaussian_number", type=int, default=25, help="从每个点处构造的高斯分布中选取Gaussian_number个点加到总样本当中")
parser.add_argument("--add_addpative_uniform", type=int, default=0, help="添加的自适应点和均匀分布点的比值")
parser.add_argument("--bound_add_number", type=int, default=50, help="每条边上加入到边界样本点中的个数,应保持bound_add_number*4*4 = num_max_index*Gaussian_number*(add_addpative_uniform+1)")
parser.add_argument("--sigma_penalty", type=float, default=5e-2, help="生成高斯分布时,方差前的修正参数,值越小高斯分布峰值越大")
parser.add_argument("--lr1", type=float, default=1e-3,
                    help="Adam 算法中 PINNs 初始时刻的learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="Adam 算法中一阶矩估计的指数衰减率")
parser.add_argument("--b2", type=float, default=0.999,
                    help="Adam 算法中二阶矩估计的指数衰减率")
parser.add_argument("--n_cpu", type=int, default=1, help="使用的cpu个数")
parser.add_argument("--input_dim", type=int, default=2, help="采样空间的维度")
parser.add_argument("--domain", type=tuple, default=[[-1, 1], [
                    -1, 1]], help="tuple类型, 必须与input_dim对应使用, 表示求解区域为[tuple[0][0],tuple[0][1]] x [tuple[1][0],tuple[1][1]]")
parser.add_argument("--epsilon", type=float, default=0.05,
                    help="生成(0,1)区域上采样点时,控制映射到边界点的区域参数, 即(0, epsilon)和(1-epsilon, 1)映射到边界")
parser.add_argument("--n_test", type=int, default=256,
                    help="在求解区域的每个维度上, 均匀n_test个测试点, 一共生成")
parser.add_argument("--beta", type=float, default=5e1, help="PINNs中边界项前面的罚参")
parser.add_argument("--alpha", type=float, default=1000, help="正则性解前面的系数")

opt = parser.parse_args()
print(opt)
#浮点精度
EPS = np.finfo('float').eps

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


## eps为最后画图是边界区域向外延伸的长度
eps = 0.05

rc1_x = 0.5
rc1_y = 0.5
rc2_x = -0.5
rc2_y = -0.5

### one peak
# source 项
def f_source(x): return -1*(torch.exp(-opt.alpha* ((x[:,0:1] - rc1_x)**2 + (x[:,1:2] - rc1_y)**2))*((-2*opt.alpha*(x[:,0:1] - rc1_x))**2 + (-2*opt.alpha*torch.sub(x[:,1:2], rc1_y))**2 - 4*opt.alpha ))
# 边界项
def g_bound(x): return torch.exp(-opt.alpha* (torch.add(torch.square(x[:,0:1] - rc1_x),torch.square(x[:,1:2] - rc1_y))))
# 真解 u 的表达式
def u(x): return torch.exp(-opt.alpha* ((x[:,0:1] - rc1_x)**2 +(x[:,1:2] - rc1_y)**2))


@torch.no_grad()
def _init_params(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

# class Block(nn.Module):
#     def __init__(self, input_size, hidden_width, output_size):
#         super(Block, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_width)
#         self.fc2 = nn.Linear(hidden_width, output_size)
#         self.apply(_init_params)

#     def forward(self, x):
#         res = torch.tanh(self.fc1(x))
#         res = torch.tanh(self.fc2(res))
#         return torch.add(x, res)


class PINNs(nn.Module):
    def __init__(self, input_size, output_size):
        super(PINNs, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 32, bias=True)
        self.fc2 = nn.Linear(32, 32, bias=True)
        self.fc3 = nn.Linear(32, 32, bias=True)
        self.fc4 = nn.Linear(32, 32, bias=True)
        self.fc5 = nn.Linear(32, 32, bias=True)
        # self.fc6 = nn.Linear(32, 32, bias=True)
        # self.fc7 = nn.Linear(32, 32, bias=True)
        self.fc8 = nn.Linear(32, output_size, bias=True)
        self.apply(_init_params)


    def forward(self, x, activate = torch.tanh):
        out = self.fc1(x)
        out = activate(out)
        out = self.fc2(out)
        out = activate(out)
        out = self.fc3(out)
        out = activate(out)
        out = self.fc4(out)
        out = activate(out)
        out = self.fc5(out)
        out = activate(out)
        # out = self.fc6(out)
        # out = activate(out)
        # out = self.fc7(out)
        # out = activate(out)
        out = self.fc8(out)

        return out

    def grad(self, x):
        xclone = x.clone().detach().requires_grad_(True)
        uforward = self.forward(xclone)

        grad = autograd.grad(uforward, xclone, grad_outputs=torch.ones_like(uforward),
                             only_inputs=True, create_graph=True)
        gradx = grad[0][:, 0].reshape((-1, 1))
        grady = grad[0][:, 1].reshape((-1, 1))
        return gradx, grady

    def laplace(self, x, clone=True):
        if clone:
            xclone = x.clone().detach().requires_grad_(True)
        else:
            xclone = x.requires_grad_()
        uforward = self.forward(xclone)

        grad = autograd.grad(uforward, xclone, grad_outputs=torch.ones_like(uforward),
                             only_inputs=True, create_graph=True, retain_graph=True)[0]

        gradgradx = autograd.grad(grad[:, 0:1], xclone, grad_outputs=torch.ones_like(grad[:, 0:1]),
                                  only_inputs=True, create_graph=True, retain_graph=True)[0][:, 0:1]
        gradgrady = autograd.grad(grad[:, 1:2], xclone, grad_outputs=torch.ones_like(grad[:, 1:2]),
                                  only_inputs=True, create_graph=True, retain_graph=True)[0][:, 1:2]

        return gradgradx, gradgrady

## load data
xi_in_load = torch.load("./GAS_one/data/xi_in.pt").cpu()
xi_bound_all = torch.load(f'./GAS_one/data/xi_bound_add_all_{epoch}.pt')
xi_add_all = torch.load(f'./GAS_one/data/xi_add_all_{epoch}.pt')
xi_bound_add_all = torch.load(f'./GAS_one/data/xi_bound_add_all_{epoch}.pt')

xi_in_all = xi_in_load[0:opt.n_sample_in + epoch * opt.num_max_index * opt.Gaussian_number, 0:3]
# xi_bound_gas = torch.load("./GAS/data/xi_bound_all.pt").cpu()


## Test Mesh
x_test = np.linspace(opt.domain[0][0], opt.domain[0][1], opt.n_test, dtype=np.float32)
y_test = np.linspace(opt.domain[1][0], opt.domain[1][1], opt.n_test, dtype=np.float32)
X, Y = np.meshgrid(x_test, y_test)
X = X.reshape((-1, 1))
Y = Y.reshape((-1, 1))
x = torch.zeros(opt.n_test*opt.n_test, 2)
# mesh point
x = np.hstack((X, Y))
x = torch.tensor(x).to(device)

u_exact = u(x).reshape((-1, 1))
u_exact = u_exact.cpu().detach().numpy()
u_exact = u_exact.reshape((-1, 1))
u_exact_re = u_exact.reshape((opt.n_test, opt.n_test))

X_re = X.reshape((opt.n_test, opt.n_test))
Y_re = Y.reshape((opt.n_test, opt.n_test))

u_exact_re = u_exact.reshape((opt.n_test, opt.n_test))
my_pinns = torch.load(f'./GAS_one/saved_model/my_pinns_model_{epoch}_{epoch_P}')

### 画图
with torch.no_grad():
    u_test = my_pinns(x)

residual_plot = np.zeros([x.shape[0],1])
for i in range(opt.n_test):
    tx_i = x[i*opt.n_test: (i+1)*opt.n_test]
    u_xx, u_yy = my_pinns.laplace(tx_i)
    residual_plot[i*opt.n_test: (i+1)*opt.n_test, 0:1] = torch.abs(u_xx + u_yy + f_source(tx_i).reshape((-1,1))).detach().cpu().numpy()
residual_plot_all = residual_plot.reshape((opt.n_test, opt.n_test))  
        
u_test_re = u_test.reshape((opt.n_test, opt.n_test))
u_test_re = u_test_re.detach().cpu().numpy()

err_absolute = np.abs(u_exact_re - u_test_re)
err_l2 = np.mean(err_absolute**2)
err_l2_format = format(err_l2, '.3E')
#err_l2 = np.sqrt(err_l2/np.mean(u_exact_re**2))
print(f"Epoch = {epoch}, epoch_P = {epoch_P}, Relative err_l2 = {err_l2_format}")


fig, axs = plt.subplots(2, 3, figsize=(10,6), sharex=True, sharey=True, dpi=300)
plt.suptitle(f'Epoch = {epoch}, epoch_P = {epoch_P}, MSE = {err_l2_format}')

plot1 = axs[0, 0].imshow(u_exact_re, cmap='jet', aspect = 'auto', extent =[opt.domain[0][0], opt.domain[0][1], opt.domain[1][0], opt.domain[1][1]], origin ='lower',vmin=0, vmax=1)
plt.colorbar(plot1, ax = axs[0, 0])
axs[0, 0].set_title("$u_{exact}$")

plot2 = axs[0, 1].imshow(u_test_re, cmap='jet', aspect = 'auto', extent =[opt.domain[0][0], opt.domain[0][1], opt.domain[1][0], opt.domain[1][1]], origin ='lower',vmin=0, vmax=1)
plt.colorbar(plot2, ax = axs[0,1])
axs[0, 1].set_title("$u_{gas}$")

plot3 = axs[1, 1].imshow(err_absolute, cmap='jet', aspect = 'auto', extent =[opt.domain[0][0], opt.domain[0][1], opt.domain[1][0], opt.domain[1][1]], origin ='lower')
plt.colorbar(plot3, ax = axs[1, 1])
axs[1, 1].set_title("$err$")

plot4 = axs[1, 0].imshow(residual_plot_all, cmap='jet', aspect = 'auto', extent =[opt.domain[0][0], opt.domain[0][1], opt.domain[1][0], opt.domain[1][1]], origin ='lower')
plt.colorbar(plot4, ax = axs[1, 0])
axs[1, 0].set_title("$residual$")

if epoch != 0:
    plot5 = axs[0, 2].scatter(torch.cat((xi_add_all[:,0:1].cpu(),xi_bound_add_all[:,0:1].cpu())), torch.cat((xi_add_all[:,1:2].cpu(),xi_bound_add_all[:,1:2].cpu())), s=0.5)
    axs[0, 2].set_title("added points")

# plot5 = axs[1, 1].scatter(torch.cat((x_in_all[:,0:1].cpu(),x_bound_all_right[:,0:1].cpu(), x_bound_all_left[:,0:1].cpu(), t_bound_all[:,0:1].cpu())), torch.cat((x_in_all[:,1:2].cpu(),x_bound_all_right[:,1:2].cpu(), x_bound_all_left[:,1:2].cpu(), t_bound_all[:,1:2].cpu())), s=0.1)
# #plt.colorbar(plot5, ax = axs[1, 1])
# axs[1, 1].set_title("points before adding")

if epoch != 0:
    plot6 = axs[1, 2].scatter(torch.cat((xi_in_all[:,0:1].cpu(),xi_bound_all[:,0:1].cpu())), torch.cat((xi_in_all[:,1:2].cpu(),xi_bound_all[:,1:2].cpu())), s=0.5)
    plot6 = axs[1, 2].scatter(torch.cat((xi_add_all[:,0:1].cpu(),xi_bound_add_all[:,0:1].cpu())), torch.cat((xi_add_all[:,1:2].cpu(),xi_bound_add_all[:,1:2].cpu())), s=0.5, c = 'r')
    #plt.colorbar(plot6, ax = axs[1, 2])
    axs[1, 2].set_title("all points")

