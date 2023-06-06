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
import scipy.io as scio

torch.manual_seed(1) # 为CPU设置随机种子
torch.cuda.manual_seed_all(1) # 为所有GPU设置随机种子
np.random.seed(1)
random.seed(1)

## epoch 是adaptive, epoch_P是 Pinns的
epoch = 1
# epoch = 3
# epoch = 5
epoch_P = 2999


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs_adaptive", type=int, default=20, help="adaptive训练的epochs总数")
parser.add_argument("--epoch_P", type=float, default=3000, help="PINNs训练时的epoch次数")
parser.add_argument("--n_sample_in", type=int, default=10000, help="内部点需要的总样本数")
parser.add_argument("--n_sample_bound_x", type=int, default=200, help="每条边上需要的样本数, 一维问题有两条边")
parser.add_argument("--n_sample_bound_t", type=int, default=200, help="每条边上需要的样本数")
parser.add_argument("--complete_batches", type=int, default=100, help="用来学习PINNs的总batch数")
parser.add_argument("--batch_size_in", type=int, default=2000, help="内部点每个batch的大小")
parser.add_argument("--batch_size_bound", type=int, default=40, help="边界点每个batch的大小, 应小于n_sample_bound/ndim")

parser.add_argument("--num_max_index", type=int, default=10, help="对残差最大的前opt.num_max_index个点构造高斯分布")
parser.add_argument("--Gaussian_number", type=int, default=200, help="从每个点处构造的高斯分布中选取Gaussian_number个点加到总样本当中")
parser.add_argument("--bound_add_number", type=int, default=40, help="每条边上加入到边界样本点中的个数")
parser.add_argument("--sigma_penalty", type=float, default=5e-2, help="生成高斯分布时,方差前的修正参数,值越小高斯分布峰值越大")
parser.add_argument("--lr1", type=float, default=1e-3,
                    help="Adam 算法中 PINNs 初始时刻的learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="Adam 算法中一阶矩估计的指数衰减率")
parser.add_argument("--b2", type=float, default=0.999,
                    help="Adam 算法中二阶矩估计的指数衰减率")
parser.add_argument("--n_cpu", type=int, default=1, help="使用的cpu个数")
parser.add_argument("--input_dim", type=int, default=2, help="采样空间的维度, 这里2维指 (t,x)两个维度")
parser.add_argument("--domain", type=tuple, default=[[0, 1], [
                    -1, 1]], help="tuple类型, 必须与input_dim对应使用, 表示求解区域为[tuple[0][0],tuple[0][1]] x [tuple[1][0],tuple[1][1]]")
parser.add_argument("--epsilon", type=float, default=0.05,
                    help="生成(0,1)区域上采样点时,控制映射到边界点的区域参数, 即(0, epsilon)和(1-epsilon, 1)映射到边界")
parser.add_argument("--n_test_t", type=int, default=201,
                    help="在求解区域的每个维度上, 均匀n_test个测试点, 一共生成")
parser.add_argument("--n_test_x", type=int, default=256,
                    help="在求解区域的每个维度上, 均匀n_test个测试点, 一共生成")
parser.add_argument("--beta", type=float, default=500, help="PINNs中边界项前面的罚参")
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

### Burgers
# def f_source(x): return torch.exp(-x[:,0:1]) * torch.sin (np.pi * x[:, 1:2]) + (-1 * torch.exp(-x[:,0:1]) * torch.sin (np.pi * x[:, 1:2])) * (-np.pi * torch.exp(-x[:,0:1]) * torch.cos (np.pi * x[:,1:2])) - (0.01/np.pi)*(np.pi**2 * torch.exp(-x[:,0:1]) * torch.sin (np.pi * x[:,1:2]))
def f_source(x): return 0*x[:,0:1]
# 时间边界项, 即 u(0, x)
def t_bound(x): return -1 * torch.sin (np.pi * x[:,1:2])
# 空间边界项
def x_bound(x): return -0 * x[:,0:1]

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
        
        self.fc1 = nn.Linear(input_size, 20, bias=True)
        self.fc2 = nn.Linear(20, 20, bias=True)
        self.fc3 = nn.Linear(20, 20, bias=True)
        self.fc4 = nn.Linear(20, 20, bias=True)
        self.fc5 = nn.Linear(20, 20, bias=True)
        self.fc6 = nn.Linear(20, 20, bias=True)
        self.fc7 = nn.Linear(20, 20, bias=True)
        self.fc8 = nn.Linear(20, 20, bias=True)
        self.fc9 = nn.Linear(20, output_size, bias=True)
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
        out = self.fc6(out)
        out = activate(out)
        out = self.fc7(out)
        out = activate(out)
        out = self.fc8(out)
        out = activate(out)
        out = self.fc9(out)

        return out

    def grad(self, x):
        xclone = x.clone().detach().requires_grad_(True)
        uforward = self.forward(xclone)

        grad = autograd.grad(uforward, xclone, grad_outputs=torch.ones_like(uforward),
                             only_inputs=True, create_graph=True)
        gradt = grad[0][:, 0].reshape((-1, 1))
        gradx = grad[0][:, 1].reshape((-1, 1))
        return gradt, gradx

    def laplace(self, x, clone=True):
        if clone:
            xclone = x.clone().detach().requires_grad_(True)
        else:
            xclone = x.requires_grad_()
        uforward = self.forward(xclone)

        grad = autograd.grad(uforward, xclone, grad_outputs=torch.ones_like(uforward),
                             only_inputs=True, create_graph=True, retain_graph=True)[0]

        gradgradt = autograd.grad(grad[:, 0:1], xclone, grad_outputs=torch.ones_like(grad[:, 0:1]),
                                  only_inputs=True, create_graph=True, retain_graph=True)[0][:, 0:1]
        gradgradx = autograd.grad(grad[:, 1:2], xclone, grad_outputs=torch.ones_like(grad[:, 1:2]),
                                  only_inputs=True, create_graph=True, retain_graph=True)[0][:, 1:2]

        return gradgradt, gradgradx


## load data
x_in_all = torch.load("./Burgers/data/x_in.pt").cpu()
x_bound_all_right = torch.load('./Burgers/data/x_bound_all_right.pt')
x_bound_all_left = torch.load('./Burgers/data/x_bound_all_left.pt')
t_bound_all = torch.load('./Burgers/data/t_bound_all.pt')

x_add_all = torch.load(f'./Burgers/data/xi_add_all_{epoch}.pt')
x_bound_add_all_left = torch.load(f'./Burgers/data/x_bound_add_all_left_{epoch}.pt')
x_bound_add_all_right = torch.load(f'./Burgers/data/x_bound_add_all_right_{epoch}.pt')
t_bound_add_all = torch.load(f'./Burgers/data/t_bound_add_all_{epoch}.pt')


x_in_all = x_in_all[0:opt.n_sample_in + epoch * opt.num_max_index * opt.Gaussian_number, 0:3]
x_bound_all_right = x_bound_all_right[0:opt.n_sample_in + epoch * opt.num_max_index * opt.Gaussian_number, 0:3]
x_bound_all_left = x_bound_all_left[0:opt.n_sample_in + epoch * opt.num_max_index * opt.Gaussian_number, 0:3]


## load test data
data = scio.loadmat('./Burgers/gen_data/Burgers_solution.mat')
t_mesh = data['t'].reshape([-1,1]).astype('float32')
x_mesh = data['x'].reshape([-1,1]).astype('float32')
u_exact = data['usol']
u_exact_re = u_exact.reshape([opt.n_test_x, opt.n_test_t]).astype('float32')

T, X = np.meshgrid(t_mesh, x_mesh)
T = T.reshape((-1, 1))
X = X.reshape((-1, 1))
tx = torch.zeros(opt.n_test_t * opt.n_test_x, 2)
# mesh point
tx = np.hstack((T, X))
tx = torch.tensor(tx).to(device)

my_pinns = torch.load(f'./Burgers/saved_model/my_pinns_model_{epoch}_{epoch_P}')

### 画图
with torch.no_grad():
    u_test = my_pinns(tx)
    
residual_plot = np.zeros([tx.shape[0],1])
for i in range(opt.n_test_x):
    tx_i = tx[i*opt.n_test_t: (i+1)*opt.n_test_t]
    residual_forward = my_pinns(tx_i)
    residual_Ggrad_t, residual_Ggrad_x = my_pinns.grad(tx_i)
    _, residual_gradgrad_x = my_pinns.laplace(tx_i)

    residual_plot[i*opt.n_test_t: (i+1)*opt.n_test_t, 0:1] = torch.abs(residual_Ggrad_t + residual_forward * residual_Ggrad_x - (0.01/np.pi) * residual_gradgrad_x).detach().cpu().numpy()
residual_plot_all = residual_plot.reshape((opt.n_test_x, opt.n_test_t))  

u_test_re = u_test.reshape((opt.n_test_x, opt.n_test_t))
u_test_re = u_test_re.detach().cpu().numpy()

err_absolute = np.abs(u_exact_re - u_test_re)
err_l2 = np.mean(err_absolute**2)
err_l2_format = format(err_l2, '.3E')


Relative_err_l2_format = np.linalg.norm(u_exact_re - u_test_re)/np.linalg.norm(u_exact_re)
Relative_err_l2_format = format(Relative_err_l2_format.item(), '.3E')

print(f"Epoch = {epoch}, epoch_P = {epoch_P}, MSE = {err_l2_format}, Relative err_l2 = {Relative_err_l2_format}")

fig, axs = plt.subplots(2, 3, figsize=(10,6), sharex=True, sharey=True, dpi=300)
plt.suptitle(f'Epoch = {epoch}, epoch_P = {epoch_P}, MSE = {err_l2_format}')

plot1 = axs[0, 0].imshow(u_exact_re, cmap='jet', aspect = 'auto', extent =[opt.domain[0][0], opt.domain[0][1], opt.domain[1][0], opt.domain[1][1]], origin ='lower',vmin=-1, vmax=1)
plt.colorbar(plot1, ax = axs[0, 0])
axs[0, 0].set_title("$u_{exact}$")

plot2 = axs[0, 1].imshow(u_test_re, cmap='jet', aspect = 'auto', extent =[opt.domain[0][0], opt.domain[0][1], opt.domain[1][0], opt.domain[1][1]], origin ='lower',vmin=-1, vmax=1)
plt.colorbar(plot2, ax = axs[0, 1])
axs[0, 1].set_title("$u_{gas}$")

plot3 = axs[1, 1].imshow(err_absolute, cmap='jet', aspect = 'auto', extent =[opt.domain[0][0], opt.domain[0][1], opt.domain[1][0], opt.domain[1][1]], origin ='lower',vmin=0, vmax=1)
plt.colorbar(plot3, ax = axs[1, 1])
axs[1, 1].set_title("$err$")

plot4 = axs[1, 0].imshow(residual_plot_all, cmap='jet', aspect = 'auto', extent =[opt.domain[0][0], opt.domain[0][1], opt.domain[1][0], opt.domain[1][1]], origin ='lower',vmin=0, vmax=1)
plt.colorbar(plot4, ax = axs[1, 0])
axs[1, 0].set_title("$residual$")

if epoch != 0:
    plot5 = axs[0, 2].scatter(torch.cat((x_add_all[:,0:1].cpu(),x_bound_add_all_right[:,0:1].cpu(), x_bound_add_all_left[:,0:1].cpu(), t_bound_add_all[:,0:1].cpu())), torch.cat((x_add_all[:,1:2].cpu(),x_bound_add_all_right[:,1:2].cpu(), x_bound_add_all_left[:,1:2].cpu(), t_bound_add_all[:,1:2].cpu())), s=0.1)
    #plt.colorbar(plot4, ax = axs[0, 2])
    axs[0, 2].set_title("added points")
5

if epoch != 0:
    plot6 = axs[1, 2].scatter(torch.cat((x_in_all[:,0:1].cpu(),x_bound_all_right[:,0:1].cpu(), x_bound_all_left[:,0:1].cpu(), t_bound_all[:,0:1].cpu())), torch.cat((x_in_all[:,1:2].cpu(),x_bound_all_right[:,1:2].cpu(), x_bound_all_left[:,1:2].cpu(), t_bound_all[:,1:2].cpu())), s=0.1)
    plot6 = axs[1, 2].scatter(torch.cat((x_add_all[:,0:1].cpu(),x_bound_add_all_right[:,0:1].cpu(), x_bound_add_all_left[:,0:1].cpu(), t_bound_add_all[:,0:1].cpu())), torch.cat((x_add_all[:,1:2].cpu(),x_bound_add_all_right[:,1:2].cpu(), x_bound_add_all_left[:,1:2].cpu(), t_bound_add_all[:,1:2].cpu())), s=0.1, c='r')
    axs[1, 2].set_title("all points")

