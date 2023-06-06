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

parser = argparse.ArgumentParser()
parser.add_argument("--domain", type=tuple, default=[[-1, 1], [
                    -1, 1]], help="tuple类型, 必须与input_dim对应使用, 表示求解区域为[tuple[0][0],tuple[0][1]] x [tuple[1][0],tuple[1][1]]")
parser.add_argument("--n_test", type=int, default=256,
                    help="在求解区域的每个维度上, 均匀n_test个测试点, 一共生成")
parser.add_argument("--alpha", type=float, default=1000, help="正则性解前面的系数")

opt = parser.parse_args()
print(opt)

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
### k = 4时, 取GAS_uniform_2960的结果, k=10,取 GAS_uniform_4400
k_adaptive = 5

xi_in_gas = torch.load("./GAS_one3/data/xi_in.pt").cpu()
if k_adaptive == 5:
    xi_in_uniform = torch.load("./GAS_uniform_2960/xi_in.pt").cpu()
elif k_adaptive == 10:
    xi_in_uniform = torch.load("./GAS_uniform_4400/xi_in.pt").cpu()
elif k_adaptive == 12:
    xi_in_uniform = torch.load("./GAS_uniform_4880/xi_in.pt").cpu()

# # 前2000个点为初始点, 即 xi_in =[0:2000,0:3], 最后一列对应的是adaptive轮数

index_end = 500 + k_adaptive * 500
xi_plot_gas = xi_in_gas[0:index_end, 0:2]


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
my_pinns = torch.load(f'./GAS_one3/saved_model/my_pinns_model_{k_adaptive}_2900')
pinns_uniform_id = 2900 * (k_adaptive + 1)
pinns_uniform = torch.load(f'./GAS_uniform_4400/saved_model/my_pinns_model_0_{pinns_uniform_id}')
### 画图
with torch.no_grad():
    u_gas = my_pinns(x)
    u_uniform = pinns_uniform(x)
    
u_gas_re = u_gas.reshape((opt.n_test, opt.n_test))
u_gas_re = u_gas_re.detach().cpu().numpy()

u_uniform_re = u_uniform.reshape((opt.n_test, opt.n_test))
u_uniform_re = u_uniform_re.detach().cpu().numpy()

err_absolute = np.abs(u_exact_re - u_gas_re)
err_l2 = np.mean(err_absolute**2)
err_l2_format = format(err_l2, '.5E')
#err_l2 = np.sqrt(err_l2/np.mean(u_exact_re**2))

# plt.figure(dpi=300)
fig, axs = plt.subplots(2, 3, figsize = (7,5), sharex=True, sharey=True, dpi=300)
# plt.suptitle(f'Epoch = {epoch}, epoch_P = {epoch_P}, MSE = {err_l2_format}')

plot1 = axs[0, 0].imshow(u_exact_re, cmap='jet', extent =[opt.domain[0][0], opt.domain[0][1], opt.domain[1][0], opt.domain[1][1]], origin ='lower',vmin=0, vmax=1)
plt.colorbar(plot1, ax = axs[0, 0])
axs[0, 0].set_title("$u_{exact}$")

plot2 = axs[0, 1].imshow(u_uniform_re, cmap='jet', extent =[opt.domain[0][0], opt.domain[0][1], opt.domain[1][0], opt.domain[1][1]], origin ='lower',vmin=0, vmax=1)
plt.colorbar(plot2, ax = axs[0,1])
axs[0, 1].set_title("$u_{pinns}$")


plot3 = axs[0, 2].imshow(u_gas_re, cmap='jet', extent =[opt.domain[0][0], opt.domain[0][1], opt.domain[1][0], opt.domain[1][1]], origin ='lower')
plt.colorbar(plot3, ax = axs[0, 2])
axs[0, 2].set_title("$u_{gas}$")


plot4 = axs[1, 0].scatter(xi_in_uniform[:,0:1], xi_in_uniform[:,1:2], s=0.1)
# plt.colorbar(plot4, ax = axs[1, 0])
axs[1, 0].set_title("uniform data")

plot5 = axs[1, 1].scatter(xi_plot_gas[:,0:1], xi_plot_gas[:,1:2], s=0.1)
# plt.colorbar(plot5, ax = axs[1, 1])
axs[1, 1].set_title("gas data")


plot6 = axs[1, 2].scatter(xi_plot_gas[0:2500,0:1], xi_plot_gas[0:2500,1:2], s=0.1)
plot6 = axs[1, 2].scatter(xi_plot_gas[2500:,0:1], xi_plot_gas[2500:,1:2], s=0.1, c = 'r')
# plt.colorbar(plot6, ax = axs[1, 2])
axs[1, 2].set_title("added points")