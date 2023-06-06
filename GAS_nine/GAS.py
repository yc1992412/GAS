# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 20:45:06 2020

@author: Fate-LD

E-mail: lidi.math@whu.edu.cn
"""


#from mpl_toolkits.mplot3d import axes3d
#import matplotlib.pyplot as plt

# problem: -\laplace u = f in [0,2*pi] and f = sin(x) ; u = 0 on x=0 and x=2*pi, exact solution u_exact = sin(x)


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
parser.add_argument("--n_epochs_adaptive", type=int, default=21, help="adaptive训练的epochs总数")
parser.add_argument("--epoch_P", type=float, default=5000, help="PINNs训练时的epoch次数")
parser.add_argument("--n_sample_in", type=int, default=1000, help="内部点需要的总样本数")
parser.add_argument("--n_sample_bound", type=int, default=100, help="每条边上需要的样本数,n_sample_bound*4=n_sample_in/4")

parser.add_argument("--complete_batches", type=int, default=100, help="用来学习PINNs的总batch数")
parser.add_argument("--batch_size_in", type=int, default=500, help="内部点每个batch的大小")
parser.add_argument("--batch_size_bound", type=int, default=50, help="边界点每个batch的大小, 应小于n_sample_bound/ndim")

parser.add_argument("--num_max_index", type=int, default=40, help="对残差最大的前opt.num_max_index个点构造高斯分布")
parser.add_argument("--Gaussian_number", type=int, default=25, help="从每个点处构造的高斯分布中选取Gaussian_number个点加到总样本当中")
parser.add_argument("--add_addpative_uniform", type=int, default=0, help="添加的自适应点和均匀分布点的比值")
parser.add_argument("--bound_add_number", type=int, default=100, help="每条边上加入到边界样本点中的个数,应保持bound_add_number*4*4 = num_max_index*Gaussian_number*(add_addpative_uniform+1)")
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

rc1_x = -0.5
rc1_y = -0.5
rc2_x = 0
rc2_y = -0.5
rc3_x = 0.5
rc3_y = -0.5
rc4_x = -0.5
rc4_y = 0
rc5_x = 0
rc5_y = 0
rc6_x = 0.5
rc6_y = 0
rc7_x = -0.5
rc7_y = 0.5
rc8_x = 0
rc8_y = 0.5
rc9_x = 0.5
rc9_y = 0.5
### one peak
# source 项
#def f_source(x): return -1*(torch.exp(-opt.alpha* ((x[:,0:1] - rc1_x)**2 + (x[:,1:2] - rc1_y)**2))*((-2*opt.alpha*(x[:,0:1] - rc1_x))**2 + (-2*opt.alpha*torch.sub(x[:,1:2], rc1_y))**2 - 4*opt.alpha ))
# 边界项
#def g_bound(x): return torch.exp(-opt.alpha* (torch.add(torch.square(x[:,0:1] - rc1_x),torch.square(x[:,1:2] - rc1_y))))
# 真解 u 的表达式
#def u(x): return torch.exp(-opt.alpha* ((x[:,0:1] - rc1_x)**2 +(x[:,1:2] - rc1_y)**2))

# #### nine peak
# # source 项
# def f_source(x): return   torch.exp(-opt.alpha* ((x[:,0:1] - rc1_x)**2 + (x[:,1:2] - rc1_y)**2))*((-2*opt.alpha*(x[:,0:1] - rc1_x))**2 + (-2*opt.alpha*torch.sub(x[:,1:2], rc1_y))**2 - 4*opt.alpha + 4*opt.alpha*x[:,0:1]*(x[:,0:1] - rc1_x) + 4*opt.alpha*x[:,1:2]*(x[:,1:2] - rc1_y)-4) + torch.exp(-opt.alpha* ((x[:,0:1] - rc2_x)**2 + (x[:,1:2] - rc2_y)**2))*((-2*opt.alpha*(x[:,0:1] - rc2_x))**2 + (-2*opt.alpha*torch.sub(x[:,1:2], rc2_y))**2 - 4*opt.alpha + 4*opt.alpha*x[:,0:1]*(x[:,0:1] - rc2_x) + 4*opt.alpha*x[:,1:2]*(x[:,1:2] - rc2_y)-4)
def f_source(x): return  (torch.exp(-opt.alpha* ((x[:,0:1] - rc1_x)**2 + (x[:,1:2] - rc1_y)**2))*((-2*opt.alpha*(x[:,0:1] - rc1_x))**2 + (-2*opt.alpha*torch.sub(x[:,1:2], rc1_y))**2 - 4*opt.alpha + 4*opt.alpha*x[:,0:1]*(x[:,0:1] - rc1_x) + 4*opt.alpha*x[:,1:2]*(x[:,1:2] - rc1_y)-4)
                        + torch.exp(-opt.alpha* ((x[:,0:1] - rc2_x)**2 + (x[:,1:2] - rc2_y)**2))*((-2*opt.alpha*(x[:,0:1] - rc2_x))**2 + (-2*opt.alpha*torch.sub(x[:,1:2], rc2_y))**2 - 4*opt.alpha + 4*opt.alpha*x[:,0:1]*(x[:,0:1] - rc2_x) + 4*opt.alpha*x[:,1:2]*(x[:,1:2] - rc2_y)-4) 
                        + torch.exp(-opt.alpha* ((x[:,0:1] - rc3_x)**2 + (x[:,1:2] - rc3_y)**2))*((-2*opt.alpha*(x[:,0:1] - rc3_x))**2 + (-2*opt.alpha*torch.sub(x[:,1:2], rc3_y))**2 - 4*opt.alpha + 4*opt.alpha*x[:,0:1]*(x[:,0:1] - rc3_x) + 4*opt.alpha*x[:,1:2]*(x[:,1:2] - rc3_y)-4) 
                        + torch.exp(-opt.alpha* ((x[:,0:1] - rc4_x)**2 + (x[:,1:2] - rc4_y)**2))*((-2*opt.alpha*(x[:,0:1] - rc4_x))**2 + (-2*opt.alpha*torch.sub(x[:,1:2], rc4_y))**2 - 4*opt.alpha + 4*opt.alpha*x[:,0:1]*(x[:,0:1] - rc4_x) + 4*opt.alpha*x[:,1:2]*(x[:,1:2] - rc4_y)-4)
                        + torch.exp(-opt.alpha* ((x[:,0:1] - rc5_x)**2 + (x[:,1:2] - rc5_y)**2))*((-2*opt.alpha*(x[:,0:1] - rc5_x))**2 + (-2*opt.alpha*torch.sub(x[:,1:2], rc5_y))**2 - 4*opt.alpha + 4*opt.alpha*x[:,0:1]*(x[:,0:1] - rc5_x) + 4*opt.alpha*x[:,1:2]*(x[:,1:2] - rc5_y)-4)
                        + torch.exp(-opt.alpha* ((x[:,0:1] - rc6_x)**2 + (x[:,1:2] - rc6_y)**2))*((-2*opt.alpha*(x[:,0:1] - rc6_x))**2 + (-2*opt.alpha*torch.sub(x[:,1:2], rc6_y))**2 - 4*opt.alpha + 4*opt.alpha*x[:,0:1]*(x[:,0:1] - rc6_x) + 4*opt.alpha*x[:,1:2]*(x[:,1:2] - rc6_y)-4)
                        + torch.exp(-opt.alpha* ((x[:,0:1] - rc7_x)**2 + (x[:,1:2] - rc7_y)**2))*((-2*opt.alpha*(x[:,0:1] - rc7_x))**2 + (-2*opt.alpha*torch.sub(x[:,1:2], rc7_y))**2 - 4*opt.alpha + 4*opt.alpha*x[:,0:1]*(x[:,0:1] - rc7_x) + 4*opt.alpha*x[:,1:2]*(x[:,1:2] - rc7_y)-4)
                        + torch.exp(-opt.alpha* ((x[:,0:1] - rc8_x)**2 + (x[:,1:2] - rc8_y)**2))*((-2*opt.alpha*(x[:,0:1] - rc8_x))**2 + (-2*opt.alpha*torch.sub(x[:,1:2], rc8_y))**2 - 4*opt.alpha + 4*opt.alpha*x[:,0:1]*(x[:,0:1] - rc8_x) + 4*opt.alpha*x[:,1:2]*(x[:,1:2] - rc8_y)-4)
                        + torch.exp(-opt.alpha* ((x[:,0:1] - rc9_x)**2 + (x[:,1:2] - rc9_y)**2))*((-2*opt.alpha*(x[:,0:1] - rc9_x))**2 + (-2*opt.alpha*torch.sub(x[:,1:2], rc9_y))**2 - 4*opt.alpha + 4*opt.alpha*x[:,0:1]*(x[:,0:1] - rc9_x) + 4*opt.alpha*x[:,1:2]*(x[:,1:2] - rc9_y)-4))
                        
# # 边界项
def g_bound(x): return  (torch.exp(-opt.alpha* ((x[:,0:1] - rc1_x)**2 +(x[:,1:2] - rc1_y)**2)) 
                  + torch.exp(-opt.alpha* ((x[:,0:1] - rc2_x)**2 +(x[:,1:2] - rc2_y)**2))
                  + torch.exp(-opt.alpha* ((x[:,0:1] - rc3_x)**2 +(x[:,1:2] - rc3_y)**2))
                  + torch.exp(-opt.alpha* ((x[:,0:1] - rc4_x)**2 +(x[:,1:2] - rc4_y)**2))
                  + torch.exp(-opt.alpha* ((x[:,0:1] - rc5_x)**2 +(x[:,1:2] - rc5_y)**2))
                  + torch.exp(-opt.alpha* ((x[:,0:1] - rc6_x)**2 +(x[:,1:2] - rc6_y)**2))
                  + torch.exp(-opt.alpha* ((x[:,0:1] - rc7_x)**2 +(x[:,1:2] - rc7_y)**2))
                  + torch.exp(-opt.alpha* ((x[:,0:1] - rc8_x)**2 +(x[:,1:2] - rc8_y)**2))
                  + torch.exp(-opt.alpha* ((x[:,0:1] - rc9_x)**2 +(x[:,1:2] - rc9_y)**2)))

# # 真解 u 的表达式
def u(x): return   (torch.exp(-opt.alpha* ((x[:,0:1] - rc1_x)**2 +(x[:,1:2] - rc1_y)**2)) 
                  + torch.exp(-opt.alpha* ((x[:,0:1] - rc2_x)**2 +(x[:,1:2] - rc2_y)**2))
                  + torch.exp(-opt.alpha* ((x[:,0:1] - rc3_x)**2 +(x[:,1:2] - rc3_y)**2))
                  + torch.exp(-opt.alpha* ((x[:,0:1] - rc4_x)**2 +(x[:,1:2] - rc4_y)**2))
                  + torch.exp(-opt.alpha* ((x[:,0:1] - rc5_x)**2 +(x[:,1:2] - rc5_y)**2))
                  + torch.exp(-opt.alpha* ((x[:,0:1] - rc6_x)**2 +(x[:,1:2] - rc6_y)**2))
                  + torch.exp(-opt.alpha* ((x[:,0:1] - rc7_x)**2 +(x[:,1:2] - rc7_y)**2))
                  + torch.exp(-opt.alpha* ((x[:,0:1] - rc8_x)**2 +(x[:,1:2] - rc8_y)**2))
                  + torch.exp(-opt.alpha* ((x[:,0:1] - rc9_x)**2 +(x[:,1:2] - rc9_y)**2)))

# def f_source(x): return torch.zeros_like(x[:,0])
# def g_bound(x): return torch.ones_like(x[:,0])
# def u(x, y): return torch.ones_like(x)

# def f_source(x): return 2*np.pi**2*torch.sin(np.pi*x[:,0])*torch.sin(np.pi*x[:,1])
# def g_bound(x): return torch.sin(np.pi*x[:,0])*torch.sin(np.pi*x[:,1])
# def u(x, y): return torch.sin(np.pi*x)*torch.sin(np.pi*y)

# #### nine peak
# # source 项
#def f_source(x): return (-1*(torch.exp(-opt.alpha* ((x[:,0:1] - rc1_x)**2 + (x[:,1:2] - rc1_y)**2))*((-2*opt.alpha*(x[:,0:1] - rc1_x))**2 + (-2*opt.alpha*torch.sub(x[:,1:2], rc1_y))**2 - 4*opt.alpha )) 
#                        -1*(torch.exp(-opt.alpha* ((x[:,0:1] - rc2_x)**2 + (x[:,1:2] - rc2_y)**2))*((-2*opt.alpha*(x[:,0:1] - rc2_x))**2 + (-2*opt.alpha*torch.sub(x[:,1:2], rc2_y))**2 - 4*opt.alpha ))
#                        -1*(torch.exp(-opt.alpha* ((x[:,0:1] - rc3_x)**2 + (x[:,1:2] - rc3_y)**2))*((-2*opt.alpha*(x[:,0:1] - rc3_x))**2 + (-2*opt.alpha*torch.sub(x[:,1:2], rc3_y))**2 - 4*opt.alpha ))
#                        -1*(torch.exp(-opt.alpha* ((x[:,0:1] - rc4_x)**2 + (x[:,1:2] - rc4_y)**2))*((-2*opt.alpha*(x[:,0:1] - rc4_x))**2 + (-2*opt.alpha*torch.sub(x[:,1:2], rc4_y))**2 - 4*opt.alpha ))
#                        -1*(torch.exp(-opt.alpha* ((x[:,0:1] - rc5_x)**2 + (x[:,1:2] - rc5_y)**2))*((-2*opt.alpha*(x[:,0:1] - rc5_x))**2 + (-2*opt.alpha*torch.sub(x[:,1:2], rc5_y))**2 - 4*opt.alpha ))
#                        -1*(torch.exp(-opt.alpha* ((x[:,0:1] - rc6_x)**2 + (x[:,1:2] - rc6_y)**2))*((-2*opt.alpha*(x[:,0:1] - rc6_x))**2 + (-2*opt.alpha*torch.sub(x[:,1:2], rc6_y))**2 - 4*opt.alpha ))
#                        -1*(torch.exp(-opt.alpha* ((x[:,0:1] - rc7_x)**2 + (x[:,1:2] - rc7_y)**2))*((-2*opt.alpha*(x[:,0:1] - rc7_x))**2 + (-2*opt.alpha*torch.sub(x[:,1:2], rc7_y))**2 - 4*opt.alpha ))
#                        -1*(torch.exp(-opt.alpha* ((x[:,0:1] - rc8_x)**2 + (x[:,1:2] - rc8_y)**2))*((-2*opt.alpha*(x[:,0:1] - rc8_x))**2 + (-2*opt.alpha*torch.sub(x[:,1:2], rc8_y))**2 - 4*opt.alpha ))
#                        -1*(torch.exp(-opt.alpha* ((x[:,0:1] - rc9_x)**2 + (x[:,1:2] - rc9_y)**2))*((-2*opt.alpha*(x[:,0:1] - rc9_x))**2 + (-2*opt.alpha*torch.sub(x[:,1:2], rc9_y))**2 - 4*opt.alpha )))
#
# # 边界项
#def g_bound(x): return (torch.exp(-opt.alpha* (torch.add(torch.square(x[:,0:1] - rc1_x),torch.square(x[:,1:2] - rc1_y)))) 
#                     + torch.exp(-opt.alpha* (torch.add(torch.square(x[:,0:1] - rc2_x),torch.square(x[:,1:2] - rc2_y))))
#                     + torch.exp(-opt.alpha* (torch.add(torch.square(x[:,0:1] - rc3_x),torch.square(x[:,1:2] - rc3_y))))
#                     + torch.exp(-opt.alpha* (torch.add(torch.square(x[:,0:1] - rc4_x),torch.square(x[:,1:2] - rc4_y))))
#                     + torch.exp(-opt.alpha* (torch.add(torch.square(x[:,0:1] - rc5_x),torch.square(x[:,1:2] - rc5_y))))
#                     + torch.exp(-opt.alpha* (torch.add(torch.square(x[:,0:1] - rc6_x),torch.square(x[:,1:2] - rc6_y))))
#                     + torch.exp(-opt.alpha* (torch.add(torch.square(x[:,0:1] - rc7_x),torch.square(x[:,1:2] - rc7_y))))
#                     + torch.exp(-opt.alpha* (torch.add(torch.square(x[:,0:1] - rc8_x),torch.square(x[:,1:2] - rc8_y))))
#                     + torch.exp(-opt.alpha* (torch.add(torch.square(x[:,0:1] - rc9_x),torch.square(x[:,1:2] - rc9_y)))))
# # 真解 u 的表达式
#def u(x): return (torch.exp(-opt.alpha* ((x[:,0:1] - rc1_x)**2 +(x[:,1:2] - rc1_y)**2)) 
#               + torch.exp(-opt.alpha* ((x[:,0:1] - rc2_x)**2 +(x[:,1:2] - rc2_y)**2))
#               + torch.exp(-opt.alpha* ((x[:,0:1] - rc3_x)**2 +(x[:,1:2] - rc3_y)**2))
#               + torch.exp(-opt.alpha* ((x[:,0:1] - rc4_x)**2 +(x[:,1:2] - rc4_y)**2))
#               + torch.exp(-opt.alpha* ((x[:,0:1] - rc5_x)**2 +(x[:,1:2] - rc5_y)**2))
#               + torch.exp(-opt.alpha* ((x[:,0:1] - rc6_x)**2 +(x[:,1:2] - rc6_y)**2))
#               + torch.exp(-opt.alpha* ((x[:,0:1] - rc7_x)**2 +(x[:,1:2] - rc7_y)**2))
#               + torch.exp(-opt.alpha* ((x[:,0:1] - rc8_x)**2 +(x[:,1:2] - rc8_y)**2))
#              + torch.exp(-opt.alpha* ((x[:,0:1] - rc9_x)**2 +(x[:,1:2] - rc9_y)**2)))



# RANDOM_SEED = 42 # any random number
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed) # CPU
#     torch.cuda.manual_seed(seed) # GPU
#     torch.cuda.manual_seed_all(seed) # All GPU
#     os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
#     torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
#     torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现
# set_seed(RANDOM_SEED)


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
        
        self.fc1 = nn.Linear(input_size, 64, bias=True)
        self.fc2 = nn.Linear(64, 64, bias=True)
        self.fc3 = nn.Linear(64, 64, bias=True)
        self.fc4 = nn.Linear(64, 64, bias=True)
        self.fc5 = nn.Linear(64, 64, bias=True)
        # self.fc6 = nn.Linear(64, 64, bias=True)
        # self.fc7 = nn.Linear(64, 64, bias=True)
        self.fc8 = nn.Linear(64, output_size, bias=True)
        self.apply(_init_params)

    # def forward(self, x):
    #     x = torch.tanh(self.fc_input(x))
    #     x = self.block1(x)
    #     x = self.block2(x)
    #     x = self.block3(x)
    #     x = self.block4(x)
    #     x = self.block5(x)
    #     x = self.block6(x)
    #     x = self.block7(x)
    #     x = self.block8(x)
    #     x = self.block9(x)
    #     x = self.block10(x)
    #     out = self.fc_output(x)

    #     return out

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

# def _gradient(outputs: Tensor, inputs: Tensor) -> Tensor:
#     grad = autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(
#         outputs), create_graph=True, only_inputs=True)
#     return grad[0]


# def grad(func: Callable[[Tensor], Tensor], x: torch.Tensor) -> Tensor:
#     x_clone = x.clone().detach().requires_grad_(True)
#     fx = func(x_clone)
#     return _gradient(fx, x_clone)


# def div(func_vec: Callable[[Tensor], Tensor], x: torch.Tensor) -> Tensor:
#     x_clone = x.clone().detach().requires_grad_(True)
#     fx_vec = func_vec(x_clone)
#     partial_f1x1 = _gradient(fx_vec[:, 0:1], x_clone)[:, 0:1]
#     partial_f2x2 = _gradient(fx_vec[:, 1:2], x_clone)[:, 1:2]
#     return torch.add(partial_f1x1, partial_f2x2)


# def laplace(func: Callable[[Tensor], Tensor], x: torch.Tensor) -> Tensor:
#     x_clone = x.clone().detach().requires_grad_(True)
#     fx = func(x_clone)
#     grad = _gradient(fx, x_clone)
#     partial_x1x1 = _gradient(grad[:, 0:1], x_clone)[:, 0:1]
#     partial_x2x2 = _gradient(grad[:, 1:2], x_clone)[:, 1:2]
#     return partial_x1x1, partial_x2x2

# X = G(Z), Z.shape = (n_samples, input), X.shape = (n_sample, output). 这里output实际上取值为维度d
# TODO:需要将得到的z限制到比求解区域稍大的区域内

def mini_batch(X_in, X_bound, mini_batch_size_in=64, mini_batch_size_bound=16, seed=0):
    np.random.seed(seed)
    m_in = X_in.shape[0]
    m_bound = X_bound.shape[0]
    mini_batches_in = []
    mini_batches_bound = []
    permutation_in = list(np.random.permutation(m_in))
    permutation_bound = list(np.random.permutation(m_bound))
    shuffle_X_in = X_in[permutation_in]
    shuffle_X_bound = X_bound[permutation_bound]
    

    num_complete_minibatches_in = int(m_in//mini_batch_size_in)
    num_complete_minibatches_bound = int(m_bound//mini_batch_size_bound)
    num_complete_minibatches = min(num_complete_minibatches_in, num_complete_minibatches_bound)
    for i in range(num_complete_minibatches):
        mini_batch_X_in = shuffle_X_in[i*mini_batch_size_in: (i+1)*mini_batch_size_in]
        mini_batch_X_bound = shuffle_X_bound[i*mini_batch_size_bound: (i+1)*mini_batch_size_bound]
        mini_batches_in.append(mini_batch_X_in)
        mini_batches_bound.append(mini_batch_X_bound)
    ## 舍弃掉后续无法成比例的数据
    return mini_batches_in, mini_batches_bound, num_complete_minibatches


def BoundPoint(xbound):
    xbound_x0 = torch.cat((opt.domain[0][0] *torch.ones_like(xbound), xbound), dim=1)
    xbound_x1 = torch.cat((opt.domain[0][1] * torch.ones_like(xbound), xbound), dim=1)
    xbound_y0 = torch.cat((xbound, opt.domain[1][0] * torch.ones_like(xbound)), dim=1)
    xbound_y1 = torch.cat((xbound, opt.domain[1][1] * torch.ones_like(xbound)), dim=1)
    xbound_all = torch.vstack([xbound_x0, xbound_x1, xbound_y0, xbound_y1])
    return xbound_all


class PINNsLoss(nn.Module):
    def __init__(self, beta, f_source, g_bound):
        super(PINNsLoss, self).__init__()
        self.beta = beta
        self.f_source = f_source
        self.g_bound = g_bound

    def forward(self, model, xi_in, xi_bound):
        # 计算内部点的loss项
        u_forward = pinns(xi_in)
        Ggrad_x1, Ggrad_x2 = pinns.grad(xi_in)
        Ggradgrad_x1, Ggradgrad_x2 = pinns.laplace(xi_in)
        xbound_value = pinns(xi_bound).to(device)
        
        loss_f = self.f_source(xi_in).reshape((-1, 1))
        loss = torch.square(Ggradgrad_x1 + Ggradgrad_x2 - 2*Ggrad_x1*xi_in[:,0:1] - 2*Ggrad_x2*xi_in[:,1:2] - 4 * u_forward - loss_f)
        loss = torch.mean(loss)
        loss_g = self.g_bound(xi_bound).reshape((-1, 1))
        loss_bound = torch.square(xbound_value - loss_g)
        loss_bound = torch.mean(loss_bound)

        loss = torch.add(loss, loss_bound, alpha=self.beta)
        if torch.isnan(loss):
            # print("rho =", rho)
            print("loss.item()",loss.item())
            print("loss_bound.item()",loss_bound.item())
            raise TypeError("nan")
        # loss = torch.mean(loss)
        return loss

def Gaussian_Distribution_2d(M=100, mu=np.zeros(2), sigma=np.ones(2)):
    '''
    Parameters
    ----------
    2 维度, 通过改变该值,可以得到高纬
    M 样本数
    mu 样本均值
    sigma: 样本方差
    
    Returns
    -------
    data  shape(M, 2), M 个 2 维服从高斯分布的样本
    Gaussian  高斯分布概率密度函数
    '''
    mean = np.zeros(2) + mu  # 均值矩阵，每个维度的均值都为 mu
    cov = np.eye(2) * sigma  # 协方差矩阵，每个维度的方差都为 sigma

    # 产生 N 维高斯分布数据
    data = np.random.multivariate_normal(mean, cov, M)
    # N 维数据高斯分布概率密度函数
    # Gaussian = multivariate_normal(mean=mean, cov=cov)
    
    return data
    # return data, Gaussian


def plot_and_save(epoch, epoch_P):
    global l2_err
    with torch.no_grad():
        u_test = pinns(x)
    
    residual_plot = np.zeros([x.shape[0],1])
    for i in range(opt.n_test):
        tx_i = x[i*opt.n_test: (i+1)*opt.n_test]
        u_forward = pinns(tx_i)
        u_x, u_y = pinns.grad(tx_i)
        u_xx, u_yy = pinns.laplace(tx_i)

        residual_plot[i*opt.n_test: (i+1)*opt.n_test, 0:1] = torch.abs(u_xx + u_yy - 2 * u_x * tx_i[:,0:1] - 2*u_y*tx_i[:,1:2] - 4 * u_forward - f_source(tx_i[:,0:2])).reshape((-1,1)).detach().cpu().numpy()
    residual_plot_all = residual_plot.reshape((opt.n_test, opt.n_test))  
        
    u_test_re = u_test.reshape((opt.n_test, opt.n_test))
    u_test_re = u_test_re.detach().cpu().numpy()

    err_absolute = np.abs(u_exact_re - u_test_re)
    err_l2 = np.mean(err_absolute**2)
    err_l2_format = format(err_l2, '.3E')
    #err_l2 = np.sqrt(err_l2/np.mean(u_exact_re**2))
    l2_err = np.append(l2_err, err_l2_format)
    print(f"Epoch = {epoch}, epoch_P = {epoch_P}, Relative err_l2 = {err_l2_format}")
    

    fig, axs = plt.subplots(2, 3, figsize=(10,6), sharex=True, sharey=True, dpi=300)
    plt.suptitle(f'$N_a$ = {epoch}, $N_p$ = {epoch_P}, MSE = {err_l2_format}')
    
    plot1 = axs[0, 0].imshow(u_exact_re, cmap='jet', aspect = 'auto',extent =[opt.domain[0][0], opt.domain[0][1], opt.domain[1][0], opt.domain[1][1]], origin ='lower',vmin=0, vmax=1)
    plt.colorbar(plot1, ax = axs[0, 0])
    axs[0, 0].set_title("$u_{exact}$")
    
    plot2 = axs[0, 1].imshow(u_test_re, cmap='jet', aspect = 'auto',extent =[opt.domain[0][0], opt.domain[0][1], opt.domain[1][0], opt.domain[1][1]], origin ='lower',vmin=0, vmax=1)
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

    plt.tight_layout()
    plt.savefig(f'./fig/pic-{epoch}-{epoch_P}.png')

    # plt.close()
    torch.save(pinns, f'saved_model/my_pinns_model_{epoch}_{epoch_P}')
    torch.save(xi_in_all, 'data/xi_in.pt')
    torch.save(xi_bound_all, 'data/xi_bound_all.pt')
    torch.save(l2_err, 'l2_err.pt')
    if epoch != 0:
        torch.save(xi_add_all, f'data/xi_add_all_{epoch}.pt')
        torch.save(xi_bound_add_all, f'data/xi_bound_add_all_{epoch}.pt')


# begin Fit

pinns = PINNs(opt.input_dim, 1)
pinns.to(device)

optimizer_P = torch.optim.Adam(
    pinns.parameters(), lr=opt.lr1, betas=(opt.b1, opt.b2), weight_decay=1e-4)

# scheduler_P = torch.optim.lr_scheduler.MultiStepLR(optimizer_P, milestones = [500, 1000, 1500, 2000, 2500, 3000], gamma = 0.7)
scheduler_P = torch.optim.lr_scheduler.MultiStepLR(optimizer_P, milestones = [500, 3000, 75000], gamma = 0.7)
# scheduler_P = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_P, mode='min', factor=0.6, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)

lossfunction = PINNsLoss(opt.beta, f_source, g_bound)
lossfunction.to(device)

error = np.zeros([opt.n_epochs_adaptive, 1])

# 生成用来进行test的数据相应的点坐标, 这里按照2维来写的

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


# all data

xi_in_all = (opt.domain[0][1]-opt.domain[0][0]) * torch.rand(opt.n_sample_in, 2) + opt.domain[0][0]
xi_in_all = torch.cat((xi_in_all, torch.zeros((xi_in_all.shape[0], 1), dtype = xi_in_all.dtype)), 1)
xi_in_all = xi_in_all.to(device)
xbound_all = (opt.domain[0][1]-opt.domain[0][0]) * torch.rand(opt.n_sample_bound, 1) + opt.domain[0][0]
xi_bound_all = BoundPoint(xbound_all)
xi_bound_all = torch.cat((xi_bound_all, torch.zeros((xi_bound_all.shape[0], 1), dtype = xi_bound_all.dtype)), 1)
xi_bound_all = xi_bound_all.to(device)

lossitem_P_list = np.array([])
l2_err = np.array([])
append_i = 0
epoch_P_init = opt.epoch_P
for epoch in range(opt.n_epochs_adaptive):

    ### 开始求解 PINNs模型
    epoch_P_init = epoch_P_init + 0* epoch
    loss_epoch_P = np.zeros([epoch_P_init, 1])
    for epoch_P in range(epoch_P_init):
        mini_batches_xi, mini_batches_xbound, complete_batches = mini_batch( 
                xi_in_all, xi_bound_all, mini_batch_size_in = opt.batch_size_in, mini_batch_size_bound = opt.batch_size_bound, seed = epoch_P)
        loss_epoch = 0
        for i in range(min(complete_batches, opt.complete_batches)):
            xi_in= mini_batches_xi[i]
            xi_bound = mini_batches_xbound[i]
            
            loss = lossfunction(pinns, xi_in[:,0:2], xi_bound[:,0:2])
            optimizer_P.zero_grad()
            loss.backward()
            optimizer_P.step()
            
            loss_epoch += loss.item()
        if epoch_P%500 == 0 or epoch_P == opt.epoch_P - 1:
            plot_and_save(epoch = epoch, epoch_P = epoch_P)
            append_i = append_i + 1
        loss_epoch_P[epoch_P] = loss_epoch/(min(complete_batches, opt.complete_batches))
        scheduler_P.step()
        lossitem_P_list = np.append(lossitem_P_list, np.log(loss_epoch_P[epoch_P]))
    print('='*10)        

    loss_epoch_P_array = np.array(loss_epoch_P)
    loss_epoch_P_array = np.log(loss_epoch_P_array)
    
    torch.save(lossitem_P_list, 'lossitem_P_list.pt')
    
    fig2 = plt.figure()
    plt.plot(loss_epoch_P_array, color = 'red', label = 'loss_P')
    plt.title(f'loss_P with epoch = {epoch}')
    plt.legend()
    plt.savefig(f'./loss/loss_P-{epoch}')
    plt.close()
    
    fig3 = plt.figure()
    plt.plot(lossitem_P_list, color = 'red', label = 'loss_P_all')
    plt.title('loss_P with all adaptive')
    plt.legend()
    plt.savefig('./loss/loss_P_all_step')
    plt.close()

    ###验证误差
    for i_index in range(min(complete_batches, opt.complete_batches)):
        xi_in= mini_batches_xi[i_index]
        residual_forward = pinns(xi_in[:,0:opt.input_dim])
        residual_Ggrad_x1, residual_Ggrad_x2 = pinns.grad(xi_in[:,0:opt.input_dim])
        residual_gradgradx, residual_gradgrady = pinns.laplace(xi_in[:,0:opt.input_dim])
        
        residual_in_validate = torch.square(residual_gradgradx + residual_gradgrady - 2*residual_Ggrad_x1*xi_in[:,0:1] - 2*residual_Ggrad_x2*xi_in[:,1:2] - 4 * residual_forward - f_source(xi_in[:,0:opt.input_dim]).reshape((-1,1)))

        # residual_index = residual_in_validate.argsort(0)[-*opt.num_max_index:]
        if i_index == 0:
            residual_all = residual_in_validate
            samples_all = xi_in
        else:
            residual_all = torch.cat((residual_all, residual_in_validate))
            samples_all = torch.cat((samples_all, xi_in))
    ## 按照残差值升序排列，并返回其最大的100个样本点对应的Index值

    # residual_gradgradx, residual_gradgrady = pinns.laplace(samples_by_residual)
    # residual_in_validate_all = torch.square(residual_gradgradx + residual_gradgrady + f_source(samples_by_residual).reshape((-1,1)))
    
    # divide = np.arange(0, residual_in_validate_all.shape[0], int(residual_in_validate_all.shape[0]/opt.num_max_index))
    # residual_index_all = residual_in_validate_all.argsort(0)[divide]
    residual_index_all = residual_all.argsort(dim=0, descending=True)
    
    samples_by_residual_all = samples_all[residual_index_all[:,0],0:opt.input_dim]
    

    samples_by_residual_grad_x, samples_by_residual_grad_y = pinns.grad(samples_by_residual_all)

    sigma = np.hstack((samples_by_residual_grad_x.detach().cpu().numpy(), samples_by_residual_grad_y.detach().cpu().numpy()))
    sigma = np.abs(sigma)
    ## 可以将sigma限制到[0,1]或计算点到边界最小的距离作为方差的最大值
    sigma = opt.sigma_penalty/np.sqrt(sigma+EPS)
    index_array = np.array([0])
    for index_i in range(samples_by_residual_all.shape[0]):
        index_j = 0
        for index_j in range(index_array.shape[0]):
            if torch.norm(samples_by_residual_all[index_i]- samples_by_residual_all[index_array[index_j]]).item() < np.linalg.norm(sigma[index_i]):
                break
            if index_j == index_array.shape[0] - 1:
                index_array = np.append(index_array, index_i)
        if index_array.shape[0] == opt.num_max_index:
            break


    samples_by_residual_all = samples_by_residual_all[index_array]
    sigma = sigma[index_array]
    mu = samples_by_residual_all.detach().cpu().numpy()
    
    ### 依据residual_grad构造二维高斯分布
    xi_add = np.array([])
    for i_samples in range(opt.num_max_index):
        while xi_add.shape[0] < opt.Gaussian_number*(i_samples+1):
            ## 增加判断生成次数和需要点数之间的差距
            data = Gaussian_Distribution_2d(M=1, mu=mu[i_samples,0:2],sigma=sigma[i_samples,0:2])
            if opt.domain[0][0] < data[0,0] < opt.domain[0][1] and opt.domain[1][0] < data[0,1] < opt.domain[1][1] :
                xi_add = np.append(xi_add, data).reshape([-1,2])
    xi_in_all = xi_in_all.detach().cpu()
    xi_add = torch.tensor(xi_add, dtype = xi_in_all.dtype)
    xi_add = torch.cat((xi_add, (epoch + 1) * torch.ones((xi_add.shape[0], 1))), 1)
    
    
    #rand_size = int(opt.Gaussian_number*opt.num_max_index/opt.add_addpative_uniform)
    #xi_add_uniform = (opt.domain[0][1]-opt.domain[0][0]) * torch.rand(rand_size, 2) + opt.domain[0][0]
    #xi_add_uniform = torch.cat((xi_add_uniform, (epoch + 1) * torch.ones((xi_add_uniform.shape[0], 1), dtype = xi_add_uniform.dtype)), 1)
    xi_add_all = xi_add
    #xi_add_all = torch.cat((xi_add, xi_add_uniform), 0)
    xi_in_all = torch.cat((xi_in_all, xi_add_all),0).to(device)

    xbound_add_all = (opt.domain[0][1]-opt.domain[0][0]) * torch.rand(opt.bound_add_number, 1) + opt.domain[0][0]
    xi_bound_add_all = BoundPoint(xbound_add_all)
    xi_bound_add_all = torch.cat((xi_bound_add_all, (epoch + 1) * torch.ones((xi_bound_add_all.shape[0], 1), dtype = xi_bound_add_all.dtype)), 1)
    
    xi_bound_all = xi_bound_all.detach().cpu()
    xi_bound_all = torch.cat((xi_bound_all, xi_bound_add_all), 0).to(device)
    5
    fig3 = plt.figure()
    plt.title(f'xi_add in epoch = {epoch}')
    plt.axis([opt.domain[0][0]-eps, opt.domain[0][1]+eps, opt.domain[1][0]-eps, opt.domain[1][1]+eps])
    plt.scatter(torch.cat((torch.tensor(xi_add_all[:,0:1]),xi_bound_add_all[:,0:1])), torch.cat((torch.tensor(xi_add_all[:,1:2]),xi_bound_add_all[:,1:2])), s=0.5)
    plt.savefig(f'./point/adaptive-{epoch}')
    plt.close()

    fig4 = plt.figure()
    plt.title(f'all points after adaptive epoch = {epoch}')
    plt.axis([opt.domain[0][0]-eps, opt.domain[0][1]+eps, opt.domain[1][0]-eps, opt.domain[1][1]+eps])
    plt.scatter(torch.cat((xi_in_all[:,0:1].cpu(),xi_bound_all[:,0:1].cpu())), torch.cat((xi_in_all[:,1:2].cpu(),xi_bound_all[:,1:2].cpu())), s=0.5)
    plt.savefig(f'./point/all-{epoch}')
    plt.close()

# lossitem_P_array = np.array(lossitem_P_list)
# fig4 = plt.figure()
# plt.plot(lossitem_P_array, color = 'red', label = 'loss_P')
# plt.title('loss_P with all adaptive')
# plt.legend()
# plt.savefig('./loss/loss_P_all_step')
# plt.close()