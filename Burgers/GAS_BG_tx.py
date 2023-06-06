# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 20:45:06 2020

@author: Fate-LD

E-mail: lidi.math@whu.edu.cn
"""


#from mpl_toolkits.mplot3d import axes3d
#import matplotlib.pyplot as plt

# problem: u_t + u * u_x - (0.01/np.pi) * u_xx = f, u(0,x) = -torch.sin(np.pi*x), u(t,-1)=u(t,1)= 0.
# solution: u_exact = -torch.exp(-t) * torch.sin(np.pi*x), u_t = torch.exp(-t)*torch.sin(np.pi*x), u_x = -np.pi*torch.exp(-t) * torch.cos(np.pi*x), u_xx = np.pi**2 * torch.exp(-t)*torch.sin(np.pi*x)


from cmath import nan
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.io as scio
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


## Burgers 方程
# source 项
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

def mini_batch(X_in, X_bound_right, X_bound_left, T_bound, mini_batch_size_in=64, mini_batch_size_bound=16, seed=0):
    np.random.seed(seed)
    m_in = X_in.shape[0]
    m_bound_x_right = X_bound_right.shape[0]
    m_bound_t = T_bound.shape[0]
    
    mini_batches_in = []
    mini_batches_bound_x_right = []
    mini_batches_bound_x_left = []
    mini_batches_bound_t = []
    permutation_in = list(np.random.permutation(m_in))
    # right和left需要用同一个打乱顺序才行
    permutation_bound_x_right = list(np.random.permutation(m_bound_x_right))
    permutation_bound_t = list(np.random.permutation(m_bound_t))
    shuffle_X_in = X_in[permutation_in]
    shuffle_X_bound_right = X_bound_right[permutation_bound_x_right]
    shuffle_X_bound_left = X_bound_left[permutation_bound_x_right]
    shuffle_T_bound = T_bound[permutation_bound_t]
    
    

    num_complete_minibatches_in = int(m_in//mini_batch_size_in)
    num_complete_minibatches_bound_x_right = int(m_bound_x_right//mini_batch_size_bound)
    num_complete_minibatches_bound_t = int(m_bound_t//mini_batch_size_bound)
    num_complete_minibatches = min(num_complete_minibatches_in, num_complete_minibatches_bound_x_right, num_complete_minibatches_bound_t)
    for i in range(num_complete_minibatches):
        mini_batch_X_in = shuffle_X_in[i*mini_batch_size_in: (i+1)*mini_batch_size_in]
        mini_batch_X_bound_right = shuffle_X_bound_right[i*mini_batch_size_bound: (i+1)*mini_batch_size_bound]
        mini_batch_X_bound_left = shuffle_X_bound_left[i*mini_batch_size_bound: (i+1)*mini_batch_size_bound]
        mini_batch_T_bound = shuffle_T_bound[i*mini_batch_size_bound: (i+1)*mini_batch_size_bound]
        mini_batches_in.append(mini_batch_X_in)
        mini_batches_bound_x_right.append(mini_batch_X_bound_right)
        mini_batches_bound_x_left.append(mini_batch_X_bound_left)
        mini_batches_bound_t.append(mini_batch_T_bound)

    return mini_batches_in, mini_batches_bound_x_right, mini_batches_bound_x_left, mini_batches_bound_t, num_complete_minibatches


class PINNsLoss(nn.Module):
    def __init__(self, beta, f_source, x_bound, t_bound):
        super(PINNsLoss, self).__init__()
        self.beta = beta
        self.f_source = f_source
        self.x_bound = x_bound
        self.t_bound = t_bound

    def forward(self, model, xi_in, xi_bound_right, xi_bound_left, ti_bound):
        # 计算内部点的loss项
        _, Gradgrad_x = pinns.laplace(xi_in)
        Grad_t, Grad_x = pinns.grad(xi_in)
        u_value = pinns(xi_in)
        
        x_bound_right_value = pinns(xi_bound_right)
        _, x_bound_right_grad = pinns.grad(xi_bound_right)
        x_bound_left_value = pinns(xi_bound_left)
        _, x_bound_left_grad = pinns.grad(xi_bound_left)
        t_bound_value = pinns(ti_bound)
        
        loss_f = self.f_source(xi_in).reshape((-1, 1))
        loss = torch.square(Grad_t + u_value * Grad_x - (0.01/np.pi) * Gradgrad_x - loss_f)
        loss = torch.mean(loss)
        
        loss_x_bound_right = self.x_bound(xi_bound_right).reshape((-1, 1))
        loss_x_bound_left = self.x_bound(xi_bound_left).reshape((-1, 1))
        loss_x_bound = torch.square(x_bound_right_value - loss_x_bound_right) + torch.square(x_bound_left_value - loss_x_bound_left)
        loss_x_bound = torch.mean(loss_x_bound)

        loss_t_bound = self.t_bound(ti_bound).reshape((-1, 1))
        loss_t_bound = torch.square(t_bound_value - loss_t_bound)
        loss_t_bound = torch.mean(loss_t_bound)

        loss = torch.add((loss + loss_x_bound), loss_t_bound, alpha=self.beta)

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
        u_test = pinns(tx)
        
    residual_plot = np.zeros([tx.shape[0],1])
    for i in range(opt.n_test_x):
        tx_i = tx[i*opt.n_test_t: (i+1)*opt.n_test_t]
        residual_forward = pinns(tx_i)
        residual_Ggrad_t, residual_Ggrad_x = pinns.grad(tx_i)
        _, residual_gradgrad_x = pinns.laplace(tx_i)

        residual_plot[i*opt.n_test_t: (i+1)*opt.n_test_t, 0:1] = torch.abs(residual_Ggrad_t + residual_forward * residual_Ggrad_x - (0.01/np.pi) * residual_gradgrad_x).detach().cpu().numpy()
    residual_plot_all = residual_plot.reshape((opt.n_test_x, opt.n_test_t))  
    
    u_test_re = u_test.reshape((opt.n_test_x, opt.n_test_t))
    u_test_re = u_test_re.detach().cpu().numpy()

    err_absolute = np.abs(u_exact_re - u_test_re)
    err_l2 = np.mean(err_absolute**2)
    err_l2_format = format(err_l2, '.3E')
    l2_err = np.append(l2_err, err_l2_format)
    
    
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

    # plot5 = axs[1, 1].scatter(torch.cat((x_in_all[:,0:1].cpu(),x_bound_all_right[:,0:1].cpu(), x_bound_all_left[:,0:1].cpu(), t_bound_all[:,0:1].cpu())), torch.cat((x_in_all[:,1:2].cpu(),x_bound_all_right[:,1:2].cpu(), x_bound_all_left[:,1:2].cpu(), t_bound_all[:,1:2].cpu())), s=0.1)
    # #plt.colorbar(plot5, ax = axs[1, 1])
    # axs[1, 1].set_title("points before adding")
    
    if epoch != 0:
        plot6 = axs[1, 2].scatter(torch.cat((x_in_all[:,0:1].cpu(),x_bound_all_right[:,0:1].cpu(), x_bound_all_left[:,0:1].cpu(), t_bound_all[:,0:1].cpu())), torch.cat((x_in_all[:,1:2].cpu(),x_bound_all_right[:,1:2].cpu(), x_bound_all_left[:,1:2].cpu(), t_bound_all[:,1:2].cpu())), s=0.1)
        plot6 = axs[1, 2].scatter(torch.cat((x_add_all[:,0:1].cpu(),x_bound_add_all_right[:,0:1].cpu(), x_bound_add_all_left[:,0:1].cpu(), t_bound_add_all[:,0:1].cpu())), torch.cat((x_add_all[:,1:2].cpu(),x_bound_add_all_right[:,1:2].cpu(), x_bound_add_all_left[:,1:2].cpu(), t_bound_add_all[:,1:2].cpu())), s=0.1, c='r')
        axs[1, 2].set_title("all points")

    plt.tight_layout()
    plt.savefig(f'./fig/pic-{epoch}-{epoch_P}.png')
    plt.close()
    torch.save(pinns, f'saved_model/my_pinns_model_{epoch}_{epoch_P}')
    torch.save(x_in_all, 'data/x_in.pt')
    torch.save(x_bound_all_right, 'data/x_bound_all_right.pt')
    torch.save(x_bound_all_left, 'data/x_bound_all_left.pt')
    torch.save(t_bound_all, 'data/t_bound_all.pt')
    
    torch.save(l2_err, 'l2_err.pt')
    if epoch != 0:
        torch.save(x_add_all, f'data/xi_add_all_{epoch}.pt')
        torch.save(x_bound_add_all_left, f'data/x_bound_add_all_left_{epoch}.pt')
        torch.save(x_bound_add_all_right, f'data/x_bound_add_all_right_{epoch}.pt')
        torch.save(t_bound_add_all, f'data/t_bound_add_all_{epoch}.pt')
        
# begin Fit

pinns = PINNs(opt.input_dim, 1)
pinns.to(device)

optimizer_P = torch.optim.Adam(
    pinns.parameters(), lr=opt.lr1, betas=(opt.b1, opt.b2), weight_decay=1e-4)

scheduler_P = torch.optim.lr_scheduler.MultiStepLR(optimizer_P, milestones = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000], gamma = 0.7)

lossfunction = PINNsLoss(opt.beta, f_source, x_bound, t_bound)
lossfunction.to(device)

error = np.zeros([opt.n_epochs_adaptive, 1])

# 生成用来进行test的数据相应的点坐标, 这里按照2维来写的

## load test data
data = scio.loadmat('./gen_data/Burgers_solution.mat')
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

T_re = T.reshape((opt.n_test_t, opt.n_test_x))
X_re = X.reshape((opt.n_test_t, opt.n_test_x))


# all data

x_in_all = torch.rand(opt.n_sample_in, 2)
#把x的范围从[0,1]拉伸到[-1,1]区间
x_in_all[:,1:2] = (opt.domain[1][1]-opt.domain[1][0]) * x_in_all[:,1:2] + opt.domain[1][0]
#增加一个维度用来label当前样本是哪一轮添加的
x_in_all = torch.cat((x_in_all, torch.zeros((x_in_all.shape[0], 1), dtype = x_in_all.dtype)), 1)
x_in_all = x_in_all.to(device)

#需要新的样本数分别在空间边界和时间边界添加新的点
## 时间的边界, 即时间 t=0
t_bound_all = (opt.domain[1][1]-opt.domain[1][0])* torch.rand(opt.n_sample_bound_t, 1) + opt.domain[1][0]
t_bound_all = torch.cat((torch.zeros((t_bound_all.shape[0], 1), dtype = t_bound_all.dtype), t_bound_all), 1)
t_bound_all = torch.cat((t_bound_all, torch.zeros((t_bound_all.shape[0], 1), dtype = t_bound_all.dtype)), 1)
t_bound_all = t_bound_all.to(device)

## 空间的边界, 即 x = 1 or x = -1
x_bound_all = torch.rand(opt.n_sample_bound_x, 1)
x_bound_all_right = torch.cat((x_bound_all, torch.ones((opt.n_sample_bound_x, 1), dtype = x_bound_all.dtype)), 1)
x_bound_all_left = torch.cat((x_bound_all, -1*torch.ones((opt.n_sample_bound_x, 1), dtype = x_bound_all.dtype)), 1)


x_bound_all_right = torch.cat((x_bound_all_right, torch.zeros((x_bound_all_right.shape[0], 1), dtype = x_bound_all_right.dtype)), 1)
x_bound_all_left = torch.cat((x_bound_all_left, torch.zeros((x_bound_all_left.shape[0], 1), dtype = x_bound_all_left.dtype)), 1)

x_bound_all_right = x_bound_all_right.to(device)
x_bound_all_left = x_bound_all_left.to(device)

lossitem_P_list = np.array([])
l2_err = np.array([])
append_i = 0
for epoch in range(opt.n_epochs_adaptive):

    ### 开始求解 PINNs模型

    loss_epoch_P = np.zeros([opt.epoch_P, 1])
    for epoch_P in range(opt.epoch_P):
        mini_batches_xi, mini_batches_bound_x_right, mini_batches_bound_x_left, mini_batches_bound_t, complete_batches = mini_batch( 
                x_in_all, x_bound_all_right, x_bound_all_left, t_bound_all, mini_batch_size_in = opt.batch_size_in, mini_batch_size_bound = opt.batch_size_bound, seed = epoch_P)
        loss_epoch = 0
        for i in range(min(complete_batches, opt.complete_batches)):
            xi_in= mini_batches_xi[i]
            xi_bound_right = mini_batches_bound_x_right[i]
            xi_bound_left = mini_batches_bound_x_left[i]
            ti_bound = mini_batches_bound_t[i]
            
            loss = lossfunction(pinns, xi_in[:,0:2], xi_bound_right[:,0:2], xi_bound_left[:,0:2], ti_bound[:,0:2])
            optimizer_P.zero_grad()
            loss.backward()
            optimizer_P.step()
            
            loss_epoch += loss.item()
        if epoch_P%100 == 0 or epoch_P == opt.epoch_P - 1:
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

##画出 误差在x维度上的L2范数随时间t的变化
    with torch.no_grad():
        u_test = pinns(tx)
        
    u_test_re = u_test.reshape((opt.n_test_x, opt.n_test_t))
    u_test_re = u_test_re.detach().cpu().numpy()
    
    err_absolute = np.abs(u_exact_re - u_test_re)
    err_absolute_x_norm2 = np.linalg.norm(err_absolute, axis=0)
    np.save(f'./err/err_absolute_x_norm2_{epoch}', err_absolute_x_norm2)
    t_plot = np.linspace(0, 1, opt.n_test_t)
    
    fig4 = plt.figure()
    plt.plot(t_plot, err_absolute_x_norm2)
    plt.title(f'err_x_norm2 with epoch = {epoch}')
    plt.savefig(f'./err/err_x_norm2_{epoch}')

###验证误差
    for i_index in range(min(complete_batches, opt.complete_batches)):
        xi_in= mini_batches_xi[i_index]
        residual_forward = pinns(xi_in[:,0:opt.input_dim])
        residual_Ggrad_t, residual_Ggrad_x = pinns.grad(xi_in[:,0:opt.input_dim])
        _, residual_gradgrad_x = pinns.laplace(xi_in[:,0:opt.input_dim])
        
        residual_in_validate = torch.square(residual_Ggrad_t + residual_forward * residual_Ggrad_x - (0.01/np.pi) * residual_gradgrad_x - f_source(xi_in[:,0:opt.input_dim])).reshape((-1,1))

        # residual_index = residual_in_validate.argsort(0)[-*opt.num_max_index:]
        if i_index == 0:
            residual_all = residual_in_validate
            samples_all = xi_in
        else:
            residual_all = torch.cat((residual_all, residual_in_validate))
            samples_all = torch.cat((samples_all, xi_in))

    residual_index_all = residual_all.argsort(dim=0, descending=True)
    
    samples_by_residual_all = samples_all[residual_index_all[:,0],0:opt.input_dim]
    

    samples_by_residual_grad_t, samples_by_residual_grad_x = pinns.grad(samples_by_residual_all)

    sigma = np.hstack((samples_by_residual_grad_t.detach().cpu().numpy(), samples_by_residual_grad_x.detach().cpu().numpy()))
    sigma = np.abs(sigma)
    ## 可以将sigma限制到[0,1]或计算点到边界最小的距离作为方差的最大值
    sigma = opt.sigma_penalty/np.sqrt(sigma+EPS)
    index_array = np.array([0])
    for index_i in range(samples_by_residual_all.shape[0]):
        index_j = 0
        for index_j in range(index_array.shape[0]):
            # if torch.norm(samples_by_residual_all[index_i]- samples_by_residual_all[index_array[index_j]]).item() < np.linalg.norm(sigma[index_i]):
            if torch.norm(samples_by_residual_all[index_i]- samples_by_residual_all[index_array[index_j]]).item() < 0.15:
            # if torch.norm(samples_by_residual_all[index_i]- samples_by_residual_all[index_array[index_j]]).item() < 0:
                break
            if index_j == index_array.shape[0] - 1:
                index_array = np.append(index_array, index_i)
        if index_array.shape[0] == opt.num_max_index:
            break

    if(index_array.shape[0] != opt.num_max_index):
        print(f'The number of local max points = {index_array.shape[0]} ,but opt.num_max_index = {opt.num_max_index}. Can be solved by reducing sigma.')
        raise("error")
    
    samples_by_residual_all = samples_by_residual_all[index_array]
    sigma = sigma[index_array]
    mu = samples_by_residual_all.detach().cpu().numpy()
    
    ### 依据residual_grad构造二维高斯分布
    x_add = np.array([])
    for i_samples in range(opt.num_max_index):
        while x_add.shape[0] < opt.Gaussian_number*(i_samples+1):
            ## 增加判断生成次数和需要点数之间的差距
            data = Gaussian_Distribution_2d(M=1, mu=mu[i_samples,0:2],sigma=sigma[i_samples,0:2])
            if opt.domain[0][0] < data[0,0] < opt.domain[0][1] and opt.domain[1][0] < data[0,1] < opt.domain[1][1] :
                x_add = np.append(x_add, data).reshape([-1,2])
    x_in_all = x_in_all.detach().cpu()
    x_add = torch.tensor(x_add, dtype = x_in_all.dtype)
    x_add = torch.cat((x_add, (epoch + 1) * torch.ones((x_add.shape[0], 1))), 1)
    x_add_all = x_add
    x_in_all = torch.cat((x_in_all, x_add_all),0).to(device)
    ##时间边界
    t_bound_add_all = (opt.domain[1][1]-opt.domain[1][0])* torch.rand(opt.bound_add_number, 1) + opt.domain[1][0]
    t_bound_add_all = torch.cat((torch.zeros((t_bound_add_all.shape[0], 1), dtype = t_bound_add_all.dtype), t_bound_add_all), 1)
    t_bound_add_all = torch.cat((t_bound_add_all, (epoch + 1) * torch.ones((t_bound_add_all.shape[0], 1), dtype = t_bound_add_all.dtype)), 1)
    t_bound_add_all = t_bound_add_all.to(device)
    t_bound_all = torch.cat((t_bound_all, t_bound_add_all), 0).to(device)


    ## 空间的边界, 即 x = 1 or x = -1
    x_bound_add_all = torch.rand(opt.bound_add_number, 1)
    x_bound_add_all_right = torch.cat((x_bound_add_all, torch.ones((x_bound_add_all.shape[0], 1), dtype = x_bound_add_all.dtype)), 1)
    x_bound_add_all_left = torch.cat((x_bound_add_all, -1*torch.ones((x_bound_add_all.shape[0], 1), dtype = x_bound_add_all.dtype)), 1)
    
    x_bound_add_all_right = torch.cat((x_bound_add_all_right, (epoch + 1) * torch.ones((x_bound_add_all_right.shape[0], 1), dtype = x_bound_add_all_right.dtype)), 1)
    x_bound_add_all_left = torch.cat((x_bound_add_all_left, (epoch + 1) * torch.ones((x_bound_add_all_left.shape[0], 1), dtype = x_bound_add_all_left.dtype)), 1)
    
    x_bound_add_all_right = x_bound_add_all_right.to(device)
    x_bound_add_all_left = x_bound_add_all_left.to(device)
    
    x_bound_all_right = torch.cat((x_bound_all_right, x_bound_add_all_right), 0).to(device)
    x_bound_all_left = torch.cat((x_bound_all_left, x_bound_add_all_left), 0).to(device)
    

# # # 画出误差图
# plot_t_x = torch.zeros([opt.n_test_t, opt.n_test_x, 2]).to(device)
# for i in range(opt.n_test_t):
#     for j in range(opt.n_test_x):
#         plot_t_x[i, j, :] = tx[i + j * opt.n_test_t, :]
       
# plot_norm2 = np.zeros([opt.n_test_t, 2])
# for k in range(opt.n_test_t):
#     plot_t = plot_t_x[k,:,:]
#     with torch.no_grad():
#         u_test_plot = pinns(plot_t)
#     u_exact_plot = u_exact(plot_t)
#     err_plot = torch.tensor((plot_t[0,0],torch.norm((u_exact_plot - u_test_plot)))).detach().numpy()
#     plot_norm2[k,:] = err_plot[:]

# plt.plot(plot_norm2[:,0],plot_norm2[:,1])