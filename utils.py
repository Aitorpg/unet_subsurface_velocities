import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import math
import numpy as np
##################################################################################
def prnModelSt(model):
        print('Number of parameters: %f million' %(nparams(model)/1e6))
        #print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        
def prnOptimSt(optim):
        print("Optimizer's state_dict:")
        for var_name in optim.state_dict():
            print(var_name, "\t", optim.state_dict()[var_name])

def nparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
##################################################################################
def setLR(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def wdstep(optimizer, wd):
  for group in optimizer.param_groups:
    for param in group['params']:
        param.data = param.data.add(-wd * group['lr'], param.data)


def save_checkpoint(dir, epoch, **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, 'checkpoint-%d.pt' % epoch)
    torch.save(state, filepath)


def testTrainSplit(dataset, fac = 0.8):
    train_sz = int(fac * len(dataset))
    test_sz = len(dataset) - train_sz
    train, test = torch.utils.data.random_split(dataset, [train_sz, test_sz])
    return train, test

def testTrainLoader(dataset, batch_size = 200, fac = 0.8):
    train_sz = int(fac * len(dataset))
    test_sz = len(dataset) - train_sz
    train, test = torch.utils.data.random_split(dataset, [train_sz, test_sz])
    train_dl = DataLoader(train, batch_size=batch_size, num_workers=12, shuffle = True)
    test_dl = DataLoader(test, batch_size=batch_size, num_workers=12, shuffle = False)
    return train_dl, test_dl

#Implementation of SGDR with cosine annealing for linearly increasing cycle length
##################################################################################
#Ti increases in an arithmetic progression
def TiTcur_ap(epoch, T0):
        alpha = 2*epoch/T0
        index = np.floor((1+(1+4*alpha)**0.5)/2.0)-1
        return T0*(index + 1), T0 * index * (index + 1)/2

#Ti is constant
def TiTcur_c(epoch, T0):
        #ti = np.floor(epoch/np.float(T0))*T0
        ti = np.floor(epoch/T0)*T0 #HW: doesn't seem that float is needed? ***
        return T0, ti

def cosineSGDR(optimizer, epoch, T0=5, eta_min=0.0, eta_max=0.1, scheme = 'constant'):
        if scheme == 'linear':
                Ti, ti = TiTcur_ap(epoch, T0)
        else:
                Ti, ti = TiTcur_c(epoch, T0)
        lr = eta_min + 0.5 * (eta_max - eta_min) *  (1 + np.cos(np.pi * (epoch - ti) / Ti)) 
        #assert(lr > 0)
        setLR(optimizer, lr)
        return lr

##################################################################################
from netCDF4 import Dataset


def ncCreate(fname, Nx, Ny, varlist):
    nco = Dataset(fname, 'w')
    nco.createDimension('time', None)
    nco.createDimension('x', Nx)
    nco.createDimension('y', Ny)
    for var in varlist:
        nco.createVariable(var, np.dtype('float32').char, ('time', 'x', 'y'))
    return nco

def addVal(nco, var, val, itime=None):
    if itime is not None:
        nco.variables[var][itime, :,:] = val.squeeze()
    else:
        nco.variables[var][:, :,:] = val.squeeze()

def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def add_gauss_perturbation(module, alpha=1e-4):
    sample = np.random.normal(0,alpha)
    perturbation  = (sample*torch.ones(module.weight.data.shape)).cuda()
    module.weight.data = module.weight.data + perturbation

def model_sharpness(model):
    for child in model.children():
        module_name = child._get_name()
        if module_name in ['Linear', 'Conv1d', 'Conv2d', 'Conv3d']:
            add_gauss_perturbation(child)
        else:
            model_sharpness(child)
