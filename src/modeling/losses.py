"""
Transfer Losses module.

This module is intended to provide a set of transfer losses for domain
adaptation application.

The code can be found here:
https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA

This file gathered some of the available losses in order to have the
functionalities in a single module.

----

MIT License

Copyright (c) 2018 Jindong Wang

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import torch 
from torch import nn 
from torch.autograd import Function 
import torch.nn.functional as F
import numpy as np

# --- CORAL ----------------------------------------------------------
def CORAL(source, target, **kwargs):
    """CORAL.

    This function computes and returns the CORAL loss for domain
    adaptation. The function is based on the paper "Deep CORAL:
    Correlation Alignment for Deep Domain Adaptation" by Sun and
    Saenko, ECCV 2016. The function takes the source and target
    tensors as inputs and calculates their covariance matrices. The
    function then computes the Frobenius norm of the difference
    between the covariance matrices and returns it as the loss.

    :param source: Source data tensor,
                   with shape (batch_size, input_dim)
    :param target: Target data tensor,
                   with shape (batch_size, input_dim)
    :param kwargs: Additional arguments. Not used
    """
    d = source.data.shape[1]
    ns, nt = source.data.shape[0], target.data.shape[0]
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm / (ns - 1)

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt / (nt - 1)

    # frobenius norm between source and target
    loss = torch.mul((xc - xct), (xc - xct))
    loss = torch.sum(loss) / (4*d*d)
    return loss

# --- Batch-Nuclear-Norm Maximization --------------------------------
def BNM(source, target):
    """BNM.
    Batch nuclear-norm maximization, CVPR 2020.

    this function computes and retruns the batch nuclear-norm 
    maximization (BNM) loss for domain adapation. The function is 
    based on the paper "Batch Nuclear-Norm Maximization for Domain 
    Generalization and Adaptation" by Li et al., CVPR 2020. The 
    function does not require the soruce domain data, only the target 
    domain data. 

    :param source: source tensor. Not used 
    :param target: a tensor, softmax target output
    """
    _, out, _ = torch.svd(target)
    loss = -torch.mean(out)
    return loss

# --- MMD ------------------------------------------------------------
class MMDLoss(nn.Module):
    """MMDLoss.

    This class inherits from the PyTorch Module class and represents a
    loss function for domain adaptation based on the maximum mean
    discrepancy (MMD) criterion.
    """

    def __init__(
        self,
        kernel_type = 'rbf',
        kernel_mul  = 2.0,
        kernel_num  = 5,
        fix_sigma   = None,
        **kwargs
    ):
        """__init__.

        :param kernel_type: type of kernel to use.
                            Possible values are:
                            * 'linear'
                            * 'rbf'
        :param kernel_mul: multiplier for the kernel bandwidth
        :param kernel_num: number of kernels to use
        :param fix_sigma:  fixed bandwidth for the kernel. If None,
                           the bandwidth will be computed from the
                           data
        :param kwargs:     additional arguments
        """
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def gaussian_kernel(
        self,
        source,
        target,
        kernel_mul,
        kernel_num,
        fix_sigma
    ):
        """gaussian_kernel.

        This method computes and returns the sum of Gaussian kernels
        for the MMD computation.

        1.  it takes the source and target tensors as inputs and
            concatenates them into a total tensor
        2.  it computes the pairwise L2 distance between the rows of
            the total tensor
        3.  it computes the bandwidth for the Gaussian kernels, either
            using the fix_sigma parameter or the data statistics
        4.  it creates a list of bandwidth values by multiplying the
            base bandwidth by the kernel_mul parameter to the power of
            the kernel index
        5.  it computes a list of kernel values by applying the
            Gaussian kernel function to the L2 distance matrix with
            each bandwidth value
        6.  it returns the sum of the kernel values as a tensor

        :param source:     source tensor
        :param target:     target tensor
        :param kernel_mul: multiplier for the kernel bandwidth
        :param kernel_num: number of kernels to use for the MMD
                           computation
        :param fix_sigma:  fixed bandwidth for the kernel. If None,
                           the bandwidth will be computed from the
                           data
        """
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        """linear_mmd2.

        This method computes and returns the linear MMD loss for the
        MMD computation.

        1. it takes the source and target tensors as inputs
        2. it computes the mean difference between them
        3. it computes the dot product of the mean difference with
           itself
        4. it returns it as a scalar tensor

        :param f_of_X: source tensor
        :param f_of_Y: target tensor
        """
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        """forward.

        This method computes and returns the MMD loss for domain
        adaptation.

        1. it takes the source and target tensors as inputs and checks
           the kernel type attribute:
           - if the kernel type is 'linear', it calls the linear_mmd2
             method and returns the result
           - if the kernel type is 'rbf', it calls the gaussian_kernel
             method and computes the MMD loss as the mean difference
             between the kernel values of the source-source,
             target-target, source-target, and target-source pairs
        2. it returns the MMD loss as a scalar tensor

        :param source: source tensor
        :param target: target tensor
        """
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.gaussian_kernel(
                source, target,
                kernel_mul = self.kernel_mul,
                kernel_num = self.kernel_num,
                fix_sigma  = self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss

# --- DARE-GRAM ------------------------------------------------------
def DARE_GRAM_LOSS(H1, H2, threshold=0.97, tradeoff_angle=0.01, tradeoff_scale=0.00001):
    device = H1.device
    b,p = H1.shape
    A = torch.cat((torch.ones(b,1).to(device), H1), 1)
    B = torch.cat((torch.ones(b,1).to(device), H2), 1)
    cov_A = (A.t()@A)
    cov_B = (B.t()@B)
    _,L_A,_ = torch.linalg.svd(cov_A)
    _,L_B,_ = torch.linalg.svd(cov_B)
    eigen_A = torch.cumsum(L_A.detach(), dim=0)/L_A.sum()
    eigen_B = torch.cumsum(L_B.detach(), dim=0)/L_B.sum()
    if(eigen_A[1]>threshold):
        T = eigen_A[1].detach()
    else:
        T = threshold
    index_A = torch.argwhere(eigen_A.detach()<=T)[-1]
    if(eigen_B[1]>threshold):
        T = eigen_B[1].detach()
    else:
        T = threshold
    index_B = torch.argwhere(eigen_B.detach()<=T)[-1]
    k = max(index_A, index_B)[0]
    A = torch.linalg.pinv(cov_A ,rtol = (L_A[k]/L_A[0]).detach())
    B = torch.linalg.pinv(cov_B ,rtol = (L_B[k]/L_B[0]).detach())
    cos_sim = nn.CosineSimilarity(dim=0,eps=1e-6)
    cos = torch.dist(torch.ones((p+1)).to(device),(cos_sim(A,B)),p=1)/(p+1)
    return tradeoff_angle*(cos) + tradeoff_scale*torch.dist((L_A[:k]),(L_B[:k]))/k

# --- Kullback-Leibler -----------------------------------------------
def KullbackLeibler(source, target, reduction='batchmean'):
    loss = torch.nn.KLDivLoss(reduction=reduction)
    return loss(F.log_softmax(source, dim=1),
                F.softmax(target, dim=1))

# --- Main -----------------------------------------------------------
class TransferLoss(nn.Module):
    """TransferLoss."""

    def __init__(self, loss_type, **kwargs):
        """__init__.

        :param loss_type:
        :param kwargs:
        """
        super(TransferLoss, self).__init__()
        self.loss_type = loss_type
        if loss_type == "mmd":
            self.loss_func = MMDLoss(**kwargs)
        elif loss_type == "coral":
            self.loss_func = CORAL
        elif loss_type == "bnm":
            self.loss_func = BNM
        elif loss_type == "kl":
            self.loss_func = KullbackLeibler
        elif loss_type == "dare":
            self.loss_func = DARE_GRAM_LOSS
        else:
            print("WARNING: No valid transfer loss function is used.")
            self.loss_func = lambda x, y: 0 # return 0
        
    def forward(self, source, target, **kwargs):
        """forward.

        :param source:
        :param target:
        :param kwargs
        """
        return self.loss_func(source, target, **kwargs)
