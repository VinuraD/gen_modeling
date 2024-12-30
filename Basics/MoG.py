'''
A Mixture of Gaussian generator.
Ref:Deep Generative Modeling by Jakub M. Tomczak
2024-12-30
'''
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MoG(nn.Module):
    def __init__(self,D,K,uniform=False):
        self.D=D
        self.K=K
        self.uniform=uniform

        self.mu=nn.Parameter(torch.randn(1,self.K,self.D)*0.25)
        self.log_var=nn.Parameter(-2.*torch.ones(1,self.K,self.D))

        if self.uniform:
            self.w=torch.zeros(1,self.K)
            self.w.requires_grad=False
        else:
            self.w=nn.Parameter(torch.zeros(1,self.K))
        self.PI = torch.from_numpy(np.asarray(np.pi))
    
    def log_diag_normal(self,x,mu,log_var,reduction='sum',dim=1):
        #gives the log probability of x (log(p(x|mu,var)))
        log_p=-0.5*torch.log(2.*self.PI)-0.5*log_var-0.5 * torch.exp(-log_var) * (x.unsqueeze(1)-mu)**2.
        return log_p
    def forward(self,x,reduction='mean'):
        #log pi is the log probability/weight of each component
        log_pi=torch.log(F.softmax(self.w,1))
        log_N = torch.sum(self.log_diag_normal(x, self.mu, self. log_var),2)

        #take the minus of the logsumexp of the observed data which is another representation of its log likelihood
        NLL_loss =-torch.logsumexp(log_pi + log_N, 1)
        if reduction == 'sum': 
            return NLL_loss.sum() 
        elif reduction == 'mean': 
            return NLL_loss.mean() 
        else: 
            raise ValueError('Either "sum" or "mean".')

    def sample(self,bs=64):
        x_sample=torch.empty(bs,self.D)
        pi=F.softmax(self.w,1)
        indices=torch.multinomial(pi,bs,replacement=True).squeeze()

        for n in range(bs):
            indx=indices[n]
            x_sample[n]=self.mu[0,indx]+torch.exp(self.log_var[0,indx])*torch.randn(self.D)
        return x_sample
    def log_prob(self,x,reduction='mean'):

        with torch.no_grad():
            log_pi=torch.log(F.softmax(self.w,1))
            log_N=torch.sum(self.log_diag_normal(x, self.mu, self. log_var),2)
            log_prob = torch.logsumexp(log_pi + log_N, 1)
            if reduction == 'sum': 
                return log_prob.sum() 
            elif reduction == 'mean': 
                return log_prob.mean() 
            else: 
                raise ValueError('Either "sum" or "mean".')