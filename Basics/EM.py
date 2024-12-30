'''
Expectation Maximization Algorithm for K cluster Gaussian Mixture Model
2024-12-30
'''

import numpy as np
from scipy.stats import norm

class EM():
    def __init__(self,data,k=2,dim=2):
        self.K=k
        self.mu = np.random.rand(self.K,dim)
        self.sigma = np.random.rand(self.K,dim)
        self.pi=np.ones(self.K)/self.K
        self.data=data
        self.epsilon = 0.0001
        self.steps=500
        self.posterior=np.zeros((len(self.data),self.K))
    

    def e_step(self):
        for i in range((self.K)):
            # post_p=self.pi[i]*np.exp(-0.5*(self.data-self.mu[i])**2/self.sigma[i])/np.sqrt(2*np.pi*self.sigma[i])
            self.posterior[:,i]=self.pi[i]*norm.pdf(self.data,self.mu[i],self.sigma[i])
            # print(post_p.shape)
            # self.posterior[:,i]=post_p.flatten()
        self.posterior/=self.posterior.sum(axis=1,keepdims=True)

        return

    def m_step(self):

        # for i in range((self.K)):
        #     self.mu[i]=np.sum(self.posterior[:,i]*self.data)/np.sum(self.posterior[:,i])
        #     self.sigma[i]=np.sum(self.posterior[:,i]*(self.data-self.mu[i])**2)/np.sum(self.posterior[:,i])
        #     self.pi[i]=self.posterior[:,i].mean()
        num_clusters = self.posterior.shape[1]
        Nk = self.posterior.sum(axis=0)
        self.mu= np.dot(self.posterior.T, self.data) / Nk
        variances = []
        for k in range(num_clusters):
            diff = self.data - self.mu[k]
            variances.append(np.sqrt(np.dot(self.posterior[:, k].T, diff**2) / Nk[k]))
        self.pi = np.mean(self.posterior, axis=0)
        self.sigma = np.array(variances)
        
        return self.mu,self.sigma,self.pi

    def run(self):
        step=0
        nll=0
        while step<self.steps:
            old_mu = self.mu.copy()
            self.e_step()
            m,s,p=self.m_step()
            # log_prob=-0.5 * np.log(2. * np.pi)-0.5 * np.log(self.sigma)-0.5 * np.exp(-np.log(self.sigma)) * (self.data-mu)**2.
            ll=0
            for i in range(self.K):
                ll+=np.sum(np.log(p[i]*norm.pdf(self.data,m[i],s[i])))
            if np.abs(ll-nll)<self.epsilon:
                break
            nll=ll
        return self.mu,self.sigma,self.pi
        
