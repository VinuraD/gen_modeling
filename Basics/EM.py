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
        self.posterior=np.zeros((len(self.data),self.K))

    def e_step(self):
        for i in range((self.K)):
            # post_p=self.pi[i]*np.exp(-0.5*(self.data-self.mu[i])**2/self.sigma[i])/np.sqrt(2*np.pi*self.sigma[i])
            post_p=self.pi[i]*norm.pdf(self.data,self.mu[i],self.sigma[i]**2)
            self.posterior[:,i]=np.squeeze(post_p,1)
        self.posterior=self.posterior/np.sum(self.posterior,axis=1)[:,np.newaxis]

        return

    def m_step(self):

        for i in range((self.K)):
            self.mu[i]=np.sum(self.posterior[:,i]*self.data)/np.sum(self.posterior[:,i])
            self.sigma[i]=np.sqrt(np.sum((self.posterior[:,i]*self.data-self.mu[i])**2)/np.sum(self.posterior[:,i]))
            self.pi[i]=self.posterior[:,i].mean()
        return self.mu,self.sigma,self.pi

    def run(self):
        nll=[]
        while True:
            old_mu = self.mu.copy()
            self.e_step()
            m,s,p=self.m_step()
            # log_prob=-0.5 * np.log(2. * np.pi)-0.5 * np.log(self.sigma)-0.5 * np.exp(-np.log(self.sigma)) * (self.data-mu)**2.
            
            if np.sum(abs(old_mu-self.mu))<self.epsilon*self.K:
                break
            return self.mu,self.sigma,self.pi