"""
    Module name: KernelSVM
    Description: Module for creating a Kernel based SVM classifier.
"""

import pandas as pd
import numpy as np
from itertools import repeat
import copy
import pickle as pkl
from scipy import stats
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split


class KernelSVM(object):
    """
        The object for Kernel based SVM classifier. Currently, the kernels
        supported are 'polynomial' and 'RBF'.
            :member attributes: kernel: kernel type
                                h: huber-loss parameter
                                lambda: regularisation parameter
                                p: polynomial order
                                sigma: bandwidth of the RBF kernel
                                gram:  gram matrix
    """
    def __init__(self,lamda,kernel='polynomial',p=7,h=0.5,sigma=0.5):
        self.kernel = kernel
        self.h = h
        self.lamda = lamda
        self.p = p
        self.sigma = sigma
        
    def __partial__(self,k,y,z,method):
        """
            Private method to calculate the partial minimisation or the evaluation
            of the huberized hinge loss function. It is decided by the argument 'method'.
        """
        if method == 'grad':
            if z > 1+self.h:
                return np.zeros(k.shape)
            elif z < 1-self.h:
                return -y*k
            else:
                return -y*k*(1+self.h-z)/(2*self.h)
        elif method == 'objective':
            if z > 1+self.h:
                return 0
            elif z < 1-self.h:
                return 1-z
            else:
                return (1+self.h-z)**2/(4*self.h)

    def __computegrad__(self,y,coef):
        """
            Private method to calculate the gradient of huberized hinge loss function.
        """
        K_a = self.gram.dot(coef)
        z = y*K_a
        rss = np.array(list(map(self.__partial__,self.gram,y,z,repeat('grad'))))
        grad = (1/len(y)) * np.sum(rss,axis=0) + 2*self.lamda*K_a
        return grad
    
    def __obj_eval__(self,y,coef):
        """
            Private method to calculate the objective value.
        """
        K_a = self.gram.dot(coef)
        z = y*K_a
        rss = np.array(list(map(self.__partial__,self.gram,y,z,repeat('objective'))))
        res = (1/len(y)) * np.sum(rss) + (self.lamda * coef.reshape(1,-1).dot(self.gram).dot(coef))
        return res

    def __backtracking__(self,y,coef,grad,t,beta=0.5,alpha=0.5,m_iter=50):
        """
            Private method that implements the backtracking algorithm to find learning rate 't'.
        """
        norm_grad = np.linalg.norm(grad)
        found_t = False
        max_iter = 0 
        while (found_t is False and max_iter < m_iter):
            if self.__obj_eval__(y,coef-t*grad) < (self.__obj_eval__(y,coef)-alpha*t*norm_grad**2):
                found_t = True
            else:
                t *= beta
                max_iter +=1
        return t
    
    def __computegram__(self, X, Z=None):
        """
            Private method that calculates the gram matrix from the given inputs. The kernel specified
            while creating the classfier object will be used.
        """
        if Z is None:
            Z = X
        if self.kernel == 'rbf':
            self.gram = np.exp(-1/(2*self.sigma**2)*((np.linalg.norm(X, axis=1)**2)[:, np.newaxis]
                                   + (np.linalg.norm(Z, axis=1)**2)[np.newaxis, :] - 2*np.dot(X, Z.T)))
        elif self.kernel == 'polynomial':
            self.gram = (X.dot(Z.T)+1)**self.p
        return self.gram

    def __kerneleval__(self, X, X_star=None):
        """
            Private method that evaluates the kernel function for the inputs.
        """
        kernelvals = self.__computegram__(X,X_star)
        return kernelvals

    def mysvm(self, X_train, y_train, eta_init=0.1, eps=0.00001,max_iterations = 50):
        """
             Method that trains the model on the inputs. Currently uses Fast Gradient Descent,
             for optimisation.
        """
        self.classes = np.unique(y_train)
        self.data = {}
        self.__kerneleval__(X_train)
        for i in range(len(self.classes)):
            y = copy.deepcopy(y_train)
            y[y != self.classes[i]] = -1
            y[y == self.classes[i]] = 1
            theta = np.zeros(self.gram.shape[0])
            t=eta_init
            coef_1 = theta
            grad = self.__computegrad__(y,theta)
            max_iter = 0
            while np.linalg.norm(grad) > eps and max_iter < max_iterations:
                t = self.__backtracking__(y,theta,grad,t,m_iter=max_iterations)
                coef = coef_1
                coef_1 = theta - t*grad
                theta = coef_1 + (t/(t+3))*(coef_1-coef)
                grad = self.__computegrad__(y,theta)
                max_iter += 1
            self.data['class_'+str(self.classes[i])] = {'beta':copy.deepcopy(theta),
                                                   'class':self.classes[i]}
        print('Training done.')

    def predict(self,X_train,X_test):
        """
            Method to calculate and return the predictions for the input test data.
        """
        def getvals(x):
            vals = x[x>0]
            if len(vals) == 0:
                return np.random.choice(self.classes)
            return np.random.choice(vals)

        self.__kerneleval__(X_train,X_test)
        pred_matrix = np.empty((X_test.shape[0],0),int)
        for key in self.data:
            pred = np.sign(np.dot(self.gram.T,self.data[key]['beta']))
            pred[pred==1] = self.data[key]['class']
            pred_matrix = np.column_stack([pred_matrix,pred])
            y_hat = np.apply_along_axis(getvals,axis=1,arr=pred_matrix)
        return y_hat
