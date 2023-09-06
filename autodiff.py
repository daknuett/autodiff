#!/usr/bin/env python3

"""
Simple layers/model based on numpy arrays with autodiff.
Also a simple gradient descent implementation (``gd``).
"""

import numpy as np
import typing
import abc
from collections import deque

class Layer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def random_weights(self):
        pass    
    @abc.abstractmethod
    def call(self, W, vin):
        pass
    @abc.abstractmethod
    def dweights_proj(self, W, vin, r):
        pass
    @abc.abstractmethod
    def dvin_proj(self, W, vin, r):
        pass

class MatrixLayer(Layer):
    def __init__(self, nin: int, nout: int):
        self._nin = nin
        self._nout = nout
        #self._weights = self.random_weights()
        
    def random_weights(self):
        return np.random.uniform(-1, 1, (self._nout, self._nin))
    
    def call(self, W, vin):
        return W @ vin
    
    def dweights_proj(self, W, vin, r):
        dW = np.zeros_like(W)
        for i in range(vin.shape[0]):
            dW[:,i] = r
        for j in range(r.shape[0]):
            dW[j,:] *= vin
        return dW
    
    def dvin_proj(self, W, vin, r):
        return W.T @ r
    
class ReluLayer(Layer):
    def __init__(self, nvals: int):
        self._nvals = nvals
        
    def random_weights(self):
        return np.random.uniform(-1, 1, self._nvals)
    
    def call(self, W, vin):
        return W * (vin > 0) * vin
    
    def dweights_proj(self, W, vin, r):
        dW = np.copy(r)
        dW *= vin * (vin > 0)
        return dW
    
    def dvin_proj(self, W, vin, r):
        dv = np.copy(W)
        dv *= r * (vin > 0)
        return dv
    
class BiasLayer(Layer):
    def __init__(self, nvals: int):
        self._nvals = nvals
        
    def random_weights(self):
        return np.random.uniform(-1, 1, self._nvals)
    
    def call(self, W, vin):
        return W + vin
    
    def dweights_proj(self, W, vin, r):
        return np.copy(r)
    
    def dvin_proj(self, W, vin, r):
        return np.copy(r)

class Model:
    def __init__(self, layers: typing.List[Layer]):
        self._layers = layers
        
    def random_weights(self):
        return [l.random_weights() for l in self._layers]
    
    def call(self, W, vin):
        v = vin
        for w, l in zip(W, self._layers):
            v = l.call(w, v)
        return v
    
    def cost(self, W, vin, b):
        return np.linalg.norm(self.call(W, vin) - b)**2
    
    def dcost(self, W, vin, b):
        # Forward propagation of values
        values = [vin]
        for w, l in zip(W, self._layers):
            values.append(l.call(w, values[-1]))
        
        # Backward accumulation of gradients
        right = (values[-1] - b)
        gradients = deque()
        for (w, l), h in zip(zip(reversed(W), reversed(self._layers)), reversed(values[:-1])):
            gradients.appendleft(l.dweights_proj(w, h, right))
            right = l.dvin_proj(w, h, right)
    
        return list(gradients)

def gd(model, Winit, vin, b, eps=1e-5, alpha=1e-3, maxiter=1000):
    """
    Gradient decent. ``eps`` is the precision. ``alpha`` is the learn rate.

    Returns ``(Weights, (converged, iterations))``.
    """
    W = [np.copy(w) for w in Winit]
    
    for k in range(maxiter):
        dW = model.dcost(W, vin, b)
        if(sum(np.sum(dw**2) for dw in dW) < eps):
            return W, (True, k)
        
        for w, dw in zip(W, dW):
            w -= alpha*dw
    
    return W, (False, k)

def adam(model, Winit, vin, b
         , eps=1e-8, alpha=1e-3, beta1=0.9, beta2=0.999, maxiter=10_000):
    """
    @misc{kingma2017adam,
          title={Adam: A Method for Stochastic Optimization}, 
          author={Diederik P. Kingma and Jimmy Ba},
          year={2017},
          eprint={1412.6980},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }
    """

    m = [0 for w in Winit]
    v = [0 for w in Winit]
    W = [np.copy(w) for w in Winit]

    for t in range(1, maxiter + 1):
        g = model.dcost(W, vin, b)

        if(sum(np.sum(dw**2) for dw in g) < eps):
            return W, (True, t)

        m = [beta1 * mi + (1 - beta1) * gi for mi,gi in zip(m, g)]
        v = [beta2 * vi + (1 - beta2) * gi**2 for vi,gi in zip(v, g)]
        mhat = [mi / (1 - beta1**t) for mi in m]
        vhat = [vi / (1 - beta1**t) for vi in v]

        W = [w - alpha*mi / (np.sqrt(vi) + eps) for w, (mi,vi) in zip(W, zip(mhat,vhat))]

    return W, (False, t)


