#!/usr/bin/env python3

"""
Simple layers/model based on numpy arrays with autodiff.
Also a simple gradient descent implementation (``gd``).
"""

import numpy as np
import typing
import abc
from collections import deque
import itertools

class Layer(metaclass=abc.ABCMeta):
    def __init__(self):
        self._dispatcher = None
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
    
    @abc.abstractproperty
    def nout(self):
        pass
    
    
    def get_weightlist_dispatcher(self):
        return list(self.generate_and_set_weightlist_dispatcher(itertools.count()))
    
    @abc.abstractmethod
    def generate_and_set_weightlist_dispatcher(self, cindex):
        pass

    @abc.abstractmethod
    def to_json(self):
        pass

    @classmethod 
    def probe_json(cls, json_obj):
        if("__type__" not in json_obj):
            return False 
        if(json_obj["__type__"] != cls.__name__):
            return False
        return True

    @classmethod
    def from_json(cls, json_obj):
        if(cls.probe_json(json_obj)):
            del(json_obj["__type__"])
            return cls(**json_obj)

        for subcls in cls.__subclasses__():
            if(subcls.probe_json(json_obj)):
                del(json_obj["__type__"])
                return subcls(**json_obj)
    
class MatrixLayer(Layer):
    def __init__(self, nin: int, nout: int):
        super().__init__()
        self._nin = nin
        self._nout = nout

    def to_json(self):
        return {"__type__": self.__class__.__name__
                , "nin": self._nin
                , "nout": self._nout}
        
    def random_weights(self):
        yield np.random.uniform(-1, 1, (self._nout, self._nin))
    
    def call(self, W, vin):
        return W[self._dispatcher] @ vin
    
    def dweights_proj(self, W, vin, r):
        dW = np.zeros_like(W[self._dispatcher])
        for i in range(vin.shape[0]):
            dW[:,i] = r
        for j in range(r.shape[0]):
            dW[j,:] *= vin
        yield dW
    
    def dvin_proj(self, W, vin, r):
        return W[self._dispatcher].T @ r
    
    @property
    def nout(self):
        return self._nout
    
    def generate_and_set_weightlist_dispatcher(self, cindex):
        self._dispatcher = next(cindex)
        yield self._dispatcher
    
class ReluLayer(Layer):
    def __init__(self, nvals: int):
        super().__init__()
        self._nvals = nvals

    def to_json(self):
        return {"__type__": self.__class__.__name__
                , "nvals": self._nvals}
        
    def random_weights(self):
        yield np.random.uniform(-1, 1, self._nvals)
    
    def call(self, W, vin):
        return W[self._dispatcher] * (vin > 0) * vin
    
    def dweights_proj(self, W, vin, r):
        dW = np.copy(r)
        dW *= vin * (vin > 0)
        yield dW
    
    def dvin_proj(self, W, vin, r):
        dv = np.copy(W[self._dispatcher])
        dv *= r * (vin > 0)
        return dv
    
    @property
    def nout(self):
        return self._nvals
    def generate_and_set_weightlist_dispatcher(self, cindex):
        self._dispatcher = next(cindex)
        yield self._dispatcher
        
class BiasLayer(Layer):
    def __init__(self, nvals: int):
        super().__init__()
        self._nvals = nvals
    def to_json(self):
        return {"__type__": self.__class__.__name__
                , "nvals": self._nvals}
        
    def random_weights(self):
        yield np.random.uniform(-1, 1, self._nvals)
    
    def call(self, W, vin):
        return W[self._dispatcher] + vin
    
    def dweights_proj(self, W, vin, r):
        yield np.copy(r)
    
    def dvin_proj(self, W, vin, r):
        return np.copy(r)
    
    @property
    def nout(self):
        return self._nvals
    def generate_and_set_weightlist_dispatcher(self, cindex):
        self._dispatcher = next(cindex)
        yield self._dispatcher
        
class SequenceLayer(Layer):
    def __init__(self, layers):
        super().__init__()
        if(not isinstance(layers[0], Layer)):
            self._layers = [Layer.from_json(l) for l in layers]
        else:
            self._layers = layers

    def to_json(self):
        return {"__type__": self.__class__.__name__
                , "layers": [layer.to_json() for layer in self._layers]}
        
    def random_weights(self):
        for l in self._layers:
            yield from l.random_weights()
    
    def call(self, W, vin):
        v = vin
        for l in self._layers:
            v = l.call(W, v)
        return v
    
    def dweights_proj(self, W, vin, r):
        # Forward propagation of values
        values = [vin]
        for l in self._layers:
            values.append(l.call(W, values[-1]))
    
        # Backward accumulation of gradients
        gradients = deque()
        right = r
        for l, h in zip(reversed(self._layers), reversed(values[:-1])):
            gradients.extend(l.dweights_proj(W, h, right))
            right = l.dvin_proj(W, h, right)
    
        yield from gradients
    
    def dvin_proj(self, W, vin, r):
        # Forward propagation of values
        values = [vin]
        for l in self._layers:
            values.append(l.call(W, values[-1]))
    
        # Backward accumulation of gradients
        right = r
        for l, h in zip(reversed(self._layers), reversed(values[:-1])):
            right = l.dvin_proj(W, h, right)
            
        return right
    
    @property
    def nout(self):
        return self._layers[-1].nout
    def generate_and_set_weightlist_dispatcher(self, cindex):
        for layer in self._layers:
            yield from layer.generate_and_set_weightlist_dispatcher(cindex)
        
class IdentityLayer(Layer):
    def __init__(self, npass):
        super().__init__()
        self._npass = npass
    
    def to_json(self):
        return {"__type__": self.__class__.__name__
                , "npass": self._npass}
    @property
    def nout(self):
        return self._npass
    
    def random_weights(self):
        yield np.array([0])
    
    def dweights_proj(self, W, vin, r):
        yield np.array([0])
    
    def dvin_proj(self, W, vin, r):
        return np.copy(r)
    
    def call(self, W, vin):
        return np.copy(vin)
    def generate_and_set_weightlist_dispatcher(self, cindex):
        self._dispatcher = next(cindex)
        yield self._dispatcher
        
class ParallelLayer(Layer):
    def __init__(self, nin, sequence_a: typing.Union[Layer, dict], sequence_b: typing.Union[Layer, dict]):
        super().__init__()
        self._nin = nin

        if(not isinstance(sequence_a, Layer)):
            self._sequence_a = Layer.from_json(sequence_a)
        else:
            self._sequence_a = sequence_a

        if(not isinstance(sequence_b, Layer)):
            self._sequence_b = Layer.from_json(sequence_b)
        else:
            self._sequence_b = sequence_b
    
    def to_json(self):
        return {"__type__": self.__class__.__name__
                , "nin": self._nin
                , "sequence_a": self._sequence_a.to_json()
                , "sequence_b": self._sequence_b.to_json()
                }

    @property
    def nout(self):
        return self._nin
    
    def call(self, W, vin):
        v1 = np.copy(vin)
        v2 = np.copy(vin)
        
        r1 = self._sequence_b.call(W, v1)
        r2 = self._sequence_b.call(W, v2)
        
        r = np.hstack([r1, r2])
        return  W[self._dispatcher] @ r
        
    def random_weights(self):
        yield from self._sequence_a.random_weights()
        yield from self._sequence_b.random_weights()
        yield np.random.uniform(-1, 1, (self.nout, self._sequence_a.nout + self._sequence_b.nout))
        
    def dweights_proj(self, W, vin, r):
        v1 = np.copy(vin)
        v2 = np.copy(vin)
        
        r1 = self._sequence_a.call(W, v1)
        r2 = self._sequence_b.call(W, v2)
        
        forwarded = np.hstack([r1, r2])
        
        dW = np.zeros_like(W[self._dispatcher])
        for i in range(forwarded.shape[0]):
            dW[:,i] = r
        for j in range(r.shape[0]):
            dW[j,:] *= forwarded
        yield dW

        right = W[self._dispatcher].T @ r
    
        yield from self._sequence_b.dweights_proj(W, vin, right[self._sequence_a.nout:])
        yield from self._sequence_a.dweights_proj(W, vin, right[:self._sequence_a.nout])
    
    def dvin_proj(self, W, vin, r):
        vcomplete = W[self._dispatcher].T @ r
        ra = vcomplete[:self._sequence_a.nout]
        rb = vcomplete[self._sequence_a.nout:]

        ra = self._sequence_a.dvin_proj(W, vin, ra)
        rb = self._sequence_b.dvin_proj(W, vin, rb)

        return ra + rb
        
    def generate_and_set_weightlist_dispatcher(self, cindex):
        yield from self._sequence_a.generate_and_set_weightlist_dispatcher(cindex)
        yield from self._sequence_b.generate_and_set_weightlist_dispatcher(cindex)
        self._dispatcher = next(cindex)
        yield self._dispatcher
    
       
class HeterogenousParallelLayer(Layer):
    def __init__(self, nout, layers: typing.List[typing.Union[Layer, dict]]):
        super().__init__()

        self._layers = [l if isinstance(l, Layer) else Layer.from_json(l) for l in layers]
        self._nout = nout
        self._n_pre_merge = sum(l.nout for l in self._layers)
    
    def to_json(self):
        return {"__type__": self.__class__.__name__
                , "layers": [l.to_json() for l in self._layers]
                , "nout": self._nout
                }

    @property
    def nout(self):
        return self._nout
    
    def call(self, W, vin):
        parallel_results = [l.call(W, np.copy(vin)) for l in self._layers]
        
        r = np.hstack(parallel_results)
        return  W[self._dispatcher] @ r
        
    def random_weights(self):
        for l in self._layers:
            yield from l.random_weights()
        yield np.random.uniform(-1, 1, (self.nout, self._n_pre_merge))
        
    def dweights_proj(self, W, vin, r):
        parallel_results = [l.call(W, np.copy(vin)) for l in self._layers]
        
        forwarded = np.hstack(parallel_results)
        
        dW = np.zeros_like(W[self._dispatcher])
        for i in range(forwarded.shape[0]):
            dW[:,i] = r
        for j in range(r.shape[0]):
            dW[j,:] *= forwarded
        yield dW

        right = W[self._dispatcher].T @ r
    
        n_outs = np.array([l.nout for l in self._layers])
        n_outs_accumulated = np.cumsum(n_outs)

        for l, (nout, nacc) in zip(reversed(self._layers),zip(reversed(n_outs), reversed(n_outs_accumulated))):
            yield from l.dweights_proj(W, vin, right[nacc-nout:nacc])

    
    def dvin_proj(self, W, vin, r):
        vcomplete = W[self._dispatcher].T @ r

        res = 0
        acc_nout = 0
        for l in self._layers:
            res += l.dvin_proj(W, vin, vcomplete[acc_nout:acc_nout + l.nout])
            acc_nout += l.nout

        return res
        
    def generate_and_set_weightlist_dispatcher(self, cindex):
        for l in self._layers:
            yield from l.generate_and_set_weightlist_dispatcher(cindex)
        self._dispatcher = next(cindex)
        yield self._dispatcher

class Model(SequenceLayer):
    def __init__(self, layers: typing.List[Layer]):
        super().__init__(layers)
        self.get_weightlist_dispatcher()
        
    def get_random_weights(self):
        return list(self.random_weights())
        
    #def call(self, W, vin):
    #    v = vin
    #    for w, l in zip(W, self._layers):
    #        v = l.call(w, v)
    #    return v
    #
    
    def cost(self, W, vin, b):
        if(not isinstance(vin, list)):
            return np.linalg.norm(self.call(W, vin) - b)**2
        return sum(np.linalg.norm(self.call(W, vini) - bi)**2 for vini, bi in zip(vin, b)) / len(vin)
    
    def dcostone(self, W, vin, b):
        # Forward propagation of values
        values = [vin]
        for l in self._layers:
            values.append(l.call(W, values[-1]))
        
        # Backward accumulation of gradients
        right = (values[-1] - b)
        gradients = deque()
        for l, h in zip(reversed(self._layers), reversed(values[:-1])):
            gradients.extend(l.dweights_proj(W, h, right))
            right = l.dvin_proj(W, h, right)
    
        return list(reversed(gradients))

    def dcost(self, W, vin, b):
        if(not isinstance(vin, list)):
            return self.dcostone(W, vin, b)
        rescale = len(vin)
        dW = self.dcostone(W, vin[0], b[0])
        for vini, bi in zip(vin, b):
            dWi = self.dcostone(W, vini, bi)
            for j, dWij in enumerate(dWi):
                dW[j] += dWij
        return [wi / rescale for wi in dW]

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


