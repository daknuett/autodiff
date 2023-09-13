#!/usr/bin/env python3


def get_dof(weights):
    def mul(iterable):
        res = 1
        for i in iterable:
            res *= i
        return res
    return sum(mul(w.shape) for w in weights) 
