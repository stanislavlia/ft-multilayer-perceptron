from autograd import Value
import matplotlib.pyplot as plt
from typing import List
import math

class Optimizer():
    """Base class for Optimizers"""
    def __init__(self, parameters: List[Value]):
        self.parameters = parameters

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0 #set to zero
    
    def step():
        raise NotImplementedError("This is base class which is not intended for usage")


class StochasticGradientDescent(Optimizer):
    def __init__(self, parameters, lr = 0.001):
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        
        #update parameters
        for i in range(len(self.parameters)):
            #gradient descent
            self.parameters[i].val -= self.lr * self.parameters[i].grad

#BONUS
class RMSProp(Optimizer):
    def __init__(self, parameters, lr = 0.001, beta=0.9):
        super().__init__(parameters)
        self.lr = lr
        self.beta = beta
        self.step_count = 0
        self.eps = 1e-9 #small const

        self.decaying_squared_grad = []

    def step(self):

        if self.step_count == 0:
            #initialize 
            self.decaying_squared_grad = [param.grad ** 2 for param in self.parameters]

        for i in range(len(self.parameters)):

            #update exponentially decaying values
            self.decaying_squared_grad[i] = self.beta * self.decaying_squared_grad[i] + (1 - self.beta) * self.parameters[i].grad ** 2
            #RMSPROP update
            self.parameters[i].val -= self.lr * self.parameters[i].grad / math.sqrt(self.decaying_squared_grad[i] + self.eps)
            
            