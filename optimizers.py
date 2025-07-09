from autograd import Value
import matplotlib.pyplot as plt
from typing import List

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

        
class RMSProp(Optimizer):
    pass

class Adam(Optimizer):
    pass