from autograd import Value
from typing import List, Optional
import random
from enum import Enum

class WeightInitializationOption(Enum):
    ZEROS = "zeros"
    UNIFORM = "uniform"
    NORMAL = "normal"

class Activation(Enum):
    TANH = "tanh"
    RELU = "relu"
    SIGMOID = "sigmoid"
    LINEAR = "linear"


class WeightInitializer():
    def __init__(self, option=WeightInitializationOption.ZEROS):
        self.option = option

    def __call__(self) -> float:
        """Returns initialized value for weight"""
        
        if self.option == WeightInitializationOption.ZEROS:
            return 0.0
        if self.option == WeightInitializationOption.NORMAL:
            return random.normalvariate(mu=0, sigma=0.1)
        if self.option == WeightInitializationOption.UNIFORM:
            return random.uniform(a=0, b=1)
        else:
            raise NotImplementedError(f"option: {self.option} is not supported")



class Neuron():
    def __init__(self, input_size: int, activation : Activation, initializer: WeightInitializer):
        
        self.weights = [initializer() for _ in range(input_size)]
        self.bias = initializer()
        self.params = self.weights + [self.bias]
        self.activation = activation

    def __call__(self, x: List[Value | float]):
        "Forward pass for single Neuron"

        weighted_sum : Value = sum([w_i * x_i for x_i, w_i in zip(x, self.weights)])
        weighted_sum += self.bias

        if self.activation == Activation.LINEAR:
            return weighted_sum
        if self.activation == Activation.RELU:
            return weighted_sum.relu()
        if self.activation == Activation.SIGMOID:
            return weighted_sum.sigmoid()
        if self.activation == Activation.TANH:
            return weighted_sum.tanh()
        else:
            raise NotImplementedError(f"Activation: {self.activation} not supported")





