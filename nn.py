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
    SOFTMAX = "softmax"


class WeightInitializer():
    def __init__(self, option=WeightInitializationOption.ZEROS):
        self.option = option

    def __call__(self) -> float:
        """Returns initialized value for weight"""
        
        if self.option == WeightInitializationOption.ZEROS:
            return Value(0.0)
        if self.option == WeightInitializationOption.NORMAL:
            return Value(random.normalvariate(mu=0, sigma=0.1))
        if self.option == WeightInitializationOption.UNIFORM:
            return Value(random.uniform(a=0, b=1))
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


    def parameters(self):
        return self.params


class Layer():
    def __init__(self,
                 n_input: int,
                 n_output: int,
                 activation: Activation,
                 initializer = WeightInitializer(option=WeightInitializationOption.UNIFORM)):
        
        self.n_input = n_input
        self.n_output = n_output
        self.activation = activation

        neuron_activation = activation if activation != Activation.SOFTMAX else Activation.LINEAR #if softmax, use linear for neurons and apply softmax on layer level

        self.neurons = [Neuron(input_size=n_input,
                               activation=neuron_activation,
                               initializer=initializer) for _ in range(n_output)]
    
    @staticmethod
    def _softmax(output : List) -> List:    
        
        softmax_results = [res.exp() for res in output] #exponentiate each
        softmax_results_normalized = [sf / sum(softmax_results) for sf in softmax_results] #normalize
        return softmax_results_normalized

    def __call__(self, x):
        assert len(x) == self.n_input

        output = [neuron(x) for neuron in self.neurons]

        if self.activation == Activation.SOFTMAX:
            output = self._softmax(output)
        
        return output

    def parameters(self):
        
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
            
        return params





if __name__ == "__main__":

    print(Layer._softmax([Value(0.1), Value(2), Value(3), Value(2.5), Value(0.0)]))

    # xs = [i for i in range(30)]
    # ys = [2 * x + 1 for x in xs]

    # # 2. Build a single‐neuron model with linear activation
    # init = WeightInitializer(option=WeightInitializationOption.NORMAL)
    # model = Neuron(input_size=1, activation=Activation.RELU, initializer=init)

    # # 3. Training hyperparameters
    # lr = 0.00001
    # epochs = 5000

    # for epoch in range(epochs):
    #     total_loss = Value(0.0)
        
    #     # zero gradients on parameters
    #     for p in model.parameters():
    #         p.grad = 0.0

    #     # sum squared error over data
    #     for x, y_true in zip(xs, ys):
    #         x_val = Value(x)
    #         y_pred = model([x_val])                 # forward
    #         loss = (y_pred - y_true) ** 2           # MSE for one point
    #         total_loss = total_loss + loss          # accumulate

    #     # backprop
    #     total_loss.backward()

    #     # update parameters
    #     for p in model.params:
    #         p.val -= lr * p.grad

    #     if epoch % 10 == 0 or epoch == epochs - 1:
    #         print(f"Epoch {epoch:2d} | Loss = {total_loss.val:.4f}")

    # # 4. Inspect learned parameters
    # w, b = model.weights[0], model.bias
    # print(f"\nLearned line:  y = {w.val:.3f} x + {b.val:.3f}")