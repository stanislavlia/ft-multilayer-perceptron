from autograd import Value
from typing import List, Optional
import random
from enum import Enum
from optimizers import StochasticGradientDescent


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






# Now implement a 2-layer network with softmax output to classify 3 classes

if __name__ == "__main__":
    random.seed(42)
    # 1) Generate synthetic 2D data for 3 classes
    xs: List[List[float]] = []
    ys: List[int] = []
    num_per_class = 60
    for cls in range(3):
        # center for each class
        center_x = [0.0, 3.0, -3.0][cls]
        center_y = [0.0, 3.0, 3.0][cls]
        for _ in range(num_per_class):
            x1 = random.gauss(center_x, 1.0)
            x2 = random.gauss(center_y, 1.0)
            xs.append([x1, x2])
            ys.append(cls)

    # 2) Build network: 2 inputs → 5 hidden (tanh) → 3 outputs (softmax)
    hidden = Layer(
        n_input=2,
        n_output=5,
        activation=Activation.RELU,
        initializer=WeightInitializer(option=WeightInitializationOption.NORMAL)
    )
    output_layer = Layer(
        n_input=5,
        n_output=3,
        activation=Activation.SOFTMAX,
        initializer=WeightInitializer(option=WeightInitializationOption.NORMAL)
    )

    # Collect parameters and setup optimizer
    params = hidden.parameters() + output_layer.parameters()
    optimizer = StochasticGradientDescent(parameters=params, lr=0.005)

    # 3) Training loop
    epochs = 200
    for epoch in range(epochs):
        total_loss = Value(0.0)
        optimizer.zero_grad()

        # forward + loss aggregation
        for x_raw, y_true in zip(xs, ys):
            # wrap inputs
            x_vals = [Value(x_raw[0]), Value(x_raw[1])]
            # forward pass
            h = hidden(x_vals)                # List[Value] of length 5
            preds = output_layer(h)          # List[Value] of length 3 (softmax)

            # one-hot encode true label
            true_vec = [1.0 if i == y_true else 0.0 for i in range(3)]
            # MSE loss between preds and true one-hot
            sample_loss = sum((p - tv) ** 2 for p, tv in zip(preds, true_vec))
            total_loss = total_loss + sample_loss

        # backward and update
        total_loss.backward()
        optimizer.step()

        # print loss every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | Loss = {total_loss.val:.4f}")

    # 4) Evaluate accuracy on training set
    correct = 0
    for x_raw, y_true in zip(xs, ys):
        x_vals = [Value(x_raw[0]), Value(x_raw[1])]
        preds = output_layer(hidden(x_vals))
        pred_label = max(range(len(preds)), key=lambda i: preds[i].val)
        if pred_label == y_true:
            correct += 1
    accuracy = correct / len(xs)
    print(f"Training accuracy: {accuracy:.2f}")
