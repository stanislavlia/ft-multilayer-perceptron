from autograd import Value
from typing import List, Optional, Tuple
import random
from enum import Enum
from optimizers import StochasticGradientDescent, Optimizer
from metrics import binary_crossentropy_loss, mean_squared_error_loss, r2_score, accuracy_score, f1_score
from loguru import logger
import json

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
    
    def state_dict(self) -> dict:

        return {
            "n_input": self.n_input,
            "n_output": self.n_output,
            "activation": self.activation.value,
            "params": [p.val for p in self.parameters()],
        }
    
    @classmethod
    def load_state_dict(cls, state: dict) -> "Layer":
        """
        Reconstructs a Layer from state_dict output.
        """

        layer = cls(
            n_input=state["n_input"],
            n_output=state["n_output"],
            activation=Activation(state["activation"]),
            initializer=WeightInitializer(option=WeightInitializationOption.ZEROS)
        )
        saved = state["params"]
        params = layer.parameters()
        if len(saved) != len(params):
            raise ValueError("Layer.load_state_dict: parameter count mismatch")
        for p_obj, v in zip(params, saved):
            p_obj.val = v
        return layer
    

class FeedForwardNN():
    def __init__(self, layers: List[Layer]):

        self.layers = layers
        self._validate_layers(layers)

        self.optimizer = None
        self.loss_func = None
        self.metric = None

        self.batch_size = None
        self.epochs = None

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

        self.train_loss_history = []
        self.val_loss_history = []
        self.train_metric_history = []
        self.val_metric_history = []


    def parameters(self):
        
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
            
        return params

    @staticmethod
    def _validate_layers(layers: List["Layer"]):
        # Ensure layers list is non-empty and all are Layer instances
        if not isinstance(layers, list) or len(layers) == 0:
            raise ValueError("`layers` must be a non-empty list of Layer instances")
        for idx, layer in enumerate(layers):
            if not isinstance(layer, Layer):
                raise TypeError(f"Element at index {idx} is not a Layer: {type(layer)}")

        # Ensure output size of each layer matches input size of next
        for i in range(len(layers) - 1):
            out_size = layers[i].n_output
            next_in = layers[i+1].n_input
            if out_size != next_in:
                raise ValueError(
                    f"Layer {i} output size ({out_size}) does not match "
                    f"Layer {i+1} input size ({next_in})"
                )
    
    def _validate_data(self,
                       X_train: List[List[float]],
                       y_train: List[float],
                       X_val: Optional[List[List[float]]] = None,
                       y_val: Optional[List[float]] = None):
        
        # Basic type & length checks
        if not isinstance(X_train, list) or not all(isinstance(x, list) for x in X_train):
            raise ValueError("X_train must be a list of feature lists")
        if not isinstance(y_train, list):
            raise ValueError("y_train must be a list")
        if len(X_train) != len(y_train):
            raise ValueError(f"Train samples ({len(X_train)}) != train labels ({len(y_train)})")

        # All train rows have same length
        feat_len = len(X_train[0])
        for i, x in enumerate(X_train):
            if len(x) != feat_len:
                raise ValueError(f"All rows in X_train must have same length; "
                                 f"row 0 is {feat_len} but row {i} is {len(x)}")

        #If validation provided, repeat checks
        if X_val is not None or y_val is not None:
            if X_val is None or y_val is None:
                raise ValueError("If you pass X_val you must also pass y_val (and vice versa)")
            if not isinstance(X_val, list) or not all(isinstance(x, list) for x in X_val):
                raise ValueError("X_val must be a list of feature lists")
            if not isinstance(y_val, list):
                raise ValueError("y_val must be a list")
            if len(X_val) != len(y_val):
                raise ValueError(f"Val samples ({len(X_val)}) != val labels ({len(y_val)})")
            for i, x in enumerate(X_val):
                if len(x) != feat_len:
                    raise ValueError(f"All rows in X_val must have same length as X_train; "
                                     f"row {i} is {len(x)} but should be {feat_len}")

        expected = self.layers[0].n_input
        if feat_len != expected:
            raise ValueError(f"Each input sample must have {expected} features, "
                             f"but data has {feat_len}")
    
    def save_params(self, filepath):

        layers_states = [l.state_dict() for l in self.layers]
        with open(filepath, "w") as f:
            json.dump({"layers" : layers_states}, f, indent=3)
            logger.info(f"Saved Neural Network Parameters to {filepath}")

    @staticmethod
    def _load_params(filepath):
        with open(filepath, "r") as f:
            params = json.load(f)
        layers_states = params["layers"]
        #reconstruct layers
        layers_built = [Layer.load_state_dict(l_state) for l_state in layers_states]
        logger.info(f"Sucessfully loaded parameters from {filepath}")
        return layers_built

    def load_params(self, filepath):
        self.layers = self._load_params(filepath)

    @classmethod
    def build_from_parameters_file(cls, filepath):
        layers = FeedForwardNN._load_params(filepath)
        return cls(layers)

    def _generate_batches(
        self,
        X: List[List[float]],
        y: List[float]
    ):
        """
        Yield successive batches from X and y.
        Shuffles indices before batching.
        """
        n_samples = len(X)
        indices = list(range(n_samples))
        random.shuffle(indices)
        for start in range(0, n_samples, self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            batch_X = [X[i] for i in batch_idx]
            batch_y = [y[i] for i in batch_idx]
            yield batch_X, batch_y

    def forward(self, x : List[float]) -> Value:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def forward_batch(self, x: List[List[float]]) -> List[Value]:

        y_pred_batch = [self.forward(xi) for xi in x]
        return y_pred_batch

    def _metric(self, y_pred, y_true) -> float:
        
        if self.metric == "accuracy":
            return accuracy_score(y_pred, y_true)
        if self.metric == "f1_score":
            return f1_score(y_pred, y_true)
        if self.metric == "r2":
            return r2_score(y_pred, y_true)
        if self.metric == "mse":
            return mean_squared_error_loss(y_pred, y_true)
        
        supported = ["accuracy", "f1_score", "r2", "mse"]
        raise NotImplementedError(
            f"Metric '{self.metric}' is not supported. Use one of {supported}."
        )


    def __call__(self, x) -> List[float]:
        return self.forward_batch(x)

    def validation_eval(self, X_val, y_val) -> Tuple:
        y_pred = self.forward_batch(X_val)
        flat_preds = [out[0] for out in y_pred]

        metric = self._metric(flat_preds, y_val)
        loss = self.loss_func(flat_preds, y_val)
        return loss, metric
    
    def fit(
        self,
        X_train: List[List[float]],
        y_train: List[float],
        optimizer: Optimizer,
        loss,
        epochs: int,
        batch_size: int = 16,
        metric: str = "accuracy",
        X_val: List[List[float]] = None,
        y_val: List[float] = None
    ):
        # Store hyperparams & data
        self.optimizer = optimizer
        self.loss_func  = loss
        self.epochs     = epochs
        self.batch_size = batch_size
        self.metric     = metric
        self.X_train, self.y_train = X_train, y_train
        self.X_val,   self.y_val   = X_val,   y_val

        self._validate_data(X_train, y_train, X_val, y_val)

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            y_train_preds = []
            y_train_true = []

            for X_batch, y_batch in self._generate_batches(X_train, y_train):
                self.optimizer.zero_grad()

                batch_out = self.forward_batch(X_batch)
                flat_preds = [out[0] for out in batch_out]

                y_train_preds.extend(flat_preds)
                y_train_true.extend(y_batch)

                # compute loss, backprop, step
                batch_loss = loss(flat_preds, y_batch)
                batch_loss.backward()
                self.optimizer.step()

                epoch_loss += batch_loss.val

            # average train loss over batches
            num_batches = len(X_train) / batch_size
            avg_train_loss = epoch_loss / num_batches

            train_metric = self._metric(y_train_preds, y_train_true)

            # validation (if provided)
            if X_val is not None and y_val is not None:
                val_loss, val_metric = self.validation_eval(X_val, y_val)
            else:
                val_loss, val_metric = None, None

            logger.info(
                f"Epoch {epoch}/{epochs}  "
                f"train_loss={avg_train_loss:.4f}  "
                f"{self.metric}: {train_metric}    "
                f"val_loss={val_loss.val:.4f}  "
                f"val_{self.metric}={val_metric:.4f}"
            )
            self.train_loss_history.append(avg_train_loss)
            self.val_loss_history.append(val_loss.val)
            self.train_metric_history.append(0.0)
            self.val_metric_history.append(val_metric)

