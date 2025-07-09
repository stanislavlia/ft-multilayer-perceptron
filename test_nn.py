import random
from autograd import Value
from typing import List
from loguru import logger

from optimizers import StochasticGradientDescent, RMSProp
from metrics import binary_crossentropy_loss
from nn import Layer, WeightInitializer, WeightInitializationOption, Activation, FeedForwardNN

if __name__ == "__main__":
    random.seed(123)
    # 1) Generate toy binary classification data
    N = 1000
    X_train: List[List[float]] = []
    y_train: List[int] = []
    for _ in range(N):
        # class 0
        X_train.append([random.gauss(7, 3), random.gauss(1.6, 0.3)])
        y_train.append(0)
        # class 1
        X_train.append([random.gauss(2, 1), random.gauss(2, 1)])
        y_train.append(1)

    # 2) Split into train/val
    split = int(0.8 * len(X_train))
    X_val, y_val = X_train[split:], y_train[split:]
    X_train, y_train = X_train[:split], y_train[:split]


    init = WeightInitializer(option=WeightInitializationOption.NORMAL)

    model = FeedForwardNN(layers=[
        Layer(n_input=2, n_output=5, activation=Activation.RELU, initializer=init),
        Layer(n_input=5, n_output=3, activation=Activation.RELU, initializer=init),
        Layer(n_input=3, n_output=1, activation=Activation.SIGMOID, initializer=init),
    ])

    # 4) Setup optimizer and loss
    params = model.parameters()
    optimizer = RMSProp(parameters=params, lr=0.01, beta=0.9) #StochasticGradientDescent(params, lr=0.01) 
    loss_fn = binary_crossentropy_loss

    # 5) Train
    model.fit(
        X_train=X_train,
        y_train=y_train,
        optimizer=optimizer,
        loss=loss_fn,
        epochs=20,
        batch_size=32,
        metric="accuracy",
        X_val=X_val,
        y_val=y_val
    )

    model.save_params("./tuned_model.json")


    loaded_model = model.build_from_parameters_file("./tuned_model.json")

    # 6) Evaluate on validation set using model.forward()
    correct = 0
    for x_raw, y_true in zip(X_val, y_val):
        # forward takes raw floats, returns List[Value] for this sample
        preds: List[Value] = loaded_model.forward(x_raw)
        pred_val = preds[0].val         # single-output neuron
        label = 1 if pred_val > 0.5 else 0
        if label == y_true:
            correct += 1

    acc = correct / len(X_val)
    logger.info(f"Validation accuracy on loaded model: {acc:.3f}")
