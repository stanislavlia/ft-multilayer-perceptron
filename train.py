import random
from autograd import Value
from typing import List
from loguru import logger
import pandas as pd
from optimizers import StochasticGradientDescent, RMSProp
from metrics import binary_crossentropy_loss
from nn import Layer, WeightInitializer, WeightInitializationOption, Activation, FeedForwardNN
import click



@click.command()
@click.option('--train-file', default='train.csv', show_default=True,
              help='Filename for the training set (written in current directory).')
@click.option('--val-file', default="val.csv", show_default=True,
              help='Filename for the validation set (written in current directory).')
@click.option('--target-idx', default=1, show_default=True,
              help='Index for target columns')
@click.option('--lr', default=0.001, show_default=True,
              help='Learning rate')
@click.option('--batch-size', default=16, show_default=True,
              help='batch size')
@click.option('--epochs', default=20, show_default=True,
              help='Number of training epochs')

def training_program(
    train_file: str,
    val_file: str,
    target_idx,
    lr: float,
    batch_size: int,
    epochs: int,
):
    
    #LOAD AND PREPROCESS
    train_df = pd.read_csv(train_file, header=None)
    val_df = pd.read_csv(val_file, header=None)

    #Drop ids
    train_df.drop(0, axis=1, inplace=True)
    val_df.drop(0, axis=1, inplace=True)

    #remove target from X
    X_train = train_df.drop(target_idx, axis=1).values.tolist()
    X_val = val_df.drop(target_idx, axis=1).values.tolist()

    n_features = len(X_train[0])

    #targets
    target_encoder = lambda target: 0 if target == "B" else 1  # Benign(Good) - 0, Malignant(Bad) - 1
    y_train = train_df[1].apply(target_encoder).to_list()
    y_val = val_df[1].apply(target_encoder).to_list()

    # SET UP NEURAL NETWORK ARCHITECTURE
    # you can build neural network from layers like Lego blocks

    initializer = WeightInitializer(option=WeightInitializationOption.UNIFORM)

    model = FeedForwardNN(
        layers=[
            Layer(n_input=n_features, #must match with input dimension
                n_output=6,
                activation=Activation.RELU,
                initializer=initializer),
            Layer(n_input=6,
                n_output=3,
                activation=Activation.RELU,
                initializer=initializer),
            Layer(n_input=3,
                n_output=1,
                activation=Activation.SIGMOID, #probability output
                initializer=initializer)
            ]
    )

    params = model.parameters()
    
    #optimizer = StochasticGradientDescent(params, lr=lr)
    optimizer = RMSProp(parameters=params, lr=lr, beta=0.9) #Use More Advanced Optimizer

    loss_fn = binary_crossentropy_loss

    model.fit(
        X_train=X_train,
        y_train=y_train,
        optimizer=optimizer,
        loss=loss_fn,
        epochs=epochs,
        batch_size=batch_size,
        metric="accuracy",
        X_val=X_val,
        y_val=y_val
    )

    model.plot_learning_history()

    



if __name__ == "__main__":
    training_program()