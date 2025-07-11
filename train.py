import random
from autograd import Value
from typing import List
from loguru import logger
import pandas as pd
from optimizers import StochasticGradientDescent, RMSProp, Adam
from sklearn.preprocessing import StandardScaler
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
@click.option('--model-path', default="trained_model_params.json", show_default=True, help="Path to save trained model")
@click.option('--opt', type=click.Choice(['sgd', 'rmsprop', 'adam'], case_sensitive=False),
              default='rmsprop', show_default=True, help='Optimizer to use: sgd, rmsprop, or adam')
def training_program(
    train_file: str,
    val_file: str,
    target_idx,
    lr: float,
    batch_size: int,
    epochs: int,
    model_path: str,
    opt: str
):
    

    #LOAD AND PREPROCESS
    train_df = pd.read_csv(train_file, header=None)
    val_df = pd.read_csv(val_file, header=None)

    #Drop ids
    train_df.drop(0, axis=1, inplace=True)
    val_df.drop(0, axis=1, inplace=True)

    #remove target from X
    X_train = train_df.drop(target_idx, axis=1).values
    X_val = val_df.drop(target_idx, axis=1).values


    #Transform features (Scale) for better convergence
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).tolist()
    X_val_scaled = scaler.fit_transform(X_val).tolist()


    n_features = len(X_train[0])

    #targets
    #encode to binary
    target_encoder = lambda target: 0 if target == "B" else 1  # Benign(Good) - 0, Malignant(Bad) - 1
    y_train = train_df[1].apply(target_encoder).to_list()
    y_val = val_df[1].apply(target_encoder).to_list()

    # SET UP NEURAL NETWORK ARCHITECTURE
    # you can build neural network from layers like Lego blocks
    initializer = WeightInitializer(option=WeightInitializationOption.UNIFORM)

    model = FeedForwardNN(
        layers=[
            Layer(n_input=n_features, #must match with input dimension
                n_output=10,
                activation=Activation.RELU,
                initializer=initializer),
            Layer(n_input=10,
                n_output=5,
                activation=Activation.RELU,
                initializer=initializer),
            Layer(n_input=5,
                n_output=1,
                activation=Activation.SIGMOID, #probability output
                initializer=initializer)
            ]
    )

    params = model.parameters()

    logger.info(f"Start training | number of parameters: {len(params)} | optimizer {opt}; learning rate={lr}")
    
    optimizer = None
    if opt == "rmsprop":
        optimizer = RMSProp(parameters=params, lr=lr, beta=0.9)
    elif opt == "adam":
        optimizer = Adam(params)
    else:
        optimizer = StochasticGradientDescent(params, lr=lr)

    loss_fn = binary_crossentropy_loss

    model.fit(
        X_train=X_train_scaled,
        y_train=y_train,
        optimizer=optimizer,
        loss=loss_fn,
        epochs=epochs,
        batch_size=batch_size,
        metric="accuracy",
        X_val=X_val_scaled,
        y_val=y_val
    )
    model.save_params(model_path)
    model.plot_learning_history()

    



if __name__ == "__main__":
    training_program()