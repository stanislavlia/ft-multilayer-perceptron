from loguru import logger
import pandas as pd
from optimizers import StochasticGradientDescent, RMSProp, Adam
from sklearn.preprocessing import StandardScaler
from metrics import categorical_crossentropy_loss
from nn import Layer, WeightInitializer, WeightInitializationOption, Activation, FeedForwardNN, xavier_std
import click
import joblib


#GLOBAL VARS: Encoding for Y
#encode to binary
target_encoder = lambda target: [1, 0] if target == "B" else [0, 1]  # Benign(Good) - 0, Malignant(Bad) - 1
reverse_target_encoder = lambda target: "B" if target == [1, 0] else "M"


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
@click.option('--opt', type=click.Choice(['sgd', 'rmsprop', "adam"], case_sensitive=False),
              default='rmsprop', show_default=True, help='Optimizer to use: sgd, rmsprop')
@click.option('--scale/--no-scale', default=True, show_default=True,
              help='Scale features using StandardScaler or not.')
@click.option('--scaler-path', default=None,
              help='Optional path to save fitted scaler (only used if --scale is True)')
def training_program(
    train_file: str,
    val_file: str,
    target_idx,
    lr: float,
    batch_size: int,
    epochs: int,
    model_path: str,
    opt: str,
    scale : bool,
    scaler_path: str
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


    #Transform features (Scale) for better convergence if scale=true
    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train).tolist()
        X_val_scaled = scaler.transform(X_val).tolist()
        if scaler_path:
            joblib.dump(scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
    else:
        X_train_scaled = X_train.tolist()
        X_val_scaled = X_val.tolist()


    n_features = len(X_train[0])

    #targets
    # One-hot encode targets. This works for binary and multi-class cases.
    num_classes = len(train_df[1].unique())
    n_features = len(X_train[0])

    #targets    
    y_train = train_df[1].apply(target_encoder).tolist()
    y_val = val_df[1].apply(target_encoder).tolist()


    # SET UP NEURAL NETWORK ARCHITECTURE
    # you can build neural network from layers like Lego blocks
    model = FeedForwardNN(
        layers=[
            Layer(n_input=n_features, #must match with input dimension
                n_output=8,
                activation=Activation.RELU,
                initializer=WeightInitializer(option=WeightInitializationOption.NORMAL, sd=xavier_std(n_features, 16))),
            Layer(n_input=8,
                n_output=4,
                activation=Activation.RELU,
                initializer=WeightInitializer(option=WeightInitializationOption.NORMAL, sd=xavier_std(16, 16))),
            Layer(n_input=4,
                n_output=num_classes, # Output layer must match the number of classes
                activation=Activation.SOFTMAX, # Softmax for multi-class probability output
                initializer=WeightInitializer(option=WeightInitializationOption.NORMAL, sd=xavier_std(16, num_classes)))
            ]
    )

    params = model.parameters()
    logger.info(f"Model created successfully | number of params: {len(params)}")

    #OPTIMIZER
    if opt == "rmsprop":
        optimizer = RMSProp(parameters=params, lr=lr, beta=0.9)
    elif opt == "adam":
        optimizer = Adam(parameters=params, lr=lr)
    else:
        optimizer = StochasticGradientDescent(parameters=params, lr=lr)

    model.fit(
        X_train=X_train_scaled,
        y_train=y_train,
        X_val=X_val_scaled,
        y_val=y_val,
        optimizer=optimizer,
        loss=categorical_crossentropy_loss, # Use categorical cross-entropy for classification
        epochs=epochs,
        batch_size=batch_size,
        metric="accuracy",
        display_each_n_step=1
    )
    model.save_params(model_path)
    model.plot_learning_history()

    



if __name__ == "__main__":
    training_program()