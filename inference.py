from loguru import logger
import pandas as pd
from sklearn.preprocessing import StandardScaler
from metrics import binary_crossentropy_loss, accuracy_score, f1_score
from nn import Layer, WeightInitializer, WeightInitializationOption, Activation, FeedForwardNN
import click
import joblib
import math

@click.command()
@click.option('--model-path', default="trained_model_params.json", show_default=True, help="Path to save trained model")
@click.option('--val-file', default="val.csv", show_default=True,
              help='Filename for the validation set (written in current directory).')
@click.option('--target-idx', default=1, show_default=True,
              help='Index for target columns')
@click.option('--scale/--no-scale', default=True, show_default=True,
              help='Scale features using StandardScaler or not.')
@click.option('--scaler-path', default=None,
              help='Optional path to save fitted scaler (only used if --scale is True)')
@click.option('--predictions-file', default=None)
@click.option('--metric', type=click.Choice(['accuracy', 'bce', 'f1'], case_sensitive=False)) #CLASSIFICATIONS ONLY METRIC
def inference_program(
    model_path,
    val_file,
    target_idx,
    scale,
    scaler_path,
    predictions_file,
    metric
):
    #PROCESS VALIDATION DATA
    val_df = pd.read_csv(val_file, header=None)
    val_df.drop(0, axis=1, inplace=True) #Drop ID
    X_val = val_df.drop(target_idx, axis=1).values
    #encode to binary
    target_encoder = lambda target: 0 if target == "B" else 1  # Benign(Good) - 0, Malignant(Bad) - 1
    reverse_target_encoder = lambda target: "B" if 0 else "M"
    y_val = val_df[1].apply(target_encoder).to_list()

    X_val_scaled = X_val
    if scale and scaler_path:
        scaler: StandardScaler = joblib.load(scaler_path)
        logger.info(f"Scaler: {scaler} loaded...")
        X_val_scaled = scaler.transform(X_val) #apply standardization
    
    X_val_scaled = X_val_scaled.tolist()

    #LOAD TRAINED MODEL
    model = FeedForwardNN.build_from_parameters_file(model_path)
    logger.info(f"Model loaded successfully | number of params: {len(model.parameters())}")

    val_predictions = model.forward_batch(X_val_scaled)
    
    print("===========FIRST 20 PREDICTIONS=========")
    for i, vp in enumerate(val_predictions[:20]):
        print(f"Prediction: {[vp[0].val, vp[1].val]}  | True = {y_val[i]}")
    
    metric_func = None
    if metric == "bce":
        metric_func = binary_crossentropy_loss
    elif metric == "f1":
        metric_func = f1_score
    else:
        metric_func = accuracy_score

    flat_preds = [out[-1] for out in val_predictions] #take prob for Y=1

    metric_value = metric_func(flat_preds, y_val)

    print()
    print()
    print('===============EVALUATION==============')
    print(f"Validation Metric {metric.upper()} = {metric_value}")


if __name__ == "__main__":
    inference_program()