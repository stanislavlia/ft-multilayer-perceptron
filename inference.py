from loguru import logger
import pandas as pd
from sklearn.preprocessing import StandardScaler
from metrics import accuracy_score, f1_score
from nn import FeedForwardNN
from train import target_encoder, reverse_target_encoder
import click
import joblib

@click.command()
@click.option('--model-path', default="trained_model_params.json", show_default=True, help="Path to trained model parameters")
@click.option('--val-file', default="val.csv", show_default=True,
              help='Filename for the validation set.')
@click.option('--target-idx', default=1, show_default=True,
              help='Index for the target column in the CSV file.')
@click.option('--scale/--no-scale', default=True, show_default=True,
              help='Scale features using a saved StandardScaler.')
@click.option('--scaler-path', default="scaler.joblib",
              help='Path to the saved scaler object.')
@click.option('--predictions-file', default="predictions.csv", show_default=True,
              help="File to save the model's predictions.")
@click.option('--metric', type=click.Choice(['accuracy', 'f1'], case_sensitive=False),
              help="Metric to evaluate the model's performance.")
def inference_program(
    model_path,
    val_file,
    target_idx,
    scale,
    scaler_path,
    predictions_file,
    metric
):
    # PROCESS VALIDATION DATA
    val_df = pd.read_csv(val_file, header=None)
    val_df.drop(0, axis=1, inplace=True) # Drop ID column
    X_val = val_df.drop(target_idx, axis=1).values
    
    # Use the imported encoder to create one-hot encoded labels
    y_val_one_hot = val_df[target_idx].apply(target_encoder).tolist()
    
    X_val_scaled = X_val
    if scale and scaler_path:
        try:
            scaler: StandardScaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")
            X_val_scaled = scaler.transform(X_val)
        except FileNotFoundError:
            logger.error(f"Scaler file not found at {scaler_path}. Proceeding without scaling.")
    
    X_val_scaled = X_val_scaled.tolist()

    # LOAD TRAINED MODEL
    model = FeedForwardNN.build_from_parameters_file(model_path)
    logger.info(f"Model loaded successfully | number of params: {len(model.parameters())}")

    # GET PREDICTIONS
    val_predictions_values = model.forward_batch(X_val_scaled)
    val_predictions_floats = [[v.val for v in p] for p in val_predictions_values]

    # EVALUATE METRIC
    if metric:
        metric_func = accuracy_score if metric == 'accuracy' else f1_score
        metric_value = metric_func(val_predictions_floats, y_val_one_hot)
        logger.info(f"Validation Metric {metric.upper()} = {metric_value:.4f}")

    # SAVE PREDICTIONS
    if predictions_file:
        # Decode one-hot predictions to class labels ('B' or 'M')
        predicted_indices = [p.index(max(p)) for p in val_predictions_floats]
        
        # Create a reverse map from index to label
        class_names = ['B', 'M'] # Assuming 'B' is index 0 from [1,0] and 'M' is index 1 from [0,1]
        predicted_labels = [class_names[i] for i in predicted_indices]
        
        predictions_df = pd.DataFrame({
            "true_label": val_df[target_idx].values,
            "predicted_label": predicted_labels,
            "benign_probability": [p[0] for p in val_predictions_floats],
            "malignant_probability": [p[1] for p in val_predictions_floats]
        })
        predictions_df.to_csv(predictions_file, index=False)
        logger.info(f"Predictions saved to {predictions_file}")

if __name__ == "__main__":
    inference_program()