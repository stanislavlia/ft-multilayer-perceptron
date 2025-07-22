from autograd import Value
from typing import List, Optional

#Losses
def categorical_crossentropy_loss(y_pred: List[List[Value]], y_true: List[List[int]]) -> Value:
    """
    Calculates the categorical cross-entropy loss for one-hot encoded labels.
    Args:
        y_pred (List[List[Value]]): Predicted probabilities for each class.
        y_true (List[List[int]]): Ground truth labels, one-hot encoded.
    Returns:
        Value: The categorical cross-entropy loss.
    """
    n_samples = len(y_true)
    loss = 0.0
    for i in range(n_samples):
        for j in range(len(y_true[i])):
            # Add a small epsilon to prevent log(0)
            loss -= y_true[i][j] * (y_pred[i][j] + 1e-9).log()
    return loss / n_samples

def mean_squared_error_loss(y_pred: List[List[Value]], y_true: List[List[float]]) -> Value:
    """
    Computes the mean squared error over a batch.
    Assumes y_pred and y_true are lists of lists (vectors). For regression,
    these are vectors with a single component.
        - y_pred: list of lists of Value predictions (e.g., [[v1], [v2]])
        - y_true: list of lists of ground-truth floats (e.g., [[t1], [t2]])
    """
    assert len(y_pred) == len(y_true), "Lengths of predictions and targets must match"
    n = len(y_true)
    
    # For regression, compare the first (and only) element of the vectors.
    loss = sum([(y_p[0] - yt_i[0])**2 for yt_i, y_p in zip(y_true, y_pred)]) / n

    return loss


#Metrics
def to_float(y: List[List[Value]]) -> List[List[float]]:

    y_float = [ [float(yij) for yij in yi]
               for yi in y]
    return y_float

def accuracy_score(y_pred: List[List[float]], y_true: List[List[float]]) -> float:
    """
    Calculates accuracy for classification tasks.
    Args:
        y_pred (List[List[float]]): Predicted probabilities.
        y_true (List[List[float]]): Ground truth labels (one-hot encoded).
    Returns:
        float: The accuracy score.
    """
    y_pred = to_float(y_pred)

    y_pred_labels = [p.index(max(p)) for p in y_pred]
    y_true_labels = [t.index(max(t)) for t in y_true]
    
    correct = sum(1 for p, t in zip(y_pred_labels, y_true_labels) if p == t)
    return correct / len(y_true_labels)

def f1_score(y_pred: List[List[float]], y_true: List[List[float]]) -> float:
    """
    Calculates the macro F1 score for multi-class classification.
    Args:
        y_pred (List[List[float]]): Predicted probabilities.
        y_true (List[List[float]]): Ground truth labels (one-hot encoded).
    Returns:
        float: The macro F1 score.
    """
    y_pred = to_float(y_pred)
    y_pred_labels = [p.index(max(p)) for p in y_pred]
    y_true_labels = [t.index(max(t)) for t in y_true]
    
    num_classes = len(y_true[0])
    f1_scores = []

    for c in range(num_classes):
        tp = sum(1 for p, t in zip(y_pred_labels, y_true_labels) if p == c and t == c)
        fp = sum(1 for p, t in zip(y_pred_labels, y_true_labels) if p == c and t != c)
        fn = sum(1 for p, t in zip(y_pred_labels, y_true_labels) if p != c and t == c)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1)
        
    return sum(f1_scores) / num_classes if num_classes > 0 else 0.0

def r2_score(y_pred: List[List[float]], y_true: List[List[float]]) -> float:
    """
    Calculates the R-squared (coefficient of determination) regression score.
    Args:
        y_pred (List[List[float]]): Predicted values, as single-element lists.
        y_true (List[List[float]]): Ground truth values, as single-element lists.
    Returns:
        float: The R-squared score.
    """
    y_pred_flat = [p[0] for p in y_pred]
    y_true_flat = [t[0] for t in y_true]

    y_true_mean = sum(y_true_flat) / len(y_true_flat)
    ss_total = sum((y - y_true_mean) ** 2 for y in y_true_flat)
    ss_residual = sum((t - p) ** 2 for t, p in zip(y_true_flat, y_pred_flat))
    
    if ss_total == 0:
        # Handle the case where all true values are the same.
        # R-squared is not well-defined here; 1.0 if predictions are also perfect, else 0.0 or undefined.
        return 1.0 if ss_residual == 0 else 0.0

    return 1 - (ss_residual / ss_total)