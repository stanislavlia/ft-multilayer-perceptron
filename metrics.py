from autograd import Value
from typing import List, Optional

#Losses
def binary_crossentropy_loss(y_pred_probs : List[Value], y_true: List[int]) -> Value:
    assert len(y_pred_probs) == len(y_true)
    n = len(y_true)
    
    loss = sum([ - y_hat.log() * y  - (1 - y_hat).log() * (1 - y) 
             for y_hat, y in zip(y_pred_probs, y_true)])
    
    return (loss / n)

def mean_squared_error_loss(y_pred: List[Value], y_true: List[float]) -> Value:
    """
    Computes the mean squared error over a batch:
        - y_pred: list of Value predictions
        - y_true: list of ground-truth floats
    """
    assert len(y_pred) == len(y_true), "Lengths of predictions and targets must match"
    n = len(y_true)
    
    loss = sum([(y_p - yt_i)**2  for yt_i, y_p in zip(y_true, y_pred)]) / n

    return loss

#Metrics

def accuracy_score(y_pred_probs: List[Value], y_true: List[int], threshold: float = 0.5) -> float:
    """
    Binary classification accuracy at given threshold.
    Returns fraction of correct predictions.
    """
    assert len(y_pred_probs) == len(y_true), "Lengths of predictions and targets must match"
    preds = [1 if y_hat.val >= threshold else 0 for y_hat in y_pred_probs]
    correct = sum(1 for p, t in zip(preds, y_true) if p == t)
    return correct / len(y_true)


def f1_score(y_pred_probs: List[Value], y_true: List[int], threshold: float = 0.5) -> float:
    """
    Binary classification F1 score at given threshold.
    Returns 0.0 if no positive predictions or no true positives.
    """
    assert len(y_pred_probs) == len(y_true), "Lengths of predictions and targets must match"
    preds = [1 if y_hat.val >= threshold else 0 for y_hat in y_pred_probs]
    tp = sum(1 for p, t in zip(preds, y_true) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(preds, y_true) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(preds, y_true) if p == 0 and t == 1)
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def r2_score(y_pred: List[Value], y_true: List[float]) -> float:
    """
    Coefficient of determination (R^2) for regression outputs.
    Returns 0.0 if true targets have zero variance.
    """
    assert len(y_pred) == len(y_true), "Lengths of predictions and targets must match"
    y_vals = [yp.val for yp in y_pred]
    n = len(y_true)
    mean_true = sum(y_true) / n
    ss_res = sum((t - yp) ** 2 for yp, t in zip(y_vals, y_true))
    ss_tot = sum((t - mean_true) ** 2 for t in y_true)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
