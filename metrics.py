from autograd import Value
from typing import List, Optional


def binary_crossentropy_loss(y_pred_probs : List[Value], y_true: List[int]) -> Value:
    assert len(y_pred_probs) == len(y_true)
    n = len(y_true)
    
    loss = sum([ - y_hat.log() * y  - (1 - y_hat).log() * (1 - y) 
             for y_hat, y in zip(y_pred_probs, y_true)])
    
    return (loss / n)




def mean_squared_error(y_pred: List[Value], y_true: List[float]) -> Value:
    assert len(y_pred) == len(y_true)
    n = len(y_true)