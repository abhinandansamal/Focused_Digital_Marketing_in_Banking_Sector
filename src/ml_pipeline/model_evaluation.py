from sklearn import metrics
from sklearn.metrics import recall_score

# function to calculate the accuracy score of a model
def evaluate_model(y_test, y_pred, method):
    """
    Evaluates the performance of a model based on the specified evaluation method.
    
    Parameters:
    -----------
    y_test : pandas.Series or numpy.ndarray
        True labels for the test data.
    y_pred : pandas.Series or numpy.ndarray
        Predicted labels from the model.
    method : str
        The evaluation metric to use. Currently, only "recall_score" is supported.
    
    Returns:
    --------
    score : float
        The accuracy score of the model if "recall_score" is used as the method.
    
    Raises:
    -------
    ValueError
        If the specified method is not supported.

    Notes:
    ------
    - Currently, the function only supports accuracy score as the evaluation metric.
    - If an unsupported method is passed, an error message will be printed.
    """
    if method == "recall_score":
        score = recall_score(y_test, y_pred)
    else:
        raise ValueError("Only 'recall_score' is supported as an evaluation metric.")

    return score
    