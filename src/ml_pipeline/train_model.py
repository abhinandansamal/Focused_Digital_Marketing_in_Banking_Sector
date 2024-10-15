from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from ml_pipeline.utils import max_val_index
from ml_pipeline.model_evaluation import evaluate_model

import warnings
warnings.simplefilter(action="ignore")

# create a function to train all the models
def train_model(X_train, y_train, X_test, y_test):
    """
    Trains multiple machine learning models and evaluates them on the test set, selecting the best model based on accuracy score.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training feature set.
    y_train : pandas.Series or numpy.ndarray
        Target labels for the training set.
    X_test : pandas.DataFrame or numpy.ndarray
        Test feature set.
    y_test : pandas.Series or numpy.ndarray
        Target labels for the test set.
    
    Returns:
    --------
    final_model : sklearn model object
        The model with the highest accuracy score on the test set.
    max_score : float
        The highest accuracy score achieved by any model on the test set.
    
    Process:
    --------
    1. Initializes a dictionary of models (Logistic Regression, Naive Bayes, SVM, Decision Tree, and Random Forest).
    2. Each model is trained on the training data (X_train, y_train).
    3. Each trained model is evaluated on the test data (X_test, y_test) using the `evaluate_model` function, with accuracy as the metric.
    4. The model with the highest accuracy score is selected as the final model.
    """
    model_dict = {"logistic_reg": LogisticRegression(solver="liblinear"),
                  "naive_bayes": GaussianNB(),
                  "svm_model": SVC(gamma=0.25, C=10),
                  "decision_tree": DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42),
                  "rfcl": RandomForestClassifier(random_state=42)
                  }
    
    fitted_model = []
    score = [] # for recall score


    for model_name in list(model_dict.keys()):
        model = model_dict[model_name]
        fitted_model.append(model.fit(X_train, y_train))
        score.append(evaluate_model(y_test, model.predict(X_test), "recall_score"))
    
    max_test = max_val_index(score) # for maximum recall score amongst all the models trained
    max_score = max_test[0]
    max_score_index = max_test[1]
    final_model = fitted_model[max_score_index]

    return final_model, max_score